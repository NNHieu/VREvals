import os, time

from collections import defaultdict
from eval_types import Eval, PerGenerationEvalResult, SamplerBase, SingleEvalResult
from evaluate import evaluate_predictions
import numpy as np
import common
# from lcb_runner.evaluation.pass_k_utils import estimate_pass_at_k
import json
from pydantic.json import pydantic_encoder


def get_task_instruction_math(question, sampler, step_by_step=False):
    if hasattr(sampler, "question_prompt_template") and sampler.question_prompt_template is not None:
        prompt = sampler.question_prompt_template.format(question)
    else:
        if not step_by_step:
            prompt = (
                'Please answer the following math question. '
                'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
                f'Question:\n{question}\n\n'
            )
        else:
            prompt = (
                'Please answer the following math question. You should think step by step to solve it.\n\n'
                'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
                f'Question:\n{question}\n\n'
            )
    if hasattr(sampler, "can_apply_chat_template") and sampler.can_apply_chat_template():
        prompt = [{"role": "user", "content": prompt}]
        prompt = sampler.apply_chat_template(prompt)
    return prompt

class MathEval(Eval):
    def __init__(self, dataset_name, split, k_list, subset_num=None, step_by_step_prompt=False, n_threads=1, data_root_dir="./data") -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.k_list = k_list
        self.n_threads = n_threads
        self.step_by_step_prompt=step_by_step_prompt

        # Paths to datasets
        dataset_paths = {
            'math500': f'{data_root_dir}/MATH-500/{split}.json',
            'gpqa': f'{data_root_dir}/GPQA/{split}.json',
            'aime24': f'{data_root_dir}/AIME/{split}.json',
            'aime25': f'{data_root_dir}/AIME2025/{split}.json',
            'amc': f'{data_root_dir}/AMC/{split}.json',
            'livecode': f'{data_root_dir}/LiveCodeBench/{split}.json',
            'simp_math': f'{data_root_dir}/SIMPMATH/{split}.json',
            'omega_func_area': '{data_root_dir}/omega/algebra_func_area.json',
            'omega_func_area_out': '{data_root_dir}/omega/algebra_func_area_out.json',
            'gsm8k': f'{data_root_dir}/gsm8k/{split}.json',
            'imobench': f'{data_root_dir}/IMOBench/{split}.json',
        }

        if dataset_name in dataset_paths:
            data_path = dataset_paths[dataset_name]
        else:
            raise ValueError(f"Unsupported dataset_name: {dataset_name}")

        # Load data
        # prepare input
        self.examples = []
        with open(data_path, mode='r', encoding='utf-8') as json_file:
            filtered_data = json.load(json_file)
        if subset_num is None:
            subset_num = 0
        for i, item in enumerate(filtered_data):
            if subset_num > 0 and i >= subset_num:
                break
            self.examples.append(item)
        print("Num eval samples is ", len(self.examples))
    
    def __call__(self, sampler: SamplerBase, output_dir):
        mode = 'gen'
        for item in self.examples:
            question = item["Question"]
            item['prompt'] = get_task_instruction_math(question, sampler, step_by_step=self.step_by_step_prompt)

        async def fn(row: dict):
            prompt = row['prompt']
            response = await sampler.acomplete(prompt, n=max(self.k_list))
            labeled_answer = row["answer"]
            
            outputs = []
            for choice in response.choices:
                metric, pred_answer = evaluate_predictions(output=choice.response_text, 
                                                           labeled_answer=labeled_answer, 
                                                           mode=mode)                
                gen_eval_result = PerGenerationEvalResult(
                    generation=choice.response_text, 
                    pred_answer=pred_answer, 
                    metric=metric,
                )

                outputs.append(gen_eval_result)
            return SingleEvalResult(
                html=None,
                score=None,
                convo=None,
                metrics={},
                example_level_metadata={
                    "outputs": outputs,
                    "num_samples": len(outputs),
                    "num_math_equal": sum([eval_result.metric['math_equal'] for eval_result in outputs])
                }
            )
        
        t_start = time.time()
        # results = common.map_with_progress(
        #     fn,
        #     self.examples,
        #     num_threads=self.n_threads,
        #     pbar=True,
        # )
        results = common.gather_with_progress(
            fn,
            self.examples,
            num_threads=self.n_threads,
            pbar=True,
        )

        avg_em, avg_acc, avg_f1, avg_math = [], [], [], []
        num_valid_answer = 0

        # compute pass@k using math_equal
        num_samples = np.array([item.example_level_metadata['num_samples'] for item in results])
        num_correct = np.array([item.example_level_metadata['num_math_equal'] for item in results])
        detail_pass_at_k = {
            f"pass@{k}": estimate_pass_at_k(num_samples, num_correct, k).tolist()
            for k in self.k_list
            if (num_samples >= k).all()
        }
        pass_at_k = {
            f"pass@{k}": estimate_pass_at_k(num_samples, num_correct, k).mean()
            for k in self.k_list
            if (num_samples >= k).all()
        }

        for input_item, output_item in zip(self.examples, results):
            metric = output_item.example_level_metadata['outputs'][0].metric
            pred_answer = output_item.example_level_metadata['outputs'][0].pred_answer
            avg_em.append(metric['em'])
            avg_acc.append(metric['acc'])
            avg_f1.append(metric['f1'])
            avg_math.append(metric['math_equal'])
            my_method_valid = pred_answer != ''
            if my_method_valid:
                num_valid_answer += 1
            input_item.update(output_item.example_level_metadata)
            
        total_time = time.time() - t_start
        t = time.localtime()
        result_json_name = f'{self.split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.json'
        metrics_json_name = f'{self.split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.metrics.json'

        # Compute overall metrics
        overall_results = {
            'em': np.mean(avg_em) if len(avg_em) > 0 else 0.0,
            'acc': np.mean(avg_acc) if len(avg_acc) > 0 else 0.0,
            'f1': np.mean(avg_f1) if len(avg_f1) > 0 else 0.0,
            'math_equal': np.mean(avg_math) if len(avg_em) > 0 else 0.0,
            'num_valid_answer': f'{num_valid_answer} of {len(results)}',
            'query_latency': f'{(total_time / len(results) * 1000):.0f} ms',
            'detail_pass_at_k': detail_pass_at_k,
            'pass_at_k': pass_at_k,
        }
        final_metrics = {'overall': overall_results}

        # Save prediction results and metrics
        with open(os.path.join(output_dir, result_json_name), mode='w', encoding='utf-8') as json_file:
            json.dump(self.examples, json_file, indent=4, ensure_ascii=False, default=pydantic_encoder)

        with open(os.path.join(output_dir, metrics_json_name), mode='w', encoding='utf-8') as json_file:
            json.dump(final_metrics, json_file, indent=4, ensure_ascii=False, default=pydantic_encoder)





