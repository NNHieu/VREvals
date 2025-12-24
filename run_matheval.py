import os
import argparse
import transformers
from sampler.chat_completion_sampler import ChatCompletionSampler, DivFirstSampler
from common import model_path_to_short_name
from math_evaluator import MathEval

from dotenv import load_dotenv
load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Run direct generation for various datasets and models.")
    
    parser.add_argument(
        '--k_list', 
        type=lambda s: [int(item) for item in s.split(',')],
        default=[1],
        help="Comma-separated list of integers for k in pass@k. E.g.: 1,2"
    )

    parser.add_argument(
        '--dataset_name', 
        type=str, 
        required=True, 
        choices=[
            'gpqa',
            'math500',
            'aime',
            'amc',
            'livecode',
            'nq',
            'triviaqa',
            'hotpotqa',
            '2wiki',
            'musique',
            'bamboogle',
            'medmcqa',
            'pubhealth',
            'simp_math',
            'omega_func_area',
            'omega_func_area_out',
            'gsm8k',
        ],
        help="Name of the dataset to use."
    )
    
    parser.add_argument(
        '--split', 
        type=str, 
        required=True, 
        choices=['test', 'diamond', 'main', 'extended', 'test_2', 'train'],
        help="Dataset split to use."
    )

    parser.add_argument(
        '--port', 
        type=int, 
    )
    
    parser.add_argument(
        '--subset_num', 
        type=int, 
        default=-1, 
        help="Number of examples to process. Defaults to all if not specified."
    )
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=True,
        help="Path to the pre-trained model."
    )
    
    parser.add_argument(
        '--temperature', 
        type=float, 
        default=1.0, 
        help="Sampling temperature."
    )
    
    parser.add_argument(
        '--top_p', 
        type=float, 
        default=None, 
        help="Top-p sampling parameter."
    )
    
    parser.add_argument(
        '--top_k', 
        type=int, 
        default=None, 
        help="Top-k sampling parameter."
    )
    
    parser.add_argument(
        '--repetition_penalty', 
        type=float, 
        default=None, 
        help="Repetition penalty. If not set, defaults based on the model."
    )
    
    parser.add_argument(
        '--max_tokens', 
        type=int, 
        default=32768, 
        help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset."
    )

    parser.add_argument(
        '--prefix_thinking',
        type=str,
        default=None,
    )

    parser.add_argument(
        '--n_threads',
        type=int,
        default=2,
        help="Number of threads to use for math evaluation."
    )

    args = parser.parse_args()
    
    if args.max_tokens is None:
        if 'qwq' in args.model_path.lower() or 'deepseek' in args.model_path.lower() or 'sky-t1' in args.model_path.lower():
            if args.dataset_name in ['aime', 'amc', 'livecode']:
                args.max_tokens = 32768
            else:
                args.max_tokens = 25600
        else:
            args.max_tokens = 3096

    # Set default repetition_penalty if not provided
    if args.repetition_penalty is None:
        args.repetition_penalty = 1.05 if 'qwq' in args.model_path.lower() or 'deepseek' in args.model_path.lower() or 'sky-t1' in args.model_path.lower() else 1.0
    
    model_short_name = model_path_to_short_name(args.model_path)
    sub_folder = 'default'
    if args.prefix_thinking:
        sub_folder = args.prefix_thinking.replace("\n", "_").replace(" ", "_")
        sub_folder = f'prefix_thinking/{sub_folder}'
    if model_short_name in ['qwq', 'ds-llama-8b', 'ds-qwen-7b', 'ds-qwen-32b', 'sky-t1']:
        if args.dataset_name in ['math500', 'gpqa', 'aime', 'amc', 'livecode', 'simp_math', 'gsm8k']:
            args.output_dir = f'./outputs/{sub_folder}/{args.dataset_name}.{model_short_name}.direct.{sub_folder}'
        else:
            args.output_dir = f'./outputs/{sub_folder}/runs.qa/{args.dataset_name}.{model_short_name}.direct'
    else:
        args.output_dir = f'./outputs/{sub_folder}/runs.baselines/{args.dataset_name}.{model_short_name}.direct'
    os.makedirs(args.output_dir, exist_ok=True)
    return args

def main():
    args = parse_args()
    eval = MathEval(args.dataset_name, 
                    args.split, 
                    args.k_list, 
                    args.subset_num, 
                    step_by_step_prompt=True,
                    n_threads=args.n_threads)
    match args.model_path:
        case "custom-qwen-math-1.5b":
            tokenizer = transformers.AutoTokenizer.from_pretrained("nnheui/qwen2.5-math-1.5b-thinking-distilled", trust_remote_code = True, revision="ckpt_48")
            sampler = ChatCompletionSampler(
                model="nnheui/qwen2.5-math-1.5b-thinking-distilled",
                system_message=None,
                base_url=f"http://localhost:{args.port}/v1",
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                api_key_name="VLLM_TOKEN",
                support_async=True,
                tokenizer=tokenizer,
                question_prompt_template="Can you solve the following math problem? {} Please reason step by step, and put your final answer within \\boxed{{}}.",
                prefix_thinking="<think>\n" + (args.prefix_thinking if args.prefix_thinking is not None else ""),
            )
        case "qwen-math-7b":
            # tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B", trust_remote_code = True)
            sampler = ChatCompletionSampler(
                model="Qwen/Qwen2.5-Math-7B",
                system_message=None,
                base_url=f"http://localhost:{args.port}/v1",
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                api_key_name="VLLM_TOKEN",
                support_async=True,
                # tokenizer=tokenizer,
                question_prompt_template="Can you solve the following math problem? {} Please reason step by step, and put your final answer within \\boxed{{}}.",
            )
        case "qwen-math-7b-divfirst":
            # tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B", trust_remote_code = True)
            sampler = DivFirstSampler(
                first_topk=10,
                model="Qwen/Qwen2.5-Math-7B",
                system_message=None,
                base_url=f"http://localhost:{args.port}/v1",
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                api_key_name="VLLM_TOKEN",
                support_async=True,
                # tokenizer=tokenizer,
                question_prompt_template="Can you solve the following math problem? {} Please reason step by step, and put your final answer within \\boxed{{}}."
            )
        case "qwen-math-grpo-7b":
            tokenizer = transformers.AutoTokenizer.from_pretrained("stellalisy/rethink_rlvr_reproduce-ground_truth-qwen2.5_math_7b-lr5e-7-kl0.00-step150", trust_remote_code = True)
            sampler = ChatCompletionSampler(
                model="stellalisy/rethink_rlvr_reproduce-ground_truth-qwen2.5_math_7b-lr5e-7-kl0.00-step150",
                system_message=None,
                base_url=f"http://localhost:{args.port}/v1",
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                api_key_name="VLLM_TOKEN",
                support_async=True,
                tokenizer=tokenizer,
                question_prompt_template="Can you solve the following math problem? {} Please reason step by step, and put your final answer within \\boxed{{}}.",
                prefix_thinking=args.prefix_thinking,
            )
        case "qwen-math-grpo-7b-divfirst":
            tokenizer = transformers.AutoTokenizer.from_pretrained("stellalisy/rethink_rlvr_reproduce-ground_truth-qwen2.5_math_7b-lr5e-7-kl0.00-step150", trust_remote_code = True)
            sampler = DivFirstSampler(
                first_topk=10,
                model="stellalisy/rethink_rlvr_reproduce-ground_truth-qwen2.5_math_7b-lr5e-7-kl0.00-step150",
                system_message=None,
                base_url=f"http://localhost:{args.port}/v1",
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                api_key_name="VLLM_TOKEN",
                support_async=True,
                tokenizer=tokenizer,
                question_prompt_template="Can you solve the following math problem? {} Please reason step by step, and put your final answer within \\boxed{{}}."
            )
        case "ds-qwen-7b":
            tokenizer = transformers.AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", trust_remote_code = True)
            sampler = ChatCompletionSampler(
                model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                system_message=None,
                base_url=f"http://localhost:{args.port}/v1",
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                api_key_name="VLLM_TOKEN",
                support_async=True,
                tokenizer=tokenizer,
                question_prompt_template="Can you solve the following math problem? {} Put your final answer within \\boxed{{}}.",
                prefix_thinking=args.prefix_thinking,
            )
        case "ds-qwen-7b-divfirst":
            tokenizer = transformers.AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", trust_remote_code = True)
            sampler = DivFirstSampler(
                first_topk=10,
                model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                system_message=None,
                base_url=f"http://localhost:{args.port}/v1",
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                api_key_name="VLLM_TOKEN",
                support_async=True,
                tokenizer=tokenizer,
                question_prompt_template="Can you solve the following math problem? {} Put your final answer within \\boxed{{}}."
            )
        case "qwen-1.5b-inst":
            tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", 
                                                                    trust_remote_code = True)
            sampler = ChatCompletionSampler(
                model="Qwen/Qwen2.5-1.5B-Instruct",
                system_message=None,
                base_url=f"http://localhost:{args.port}/v1",
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                api_key_name="VLLM_TOKEN",
                support_async=True,
                tokenizer=tokenizer,
                question_prompt_template="Can you solve the following math problem? {} Put your final answer within \\boxed{{}}.",
                prefix_thinking=args.prefix_thinking,
            )
        case "qwen-math-1.5b-inst":
            tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct", trust_remote_code = True)
            sampler = ChatCompletionSampler(
                model="Qwen/Qwen2.5-Math-1.5B-Instruct",
                system_message=None,
                base_url=f"http://localhost:{args.port}/v1",
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                api_key_name="VLLM_TOKEN",
                support_async=True,
                tokenizer=tokenizer,
                question_prompt_template="Can you solve the following math problem? {} Put your final answer within \\boxed{{}}.",
                prefix_thinking=args.prefix_thinking,
            )
        case "qwen-math-1.5b-inst-distilled":
            tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct", 
                                                                    trust_remote_code = True,
                                                                    eos_token="<|im_end|>")
            sampler = ChatCompletionSampler(
                model="nnheui/thinking_distilled-qwen2.5-1.5b-math-instruct-mot",
                system_message=None,
                base_url=f"http://localhost:{args.port}/v1",
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                api_key_name="VLLM_TOKEN",
                support_async=True,
                tokenizer=tokenizer,
                question_prompt_template="Can you solve the following math problem? {} Put your final answer within \\boxed{{}}.",
                prefix_thinking="<think>\n" + (args.prefix_thinking if args.prefix_thinking is not None else ""),
            )
        case "gemma2bit":
            tokenizer = transformers.AutoTokenizer.from_pretrained("google/gemma-2-2b-it", 
                                                                    trust_remote_code = True)
            sampler = ChatCompletionSampler(
                model="google/gemma-2-2b-it",
                system_message=None,
                base_url=f"http://localhost:{args.port}/v1",
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                api_key_name="VLLM_TOKEN",
                support_async=True,
                tokenizer=tokenizer,
                question_prompt_template="Can you solve the following math problem? {} Put your final answer within \\boxed{{}}.",
                prefix_thinking=args.prefix_thinking,
            )
        case "gemma2bit-distilled":
            tokenizer = transformers.AutoTokenizer.from_pretrained("google/gemma-2-2b-it", 
                                                                    trust_remote_code = True)
            sampler = ChatCompletionSampler(
                model="nnheui/thinking_distilled-gemma-2-2b-it-mot",
                system_message=None,
                base_url=f"http://localhost:{args.port}/v1",
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                api_key_name="VLLM_TOKEN",
                support_async=True,
                tokenizer=tokenizer,
                question_prompt_template="Can you solve the following math problem? {} Put your final answer within \\boxed{{}}.",
                prefix_thinking=args.prefix_thinking,
            )
        case _:
            # raise Exception(f"Unrecognized eval type: {eval_name}")
            sampler = ChatCompletionSampler(
                model=args.model_path,
                system_message=None,
                base_url=f"http://localhost:{args.port}/v1",
                top_p=args.top_p,
                top_k=args.top_k,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                api_key_name="VLLM_TOKEN",
                support_async=True,
                prefix_thinking=args.prefix_thinking,
            )

    eval(sampler, args.output_dir)

if __name__ == "__main__":
    main()
