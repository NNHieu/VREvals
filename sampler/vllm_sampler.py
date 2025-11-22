import time
from typing import Any, List, Optional

from eval_types import MessageList, ResponseChoice, SamplerBase, SamplerResponse

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

class VLLMSampler(SamplerBase):
    """
    Sampler using vllm engine.
    """
    def __init__(
        self,
        model_name_or_path: str,
        tensor_parallel_size: int = 1,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        top_k: int = 20,
        top_p: float = 0.8,
        tokenizer: Any = None,
        question_prompt_template: Optional[Any] = None,
        trust_remote_code: bool = False,
        **llm_kwargs
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.tokenizer = tokenizer
        self.question_prompt_template = question_prompt_template
        self.model_name_or_path = model_name_or_path

        # Defensive import to allow fail if vllm not available
        if LLM is None:
            raise ImportError(
                "vllm is not installed. Please install vllm to use VLLMSampler."
            )
        # vllm.LLM can be heavy to initialize, so allow multiple kwargs
        self.llm = LLM(
            model=model_name_or_path,
            tokenizer=tokenizer,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            **llm_kwargs
        )

    def can_apply_chat_template(self):
        # If tokenizer supports chat template
        return (self.tokenizer is not None) and (hasattr(self.tokenizer, "apply_chat_template"))

    def apply_chat_template(self, messages):
        assert self.can_apply_chat_template()
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def convert_message_list_to_prompt(self, message_list: MessageList) -> str:
        # By default, just join as plain text, or use question_prompt_template if available
        if self.can_apply_chat_template():
            return self.apply_chat_template(message_list)
        elif self.question_prompt_template is not None:
            return self.question_prompt_template(message_list)
        else:
            # Fallback: naive concatenation
            return "\n".join([f"{msg['role']}: {msg['content']}" for msg in message_list])

    def complete(self, prompts: str, n: int = 1) -> SamplerResponse:
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_k=self.top_k,
            top_p=self.top_p,
            n=n,
        )
        resonpses = []
        try:
            # vllm supports batch generation, each prompt can be replicated n times
            outputs = self.llm.generate(prompts, sampling_params)
            for out in outputs:
                choices = []
                # Each output may contain multiple generations
                for g in out.outputs:
                    response_text = g.text
                    choices.append(ResponseChoice(response_text=response_text))
                # # clip to n if more generated
                # choices = choices[:n]
                resonpses.append(
                    SamplerResponse(
                        status="OK",
                        choices=choices,
                        response_metadata={},
                        actual_queried_message_list=[{"prompt": out.prompt}],
                    )
                )
            return resonpses
        except Exception as e:
            print("VLLM Exception during sampling:", e)
            return [SamplerResponse(
                    status=f"Error: {e!r}",
                    choices=[],
                    response_metadata={},
                    actual_queried_message_list=[{"prompt": ""}],
                )]

    def __call__(self, message_list: MessageList, n: int = 1) -> SamplerResponse:
        # Apply chat template to convert to prompt as needed
        prompt = self.convert_message_list_to_prompt(message_list)
        return self.complete(prompt, n=n)

