import os
import time
from typing import Any

import openai
from openai import OpenAI, AsyncOpenAI

from ..eval_types import MessageList, ResponseChoice, SamplerBase, SamplerResponse

OPENAI_SYSTEM_MESSAGE_API = "You are a helpful assistant."
OPENAI_SYSTEM_MESSAGE_CHATGPT = (
    "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."
    + "\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01"
)

SYSTEM_MESSAGE_OTHER = (
    "You are a helpful assistant."
    + "\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01" 
)

import asyncio

class ChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        top_k=20,
        top_p=0.8,
        base_url: str | None = None,
        api_key_name = "OPENAI_API_KEY",
        support_async = False,
        tokenizer=None,
        question_prompt_template=None,
        prefix_thinking=None,
    ):
        self.api_key_name = api_key_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=os.environ.get(self.api_key_name),
            timeout=6000,
        )
        if support_async:
            self.async_client = AsyncOpenAI(
                base_url=base_url,
                api_key=os.environ.get(self.api_key_name),
                timeout=6000,
            )
        else:
            self.async_client = None
        # using api_key=os.environ.get("OPENAI_API_KEY")  # please set your API_KEY
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"
        self.top_k=top_k
        self.top_p=top_p
        
        self.tokenizer = tokenizer
        self.question_prompt_template = question_prompt_template
        self.prefix_thinking = prefix_thinking

    def _handle_image(
        self,
        image: str,
        encoding: str = "base64",
        format: str = "png",
        fovea: int = 768,
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role),
                "content": content}
    
    def can_apply_chat_template(self):
        return (self.tokenizer is not None) and (hasattr(self.tokenizer, "apply_chat_template"))
    
    def apply_chat_template(self, messages):
        assert self.can_apply_chat_template()
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if self.prefix_thinking is not None:
            prompt += self.prefix_thinking
        return prompt

    def complete(self, prompt, n=1) -> SamplerResponse:
        try:
            response = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                n=n,
                extra_body={"top_k": self.top_k}
            )
            content = response.choices[0].text
            if content is None:
                raise ValueError("OpenAI API returned empty response; retrying")
            return SamplerResponse(
                status="OK",
                choices=[ResponseChoice(response_text=choice.text) for choice in response.choices],
                response_metadata={"usage": response.usage},
                actual_queried_message_list=[{"prompt": prompt}],
            )
        except openai.BadRequestError as e:
            print("Bad Request Error", e)
            return SamplerResponse(
                status="No response (bad request).",
                choices=[],
                response_metadata={"usage": None},
                actual_queried_message_list=[{"prompt": prompt}],
            )
        except Exception as e:
            # handle async/regular retry for complete if needed (mirroring sync behavior)
            raise

    async def acomplete(self, prompt, n=1) -> SamplerResponse:
        """Asynchronous version of complete."""
        trial = 0
        response = None
        while True:
            try:
                client = getattr(self, "async_client", None)
                if client is None:
                    raise RuntimeError("AsyncOpenAI client not initialized or openai python package too old.")
                response = await client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    n=n,
                    extra_body={"top_k": self.top_k}
                )
                content = response.choices[0].text
                if content is None:
                    raise ValueError("OpenAI API returned empty response; retrying")
                return SamplerResponse(
                    status="OK",
                    choices=[ResponseChoice(response_text=choice.text) for choice in response.choices],
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=[{"prompt": prompt}],
                )
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return SamplerResponse(
                    status="No response (bad request).",
                    choices=[],
                    response_metadata={"usage": None},
                    actual_queried_message_list=[{"prompt": prompt}],
                )
            except Exception as e:
                exception_backoff = 2 ** trial
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec (async)",
                    e,
                )
                await asyncio.sleep(exception_backoff)
                trial += 1

    def __call__(self, message_list: MessageList, n=1) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list
        trial = 0
        response = None
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_completion_tokens=self.max_tokens,
                    top_p=self.top_p,
                    n=n,
                    extra_body={
                        "top_k": self.top_k
                    }
                )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("OpenAI API returned empty response; retrying")
                return SamplerResponse(
                    status="OK",
                    choices=[ResponseChoice(response_text=choice.message.content) for choice in response.choices],
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=message_list,
                )
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return SamplerResponse(
                    status="No response (bad request).",
                    choices=[],
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                if isinstance(response, str):
                    content = response
                    return SamplerResponse(
                        status="OK",
                        choices=[ResponseChoice(response_text=content)],
                        response_metadata={"usage": None},
                        actual_queried_message_list=message_list,
                    )
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception

    async def acall(self, message_list: MessageList, n=1) -> SamplerResponse:
        """
        Async version of __call__, to sample from the OpenAI chat completion API asynchronously.

        Returns a SamplerResponse.
        """
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list
        trial = 0
        response = None
        while True:
            try:
                client = getattr(self, "async_client", None)
                if client is None:
                    raise RuntimeError("AsyncOpenAI client not initialized or openai python package too old.")
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_completion_tokens=self.max_tokens,
                    top_p=self.top_p,
                    n=n,
                    extra_body={
                        "top_k": self.top_k
                    }
                )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("OpenAI API returned empty response; retrying")
                return SamplerResponse(
                    status="OK",
                    choices=[ResponseChoice(response_text=choice.message.content) for choice in response.choices],
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=message_list,
                )
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return SamplerResponse(
                    status="No response (bad request).",
                    choices=[],
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                if isinstance(response, str):
                    content = response
                    return SamplerResponse(
                        status="OK",
                        choices=[ResponseChoice(response_text=content)],
                        response_metadata={"usage": None},
                        actual_queried_message_list=message_list,
                    )
                exception_backoff = 2**trial  # exponential back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec (async)",
                    e,
                )
                await asyncio.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception


class DivFirstSampler(ChatCompletionSampler):
    def __init__(
        self,
        first_topk:int,
        model: str = "gpt-3.5-turbo",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        top_k=20,
        top_p=0.8,
        base_url: str | None = None,
        api_key_name = "OPENAI_API_KEY",
        support_async = False,
        tokenizer=None,
        question_prompt_template=None,
    ):
        super().__init__(
            model=model,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            base_url=base_url,
            api_key_name=api_key_name,
            support_async=support_async,
            tokenizer=tokenizer,
            question_prompt_template=question_prompt_template,
        )
        self.first_topk = first_topk


    async def acomplete(self, prompt, n=1) -> SamplerResponse:
        """Asynchronous version of complete."""
        trial = 0
        response = None
        while True:
            try:
                client = getattr(self, "async_client", None)
                if client is None:
                    raise RuntimeError("AsyncOpenAI client not initialized or openai python package too old.")
                
                # get_first_topk_tokens
                response = await client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    temperature=0,
                    max_tokens=1,
                    top_p=1,
                    n=1,
                    extra_body={
                        "top_k": self.first_topk,
                        "logprobs": self.first_topk,
                        "top_logprobs": self.first_topk,
                    }
                )
                choice = response.choices[0]
                topk_tokens = {'decoded': [], 'probs': [], 'token_id': [], 'logprobs': []}
                # print(prompt)
                # print(choice)
                for token, logprob in choice.logprobs.top_logprobs[0].items():
                    # print(f"Token: {token_info.token}, Logprob: {token_info.logprob}, Top Logprobs: {token_info.top_logprobs}")
                    topk_tokens['decoded'].append(token)
                    topk_tokens['logprobs'].append(logprob)
                
                # INSERT_YOUR_CODE
                # Repeat topk_tokens to ensure all fields have length n
                if len(topk_tokens['decoded']) == 0:
                    raise RuntimeError("No top-k tokens returned from OpenAI API; cannot proceed.")
                    # Defensive: fallback in case no topk tokens
                    # topk_tokens = {'decoded': ["" for _ in range(n)], 'probs': [0.0 for _ in range(n)], 'token_id': [None for _ in range(n)], 'logprobs': [float('-inf') for _ in range(n)]}
                else:
                    num_repeats = (n + len(topk_tokens['decoded']) - 1) // len(topk_tokens['decoded'])
                    for key in topk_tokens:
                        topk_tokens[key] = (topk_tokens[key] * num_repeats)[:n]
                
                # print(topk_tokens['decoded'])

                appended_prompts = [prompt + token for token in topk_tokens['decoded']] # K questions.
                responses = await client.completions.create(
                    model=self.model,
                    prompt=appended_prompts,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    n=1,
                    extra_body={"top_k": self.top_k}
                )
                # Collect all choices' text from the responses
                choices = []
                for first_token, choice in zip(topk_tokens['decoded'], responses.choices):
                    if choice.text is None:
                        raise ValueError("OpenAI API returned empty response; retrying")
                    choices.append(ResponseChoice(response_text=first_token + choice.text))
                return SamplerResponse(
                    status="OK",
                    choices=choices,
                    response_metadata={"usage": responses.usage},
                    actual_queried_message_list=[{"prompt": prompt}],
                )
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return SamplerResponse(
                    status="No response (bad request).",
                    choices=[],
                    response_metadata={"usage": None},
                    actual_queried_message_list=[{"prompt": prompt}],
                )
            except Exception as e:
                exception_backoff = 2 ** trial
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec (async)",
                    e,
                )
                await asyncio.sleep(exception_backoff)
                trial += 1