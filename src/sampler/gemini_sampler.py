import os
import time
from typing import Any

from google import genai
from google.genai import types

from ..eval_types import MessageList, ResponseChoice, SamplerBase, SamplerResponse

GEMINI_SYSTEM_PROMPT = (
    "You are a helpful assistant."
    "\nKnowledge cutoff: 2023-12\nCurrent date: {currentDateTime}" 
).format(currentDateTime="2024-04-01")

class GeminiSampler(SamplerBase):
    """
    Sample from Google Gemini API
    """

    def __init__(
        self,
        model: str = "gemini-pro",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        reasoning_model: bool = False,  # Not used in Gemini
        reasoning_effort: str | None = None,  # Not used in Gemini
        api_key_env: str = "GEMINI_API_KEY",
    ):
        self.api_key_name = api_key_env
        self.api_key = os.environ.get(self.api_key_name)
        assert self.api_key, f"Please set {self.api_key_name}"
        # genai.configure(api_key=self.api_key)
        self.client = genai.Client(api_key=self.api_key)
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "base64"  # Gemini supports base64 images
        # Gemini does not support reasoning_model or reasoning_effort

    def _handle_image(
        self,
        image: str,
        encoding: str = "base64",
        format: str = "png",
        fovea: int = 768,
    ) -> dict[str, Any]:
        # Gemini expects images as base64-encoded bytes
        return {
            "type": "image",
            "data": image,
            "mime_type": f"image/{format}",
        }

    def _handle_text(self, text: str) -> dict[str, Any]:
        return {"text": text}

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        # Gemini expects a list of dicts with "role" and "parts"
        # "parts" is a list of dicts, each with "text" or "image"
        if isinstance(content, dict) and "type" in content:
            parts = [content]
        elif isinstance(content, list):
            parts = content
        else:
            parts = [{"text": str(content)}]
        return {"role": role, "parts": parts}

    def _convert_message_list(self, message_list: MessageList):
        # Convert message_list to Gemini's expected format
        gemini_messages = []
        for msg in message_list:
            role = msg.get("role", "user")
            if role == "assistant":
                role = "model"
            content = msg.get("content", "")
            if isinstance(content, list):
                parts = []
                for c in content:
                    if isinstance(c, dict) and "type" in c:
                        parts.append(c)
                    else:
                        parts.append({"text": str(c)})
            elif isinstance(content, dict) and "type" in content:
                parts = [content]
            else:
                parts = [{"text": str(content)}]
            gemini_messages.append({"role": role, "parts": parts})
        return gemini_messages

    def __call__(self, message_list: MessageList, n=1) -> SamplerResponse:
        # Gemini expects a list of messages, each with "role" and "parts"
        # if self.system_message:
        #     system_msg = self._pack_message("system", self.system_message)
        #     message_list = [system_msg] + message_list
        gemini_messages = self._convert_message_list(message_list)
        trial = 0
        while True:
            try:
                # model = genai.GenerativeModel(self.model)
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=gemini_messages,
                    # generation_config={
                    #     "temperature": self.temperature,
                    #     "max_output_tokens": self.max_tokens,
                    # },
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_message,
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens
                    )
                )
                # Gemini's response.text contains the output
                # print(response.candidates[0])
                response_text = response.candidates[0].content.parts[0].text
                return SamplerResponse(
                    status="OK",
                    choices=[ResponseChoice(response_text=response_text)],
                    response_metadata={},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                exception_backoff = 2 ** trial  # exponential backoff
                print(
                    f"Exception occurred, waiting and retrying (trial {trial}) after {exception_backoff} sec:",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception