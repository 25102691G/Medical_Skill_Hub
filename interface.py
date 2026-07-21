from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(Path(__file__).resolve().parent / ".env")


def _argument_or_env(
    args: Any,
    argument_name: str,
    environment_name: str,
    default: str = "",
) -> str:
    argument_value = getattr(args, argument_name, None)
    if argument_value:
        return str(argument_value).strip()
    return os.getenv(environment_name, default).strip()


class Openai_api:
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY or --openai_apikey is required.")
        if not model:
            raise ValueError("OPENAI_MODEL or --openai_model is required.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def get_completion(
        self,
        system_prompt: str,
        prompt: str,
        seed: int = 42,
    ) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            seed=seed,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return str(completion.choices[0].message.content or "")

    def openai_summarize(self, text: str) -> str:
        try:
            output = self.get_completion(
                (
                    "Assume you are a doctor. Summarize this medical article into "
                    "one paragraph, retaining only key messages and focusing on "
                    "phenotypes and related diseases."
                ),
                text,
            )
            if "not a medical-related page" in output.lower():
                return ""
            return output
        except Exception:
            print("Error in summarizing the text. Return the first 1000 characters.")
            return text[:1000]

    def mini_completion(
        self,
        system_prompt: str,
        prompt: str,
        seed: int = 42,
    ) -> str:
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            seed=seed,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return str(completion.choices[0].message.content or "")

    def get_embedding(
        self,
        text: str,
        model: str = "text-embedding-3-small",
    ) -> list[float]:
        return self.client.embeddings.create(
            input=[text],
            model=model,
        ).data[0].embedding


class deepseek_api:
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        self.model = model

    def get_completion(
        self,
        system_prompt: str,
        prompt: str,
        seed: int = 42,
    ) -> str:
        del seed
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        return str(completion.choices[0].message.content or "")


class gemini_api:
    def __init__(self, api_key: str, model: str):
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError(
                "Gemini requires the optional google-generativeai package."
            ) from exc

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def get_completion(
        self,
        system_prompt: str,
        prompt: str,
        seed: int = 42,
    ) -> str:
        del seed
        full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
        response = self.model.generate_content(full_prompt)
        return str(response.text)


class claude_api:
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("CLAUDE_API_KEY or --claude_apikey is required.")
        if not model:
            raise ValueError("CLAUDE_MODEL or --claude_model is required.")
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "Claude requires the optional anthropic package."
            ) from exc

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def get_completion(
        self,
        system_prompt: str,
        prompt: str,
        seed: int = 42,
    ) -> str:
        del seed
        message = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        return str(message.content[0].text)


class LLM_handler:
    def __init__(self, args: Any):
        provider = str(getattr(args, "model", "")).strip().lower()
        if provider == "openai":
            self.handler = Openai_api(
                _argument_or_env(args, "openai_apikey", "OPENAI_API_KEY"),
                _argument_or_env(args, "openai_model", "OPENAI_MODEL", "gpt-4o"),
            )
        elif provider == "gemini":
            self.handler = gemini_api(
                _argument_or_env(args, "gemini_apikey", "GEMINI_API_KEY"),
                _argument_or_env(
                    args,
                    "gemini_model",
                    "GEMINI_MODEL",
                    "gemini-2.0-flash",
                ),
            )
        elif provider == "deepseek":
            self.handler = deepseek_api(
                _argument_or_env(args, "deepseek_apikey", "DEEPSEEK_API_KEY"),
                _argument_or_env(
                    args,
                    "deepseek_model",
                    "DEEPSEEK_MODEL",
                    "deepseek-chat",
                ),
            )
        elif provider == "claude":
            self.handler = claude_api(
                _argument_or_env(args, "claude_apikey", "CLAUDE_API_KEY"),
                _argument_or_env(args, "claude_model", "CLAUDE_MODEL"),
            )
        else:
            raise ValueError(
                "Invalid model name. Choose from: openai, gemini, deepseek, claude."
            )
