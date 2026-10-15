"""Getting completions out of a model.

Generation is the only part of evaluation that needs a GPU, so it is kept
separate from scoring: `mirage generate` writes raw completions to
`predictions.jsonl` (on Kaggle), and `mirage score` turns them into numbers
(anywhere, in seconds). If the scoring rules change, every past run can be
re-scored without regenerating anything.

Backends:

* `hf`     — transformers `generate`. Always available, slowest, the reference.
* `vllm`   — vLLM offline engine; many times faster for sampling n>1. Optional;
             check that the installed vLLM supports the T4 before relying on it.
* `openai` — any OpenAI-compatible server (a `vllm serve` process, for
             example). Also what Vantage talks to.
* `mock`   — a scripted policy for tests and the GPU-free demo.

All of them take the same chat messages from `data/prompts.py`.
"""

from __future__ import annotations

import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

from mirage.errors import ConfigError, MissingDependency


@dataclass
class GenConfig:
    backend: str = "hf"
    model: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    adapter: str | None = None  # path to a LoRA adapter (SFT or GRPO output)
    dtype: str = "float16"  # T4 has no bf16
    max_new_tokens: int = 192
    batch_size: int = 16
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)  # e.g. {"enable_thinking": False} for Qwen3
    base_url: str | None = None  # openai backend
    api_key_env: str = "OPENAI_API_KEY"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.85
    seed: int = 0


class Generator(Protocol):
    def generate(self, prompts: list[list[dict]], n: int, temperature: float) -> list[list[str]]: ...


# --------------------------------------------------------------------------- #


class HFGenerator:
    def __init__(self, cfg: GenConfig):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise MissingDependency("transformers") from e
        self.cfg = cfg
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(cfg.model)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[cfg.dtype]
        model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=dtype)
        if cfg.adapter:
            try:
                from peft import PeftModel
            except ImportError as e:
                raise MissingDependency("peft") from e
            model = PeftModel.from_pretrained(model, cfg.adapter)
            model = model.merge_and_unload()  # faster generation; scoring is unaffected
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device).eval()

    def generate(self, prompts: list[list[dict]], n: int, temperature: float) -> list[list[str]]:
        torch = self.torch
        out: list[list[str]] = []
        texts = [
            self.tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True, **self.cfg.chat_template_kwargs)
            for p in prompts
        ]
        bs = max(1, self.cfg.batch_size // max(1, n))
        torch.manual_seed(self.cfg.seed)
        for i in range(0, len(texts), bs):
            enc = self.tok(texts[i : i + bs], return_tensors="pt", padding=True).to(self.device)
            kw: dict[str, Any] = {"max_new_tokens": self.cfg.max_new_tokens, "pad_token_id": self.tok.pad_token_id}
            if temperature > 0:
                kw.update(do_sample=True, temperature=temperature, top_p=1.0, num_return_sequences=n)
            else:
                kw.update(do_sample=False, num_return_sequences=1)
            with torch.no_grad():
                gen = self.model.generate(**enc, **kw)
            gen = gen[:, enc["input_ids"].shape[1] :]
            decoded = self.tok.batch_decode(gen, skip_special_tokens=True)
            per = n if temperature > 0 else 1
            for j in range(0, len(decoded), per):
                out.append(decoded[j : j + per])
        return out


class VLLMGenerator:
    def __init__(self, cfg: GenConfig):
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise MissingDependency("vllm", extra="vllm") from e
        self.cfg = cfg
        self.SamplingParams = SamplingParams
        self.llm = LLM(
            model=cfg.model,
            dtype="half" if cfg.dtype == "float16" else cfg.dtype,
            enable_lora=bool(cfg.adapter),
            max_model_len=cfg.max_model_len,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            seed=cfg.seed,
        )
        self.lora = None
        if cfg.adapter:
            from vllm.lora.request import LoRARequest

            self.lora = LoRARequest("mirage", 1, cfg.adapter)

    def generate(self, prompts: list[list[dict]], n: int, temperature: float) -> list[list[str]]:
        sp = self.SamplingParams(
            n=n if temperature > 0 else 1,
            temperature=temperature,
            max_tokens=self.cfg.max_new_tokens,
            seed=self.cfg.seed,
        )
        res = self.llm.chat(
            prompts, sp, lora_request=self.lora, chat_template_kwargs=self.cfg.chat_template_kwargs or None
        )
        return [[o.text for o in r.outputs] for r in res]


class OpenAIGenerator:
    """Talks to any OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, cfg: GenConfig, workers: int = 8):
        if not cfg.base_url:
            raise ConfigError("the openai backend needs base_url (e.g. http://localhost:8000/v1)")
        self.cfg = cfg
        self.key = os.environ.get(cfg.api_key_env, "EMPTY")
        self.workers = workers

    def _one(self, messages: list[dict], n: int, temperature: float) -> list[str]:
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "max_tokens": self.cfg.max_new_tokens,
            "temperature": temperature,
            "n": n if temperature > 0 else 1,
            "seed": self.cfg.seed,
        }
        if self.cfg.chat_template_kwargs:
            body["chat_template_kwargs"] = self.cfg.chat_template_kwargs
        req = urllib.request.Request(
            self.cfg.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        return [c["message"]["content"] or "" for c in data["choices"]]

    def generate(self, prompts: list[list[dict]], n: int, temperature: float) -> list[list[str]]:
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(lambda p: self._one(p, n, temperature), prompts))


def build_generator(cfg: GenConfig, mock_policy=None) -> Generator:
    if cfg.backend == "hf":
        return HFGenerator(cfg)
    if cfg.backend == "vllm":
        return VLLMGenerator(cfg)
    if cfg.backend == "openai":
        return OpenAIGenerator(cfg)
    if cfg.backend == "mock":
        if mock_policy is None:
            raise ConfigError("the mock backend needs a policy (used by tests and `mirage demo`)")
        return mock_policy
    raise ConfigError(f"unknown generation backend {cfg.backend!r} (hf, vllm, openai, mock)")
