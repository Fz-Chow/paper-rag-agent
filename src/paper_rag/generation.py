"""Shared local generation; no truncation or retrieval decisions."""
from __future__ import annotations
import os
import platform
import threading
import time
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version

TASKS = {
    "compare": "比较证据实际支持的方法、流程和适用场景。",
    "summarize": "总结证据支持的研究问题、方法和结论。",
    "question_answering": "直接回答问题，并解释关键依据。",
    "experiment": '只输出一个 JSON 对象：{"answer":"简短答案","citations":["证据编号"]}。引用支持答案的全部必要证据编号；证据不足时 answer 写“证据不足”。',
}

@dataclass
class GenerationResult:
    answer: str
    prompt: str
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float
    device: str
    gpu: dict | None
    finish_reason: str

    def to_dict(self):
        return asdict(self)


def environment() -> dict:
    packages = {}
    for name in ("torch", "transformers", "accelerate", "tokenizers"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def gpu_metrics(device="cuda") -> dict | None:
    try:
        import torch
    except ImportError:
        return None
    if device != "cuda" or not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return {
        "device_name": torch.cuda.get_device_name(),
        "hip_version": getattr(torch.version, "hip", None),
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
        "free_mb": free / 1024**2, "total_mb": total / 1024**2,
    }


def validate_input_length(count, max_input_tokens, max_new_tokens, context_limit):
    if count > max_input_tokens or count + max_new_tokens > context_limit:
        raise ValueError(f"Input budget exceeded: input={count}, output_budget={max_new_tokens}, context={context_limit}; truncation disabled")


class LocalGenerator:
    def __init__(self, model_name=None, max_new_tokens=None, max_input_tokens=None):
        from .core import resolve_device
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_name = model_name or os.environ["PAPER_RAG_LLM_MODEL"]
        self.revision = os.getenv("PAPER_RAG_MODEL_REVISION") or "main"
        self.max_new_tokens = int(max_new_tokens or os.getenv("PAPER_RAG_MAX_NEW_TOKENS", "256"))
        self.device = resolve_device()
        self.lock = threading.Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, revision=self.revision,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        limits = [getattr(self.model.config, "max_position_embeddings", None), self.tokenizer.model_max_length]
        self.context_limit = min(int(x) for x in limits if x and 0 < x < 10**9)
        self.max_input_tokens = int(max_input_tokens or os.getenv("PAPER_RAG_MAX_INPUT_TOKENS", "8192"))
        if min(self.max_new_tokens, self.max_input_tokens) < 1:
            raise ValueError("Token budgets must be positive")

    def metadata(self):
        return {"model": self.model_name, "revision": self.revision,
                "resolved_revision": getattr(self.model.config, "_commit_hash", None),
                "device": self.device, "context_limit": self.context_limit,
                "max_input_tokens": self.max_input_tokens, "max_new_tokens": self.max_new_tokens,
                "do_sample": False, "num_beams": 1, "gpu": gpu_metrics(self.device)}

    def prompt(self, question, evidence, task="experiment"):
        format_instruction = "" if task == "experiment" else "使用中文；每个关键结论都用 [1] 这样的编号引用证据。证据不足时明确说明。"
        messages = [
            {"role": "system", "content": "只根据给定证据回答，不补充外部事实。" + format_instruction + TASKS[task]},
            {"role": "user", "content": f"问题：{question}\n\n证据：\n{evidence}"},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def check_length(self, count):
        validate_input_length(count, self.max_input_tokens, self.max_new_tokens, self.context_limit)

    def generate(self, question, evidence, task="experiment"):
        import torch
        prompt = self.prompt(question, evidence, task)
        count = self.count_tokens(prompt)
        self.check_length(count)
        with self.lock, torch.inference_mode():
            if self.device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt", truncation=False).to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                do_sample=False, num_beams=1, pad_token_id=self.tokenizer.eos_token_id)
            if self.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            ids = outputs[0][count:]
            answer = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            return GenerationResult(answer, prompt, count, len(ids), elapsed, self.device,
                                    gpu_metrics(self.device), "length" if len(ids) >= self.max_new_tokens else "stop")

_generator = None
_load_lock = threading.Lock()

def get_generator():
    global _generator
    if not os.getenv("PAPER_RAG_LLM_MODEL", "").strip():
        return None
    with _load_lock:
        if _generator is None:
            _generator = LocalGenerator()
    return _generator
