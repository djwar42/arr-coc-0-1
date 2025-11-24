# vLLM Deployment

**Full guide**: `RESEARCH/DeepSeekOCR/HF.md` lines 750-950

## Why vLLM?

**10-20× faster** than HuggingFace Transformers!

**Optimizations**:
- PagedAttention (efficient KV cache)
- Continuous batching
- Optimized CUDA kernels

## Installation

```bash
pip install vllm
```

## Basic Usage

```python
from vllm import LLM

llm = LLM(model="deepseek-ai/DeepSeek-OCR", trust_remote_code=True)

outputs = llm.generate(prompts=["<image>\nFree OCR."], images=["doc.jpg"])
print(outputs[0].text)
```

## Server Deployment

```bash
# Start server (OpenAI-compatible API)
vllm serve deepseek-ai/DeepSeek-OCR --trust-remote-code

# Use with OpenAI client
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-OCR",
    messages=[{"role": "user", "content": "<image>\nFree OCR."}]
)
```

## Production Settings

```bash
vllm serve deepseek-ai/DeepSeek-OCR \
    --trust-remote-code \
    --tensor-parallel-size 2 \    # Multi-GPU
    --max-model-len 4096 \         # Context length
    --gpu-memory-utilization 0.9   # Use 90% VRAM
```

## Performance

**HuggingFace**: ~2-3 images/sec
**vLLM**: ~30-50 images/sec

**20k+ pages/day** on single A100 with vLLM!

**See HF.md** for complete vLLM setup and benchmarks!
