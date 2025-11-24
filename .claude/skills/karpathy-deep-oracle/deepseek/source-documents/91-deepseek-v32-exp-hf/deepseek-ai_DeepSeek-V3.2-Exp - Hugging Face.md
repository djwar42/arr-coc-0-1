---
sourceFile: "deepseek-ai/DeepSeek-V3.2-Exp - Hugging Face"
exportedBy: "Kortex"
exportDate: "2025-10-29T02:37:55.122Z"
---

# deepseek-ai/DeepSeek-V3.2-Exp - Hugging Face

61df6094-2252-4e1f-ae8c-68c8e6b60bd9

deepseek-ai/DeepSeek-V3.2-Exp - Hugging Face

d55c440f-1200-4e9d-bade-fcb8c3a14a4c

https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp

## Hugging Face

## Enterprise

deepseek-ai

DeepSeek-V3.2-Exp

https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp

## Text Generation

https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp

## Transformers

https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp

## Safetensors

https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp

deepseek_v32

https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp

conversational

https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp

https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp

## Model card

https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp

## Files   Files and versions   xet

https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp

Community  28

https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp

DeepSeek-V3.2-Exp

## Introduction

We are excited to announce the official release of DeepSeek-V3.2-Exp, an experimental version of our model. As an intermediate step toward our next-generation architecture, V3.2-Exp builds upon V3.1-Terminus by introducing DeepSeek Sparse Attention—a sparse attention mechanism designed to explore and validate optimizations for training and inference efficiency in long-context scenarios.

This experimental release represents our ongoing research into more efficient transformer architectures, particularly focusing on improving computational efficiency when processing extended text sequences.

DeepSeek Sparse Attention (DSA) achieves fine-grained sparse attention for the first time, delivering substantial improvements in long-context training and inference efficiency while maintaining virtually identical model output quality.

To rigorously evaluate the impact of introducing sparse attention, we deliberately aligned the training configurations of DeepSeek-V3.2-Exp with V3.1-Terminus. Across public benchmarks in various domains, DeepSeek-V3.2-Exp demonstrates performance on par with V3.1-Terminus.

Benchmark   DeepSeek-V3.1-Terminus   DeepSeek-V3.2-Exp

Reasoning Mode w/o Tool Use

MMLU-Pro   85.0   85.0   GPQA-Diamond   80.7   79.9   Humanity's Last Exam   21.7   19.8   LiveCodeBench   74.9   74.1   AIME 2025   88.4   89.3   HMMT 2025   86.1   83.6   Codeforces   2046   2121   Aider-Polyglot   76.1   74.5

## Agentic Tool Use

BrowseComp   38.5   40.1   BrowseComp-zh   45.0   47.9   SimpleQA   96.8   97.1   SWE Verified   68.4   67.8   SWE-bench Multilingual   57.8   57.9   Terminal-bench   36.7   37.7

## How to Run Locally

## HuggingFace

## We provide an updated inference demo code in the

https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/tree/main/inference

folder to help the community quickly get started with our model and understand its architectural details.

First convert huggingface model weights to the the format required by our inference demo. Set  MP  to match your available GPU count:

cd  inference  export  EXPERTS=256 python convert.py --hf-ckpt-path  ${HF_CKPT_PATH}  --save-path  ${SAVE_PATH}  --n-experts  ${EXPERTS}  --model-parallel  ${MP}

Launch the interactive chat interface and start exploring DeepSeek's capabilities:

export  CONFIG=config_671B_v3.2.json torchrun --nproc-per-node  ${MP}  generate.py --ckpt-path  ${SAVE_PATH}  --config  ${CONFIG}  --interactive

## Installation with Docker

# H200 docker pull lmsysorg/sglang:dsv32 # MI350 docker pull lmsysorg/sglang:dsv32-rocm # NPUs docker pull lmsysorg/sglang:dsv32-a2 docker pull lmsysorg/sglang:dsv32-a3

## Launch Command

python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --dp 8 --enable-dp-attention

vLLM provides day-0 support of DeepSeek-V3.2-Exp. See the

https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_2-Exp.html

for up-to-date details.

Open-Source Kernels

## For TileLang kernels with

better readability and research-purpose design

, please refer to

https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32

high-performance CUDA kernels

, indexer logit kernels (including paged versions) are available in

https://github.com/deepseek-ai/DeepGEMM/pull/200

. Sparse attention kernels are released in

https://github.com/deepseek-ai/FlashMLA/pull/98

## This repository and the model weights are licensed under the

## MIT License

https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/LICENSE

@misc{deepseekai2024deepseekv32, title={DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention}, author={DeepSeek-AI}, year={2025}, }

If you have any questions, please raise an issue or contact us at

service@deepseek.com

https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/service@deepseek.com

Downloads last month 101,711  Safetensors

Model size   685B params   Tensor type   BF16  · F8_E4M3  · F32  ·

## Text Generation

https://huggingface.co/docs/safetensors

## Examples   Input a message to start chatting with

deepseek-ai/DeepSeek-V3.2-Exp

## Open Playground

https://huggingface.co/playground?modelId=deepseek-ai/DeepSeek-V3.2-Exp&provider=novita

Model tree for  deepseek-ai/DeepSeek-V3.2-Exp

deepseek-ai/DeepSeek-V3.2-Exp-Base

https://huggingface.co/docs/hub/model-cards#specifying-a-base-model

Finetuned   (

https://huggingface.co/models?other=base_model:finetune:deepseek-ai/DeepSeek-V3.2-Exp-Base

)  this model  Adapters

https://huggingface.co/models?other=base_model:adapter:deepseek-ai/DeepSeek-V3.2-Exp

## Finetunes

https://huggingface.co/models?other=base_model:finetune:deepseek-ai/DeepSeek-V3.2-Exp

## Quantizations

https://huggingface.co/models?other=base_model:quantized:deepseek-ai/DeepSeek-V3.2-Exp

Spaces using  deepseek-ai/DeepSeek-V3.2-Exp   66

Collection including  deepseek-ai/DeepSeek-V3.2-Exp

DeepSeek-V3.2

