---
sourceFile: "33% faster LLM inference with FP8 quantization - Baseten"
exportedBy: "Kortex"
exportDate: "2025-10-29T02:37:17.424Z"
---

# 33% faster LLM inference with FP8 quantization - Baseten

3e77c161-8409-4bcb-a64a-a52930b7e2d3

33% faster LLM inference with FP8 quantization - Baseten

e6178dff-6a4f-4199-bea9-9fcc65c3c066

https://www.baseten.co/blog/33-faster-llm-inference-with-fp8-quantization/

## Pricing Pricing

## Get started

## Model performance

33% faster LLM inference with FP8 quantization

Quantizing open-source LLMs to FP8 resulted in near-zero perplexity gains and yielded material performance improvements across latency, throughput, and cost.

## Pankaj Gupta

## Philip Kiely

## Last updated

May 18, 2025

To achieve higher inference performance at lower cost, we quantized Mistral 7B to FP8, a data format supported by recent GPUs such as the H100. FP8 unlocked a meaningful increase in throughput as well as a small improvement in time to first token, but before we could take advantage of that performance in production, we had to rigorously assess model output quality with both quantitative and qualitative checks for any degradation in generated text quality. FP8 easily passes those checks, making it a great choice for efficient inference in production.

## Quantization

is the process of mapping model parameters from one data format (most commonly FP16 for LLMs) to a smaller data format, like INT8 or FP8. Quantizing a model offers faster, less expensive inference. By quantizing Mistral 7B to FP8, we observed the following improvements vs FP16 (both using TensorRT-LLM on an H100 GPU):

An 8.5% decrease in latency in the form of time to first token

A 33% improvement in speed, measured as output tokens per second

A 31% increase in throughput in terms of total output tokens

A 24% reduction in cost per million tokens

These benchmarks are for a specific batch size (32 max requests) and sequence shape (80 input and 100 output tokens per request) — we’ll dive into a wider range of benchmarks later in the article. First, we have to validate that the quantized model is suitable for production use by checking its output quality against perplexity benchmarks and manually analyzing sample responses.

Model output quality for FP8 Mistral 7B

Running a model in FP8 means you’re spending less money for faster inference. By going from 16-bit numbers to 8-bit numbers, model inference uses substantially less VRAM, bandwidth, and compute. But quantizing has a risk. If done wrong, it can degrade model output quality to the point of being unusable — a problem we encountered before switching to FP8.

Previously, we’ve tried two approaches to 8-bit quantization with INT8. First, we created weights-only quantizations of LLMs. While this approach preserved output quality, it required activations to still run in FP16, limiting speed improvements. We also tried SmoothQuant to quantize all components of a given LLM into INT8, but found that it degraded model output quality to unacceptable levels for this model and use case.

For Mistral 7B, we used a pre-release library created by NVIDIA compatible with the TensorRT-LLM ecosystem to quantize the model to a different 8-bit data format, FP8.

FP8 is a newly supported data format

https://www.baseten.co#model-output-quality-for-fp8-mistral-7b

that promises the same benefits of 8-bit quantization without loss of output quality. It’s only supported on NVIDIA’s most recent GPU architectures, Ada Lovelace and Hopper, so you can run LLMs like Mistral 7B in FP8 on

powerful hardware like the H100 GPU

https://www.baseten.co/blog/unlocking-the-full-power-of-nvidia-h100-gpus-for-ml-inference-with-tensorrt/

✕ Visualizing FP32, FP16, FP8, and INT8 precisions

FP8 has a higher dynamic range than INT8, which makes it suitable for quantizing performance-critical components of the LLM, including weights, activations, and KV cache. TensorRT-LLM includes kernel implementations that take advantage of FP8 compute capabilities on H100 GPUs, enabling FP8 inference.

FP8 has a lower memory footprint than FP16, only requiring 7GB of VRAM instead of 16GB. This is especially relevant when using multi-instance GPUs to split H100s into multiple parts, which can have as little as 10GB of VRAM each. But more generally, FP8 offers incredible performance improvements for latency and throughput during inference. However, none of that matters if the output quality is unusable.

Fortunately, the FP8 quantization of Mistral 7B provides nearly identical output quality to the FP16 base model, as observed both mathematically and anecdotally.

Model output quality: perplexity benchmark

Perplexity is a measure of model output quality that can tell us how well a model survived quantization. Perplexity is an interesting metric; it’s calculated by giving the model high-quality output samples and checking if those align with what the model would provide. A lower perplexity score is better as it means the model was less “surprised” by those output samples.

Absolute values for perplexity scores vary a lot based on prompt content and sequence lengths. When evaluating a quantized model, we care about the relative score between the original and quantized model rather than the absolute score. Here’s how the perplexity score breaks down for FP16, FP8, and INT8.

The FP8 quantization shows a comparable perplexity to FP16 — in fact some benchmark runs showed FP8 at a lower perplexity which indicates that these slight differences are just noise — but INT8 with SmoothQuant is clearly unusable for this model at nearly double the FP16 baseline perplexity.

Model output quality: anecdotal evaluation

While the math behind perplexity is solid, it’s always useful to double-check any change to your ML model by manually verifying a bit of output. We ran the FP16 and FP8 models under identical circumstances with three different prompts across factual recall, coding, and poetry to see how well the quantized model really works. Here are side-by-side comparisons showing that FP8 output is comparable in quality to FP16 output.

Prompt: How far is LA from SFO?

We found that FP8 equals FP16 in accuracy for this sample. The FP8 answer was also somewhat more concise, while the FP16 answer had greater detail, but it’s hard to generalize from this tiny sample size.

Prompt: Write a Python script for calculating the fibonacci sequence.

Omitted are the near-identical text descriptions that both models gave for their answer. The code generated by both models is comparable with minor stylistic differences. Notably, both answers have the same typo: an extra closing parentheses in the line that reads user input.

Prompt: Write me a poem about Lagos traffic

Each model wrote a dozen lines that varied after the beginning but were consistent in topic, structure, and poetic quality (or lack thereof). Again, the FP8 model happens to write shorter lines.

While the FP16 and FP8 model results aren't identical, they're comparable. Confident that the FP8 model output is of equivalent quality to the FP16 original, we turn our attention to the improvements in latency, throughput, and cost that FP8 offers for production inference workloads.

Benchmarks for FP8 Mistral 7B inference

Once we’re confident in the model’s output quality, we can turn to performance. This is where quantization gets exciting. The motivation for quantizing the LLM to FP8 was to use 8-bit floating point numbers instead of 16-bit numbers to reduce the load on compute during prefill and memory bandwidth during token generation.

Benchmark performance varies along two axis:

Batch size: more queries per second means more load on the system.

Sequence lengths: longer input sequences take more time to process in prefill.

As each of these increase, TTFT and TPS worsen, but total tokens generated and cost per token improve. We benchmarked a wide range of batch sizes and sequence lengths and can use that data to find the right balance that makes sense for your traffic patterns.

## Benchmarks across batch sizes

Time to first token increases with batch size, but FP8 consistently offers an advantage for lower latency. We’ll see a larger improvement in latency for longer sequence lengths in the next section, where FP8 makes a greater impact. With the same 80x100 sequence shape, here’s how TTFT measures up over various batch sizes:

✕ Mistral 7B time to first token across batch sizes

Time to first token is based on prefill, a compute-bound process during LLM inference. As batch size (and thus queries per second) increases, the number of context phase slots required for prefill increases. When space for these processes gets saturated, there’s a chance of slots colliding. The chances of a collision isn’t linear; as batch size increases TTFT will slowly increase until collisions begin to occur, at which point it will spike.

Where FP8 provides a bigger lift is in tokens per second generated. Our FP8 Mistral implementation benchmarked as much as 33% higher in tokens per second than the FP16 model at some batch sizes.

✕ Mistral 7B output tokens per second across batch sizes

While an individual user’s perceived tokens per second decreases with larger batch sizes, the overall throughput of the GPU increases. At a batch size of 128, a model server with Mistral 7B in FP8 on an H100 GPU can generate more than 16,000 total tokens per second across all requests. With a fixed price per minute of GPU time, these higher batch sizes enable substantial cost savings on a per-token basis.

## Benchmarks across sequence lengths

80 input tokens and 100 output tokens is not a very big sequence shape. While it’s realistic for some use cases, like short customer service chats, many use cases have much longer input and output sequences, like summarization.

With a longer input sequence, the model takes more time in the prefill step before it can generate a first output token. Here are time to first token measurements for 100, 250, 500, and 1000 input tokens (each with 1000 output tokens at a batch size of 32):

✕ Mistral 7B time to first token across sequence lengths

As the input sequence lengthens, we see the impact that FP8 brings to model performance. For the long input sequences that require substantial prefill computation, the efficiency gains from FP8 are clear.

One thing to watch out for is the interaction between input sequence length and batch size. As input sequences get longer, you can process fewer of them at once before running out of compute slots, which can skyrocket TTFT from a couple hundred milliseconds to several seconds. For example, an input sequence of 1000 tokens works well at a batch size of 72, but at a batch size of 96 TTFT spikes to over 10 seconds.

Baseten addresses this issue through autoscaling infrastructure. After carefully selecting batch sizes and input sequence lengths conducive to high performance, we set a max concurrency and spin up new replicas of the deployed model in response to traffic spikes rather than degrading performance.

On the tokens per second side, a longer input sequence and time to first token does reduce the user-facing tokens per second.

✕ Mistral 7B output tokens per second across sequence lengths

However, like with larger batch sizes, the overall throughput of the GPU increases as sequence length expands. Batch size has a greater impact than sequence length, but going from 100 to 1000 input tokens with a fixed batch size increases the overall throughput by about 50%.

FP8 beyond Mistral 7B

The process we used for quantizing Mistral to FP8 applies to similar models, and we expect similar performance gains when working with other 7B (and larger) LLMs from model families like Llama, Gemma, and Qwen. Next, we look forward to bringing FP8 to LLMs with more complicated architectures like

https://www.baseten.co/library/mixtral-8x7b-instruct/

, where we’ve

previously had success with weights-only INT8 quantization

https://www.baseten.co/blog/faster-mixtral-inference-with-tensorrt-llm-and-quantization/

, as well as exploring its utility for other families of models like Stable Diffusion.

Quantization is just the first step for efficient inference in production. After quantizing a model, we validate the output quality with both a perplexity comparison and a manual check on sample outputs. If the quantized model’s quality is acceptable, we package it for production use and serve it in production with TensorRT-LLM for optimized inference.

By benchmarking the model to understand its performance at different batch sizes, we can make appropriate tradeoffs between cost and performance and build optimized serving engines to target specific traffic patterns and use cases. If you’re interested in serving a model using FP8 and TensorRT-LLM on robust autoscaling infrastructure, we’re here to help at

support@baseten.co

mailto:support@baseten.co

## Subscribe to our newsletter

Stay up to date on model performance, GPUs, and more.

## Related posts

## View all  Model performance

mailto:support@baseten.co

## Model performance

How we made the fastest GPT-OSS on NVIDIA GPUs 60% faster

mailto:support@baseten.co

Tri Dao 2   others Model performance

How Baseten achieved 2x faster inference with NVIDIA Dynamo

mailto:support@baseten.co

Abu Qader 2   others Model performance

How we run GPT OSS 120B at 500+ tokens per second on NVIDIA GPUs

mailto:support@baseten.co

Amir Haghighat 4   others

## Explore Baseten today

## Start deploying

mailto:support@baseten.co

## Talk to an engineer

mailto:support@baseten.co

all systems normal

© 2025 Baseten

## Dedicated deployments

mailto:support@baseten.co

mailto:support@baseten.co

mailto:support@baseten.co

## Inference Stack

## Model Runtimes

mailto:support@baseten.co

## Infrastructure

mailto:support@baseten.co

Multi-cloud Capacity Management

mailto:support@baseten.co

## Developer Experience

mailto:support@baseten.co

## Model management

mailto:support@baseten.co

## Deployment options

## Baseten Cloud

mailto:support@baseten.co

Self-hosted

mailto:support@baseten.co

mailto:support@baseten.co

mailto:support@baseten.co

## Transcription

mailto:support@baseten.co

## Image generation

mailto:support@baseten.co

Text-to-speech

mailto:support@baseten.co

## Large language models

mailto:support@baseten.co

## Compound AI

mailto:support@baseten.co

mailto:support@baseten.co

mailto:support@baseten.co

mailto:support@baseten.co

mailto:support@baseten.co

mailto:support@baseten.co

mailto:support@baseten.co

mailto:support@baseten.co

mailto:support@baseten.co

## Documentation

mailto:support@baseten.co

## Model library

mailto:support@baseten.co

mailto:support@baseten.co

## Popular models

GPT OSS 120B

mailto:support@baseten.co

GPT OSS 20B

mailto:support@baseten.co

Kimi K2 0905

mailto:support@baseten.co

mailto:support@baseten.co

## Orpheus TTS

mailto:support@baseten.co

Qwen3 Coder 480B

mailto:support@baseten.co

## Explore all

mailto:support@baseten.co

## Terms and Conditions

mailto:support@baseten.co

## Privacy Policy

mailto:support@baseten.co

## Service Level Agreement

mailto:support@baseten.co

all systems normal

© 2025 Baseten

