---
sourceFile: "Understanding DeepSeek R1 | Christian B. B. Houmann"
exportedBy: "Kortex"
exportDate: "2025-10-29T02:37:51.951Z"
---

# Understanding DeepSeek R1 | Christian B. B. Houmann

bd6df5e6-d860-4435-a876-b49487466f47

Understanding DeepSeek R1 | Christian B. B. Houmann

f4383e42-8e23-481b-8d73-f37371b0f696

https://bagerbach.com/blog/understanding-deepseek-r1/

~/cbbh   _

Understanding DeepSeek R1

Feb 1, 2025   | 11 min read

DeepSeek-R1 is an open-source language model built on DeepSeek-V3-Base that’s been making waves in the AI community. Not only does it match—or even surpass—OpenAI’s o1 model in many benchmarks, but it also comes with fully MIT-licensed weights. This marks it as the first non-OpenAI/Google model to deliver strong reasoning capabilities in an open and accessible manner.

What makes DeepSeek-R1 particularly exciting is its transparency. Unlike the less-open approaches from some industry leaders, DeepSeek has published a detailed training methodology in their

https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf

. 
  The model is also remarkably cost-effective, with input tokens costing just $0.14-0.55 per million (vs o1’s $15) and output tokens at $2.19 per million (vs o1’s $60).

✨   Why it matters

Until ~GPT-4, the common wisdom was that better models required more data and compute. While that’s still valid, models like o1 and R1 demonstrate an alternative: inference-time scaling through reasoning.

## The Essentials

DeepSeek-R1 paper

https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf

presented multiple models, but primary amongst them were R1 and R1-Zero. Following these are a series of distilled models that, while interesting, I won’t discuss here.

DeepSeek-R1 uses two major ideas:

A multi-stage pipeline where a small set of cold-start data kickstarts the model, followed by large-scale RL.

Group Relative Policy Optimization (GRPO), a reinforcement learning method that relies on comparing multiple model outputs per prompt to avoid the need for a separate critic.

R1 and R1-Zero are both reasoning models. This essentially means they do

Chain-of-Thought

https://www.promptingguide.ai/techniques/cot

before answering. For the R1 series of models, this takes form as thinking within a  <think>  tag, before answering with a final summary.

R1-Zero vs R1

R1-Zero applies Reinforcement Learning (RL) directly to DeepSeek-V3-Base with no supervised fine-tuning (SFT). RL is used to optimize the model’s policy to maximize reward. 
  R1-Zero achieves excellent accuracy but sometimes produces confusing outputs, such as mixing multiple languages in a single response. R1 fixes that by incorporating limited supervised fine-tuning and multiple RL passes, which improves both correctness and readability.

It is interesting how some languages may express certain ideas better, which leads the model to choose the most expressive language for the task.

## Training Pipeline

The training pipeline that DeepSeek published in the R1 paper is immensely interesting. It showcases how they created such strong reasoning models, and what you can expect from each phase. This includes the problems that the resulting models from each phase have, and how they solved it in the next phase.

It’s interesting that their training pipeline varies from the usual:

The usual training strategy:

Pretraining on large dataset (train to predict next word) to get the base model → supervised fine-tuning → preference tuning via RLHF

Pretrained → RL

Pretrained → Multistage training pipeline with multiple SFT and RL stages

A high-level pipeline diagram for DeepSeek-R1 posted by @SirrahChan on X

Cold-Start Fine-Tuning:

Fine-tune DeepSeek-V3-Base on a few thousand Chain-of-Thought (CoT) samples to ensure the RL process has a decent starting point. This gives a good model to start RL.

First RL Stage:

Apply GRPO with rule-based rewards to improve reasoning correctness and formatting (such as forcing chain-of-thought into thinking tags). When they were near convergence in the RL process, they moved to the next step. The result of this step is a strong reasoning model but with weak general capabilities, e.g., poor formatting and language mixing.

## Rejection Sampling

https://www.promptingguide.ai/techniques/cot

+ general data:

Create new SFT data through rejection sampling on the RL checkpoint (from step 2), combined with supervised data from the DeepSeek-V3-Base model. They collected around 600k high-quality reasoning samples.

Second Fine-Tuning:

Fine-tune DeepSeek-V3-Base again on 800k total samples (600k reasoning + 200k general tasks) for broader capabilities. This step resulted in a strong reasoning model with general capabilities.

Second RL Stage:

Add more reward signals (helpfulness, harmlessness) to refine the final model, in addition to the reasoning rewards. The result is DeepSeek-R1.

They also did model distillation for several Qwen and Llama models on the reasoning traces to get distilled-R1 models.

🧑‍🏫   Model distillation

Model distillation is a technique where you use a teacher model to improve a student model by generating training data for the student model. 
  The teacher is typically a larger model than the student.

Group Relative Policy Optimization (GRPO)

The basic idea behind using reinforcement learning for LLMs is to fine-tune the model’s policy so that it naturally produces more accurate and useful answers. 
  They used a reward system that checks not only for correctness but also for proper formatting and language consistency, so the model gradually learns to favor responses that meet these quality criteria.

In this paper, they encourage the R1 model to generate chain-of-thought reasoning through RL training with GRPO. 
  Rather than adding a separate module at inference time, the training process itself nudges the model to produce detailed, step-by-step outputs—making the chain-of-thought an emergent behavior of the optimized policy.

What makes their approach particularly interesting is its reliance on straightforward, rule-based reward functions. 
  Instead of depending on expensive external models or human-graded examples as in traditional RLHF, the RL used for R1 uses simple criteria: it might give a higher reward if the answer is correct, if it follows the expected  <think> / <answer>  formatting, and if the language of the answer matches that of the prompt. 
  Not relying on a reward model also means you don’t have to spend time and effort training it, and it doesn’t take memory and compute away from your main model.

## GRPO was introduced in the

## DeepSeekMath paper

https://arxiv.org/abs/2402.03300

. Here’s how GRPO works:

For each input prompt, the model generates   different responses.

Each response receives a scalar reward based on factors like accuracy, formatting, and language consistency.

Rewards are adjusted relative to the group’s performance, essentially measuring how much better each response is compared to the others.

The model updates its strategy slightly to favor responses with higher relative advantages. It only makes slight adjustments—using techniques like clipping and a KL penalty—to ensure the policy doesn’t stray too far from its original behavior.

A cool aspect of GRPO is its flexibility. You can use simple rule-based reward functions—for instance, awarding a bonus when the model correctly uses the  <think>  syntax—to guide the training.

While DeepSeek used GRPO, you could use

alternative methods instead (PPO or PRIME)

https://x.com/jiayi_pirate/status/1882839504899420517

For those looking to dive deeper, Will Brown has written quite a nice

implementation

https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb

of training an LLM with RL using GRPO. GRPO has also already been added to the

Transformer Reinforcement Learning (TRL)

https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py

library, which is another good resource. 
  Finally,

## Yannic Kilcher

https://www.youtube.com/@YannicKilcher

great video

https://www.youtube.com/watch?v=bAWV_yrqx4w

explaining GRPO by going through the DeepSeekMath paper.

Is RL on LLMs the path to AGI?

As a final note on describing DeepSeek-R1 and the methodologies they’ve presented in their paper, I want to highlight a passage from the DeepSeekMath paper, based on a point Yannic Kilcher made in his

https://www.youtube.com/watch?v=bAWV_yrqx4w

These findings indicate that RL enhances the model’s overall performance by rendering the output distribution more robust, in other words, it seems that the improvement is attributed to boosting the correct response from TopK rather than the enhancement of fundamental capabilities.

In other words, RL fine-tuning tends to shape the output distribution so that the highest-probability outputs are more likely to be correct, even though the overall capability (as measured by the diversity of correct answers) is largely present in the pretrained model.

This suggests that reinforcement learning on LLMs is more about refining and “shaping” the existing distribution of responses rather than endowing the model with entirely new capabilities. 
  Consequently, while RL techniques such as PPO and GRPO can produce substantial performance gains, there appears to be an inherent ceiling determined by the underlying model’s pretrained knowledge.

It is unclear to me how far RL will take us. Perhaps it will be the stepping stone to the next big milestone. I’m excited to see how it unfolds!

Running DeepSeek-R1

I’ve used DeepSeek-R1 via the

official chat interface

https://chat.deepseek.com

for various problems, which it seems to solve well enough. The additional search functionality makes it even nicer to use.

Interestingly, o3-mini(-high) was released as I was writing this post. From my initial testing, R1 seems stronger at math than o3-mini.

I also rented a single H100 via

## Lambda Labs

https://lambdalabs.com

for $2/h (26 CPU cores, 214.7 GB RAM, 1.1 TB SSD) to run some experiments. 
  The primary objective was to see how the model would perform when deployed on a single H100 GPU—not to extensively test the model’s capabilities.

671B via Llama.cpp

DeepSeek-R1 1.58-bit (UD-IQ1_S)

https://lambdalabs.com

quantized model by Unsloth, with a 4-bit quantized KV-cache and partial GPU offloading (29 layers running on the GPU), running via

llama.cpp

https://github.com/ggerganov/llama.cpp

.llama-cli  \    --model  DeepSeek-R1-GGUF/DeepSeek-R1-UD-IQ1_S/DeepSeek-R1-UD-IQ1_S-00001-of-00003.gguf  \    --cache-type-k  q4_0  \    --threads  16  \    --n-gpu-layers  29  \    --prio  2  \    --temp  0.6  \    --ctx-size  8192  \    --seed  3407  \    --prompt  "<|User|>Create a Flappy Bird game in Python.<|Assistant|>"

29 layers seemed to be the sweet spot given this configuration.

Performance:

llama_perf_sampler_print: sampling time = 130.46 ms / 1690 runs ( 0.08 ms per token, 12954.06 tokens per second)   llama_perf_context_print: load time = 20112.10 ms   llama_perf_context_print: prompt eval time = 8959.28 ms / 19 tokens ( 471.54 ms per token, 2.12 tokens per second)   llama_perf_context_print: eval time = 387400.66 ms / 1689 runs ( 229.37 ms per token, 4.36 tokens per second)   llama_perf_context_print: total time = 398219.39 ms / 1708 tokens

A r/localllama user

described

https://www.reddit.com/r/LocalLLaMA/comments/1idseqb/deepseek_r1_671b_over_2_toksec_without_gpu_on/

that they were able to get over 2 tok/sec with DeepSeek R1 671B, without using their GPU on their local gaming setup. 
  Digital Spaceport wrote a full guide on

how to run Deepseek R1 671b fully locally on a $2000 EPYC server

https://digitalspaceport.com/how-to-run-deepseek-r1-671b-fully-locally-on-2000-epyc-rig/

, on which you can get ~4.25 to 3.5 tokens per second.

As you can see, the tokens/s isn’t quite bearable for any serious work, but it’s fun to run these large models on accessible hardware.

What matters most to me is a combination of usefulness and time-to-usefulness in these models. Since reasoning models need to think before answering, their time-to-usefulness is usually higher than other models, but their usefulness is also usually higher. 
  We need to both maximize usefulness and minimize time-to-usefulness.

70B via Ollama

70.6b params, 4-bit KM

https://digitalspaceport.com/how-to-run-deepseek-r1-671b-fully-locally-on-2000-epyc-rig/

quantized DeepSeek-R1 running via

https://ollama.com

ollama  run  deepseek-r1:70b

GPU utilization shoots up here, as expected when compared to the mostly CPU-powered run of 671B that I showcased above.

DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

https://ollama.com

[2402.03300] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

https://ollama.com

DeepSeek R1 - Notion

https://ollama.com

Building a fully local “deep researcher” with DeepSeek-R1 - YouTube

https://www.youtube.com/watch?v=sGUjmyfof4Q

DeepSeek R1’s recipe to replicate o1 and the future of reasoning LMs

https://www.youtube.com/watch?v=sGUjmyfof4Q

The Illustrated DeepSeek-R1 - by Jay Alammar

https://www.youtube.com/watch?v=sGUjmyfof4Q

Explainer: What’s R1 & Everything Else? - Tim Kellogg

https://www.youtube.com/watch?v=sGUjmyfof4Q

DeepSeek R1 Explained to your grandma - YouTube

https://www.youtube.com/watch?v=sGUjmyfof4Q

chat.deepseek.com

https://chat.deepseek.com

GitHub - deepseek-ai/DeepSeek-R1

https://chat.deepseek.com

deepseek-ai/Janus-Pro-7B · Hugging Face

https://chat.deepseek.com

(January 2025): Janus-Pro is a novel autoregressive framework that unifies multimodal understanding and generation. It can both understand and generate images.

DeepSeek-R1: Incentivizing Reasoning Capability in Large Language Models via Reinforcement Learning

https://chat.deepseek.com

(January 2025) This paper introduces DeepSeek-R1, an open-source reasoning model that rivals the performance of OpenAI’s o1. It presents a detailed methodology for training such models using large-scale reinforcement learning techniques.

DeepSeek-V3 Technical Report

https://chat.deepseek.com

(December 2024) This report discusses the implementation of an FP8 mixed precision training framework validated on an extremely large-scale model, achieving both accelerated training and reduced GPU memory usage.

DeepSeek LLM: Scaling Open-Source Language Models with Longtermism

https://chat.deepseek.com

(January 2024) This paper delves into scaling laws and presents findings that facilitate the scaling of large-scale models in open-source configurations. It introduces the DeepSeek LLM project, dedicated to advancing open-source language models with a long-term perspective.

DeepSeek-Coder: When the Large Language Model Meets Programming—The Rise of Code Intelligence

https://chat.deepseek.com

(January 2024) This research introduces the DeepSeek-Coder series, a range of open-source code models trained from scratch on 2 trillion tokens. The models are pre-trained on a high-quality project-level code corpus and employ a fill-in-the-blank task to enhance code generation and infilling.

DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model

https://chat.deepseek.com

(May 2024) This paper presents DeepSeek-V2, a Mixture-of-Experts (MoE) language model characterized by economical training and efficient inference.

DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence

https://chat.deepseek.com

(June 2024) This research introduces DeepSeek-Coder-V2, an open-source Mixture-of-Experts (MoE) code language model that achieves performance comparable to GPT-4 Turbo in code-specific tasks.

## Interesting events

## Hong Kong University

replicates R1 results

https://hkust-nlp.notion.site/simplerl-reason

(Jan 25, ‘25)

## Huggingface announces

huggingface/open-r1: Fully open reproduction of DeepSeek-R1

https://github.com/huggingface/open-r1

to replicate R1, fully open source (Jan 25, ‘25)

## OpenAI researcher

https://x.com/markchen90/status/1884303237186216272

the DeepSeek team independently found and used some core ideas the OpenAI team used on the way to o1

Liked this post? Join the newsletter.

Get notified whenever I post something new.

© 2025 Christian B. B. Houmann

