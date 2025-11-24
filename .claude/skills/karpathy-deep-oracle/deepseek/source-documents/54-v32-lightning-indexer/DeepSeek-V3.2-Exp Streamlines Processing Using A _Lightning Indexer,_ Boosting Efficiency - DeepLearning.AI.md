---
sourceFile: "DeepSeek-V3.2-Exp Streamlines Processing Using A "Lightning Indexer," Boosting Efficiency - DeepLearning.AI"
exportedBy: "Kortex"
exportDate: "2025-10-29T02:37:30.586Z"
---

# DeepSeek-V3.2-Exp Streamlines Processing Using A "Lightning Indexer," Boosting Efficiency - DeepLearning.AI

d6141a19-ec2e-4544-a984-739625b7e0fa

DeepSeek-V3.2-Exp Streamlines Processing Using A "Lightning Indexer," Boosting Efficiency - DeepLearning.AI

33575dcd-ddd0-45ab-8750-42d5e95967ac

https://www.deeplearning.ai/the-batch/deepseek-v3-2-exp-streamlines-processing-using-a-lightning-indexer-boosting-efficiency/

✨ New course! Enroll in

Fine-tuning and Reinforcement Learning for LLMs: Intro to Post-Training

https://bit.ly/4ofRjka

## Start Learning

https://bit.ly/4ofRjka

## Weekly Issues

https://bit.ly/4ofRjka

Andrew's Letters

https://bit.ly/4ofRjka

## Data Points

https://bit.ly/4ofRjka

## ML Research

https://bit.ly/4ofRjka

https://bit.ly/4ofRjka

https://bit.ly/4ofRjka

https://bit.ly/4ofRjka

https://bit.ly/4ofRjka

https://bit.ly/4ofRjka

https://bit.ly/4ofRjka

DeepSeek Cuts Inference Costs  DeepSeek-V3.2-Exp streamlines processing using a "lightning indexer," boosting efficiency

## Machine Learning Research

https://bit.ly/4ofRjka

Large Language Models (LLMs)

https://bit.ly/4ofRjka

## Transformers

https://bit.ly/4ofRjka

Oct 15, 2025

https://www.deeplearning.ai/the-batch/tag/oct-15-2025/

Reading time 3  min read Share

## Loading the

## Elevenlabs Text to Speech

https://elevenlabs.io/text-to-speech

AudioNative Player...

DeepSeek’s latest large language model can cut inference costs by more than half and processes long contexts dramatically faster relative to its predecessor.

What’s new:

## DeepSeek released weights for

DeepSeek-V3.2-Exp

https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp?utm_campaign=The%20Batch&utm_source=hs_email&utm_medium=email&_hsenc=p2ANqtz-_L4bJJ6vXg6Hs10je6IoHKFICRZs2QVtY2uv4WUTWxZX4HrpAhzRVhAc_65PiMrUfftu9f

, a variation on DeepSeek-V3.1-Terminus, which was released in late September. It streamlines processing using a dynamic variation on

sparse attention

https://arxiv.org/abs/1904.10509?utm_campaign=The%20Batch&utm_source=hs_email&utm_medium=email&_hsenc=p2ANqtz-_L4bJJ6vXg6Hs10je6IoHKFICRZs2QVtY2uv4WUTWxZX4HrpAhzRVhAc_65PiMrUfftu9f

that enables inference speed to scale linearly with input length. The code supports AI chips designed by Huawei, and other Chinese chip designers have adapted it for their products, helping developers in China to use domestic alternatives to U.S.-designed Nvidia GPUs.

Input/output:

Text in (up to 128,000 tokens), text out (up to 8,000 tokens)

Architecture:

Mixture-of-experts transformer, 685 billion total parameters, approximately 37 billion active parameters per token

Availability:

Free via web interface or app, weights

https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp?utm_campaign=The%20Batch&utm_source=hs_email&utm_medium=email&_hsenc=p2ANqtz-_L4bJJ6vXg6Hs10je6IoHKFICRZs2QVtY2uv4WUTWxZX4HrpAhzRVhAc_65PiMrUfftu9f

for noncommercial and commercial uses under MIT license, $0.28/$0.028/$0.42 per million input/cached/output tokens via

https://api-docs.deepseek.com/quick_start/pricing?utm_campaign=The%20Batch&utm_source=hs_email&utm_medium=email&_hsenc=p2ANqtz-_L4bJJ6vXg6Hs10je6IoHKFICRZs2QVtY2uv4WUTWxZX4HrpAhzRVhAc_65PiMrUfftu9f

Performance:

Comparable to DeepSeek-V3.1-Terminus across many benchmarks, processing inputs over 7,000 tokens 2 to 3 times faster

How it works:

The team modified DeepSeek-V3.1-Terminus with a sparse attention mechanism that, rather than attending to the entire input context, selectively processes only the most relevant tokens.

During training, a “lightning indexer,” a weighted similarity function, learned from 2.1 billion tokens of text to predict which tokens DeepSeek-V3.1-Terminus’ dense attention mechanism would focus on. Then the team fine-tuned all parameters on around 100 billion tokens of text to work with the indexer’s sparse token selections.

The team further fine-tuned the model by distilling five specialist models (versions of the pretrained DeepSeek-V3.2 base fine-tuned for reasoning, math, coding, agentic coding, and agentic search) into DeepSeek-V3.2-Exp.

## The team applied

https://arxiv.org/abs/2402.03300?utm_campaign=The%20Batch&utm_source=hs_email&utm_medium=email&_hsenc=p2ANqtz-_L4bJJ6vXg6Hs10je6IoHKFICRZs2QVtY2uv4WUTWxZX4HrpAhzRVhAc_65PiMrUfftu9f

to merge reasoning, agentic, and alignment training into a single stage. This approach avoided the catastrophic forgetting problem, in which new learning displaces old, that typically bedevils multi-stage reinforcement learning.

At inference, the indexer scores the relevance of each past token to the token being generated. It uses simple operations and FP8 precision (8-bit floating point numbers that are relatively imprecise but require less computation to process) to compute these scores quickly.

Based on these scores, instead of computing attention across all tokens in the current input context, the model selects and computes attention across the top 2,048 highest-scoring tokens, dramatically reducing computational cost.

In DeepSeek’s benchmark tests, DeepSeek-V3.2-Exp achieved substantial gains in efficiency with modest trade-offs in performance relative to its predecessor DeepSeek-V3.1-Terminus.

DeepSeek-V3.2-Exp cut inference costs for long input contexts by 6 to 7 times compared to DeepSeek-V3.1 Terminus. Processing 32,000 tokens of context, DeepSeek-V3.2-Exp cost around $0.10 per 1 million tokens versus $0.60. Processing 128,000 tokens of context, it cost $0.30 per 1 million tokens compared to $2.30.

DeepSeek-V3.2-Exp showed gains on tasks that involved coding and agentic behavior as well as some math problems. It surpassed DeepSeek-V3.1-Terminus on Codeforces coding challenges (2121 Elo versus 2046 Elo) and BrowseComp the browser-based agentic tasks (40.1 percent versus 38.5 percent). It also surpassed its predecessor on AIME 2025’s competition high-school math problems (89.3 percent versus 88.4 percent), which are more structured and have clearer solutions than those in HMMT (see below).

However, its performance showed slight degradation relative to DeepSeek-V3.2-Terminus across several tasks. It trailed its predecessor on GPQA-Diamond’s graduate-level science questions (79.9 percent versus 80.7 percent), HLE’s abstract-thinking challenges (19.8 percent versus 21.7 percent), HMMT 2025’s competitive high-school math problems (83.6 percent versus 86.1 percent), and Aider-Polyglot’s coding tasks (74.5 percent versus 76.1 percent).

Behind the news:

DeepSeek-V3.2-Exp is among the first large language models to

https://www.tomshardware.com/tech-industry/deepseek-new-model-supports-huawei-cann?utm_campaign=The%20Batch&utm_source=hs_email&utm_medium=email&_hsenc=p2ANqtz-_L4bJJ6vXg6Hs10je6IoHKFICRZs2QVtY2uv4WUTWxZX4HrpAhzRVhAc_65PiMrUfftu9f

with optimizations for domestic chips rather than adding these as an afterthought. The software has been adapted to run on chips by Huawei, Cambricon, and Hygon, following an

https://www.bloomberg.com/news/articles/2025-09-17/china-tells-companies-to-stop-buying-nvidia-s-repurposed-ai-chip?utm_campaign=The%20Batch&utm_source=hs_email&utm_medium=email&_hsenc=p2ANqtz-_L4bJJ6vXg6Hs10je6IoHKFICRZs2QVtY2uv4WUTWxZX4HrpAhzRVhAc_65PiMrUfftu9f

by China’s government to domestic AI companies not to use Nvidia chips. The government’s order followed reports that Chinese AI companies had

https://www.deeplearning.ai/the-batch/china-reconsiders-u-s-ai-processors-nvidia-and-amd-must-reassure-china-their-high-end-gpus-dont-pose-security-risk/?utm_campaign=The%20Batch&utm_source=hs_email&utm_medium=email&_hsenc=p2ANqtz-_L4bJJ6vXg6Hs10je6IoHKFICRZs2QVtY2uv4WUTWxZX4HrpAhzRVhAc_65PiMrUfftu9f

to use domestic chips rather than Nvidia chips, which are subject to U.S. export restrictions.

Why it matters:

## Even as prices have

https://www.deeplearning.ai/the-batch/falling-llm-token-prices-and-what-they-mean-for-ai-companies/?utm_campaign=The%20Batch&utm_source=hs_email&utm_medium=email&_hsenc=p2ANqtz-_L4bJJ6vXg6Hs10je6IoHKFICRZs2QVtY2uv4WUTWxZX4HrpAhzRVhAc_65PiMrUfftu9f

, the cost of processing LLM output tokens can make it prohibitively expensive to perform long-context tasks like analyzing large collections of documents, conversing across long periods of time, and refactoring large code repositories. DeepSeek’s implementation of sparse attention goes some distance toward remedying the issue.

We’re thinking:

DeepSeek-V3.2-Exp joins

https://www.deeplearning.ai/the-batch/alibabas-new-model-uses-hybrid-attention-layers-and-a-sparse-moe-architecture-for-speed-and-performance/?utm_campaign=The%20Batch&utm_source=hs_email&utm_medium=email&_hsenc=p2ANqtz-_L4bJJ6vXg6Hs10je6IoHKFICRZs2QVtY2uv4WUTWxZX4HrpAhzRVhAc_65PiMrUfftu9f

in experimenting with self-attention alternatives to improve the efficiency of large transformers. While Qwen3-Next combines Gated DeltaNet layers with gated attention layers, DeepSeek-V3.2-Exp uses dynamic sparse attention, suggesting that there’s still more efficiency to be gained by tweaking the transformer architecture.

## Subscribe to The Batch

## Stay updated with weekly AI News and Insights delivered to your inbox

