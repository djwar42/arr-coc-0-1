---
sourceFile: "DeepSeek R1: Understanding GRPO and Multi-Stage Training | by BavalpreetSinghh"
exportedBy: "Kortex"
exportDate: "2025-10-29T02:37:23.921Z"
---

# DeepSeek R1: Understanding GRPO and Multi-Stage Training | by BavalpreetSinghh

2612f27a-3d77-482e-b1fb-d094cf782322

DeepSeek R1: Understanding GRPO and Multi-Stage Training | by BavalpreetSinghh

483e8ade-e4a9-411f-9f61-e4deef0f3661

https://ai.plainenglish.io/deepseek-r1-understanding-grpo-and-multi-stage-training-5e0bbc28a281

## Open in app

## Medium Logo

## Artificial Intelligence in Plain English

New AI, ML and Data Science articles every day. Follow to join our 3.5M+ monthly readers.

DeepSeek R1: Understanding GRPO and Multi-Stage Training

## BavalpreetSinghh

8 min read · Jan 28, 2025

A rtificial intelligence has taken a significant leap forward with the release of

DeepSeek R1

, an open model that challenges OpenAI’s o1 in advanced reasoning tasks. Developed using an innovative technique called

Group Relative Policy Optimisation (GRPO)

and a multi-stage training approach, DeepSeek R1 sets new benchmarks for AI models in mathematics, coding, and general reasoning.

What sets DeepSeek R1 apart is its ability to solve complex tasks with remarkable accuracy and reasoning depth while maintaining a streamlined training process. This blog dives into the foundational methods, the training pipeline, and the innovations that make DeepSeek R1 an exceptional model in AI research.

Press enter or click to view image in full size Source: OpenAI

Understanding Group Relative Policy Optimization (GRPO)

Group Relative Policy Optimisation (GRPO)

is the core innovation driving DeepSeek R1’s exceptional reasoning abilities. Introduced in the DeepSeekMath paper, this reinforcement learning algorithm enhances model training by rethinking how rewards and optimisation are handled. GRPO replaces traditional methods like Proximal Policy Optimisation (PPO) with a simpler and more efficient approach tailored for large language models.

If you’re new to PPO and similar methods, you can check out my previous

https://medium.com/@bavalpreetsinghh/rlhf-ppo-vs-dpo-26b1438cf22b

to get an overview of what they are and how they work.

## Key Features of GRPO

No Value Function Model:

Unlike PPO, GRPO eliminates the need for a separate value function model. This simplifies training and reduces memory usage, making it more efficient.

Group-Based Advantage Calculation:

GRPO leverages a group of outputs for each input, calculating the baseline reward as the average score of the group. This group-based approach aligns better with reward model training, especially for reasoning tasks.

Direct KL Divergence Optimisation:

Instead of incorporating KL divergence into the reward signal (as in PPO), GRPO integrates it directly into the loss function, providing finer control during optimisation.

## Glimpse of how GRPO Works

The model generates multiple outputs for each prompt using the current policy.

Reward Scoring:

Each output is scored using a reward function. These scores can be rule-based (e.g., format or accuracy) or outcome-based (e.g., correctness in math or coding).

Advantage Calculation:

The average reward from the group serves as the baseline. The relative advantage of each output is calculated based on this baseline, with rewards normalised within the group.

Policy Optimisation:

Using the calculated advantages, the policy updates itself to maximise performance. The KL divergence term is incorporated directly into the loss function, ensuring the model balances exploration and stability.

## Press enter or click to view image in full size

## Deepdive into GRPO

If you’re just here for an overview, you can skip this part—the previous section should be enough. I don’t want you to feel overwhelmed, so no need to dive into this if it’s not necessary.

Group Relative Policy Optimisation (GRPO), a variant of Proximal Policy Optimisation (PPO), enhances mathematical reasoning abilities while concurrently optimising the memory usage of PPO.

Group Relative Policy Optimization: Comprehensive Explanation

### 1. Comparison Between PPO and GRPO

## The key difference between

Proximal Policy Optimization (PPO)

Group Relative Policy Optimization (GRPO)

lies in their approach to advantage estimation and computational efficiency. While PPO relies on a separate value model, GRPO eliminates this dependency, replacing it with group-based relative advantage estimation, reducing memory and computation costs.

Press enter or click to view image in full size Source: DeepSeekMath

### 2. Diagram Overview

In the diagram:

## The policy model generates outputs

for a given input

## A separate

value model

predicts a baseline v, used with Generalised Advantage Estimation (GAE) to compute advantages

## The reward

includes a KL penalty term computed using a reference model and reward model.

This architecture results in significant resource overhead.

## Multiple outputs

{o1,o2,…,oG}

are generated for each q, and their rewards

{r1,r2,…,rG}

are computed using the reward model.

Group computation normalises these rewards, providing relative advantages

A1, A2,..., AG

without a value model.

The KL divergence between the trained policy and reference model is added directly to the loss, simplifying training.

PPO: Mathematical Formulation

## PPO is an RL algorithm that optimises a

policy model

by maximising a

surrogate objective function

while ensuring training stability through clipping-based constraints. The key aspects are described below:

Press enter or click to view image in full size Press enter or click to view image in full size Press enter or click to view image in full size

## Takeaways from PPO

## Advantage Calculation

: PPO uses the GAE to reduce the variance in At, leveraging a learned value function Vψ as a baseline.

## Clipping Regularization

: Clipping in the surrogate objective ensures stability and prevents excessively large policy updates.

## KL Divergence Regularization

: The KL penalty in the reward discourages the policy from diverging too much from the reference model, promoting stable learning.

GRPO: Mathematical Formulation

Group Relative Policy Optimization (GRPO) simplifies PPO by removing the value model and using

group-based relative rewards

for baseline estimation. It is designed to efficiently fine-tune large language models (LLMs) while reducing computational overhead.

Press enter or click to view image in full size Press enter or click to view image in full size

## Takeaways from GRPO

## Eliminates the Value Model

: GRPO replaces the computationally expensive value model with group-based reward normalization, significantly reducing resource requirements.

## Leverages Group Comparisons

: By normalizing rewards within a group, GRPO aligns with the

pairwise comparison nature

of most reward models, ensuring better relative reward estimation.

## Simplifies KL Regularization

: GRPO directly regularizes the policy with a KL divergence term, avoiding the need for complex KL penalties in the reward.

## Outcome Supervision RL with GRPO

## Press enter or click to view image in full size

## Process Supervision RL with GRPO

Press enter or click to view image in full size Press enter or click to view image in full size

GRPO training involves iteratively updating the policy and reward model to maintain alignment. The steps are:

## Press enter or click to view image in full size

## Key Differences Between PPO and GRPO

## Value Model

: PPO uses a value model for advantage estimation, while GRPO eliminates it and relies on group-normalized rewards.

## KL Regularization

: PPO includes a KL penalty in the reward; GRPO directly regularizes the loss with a KL divergence term.

Reward Granularity:

PPO computes token-level rewards directly, while GRPO leverages group-relative rewards normalized across sampled outputs.

## Computational Efficiency

: GRPO is more efficient due to the removal of the value model and simpler advantage estimation.

Multi-Stage Training of DeepSeek R1

Training an advanced reasoning model like DeepSeek R1 requires more than just raw computational power — it demands a carefully structured training pipeline. To achieve superior reasoning and coherence, the DeepSeek team designed a

multi-stage training process

that combines supervised fine-tuning (SFT) with reinforcement learning (RL) using GRPO. This approach overcomes challenges like early instability in RL training and ensures that the model excels in diverse tasks.

Stage 1: Base to Supervised Fine-Tuning (SFT)

The journey began with fine-tuning the DeepSeek V3 base model using high-quality, chain-of-thought (CoT) data.

Data Collection:

Generated up to 10k token-long reasoning completions (CoT) using the R1-zero model and human annotators.

Enhance readability, coherence, and logical flow in the model’s outputs.

A solid foundation for reinforcement learning, reducing instability during subsequent training stages.

Stage 2: RL for Reasoning

GRPO was introduced to refine the model’s reasoning capabilities in tasks like mathematics, coding, and structured problem-solving.

Rule-Based Rewards:

Focused on accuracy (e.g., solving coding problems, verifying mathematical results).

Enforced formatting rules to ensure clarity, such as enclosing thought processes within specific tags (e.g.,  ‘reasoning’ ).

New Reward Signal:

A “language consistency” reward encouraged the model to maintain the same language throughout its outputs.

Significant improvements in reasoning performance, as evidenced by the AIME 2024 pass@1 score jump to 71.0%.

Stage 3: Rejection Sampling and SFT

To expand the model’s capabilities, a large synthetic dataset was generated using

Rejection Sampling (RS)

Dataset Creation:

The model from Stage 2 generated 600k reasoning-related samples.

Additional 200k samples focused on general-purpose tasks like writing and role-playing.

Data sourced from DeepSeek V3’s SFT dataset or regenerated with chain-of-thought included.

Broaden the model’s expertise beyond reasoning tasks into creative and general-purpose domains.

The model demonstrated greater versatility and coherence across a wider range of tasks.

Stage 4: RL for Helpfulness

In the final stage, GRPO was applied once again, but with a broader focus on

helpfulness and harmlessness

Combination of Reward Models:

Rule-based rewards ensured continued improvement in reasoning and accuracy.

Outcome-based rewards encouraged helpful and safe outputs.

A balanced model capable of handling complex reasoning tasks while maintaining clarity, safety, and user alignment.

Key Insights from Multi-Stage Training

Early SFT Stabilizes RL Training:

Fine-tuning the base model before applying RL techniques reduces training instability and accelerates convergence.

Rule-Based Rewards Are Effective:

Simple, targeted rewards (accuracy, format) often outperform complex reward models.

Rejection Sampling Improves Versatility:

Synthetic datasets generated through rejection sampling enhance the model’s adaptability to varied tasks.

By strategically alternating between supervised fine-tuning and reinforcement learning, the DeepSeek team overcame the challenges of RL cold starts and task-specific overfitting. This multi-stage pipeline ensured that DeepSeek R1 could excel in both reasoning and broader applications.

Stay tuned, will share more insights on it soon!

## GRPO Trainer

We're on a journey to advance and democratize artificial intelligence through open source and open science.

huggingface.co

GitHub - rsshyam/GRPO

Contribute to rsshyam/GRPO development by creating an account on GitHub.

Deepseek R1 for Everyone | Notion

made by neuralnets

trite-song-d6a.notion.site

Bite: How Deepseek R1 was trained

5 Minute Read on how Deepseek R1 was trained using Group Relative Policy Optimization (GRPO) and RL-focused multi-stage…

www.philschmid.de

Deepseek-v3 101 | Notion

author: @himanshutwts

lunar-joke-35b.notion.site

## Thank you for being a part of the community

Before you go:

## Be sure to

the writer ️👏

Follow us:

Check out CoFeed, the smart way to stay up-to-date with the latest in tech

Start your own free AI-powered blog on Differ

## Join our content creators community on Discord

For more content, visit

plainenglish.io

stackademic.com

https://stackademic.com/

https://stackademic.com/

https://stackademic.com/

https://stackademic.com/

Deepseek R1

https://stackademic.com/

## Published in  Artificial Intelligence in Plain English

32K followers

https://stackademic.com/

Last published  9 hours ago

https://ai.plainenglish.io/ai-threats-have-moved-out-of-the-lab-91c8fd8b32d0?source=post_page---post_publication_info--5e0bbc28a281---------------------------------------

New AI, ML and Data Science articles every day. Follow to join our 3.5M+ monthly readers.

## Written by  BavalpreetSinghh

522 followers

https://ai.plainenglish.io/ai-threats-have-moved-out-of-the-lab-91c8fd8b32d0?source=post_page---post_publication_info--5e0bbc28a281---------------------------------------

47 following

https://medium.com/@bavalpreetsinghh/following?source=post_page---post_author_info--5e0bbc28a281---------------------------------------

Consultant Data Scientist and AI ML Engineer @ CloudCosmos | Ex Data Scientist at Tatras Data | Reseacher @ Humber College | Ex Consultant @ SL2

Responses ( 4 )

## Text to speech

