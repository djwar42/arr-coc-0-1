---
sourceFile: "DeepSeek - Where there is a will, there is a way... - Dr. Robert Li"
exportedBy: "Kortex"
exportDate: "2025-10-29T02:37:22.399Z"
---

# DeepSeek - Where there is a will, there is a way... - Dr. Robert Li

d6a0affa-9c67-4b8c-adde-88e27bd98e24

DeepSeek - Where there is a will, there is a way... - Dr. Robert Li

75bc7cb1-e649-4c67-b125-40298881b612

https://drli.blog/posts/DeepSeek/

Dr. Robert Li

DeepSeek - Where there is a will, there is a way...

02 Feb 2025

## Model Development

## Architecture

Geo-Politics

## Optimization

## Introduction

Last month, a Chinese startup named DeepSeek released an open-source language model that outperformed many Western counterparts while using fewer resources. The story behind this achievement reveals how limitations can fuel creativity, reshape industries, and challenge assumptions about technological dominance.

There were three critical factors in DeepSeek’s rise:

## Architectural breakthroughs that redefined efficiency

## Hardware optimizations born from necessity

## The unintended consequences of geopolitical trade restrictions

This is an important moment that signals an inflection point for open-source AI model development and it shows that human ingenuity knows no bounds. Here are some of my quick thoughts…

## Architectural Efficiency

Rethinking Mixture-of-Experts (MoE)

Traditional MoE models activate subsets of neural networks (“experts”) based on input needs. DeepSeekMoE introduced two critical changes:

Specialized vs. Shared Experts

: 64 task-specific experts handle niche problems, while 2 shared experts manage general patterns. This split reduced redundant calculations by 40% compared to standard MoE designs.

## Load Balancing Without Auxiliary Loss

: Earlier MoE models used extra computations to distribute workloads evenly. DeepSeekMoE achieved balance through smarter routing, cutting training costs by 15%.

The result? A 671B-parameter model that activates only 37B parameters per task—like having a toolbox where each tool automatically adjusts to the job.

## Memory Efficiency Through Compression

The Multi-Head Latent Attention (MLA) system tackled a major bottleneck: memory usage. Traditional models store detailed “key-value” data for every token in a conversation. MLA compresses this data into latent vectors, shrinking memory requirements by 90% while maintaining accuracy.

Imagine condensing a 300-page manual into a 30-page cheat sheet without losing essential information.

GRPO: Reinventing RL

The real breakthrough described in DeepSeek’s paper:

DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

https://arxiv.org/abs/2402.03300

, however, was

Group Relative Policy Optimization (GRPO)

which reimagines how language models learn reasoning skills. Unlike traditional methods like Proximal Policy Optimization (PPO), which rely on separate critic models to evaluate outputs, GRPO eliminates this component entirely.

## How GRPO Works

## Group Sampling

: For each prompt, GRPO generates multiple responses (e.g., 64 variations of a math solution).

## Relative Advantage Calculation

: Instead of using a critic model to score each response, GRPO compares outputs within the group. The advantage for each response is calculated as:

\[A_i = \frac{R(r_i) - \text{mean}(R(\mathcal{G}))}{\text{std}(R(\mathcal{G}))}\]

where \(R(r_i)\) is the reward for response \(i , \mathcal{G}\) is the group.

## Optimization

: The model updates its policy to favor responses with higher relative advantages, using a clipped loss function excluding extreme solutions, to ensure the model remained relatively stable.

This has already been added to the HuggingFace Transformers library for easy and immediate use:

## GRPO Trainer

https://huggingface.co/docs/trl/main/en/grpo_trainer

. It supports generic Outcome Reward Models (ORM) and custom reward functions, that can be used to implement Rule-Based Reward Models.

For Deepseek R1, DeepSeek implemented a 3 tiered Rule-Based Reward Models to verify the correctness of generated solutions:

## Accuracy Rewards

## Binary rewards based on whether the final answer was correct

For math problems, answers had to be in a specified format and reverse-engineered to ensure matching the original problem

For coding problems, compilers tested solutions against predefined test cases

## Format Rewards

Rewarded based on the proper use of thinking tags ('<think>' and '<\/think>')

Cosine similarity scoring against templates to encourage outputting of clear structured reasoning.

## Language Quality

## Percentage of tokens that matched the target language

## Rewarded based on consistent language style

The goal was to prevent language mixing in Chain of Thought (CoT) reasoning

As DeepSeek’s CTO noted:

“GRPO isn’t just an algorithm—it’s a philosophy. Why build complex systems when simple comparisons work better?”

. This approach has already influenced Meta and Google to rethink their RL strategies.

## Key Innovations

## No Critic Model

: Removes the need for a separate neural network to estimate value functions, cutting memory usage by 50%.

## Simplified Training

: Group comparisons replace complex reward modeling, reducing computational overhead.

## Efficient Exploration

: Generating multiple responses per prompt encourages diverse reasoning paths without costly external feedback.

Real-World Impact

## DeepSeek applied GRPO to train

DeepSeek-R1

, a 32B-parameter model that achieved:

60.9% accuracy

on the MATH benchmark via self-consistency checks

95% lower training costs

compared to PPO-based approaches

Allowing for single-GPU fine-tuning

to create specialised models.

## Low Level Optimization of Hardware

## Bypassing CUDA for PTX

Nvidia’s CUDA framework simplifies GPU programming but limits low-level control. DeepSeek engineers dropped to PTX—Nvidia’s assembly-like language—to:

Reprogram 20 out of 132 streaming multiprocessors (SMs) on each H800 GPU for data routing

## Create custom kernels that merged computation and communication phases

Implement FP8 precision with 4-bit exponents and 3-bit mantissas, reducing memory traffic by 60%

These optimizations compensated for the H800’s limited memory bandwidth compared to banned H100 chips.

## The DualPipe System

Training large models requires constant data shuffling between GPUs. DeepSeek’s DualPipe algorithm:

## Overlaps computation and communication phases

Uses spare SMs as on-chip network routers

## Balances NVLink and InfiniBand traffic dynamically

This approach cut idle GPU time from 30% to under 5%, making their 2,048 H800 cluster perform like a 10,000-GPU setup.

## Chip Ban Forcing Function

## Forced Innovation Cycle

The 2022 U.S. export controls barred Chinese firms from obtaining H100 GPUs. DeepSeek’s response followed a clear pattern:

: H800 chips had half the memory bandwidth of H100s

: MLA compression reduced memory needs

: Comparable performance at lower cost

As Lawrence Ward, a partner at Dorsey & Whitney, noted: “Sanctions initiated innovation. DeepSeek optimized software because they couldn’t access better hardware”.

## Cost Efficiency Gains

DeepSeek-V3

## Comparable Model

Training Cost   $5.6M   $100M+   Active Parameters   37B   110B   Energy per Query   0.8 kWh   3.2 kWh

Data sourced from DeepSeek’s technical reports

The Sanctions Paradox: Why Restrictions Often Fail

Lesson 1: Scarcity Breeds Creativity

The chip ban forced Chinese engineers to rethink every layer of AI development:

Limited GPU memory bandwidth   MLA compression for smaller data footprints   Banned H100 chips   Distributed training across H800 clusters   No access to advanced EDA tools   Open-source alternatives + legacy software hacks

As Brian Colello from Morningstar observed:

“Processor constraints led to creative training techniques. DeepSeek’s performance defies expectations given their hardware”.

Lesson 2: The Resilience of Open Ecosystems

## Collaborative Resource Pooling

: Startups shared GPU clusters via platforms like GMI Cloud, reducing individual costs.

## Black Market Adaptability

: Despite sanctions, Nvidia chips reached China through third countries, with smuggling networks routing GPUs via Malaysia and Vietnam.

Lesson 3: The Law of Diminishing Returns

Sanctions initially slowed China’s AI progress but couldn’t halt it:

: Chinese models lagged 2-3 years behind U.S. counterparts

: DeepSeek-R1 matches GPT-4o1 in reasoning benchmarks

As Stanford’s AI Index 2025 notes:

“Compute efficiency improvements are closing the gap faster than expected.”

## Why Sanctions Have Limits

### 1. Global Supply Chains Are Leaky

Third-Party Workarounds

: Chinese firms leased cloud compute from U.S. providers (AWS, Google Cloud) to train models indirectly.

Chip “Laundering”

: Export-controlled H100s reached China via shell companies in Singapore and Saudi Arabia.

### 2. Innovation Can’t Be Bottled

## Legacy Hardware Gains

: Older chips saw 3-5x efficiency boosts through software alone (DeepSeek’s MLA).

## Algorithmic Breakthroughs

: Chain-of-thought reasoning and MoE architectures reduced reliance on brute-force compute.

### 3. Markets Adapt Faster Than Policies

Cost-Driven Shifts

: DeepSeek’s $6M training cost (vs. OpenAI’s $100M+) reshaped industry priorities toward efficiency over scale.

## Domestic Chip Growth

: SMIC’s 7nm process and Huawei’s Ascend 910B filled gaps, albeit with 30% lower yields than TSMC.

History shows prohibitions often backfire:

: U.S. oil embargo on Japan accelerated synthetic fuel research

: COCOM restrictions spurred China’s semiconductor industry

: Chip bans fueled AI efficiency breakthroughs

DeepSeek’s story isn’t about circumventing rules—it’s about human ingenuity thriving under pressure. As Marc Andreessen noted, this is AI’s

“Sputnik moment”

: a reminder that barriers breed creativity.

The buds of Open-Source AI

## Democratizing AI Development

DeepSeek’s approach to reinforcement learning and releasing it, the model, the weights and a research paper under an open source MIT license provides the opportunity for a significant shift in AI model development:

Cost-Effective Innovation

Pure reinforcement learning eliminates the need for extensive labeled datasets, dramatically reducing development costs. DeepSeek’s training method costs approximately 95% less than traditional approaches, making advanced AI development accessible to smaller teams and organizations.

## Technical Advantages

The pure RL approach offers several benefits:

Models develop reasoning capabilities through self-directed exploration rather than mimicking labeled examples

The GRPO framework removes the need for an LLM critic, further reducing computational overhead

Training can be targeted to specific domains without requiring domain-specific labeled data

## Scalable Development

DeepSeek’s methodology shows particular promise in specialized applications:

The 32B distilled model achieves performance comparable to larger proprietary models

Smaller teams can fine-tune models for specific use cases with limited hardware

## The approach allows for continuous improvement without additional labeled data

## Industry Impact

This development path creates new opportunities:

: A 3-person team can fine-tune the 1.5B model on a single GPU

## Researchers

: Academic groups replicated DeepSeek’s results within 72 hours of release

: A logistics company cut document processing costs by 60% using the 7B model

## Future Implications

The success of DeepSeek’s RL-first approach suggests a shift in AI development - this is the reason why investment analysts have been fearing for the long term growth potential of nVidia and power suppliers who rely on brute force energy consumption. This is a good thing:

We are moving away from data-hungry supervised learning

This can lower prices (OpenAI reduced GPT-4 API costs by 40% post-DeepSeek)

This forces improved transparency (Google published previously internal efficiency metrics)

This accelerates innovation cycles (Meta’s Llama 4 release moved up by 6 months)

DeepSeek’s advances should, however, be seen as a boon to all players in the AI ecosystem - the inverse of what perhaps chip manufacturer’s outlooks are.

DeepSeek’s approach to low level optimization is highly hardware locked, and therefore it’s benefits are limited to this generation of model and this specific hardware. The actionable lesson is that PTX optimization can be applied to newer generation hardware as well for even more efficient performance, although that is in exchange for significantly more technical debt. It teaches engineers that there are more optimizations, albeit, more difficult optimizations, to be had, versus simply “moar powa”.

For analysts, this surfaces the potential for vertical integration of the hardware and software stack as being a defensible moat. Trade barriers make it hard for Chinese competitors as there is still a significant gap in the capabilities of local chip manufacturers (such as SMIC), however, they could throw all their efforts into creating NPU or even more specialised ASIC chipsets, moving laterally to compete directly with the likes of nVidia and, in the mobile sector, Qualcomm.

This is, therefore an opportunity for non-Chinese companies to more easily create defensible moats through partnerships with these kind of chip manufacturers or even to commission to design and manufacturing of their own. This is what to look out for in this sector.

DeepSeek’s story echoes historical precedents:

1940s: Radar miniaturization (forced by aircraft weight limits)

1970s: Japanese fuel-efficient cars (sparked by oil crises)

now in the 2020s: Efficient AI models (born from compute constraints)

Constraints don’t stifle progress—they redirect it. As is often said, necessity is the mother of invention. The companies that succeed are often not those with the most resources, but those who can most innovatively deploy what resources they have.

True open-source AI (not OpenAI) now has its proof point. The next breakthroughs won’t come from labs with infinite budgets, but from engineers working under stifling constraints (even if that is the compute efficient frontier) rearchitecting and reimagining model structures, like those at DeepSeek.

[1] https://www.aljazeera.com/economy/2025/1/28/why-chinas-ai-startup-deepseek-is-sending-shockwaves-through-global-tech

[2] https://www.businessinsider.com/china-deepseek-chip-restrictions-exports-imports-2025-1

[3] https://www.csis.org/analysis/choking-chinas-access-future-ai

[4] https://www.sify.com/ai-analytics/unbelievable-how-chinas-outsmarting-us-chip-ban-to-dominate-ai/

[5] https://www.bbc.com/news/articles/czepw096wy4o

[6] https://www.brookings.edu/articles/understanding-the-limits-of-sanctions/

[7] https://www.csis.org/analysis/collateral-damage-domestic-impact-us-semiconductor-export-controls

[8] https://www.bbc.com/news/articles/c0qw7z2v1pgo

[9] https://www.technologyreview.com/2025/01/24/1110526/china-deepseek-top-ai-despite-sanctions/

[10] https://tecex.com/ai-chips-trade-wars-sanctions/

[11] https://qz.com/u-s-investigates-chinas-deepseek-for-potential-use-of-1851752981

[12] https://www.newsweek.com/joe-biden-china-deepseek-artificial-intelligence-2021974

[13] https://www.bbc.com/news/articles/cwy7x84qvv4o

[14] https://www.reuters.com/technology/us-looking-into-whether-deepseek-used-restricted-ai-chips-source-says-2025-01-31/

[15] https://www.cio.com/article/3805170/us-gpu-export-limits-could-bring-cold-war-to-ai-data-center-markets.html

[16] https://www.youtube.com/watch?v=al9kE8DWfgM

[17] https://www.cnn.com/2025/01/27/tech/deepseek-stocks-ai-china/index.html

[18] https://www.mi-3.com.au/27-01-2025/massive-gen-ai-disruption-incoming-deepseekr1-trigger-massive-gen-ai-commoditisation

[19] https://www.reuters.com/technology/artificial-intelligence/what-is-deepseek-why-is-it-disrupting-ai-sector-2025-01-27/

[20] https://www.nytimes.com/2025/01/28/business/economy/deepseek-china-us-chip-controls.html

[21] https://research.oregonstate.edu/ori/export-control-international-compliance/embargoed-and-sanctioned-countries

[22] https://www.lowyinstitute.org/publications/chips-subsidies-security-great-power-competition

[23] https://ia.acs.org.au/article/2025/us-to-bar-china–russia-from-advanced-ai-chips.html

[24] https://www.dfat.gov.au/international-relations/security/sanctions/about-sanctions

[25] https://www.csis.org/analysis/balancing-ledger-export-controls-us-chip-technology-china

[26] https://www.reuters.com/technology/lawmakers-urge-trump-consider-new-curbs-nvidia-chips-used-by-chinas-deepseek-2025-01-30/

[27] https://www.rahmanravelli.co.uk/expertise/sanctions/articles/sanctions-vs-embargoes-international-trade-restrictions-explained/

[28] https://www.scmp.com/news/china/article/3296994/reacting-deepseek-us-senate-bill-would-separate-us-and-china-efforts-develop-ai

[29] https://world-toolkit.yale.edu/regulated-activity/economic-embargoes-trade-sanctions-prohibited-parties-and-anti-boycotting-laws

[30] https://www.economist.com/science-and-technology/2024/09/19/chinas-ai-firms-are-cleverly-innovating-around-chip-bans

[31] https://www.cyberdaily.au/digital-transformation/11656-deepseek-allegedly-used-banned-chips-says-scale-ai-ceo

[32] https://time.com/7204164/china-ai-advances-chips/

[33] https://www.tookitaki.com/compliance-hub/embargo-vs-sanction-understanding-international-trade-restrictions

[34] https://business.nab.com.au/wp-content/uploads/2014/05/sanctions-embargoes.pdf

[35] https://www.cigionline.org/articles/in-the-global-ai-chips-race-china-is-playing-catch-up/

[36] https://www.restack.io/p/fine-tuning-answer-supervised-vs-reinforcement-cat-ai

[37] https://arxiv.org/html/2501.12948v1

[38] https://www.vellum.ai/blog/the-training-of-deepseek-r1-and-ways-to-use-it

[39] https://www.grokmountain.com/p/deepseeks-ai-dojo-harnessing-the

[40] https://victorysquarepartners.com/training-ai-with-pure-reinforcement-learning-insights-from-deepseek-r1/

[41] https://fireworks.ai/blog/deepseek-r1-deepdive

[42] https://www.technologyreview.com/2025/01/31/1110740/how-deepseek-ripped-up-the-ai-playbook-and-why-everyones-going-to-follow-it/

[43] https://www.linkedin.com/pulse/explaining-methodology-behind-deepseek-r1-rana-gujral-ajmcc

[44] https://fireworks.ai/blog/reinforcement-learning-with-verifiable-reward

[45] https://www.linkedin.com/pulse/beyond-hype-why-deepseek-r1s-pure-rl-approach-ai-marian-dumitrascu-vxk2c

[46] https://towardsdatascience.com/exploring-deepseeks-r1-training-process-5036c42deeb1

[47] https://www.forbes.com/sites/janakirammsv/2025/01/26/all-about-deepseekthe-chinese-ai-startup-challenging-the-us-big-tech/

[48] https://www.datacamp.com/tutorial/fine-tuning-deepseek-r1-reasoning-model

[49] https://c3.unu.edu/blog/deepseek-r1-pioneering-open-source-thinking-model-and-its-impact-on-the-llm-landscape

[50] https://www.linkedin.com/pulse/deepseek-r1-reinforcement-learning-vs-gpt-based-llms-michael-kirch-np49c

[51] https://www.linkedin.com/pulse/deepseeks-revolutionary-approach-crafting-r1-model-through-mersch-eownc

[52] https://www.techtarget.com/whatis/feature/DeepSeek-explained-Everything-you-need-to-know

[53] https://venturebeat.com/ai/deepseek-r1s-bold-bet-on-reinforcement-learning-how-it-outpaced-openai-at-3-of-the-cost/

[54] https://www.deeplearning.ai/the-batch/how-deepseek-r1-and-kimi-k1-5-use-reinforcement-learning-to-improve-reasoning/

[55] https://www.reddit.com/r/reinforcementlearning/comments/1icrgt5/who_is_rewarding_deepseek_r1_in_rl_you_need_some/

[56] https://www.youtube.com/watch?v=DCqqCLlsIBU

[57] https://www.nature.com/articles/d41586-025-00259-0

[58] https://dev.to/aws/takeaways-from-the-deepseek-r1-model-2dli

[59] https://www.inferless.com/learn/the-ultimate-guide-to-deepseek-models

[60] https://www.reddit.com/r/LLMDevs/comments/1ibhpqw/how_was_deepseekr1_built_for_dummies/*

[61] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/13695847/cbd7998d-74f3-4829-8da6-ced05dfd9213/paste.txt

[62] https://pub.towardsai.net/group-relative-policy-optimization-grpo-illustrated-breakdown-explanation-684e71b8a3f2?gi=3049aee435c4

[63] https://blog.dataopslabs.com/deepseek-r1-efficient-reinforcement-learning-with-grpo

[64] https://www.thoughtworks.com/en-au/insights/blog/generative-ai/demystifying-deepseek

[65] https://towardsai.net/p/l/group-relative-policy-optimization-grpo-illustrated-breakdown-explanation

[66] https://yugeten.github.io/posts/2025/01/ppogrpo/

[67] https://arxiv.org/html/2501.12948v1

[68] https://www.youtube.com/watch?v=bAWV_yrqx4w

[69] https://planetbanatt.net/articles/deepseek.html

[70] https://www.linkedin.com/pulse/group-relative-policy-optimization-grpo-reinforcement-zahir-shaikh-f2cuf

[71] https://medium.com/@aipapers/deepseek-r1-paper-explained-a-new-rl-llms-era-in-ai-6d3271401270

[72] https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/mini-deepseek-r1-aha-grpo.ipynb

[73] https://www.linkedin.com/pulse/how-grpo-used-deepseek-could-revolutionize-ravi-lingarkar-jdglf

[74] https://arxiv.org/pdf/2402.03300.pdf

[75] https://www.youtube.com/watch?v=QdEuh2UVbu0

[76] https://www.linkedin.com/pulse/deepseek-r1-reinforcement-learning-llm-group-relative-mitul-tiwari-c8gmf

[77] https://www.linkedin.com/pulse/deepseek-ai-revolution-you-didnt-see-coming-valtech-aksfe

[78] https://github.com/deepseek-ai/DeepSeek-Math/actions

[79] https://www.reddit.com/r/ChatGPTPro/comments/1ibph6u/grpo_group_relative_policy_optimization/

