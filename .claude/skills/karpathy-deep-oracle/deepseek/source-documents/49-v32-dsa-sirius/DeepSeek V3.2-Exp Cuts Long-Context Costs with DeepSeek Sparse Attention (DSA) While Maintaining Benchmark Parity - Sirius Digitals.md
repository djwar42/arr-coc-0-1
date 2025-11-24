---
sourceFile: "DeepSeek V3.2-Exp Cuts Long-Context Costs with DeepSeek Sparse Attention (DSA) While Maintaining Benchmark Parity - Sirius Digitals"
exportedBy: "Kortex"
exportDate: "2025-10-29T02:37:24.634Z"
---

# DeepSeek V3.2-Exp Cuts Long-Context Costs with DeepSeek Sparse Attention (DSA) While Maintaining Benchmark Parity - Sirius Digitals

9a4c151a-30cc-4f2d-abc6-edfcaa42e666

DeepSeek V3.2-Exp Cuts Long-Context Costs with DeepSeek Sparse Attention (DSA) While Maintaining Benchmark Parity - Sirius Digitals

9cc5c65f-4907-4269-b162-66a6bf43842f

https://siriusdigital.us/deepseek-v3-2-exp-cuts-long-context-costs-with-deepseek-sparse-attention-dsa-while-maintaining-benchmark-parity/

info@siriusdigital.us

https://siriusdigital.us/contact/

https://siriusdigital.us/contact/

https://siriusdigital.us/contact/

+234 9065656120

https://siriusdigital.us/contact/

https://siriusdigital.us/contact/

https://siriusdigital.us/contact/

## Digital Marketing

https://siriusdigital.us/contact/

## Website Development

https://siriusdigital.us/contact/

## Graphic Designing

https://siriusdigital.us/contact/

## Editorial Design

https://siriusdigital.us/contact/

SEO & Content Writing

https://siriusdigital.us/contact/

https://siriusdigital.us/contact/

https://siriusdigital.us/contact/

https://siriusdigital.us/contact/

https://siriusdigital.us/contact/

## Save Cost and Increase Efficiency

## Get a Quote   Get a Quote

https://siriusdigital.us#

https://siriusdigital.us#

https://siriusdigital.us#

## Digital Marketing

https://siriusdigital.us#

## Website Development

https://siriusdigital.us#

## Graphic Designing

https://siriusdigital.us#

## Editorial Design

https://siriusdigital.us#

SEO & Content Writing

https://siriusdigital.us#

https://siriusdigital.us#

https://siriusdigital.us#

https://siriusdigital.us#

https://siriusdigital.us#

## Save Cost and Increase Efficiency

DeepSeek V3.2-Exp Cuts Long-Context Costs with DeepSeek Sparse Attention (DSA) While Maintaining Benchmark Parity

https://siriusdigital.us#

DeepSeek V3.2-Exp Cuts Long-Context Costs with DeepSeek Sparse Attention (DSA) While Maintaining Benchmark Parity   September 30, 2025   by

## Siriusdigitals

https://siriusdigital.us/author/siriusdigitals/

https://siriusdigital.us/author/siriusdigitals/

DeepSeek V3.2-Exp Cuts Long-Context Costs with DeepSeek Sparse Attention (DSA) While Maintaining Benchmark Parity

## Table of contents

## DeepSeek released

DeepSeek-V3.2-Exp,

an “intermediate” update to V3.1 that adds

DeepSeek Sparse Attention (DSA)

—a trainable sparsification path aimed at long-context efficiency. DeepSeek also reduced

API prices by 50%+

, consistent with the stated efficiency gains.

DeepSeek-V3.2-Exp

keeps the V3/V3.1 stack (MoE + MLA) and inserts a

two-stage attention

path: (i) a lightweight “indexer” that scores context tokens; (ii) sparse attention over the selected subset.

https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf

FP8 index → top-k selection → sparse core attention

DeepSeek Sparse Attention (DSA)

splits the attention path into two compute tiers:

(1) Lightning indexer (FP8, few heads): For each query token 
  ℎ 
  𝑡 
  ∈ 
  𝑅 
  𝑑 
  h 
  t

∈R 
  d 
  , a lightweight scoring function computes index logits 
  𝐼 
  𝑡 
  , 
  𝑠 
  I 
  t,s

against preceding tokens 
  ℎ 
  𝑠 
  h 
  s

. It uses small indexer heads with a ReLU nonlinearity for throughput. Because this stage runs in FP8 and with few heads, its wall-time and FLOP cost are minor relative to dense attention.

(2) Fine-grained token selection (top-k): The system selects only the top-k=2048 key-value entries for each query and then performs standard attention only over that subset. This changes the dominant term from 
  𝑂 
  ( 
  𝐿 
  2 
  ) 
  O(L 
  2 
  ) to 
  𝑂 
  ( 
  𝐿 
  𝑘 
  ) 
  O(Lk) with 
  𝑘 
  ≪ 
  𝐿 
  k≪L, while preserving the ability to attend to arbitrarily distant tokens when needed.

Training signal:

The indexer is trained to imitate the dense model’s head-summed attention distribution via

KL-divergence

, first under a short

dense warm-up

(indexer learns targets while the main model is frozen), then during

sparse training

where gradients for the indexer remain separate from the main model’s language loss. Warm-up uses ~

tokens; sparse stage uses ~

tokens with

for the main model.

Instantiation:

## DSA is implemented under

(Multi-head Latent Attention) in

for decoding so each latent KV entry is shared across query heads, aligning with the kernel-level requirement that KV entries be reused across queries for throughput.

https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf

Lets Talk about it’s efficiency and accuracy

Costs vs. position (128k)

: DeepSeek provides per-million-token cost curves for

clusters (reference price $2/GPU-hour). Decode costs fall substantially with DSA; prefill also benefits via a masked MHA simulation at short lengths. While the exact 83% figure circulating on social media maps to “

~6× cheaper decode at 128k

,” treat it as

DeepSeek-reported

until third-party replication lands.

Benchmark parity:

## The released table shows

MMLU-Pro = 85.0

(unchanged), small movement on GPQA/HLE/HMMT due to fewer reasoning tokens, and flat/positive movement on agentic/search tasks (e.g., BrowseComp 40.1 vs 38.5). The authors note the gaps close when using intermediate checkpoints that produce comparable token counts.

Operational signals:

Day-0 support in

suggests the kernels and scheduler changes are production-aimed, not research-only. DeepSeek also references

(indexer logits), and

(sparse kernels) for open-source kernels.

## DeepSeek says API prices were cut by

, consistent with model-card messaging about efficiency and Reuters/TechCrunch coverage that the release targets lower long-context inference economics.

DeepSeek V3.2-Exp shows that trainable sparsity (DSA) can hold benchmark parity while materially improving long-context economics: official docs commit to

50%+ API price cuts

day-0 runtime support

already available, and community threads claim larger decode-time gains at

that warrant independent replication under matched batching and cache policies. The near-term takeaway for teams is simple: treat V3.2-Exp as a drop-in A/B for RAG and long-document pipelines where O(L2)O(L^2)O(L2) attention dominates costs, and validate end-to-end throughput/quality on your stack.

1) What exactly is DeepSeek V3.2-Exp?

V3.2-Exp is an

experimental, intermediate

update to V3.1-Terminus that introduces DeepSeek Sparse Attention (DSA) to improve long-context efficiency.

2) Is it truly open source, and under what license?

Yes. The repository and model weights are licensed under

, per the official Hugging Face model card (License section).

3) What is DeepSeek Sparse Attention (DSA) in practice?

DSA adds a lightweight indexing stage to score/select a small set of relevant tokens, then runs attention only over that subset—yielding “fine-grained sparse attention” and reported long-context training/inference efficiency gains while keeping output quality on par with V3.1.

## Check out the

. Feel free to check out our

. Also, feel free to follow us on

and don’t forget to join our

and Subscribe to

The post appeared first on .

ChatGPT Pulse — A new AI feature that turns chats into custom morning briefs

Delinea Released an MCP Server to Put Guardrails Around AI Agents Credential Access

## Leave a Comment

## Cancel reply

## Recent Posts

## PayPal Unveils Open Source AI Initiative to Shape the Future of Commerce

Google Gemini Unveils Voice Assistant with 100 Innovative Features for Home Use

How Exploration Agents like Q-Learning, UCB, and MCTS Collaboratively Learn Intelligent Problem-Solving Strategies in Dynamic Grid Environments

## Missouri Modernizes Financial Operations Using Oracle Fusion Cloud Suite

IBM Study: 66% of EMEA Enterprises See Major Productivity Boosts from AI

## Recent Comments

No comments to show.

## Latest Posts

## Siriusdigitals

https://siriusdigital.us/author/siriusdigitals/

## PayPal Unveils Open Source AI Initiative to Shape the Future of Commerce

https://siriusdigital.us/author/siriusdigitals/

## Siriusdigitals

https://siriusdigital.us/author/siriusdigitals/

Google Gemini Unveils Voice Assistant with 100 Innovative Features for Home Use

https://siriusdigital.us/author/siriusdigitals/

## Siriusdigitals

https://siriusdigital.us/author/siriusdigitals/

How Exploration Agents like Q-Learning, UCB, and MCTS Collaboratively Learn Intelligent Problem-Solving Strategies in Dynamic Grid Environments

https://siriusdigital.us/author/siriusdigitals/

## AI for Business

https://siriusdigital.us/author/siriusdigitals/

AI Tools & Reviews

https://siriusdigital.us/author/siriusdigitals/

Branding & Identity

https://siriusdigital.us/author/siriusdigitals/

## Business Tips

https://siriusdigital.us/author/siriusdigitals/

## Business Website Optimization

https://siriusdigital.us/author/siriusdigitals/

## Design Inspiration

https://siriusdigital.us/author/siriusdigitals/

## Digital Marketing

https://siriusdigital.us/author/siriusdigitals/

GPT & Chatbots

https://siriusdigital.us/author/siriusdigitals/

## Tech for Creatives

https://siriusdigital.us/author/siriusdigitals/

Tools & Software

https://siriusdigital.us/author/siriusdigitals/

## Uncategorized

https://siriusdigital.us/author/siriusdigitals/

## AI automation

https://siriusdigital.us/author/siriusdigitals/

## AI for business

https://siriusdigital.us/author/siriusdigitals/

## AI in web design

https://siriusdigital.us/author/siriusdigitals/

## AI tools and apps

https://siriusdigital.us/author/siriusdigitals/

## Branding and identity

https://siriusdigital.us/author/siriusdigitals/

## ChatGPT for business

https://siriusdigital.us/author/siriusdigitals/

## Client acquisition

https://siriusdigital.us/author/siriusdigitals/

## Design tools and software

https://siriusdigital.us/author/siriusdigitals/

## Digital transformation

https://siriusdigital.us/author/siriusdigitals/

## Landing page ideas

https://siriusdigital.us/author/siriusdigitals/

## Logo design inspiration

https://siriusdigital.us/author/siriusdigitals/

## Marketing strategies

https://siriusdigital.us/author/siriusdigitals/

Mobile-friendly websites

https://siriusdigital.us/author/siriusdigitals/

## Online branding

https://siriusdigital.us/author/siriusdigitals/

## Productivity hacks

https://siriusdigital.us/author/siriusdigitals/

## Small business growth

https://siriusdigital.us/author/siriusdigitals/

## Tech tools for business

https://siriusdigital.us/author/siriusdigitals/

## Website optimization

https://siriusdigital.us/author/siriusdigitals/

Support@siriusdigital.us

Mon - Sat: 8 AM to 6 PM  Sunday: CLOSED

We work with a passion of taking challenges and creating new ones in advertising sector.

## Meet the Team

## Our Services

## Latest News

## Improve Your Digital Business

## Improve Your Digital Business There are many varia

## Five Tips How To Start Business

https://siriusdigital.us/project/improve-your-digital-business-3/

## Five Tips How To Start Business There are many var

## Digital Marketing Pricing Guide

https://siriusdigital.us/project/five-tips-how-to-start-business-3/

## Digital Marketing Pricing Guide There are many var

## How Do We Keep Up with Digital Marketing

https://siriusdigital.us/project/digital-marketing-pricing-guide-3/

## How Do We Keep It There are many variations of pas

## Improve Your Digital Business

https://siriusdigital.us/project/how-do-we-keep-it-2/

## Improve Your Digital Business There are many varia

## Get the Website Solutions

https://siriusdigital.us/project/improve-your-digital-business-2/

## Get the Website Solutions There are many variation

Subscribe our newsletter to get our latest update & news

[mc4wp_form]

© All Copyright 2025 by siriusdigital.us

## Terms of Use

https://siriusdigital.us/project/get-the-website-solutions-4/

