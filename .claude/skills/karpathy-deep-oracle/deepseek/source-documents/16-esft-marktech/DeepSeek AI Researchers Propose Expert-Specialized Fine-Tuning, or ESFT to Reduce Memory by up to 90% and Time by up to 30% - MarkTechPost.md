---
sourceFile: "DeepSeek AI Researchers Propose Expert-Specialized Fine-Tuning, or ESFT to Reduce Memory by up to 90% and Time by up to 30% - MarkTechPost"
exportedBy: "Kortex"
exportDate: "2025-10-29T02:37:23.139Z"
---

# DeepSeek AI Researchers Propose Expert-Specialized Fine-Tuning, or ESFT to Reduce Memory by up to 90% and Time by up to 30% - MarkTechPost

8830f511-3a8e-4bae-8901-5392f0731cff

DeepSeek AI Researchers Propose Expert-Specialized Fine-Tuning, or ESFT to Reduce Memory by up to 90% and Time by up to 30% - MarkTechPost

9ca25ed0-b2fd-4e14-a68c-219de56f4ff9

https://www.marktechpost.com/2024/07/06/deepseek-ai-researchers-propose-expert-specialized-fine-tuning-or-esft-to-reduce-memory-by-up-to-90-and-time-by-up-to-30/

Open Source/Weights

## Enterprise AI

## Sponsorship

## Premium Content

## Read our exclusive articles

Open Source/Weights

## Enterprise AI

## Sponsorship

Open Source/Weights

## Enterprise AI

## Sponsorship

## AI Paper Summary

DeepSeek AI Researchers Propose Expert-Specialized Fine-Tuning, or ESFT to Reduce Memory by...

## AI Paper Summary

## Artificial Intelligence

## Applications

## Editors Pick

## Language Model

## Large Language Model

## Machine Learning

DeepSeek AI Researchers Propose Expert-Specialized Fine-Tuning, or ESFT to Reduce Memory by up to 90% and Time by up to 30%

## Asif Razzaq

https://www.marktechpost.com/author/6flvq/

-  July 6, 2024
Natural language processing is advancing rapidly, focusing on optimizing large language models (LLMs) for specific tasks. These models, often containing billions of parameters, pose a significant challenge in customization. The aim is to develop efficient and better methods for fine-tuning these models to specific downstream tasks without prohibitive computational costs. This requires innovative approaches to parameter-efficient fine-tuning (PEFT) that maximize performance while minimizing resource usage.

One major problem in this domain is the resource-intensive nature of customizing LLMs for specific tasks. Traditional fine-tuning methods typically update all model parameters, which can lead to high computational costs and overfitting. Given the scale of modern LLMs, such as those with sparse architectures that distribute tasks across multiple specialized experts, there is a pressing need for more efficient fine-tuning techniques. The challenge lies in optimizing performance while ensuring the computational burden remains manageable.

Existing methods for PEFT in dense-architecture LLMs include low-rank adaptation (LoRA) and P-Tuning. These methods generally involve adding new parameters to the model or selectively updating existing ones. For instance, LoRA decomposes weight matrices into low-rank components, which helps reduce the number of parameters that need to be trained. However, these approaches have primarily focused on dense models and do not fully exploit the potential of sparse-architecture LLMs. In sparse models, different tasks activate different subsets of parameters, making traditional methods less effective.

DeepSeek AI and Northwestern University researchers have introduced a novel method called

Expert-Specialized Fine-Tuning (ESFT)

tailored for sparse-architecture LLMs, specifically those using a mixture-of-experts (MoE) architecture. This method aims to fine-tune only the most relevant experts for a given task while freezing the other experts and model components. By doing so, ESFT enhances tuning efficiency and maintains the specialization of the experts, which is crucial for optimal performance. The ESFT method capitalizes on the MoE architecture’s inherent ability to assign different tasks to experts, ensuring that only the necessary parameters are updated.

In more detail, ESFT involves calculating the affinity scores of experts to task-specific data and selecting a subset of experts with the highest relevance. These selected experts are then fine-tuned while the rest of the model remains unchanged. This selective approach significantly reduces the computational costs associated with fine-tuning. For instance, ESFT can reduce storage requirements by up to 90% and training time by up to 30% compared to full-parameter fine-tuning. This efficiency is achieved without compromising the model’s overall performance, as demonstrated by the experimental results.

In various downstream tasks, ESFT not only matched but often surpassed the performance of traditional full-parameter fine-tuning methods. For example, in tasks like math and code, ESFT achieved significant performance gains while maintaining a high degree of specialization. The method’s ability to efficiently fine-tune a subset of experts, selected based on their relevance to the task, highlights its effectiveness. The results showed that ESFT maintained general task performance better than other PEFT methods like LoRA, making it a versatile and powerful tool for LLM customization.

In conclusion, the research introduces Expert-Specialized Fine-Tuning (ESFT) as a solution to the problem of resource-intensive fine-tuning in large language models. By selectively tuning relevant experts, ESFT optimizes both performance and efficiency. This method leverages the specialized architecture of sparse-architecture LLMs to achieve superior results with reduced computational costs. The research demonstrates that ESFT can significantly improve training efficiency, reduce storage and training time, and maintain high performance across various tasks. This makes ESFT a promising approach for future developments in customizing large language models.

## Check out the

https://github.com/deepseek-ai/ESFT

All credit for this research goes to the researchers of this project. Also, don’t forget to follow us on

## Telegram Channel

## LinkedIn Gr

If you like our work, you will love our

newsletter..

Don’t Forget to join our

46k+ ML SubReddit

## Asif Razzaq

+ posts Bio

https://www.marktechpost.com#

Asif Razzaq is the CEO of Marktechpost Media Inc.. As a visionary entrepreneur and engineer, Asif is committed to harnessing the potential of Artificial Intelligence for social good. His most recent endeavor is the launch of an Artificial Intelligence Media Platform, Marktechpost, which stands out for its in-depth coverage of machine learning and deep learning news that is both technically sound and easily understandable by a wide audience. The platform boasts of over 2 million monthly views, illustrating its popularity among audiences.

Asif Razzaq https://www.marktechpost.com/author/6flvq/

How Exploration Agents like Q-Learning, UCB, and MCTS Collaboratively Learn Intelligent Problem-Solving Strategies in Dynamic Grid Environments

Asif Razzaq https://www.marktechpost.com/author/6flvq/

Zhipu AI Releases ‘Glyph’: An AI Framework for Scaling the Context Length through Visual-Text Compression

Asif Razzaq https://www.marktechpost.com/author/6flvq/

How to Build a Fully Interactive, Real-Time Visualization Dashboard Using Bokeh and Custom JavaScript?

Asif Razzaq https://www.marktechpost.com/author/6flvq/

How to Build an Agentic Decision-Tree RAG System with Intelligent Query Routing, Self-Checking, and Iterative Refinement?

🙌 Follow MARKTECHPOST: Add us as a preferred source on Google.

## RELATED ARTICLES

## MORE FROM AUTHOR

How Exploration Agents like Q-Learning, UCB, and MCTS Collaboratively Learn Intelligent Problem-Solving Strategies in Dynamic Grid Environments

MiniMax Releases MiniMax M2: A Mini Open Model Built for Max Coding and Agentic Workflows at 8% Claude Sonnet Price and ~2x Faster

Zhipu AI Releases ‘Glyph’: An AI Framework for Scaling the Context Length through Visual-Text Compression

Meet Pyversity Library: How to Improve Retrieval Systems by Diversifying the Results Using Pyversity?

How to Build a Fully Interactive, Real-Time Visualization Dashboard Using Bokeh and Custom JavaScript?

How to Build an Agentic Decision-Tree RAG System with Intelligent Query Routing, Self-Checking, and Iterative Refinement?

How Exploration Agents like Q-Learning, UCB, and MCTS Collaboratively Learn Intelligent Problem-Solving Strategies in...

## Asif Razzaq

-   October 28, 2025
https://www.marktechpost.com/2025/10/28/how-exploration-agents-like-q-learning-ucb-and-mcts-collaboratively-learn-intelligent-problem-solving-strategies-in-dynamic-grid-environments/#respond

In this tutorial, we explore how exploration strategies shape intelligent decision-making through agent-based problem solving. We build and train three agents, Q-Learning with epsilon-greedy...

MiniMax Releases MiniMax M2: A Mini Open Model Built for Max Coding and Agentic Workflows at 8% Claude...

https://www.marktechpost.com/2025/10/28/minimax-open-sources-minimax-m2-a-mini-model-built-for-max-coding-and-agentic-workflows-at-8-claude-sonnet-price-and-2x-faster/

## Michal Sutter

https://www.marktechpost.com/2025/10/28/minimax-open-sources-minimax-m2-a-mini-model-built-for-max-coding-and-agentic-workflows-at-8-claude-sonnet-price-and-2x-faster/

-   October 28, 2025
https://www.marktechpost.com/2025/10/28/minimax-open-sources-minimax-m2-a-mini-model-built-for-max-coding-and-agentic-workflows-at-8-claude-sonnet-price-and-2x-faster/#respond

Can an open source MoE truly power agentic coding workflows at a fraction of flagship model costs while sustaining long-horizon tool use across MCP,...

Zhipu AI Releases ‘Glyph’: An AI Framework for Scaling the Context Length through Visual-Text...

https://www.marktechpost.com/2025/10/28/zhipu-ai-releases-glyph-an-ai-framework-for-scaling-the-context-length-through-visual-text-compression/

## Asif Razzaq

https://www.marktechpost.com/2025/10/28/zhipu-ai-releases-glyph-an-ai-framework-for-scaling-the-context-length-through-visual-text-compression/

-   October 28, 2025
https://www.marktechpost.com/2025/10/28/zhipu-ai-releases-glyph-an-ai-framework-for-scaling-the-context-length-through-visual-text-compression/#respond

Can we render long texts as images and use a VLM to achieve 3–4× token compression, preserving accuracy while scaling a 128K context toward...

Meet Pyversity Library: How to Improve Retrieval Systems by Diversifying the Results Using Pyversity?

https://www.marktechpost.com/2025/10/27/meet-pyversity-library-how-to-improve-retrieval-systems-by-diversifying-the-results-using-pyversity/

## Arham Islam

https://www.marktechpost.com/2025/10/27/meet-pyversity-library-how-to-improve-retrieval-systems-by-diversifying-the-results-using-pyversity/

-   October 27, 2025
https://www.marktechpost.com/2025/10/27/meet-pyversity-library-how-to-improve-retrieval-systems-by-diversifying-the-results-using-pyversity/#respond

Pyversity is a fast, lightweight Python library designed to improve the diversity of results from retrieval systems. Retrieval often returns items that are very...

How to Build a Fully Interactive, Real-Time Visualization Dashboard Using Bokeh and Custom JavaScript?

https://www.marktechpost.com/2025/10/27/how-to-build-a-fully-interactive-real-time-visualization-dashboard-using-bokeh-and-custom-javascript/

## Asif Razzaq

https://www.marktechpost.com/2025/10/27/how-to-build-a-fully-interactive-real-time-visualization-dashboard-using-bokeh-and-custom-javascript/

-   October 27, 2025
https://www.marktechpost.com/2025/10/27/how-to-build-a-fully-interactive-real-time-visualization-dashboard-using-bokeh-and-custom-javascript/#respond

In this tutorial, we create a fully interactive, visually compelling data visualization dashboard using Bokeh. We start by turning raw data into insightful plots,...

How to Build an Agentic Decision-Tree RAG System with Intelligent Query Routing, Self-Checking, and...

https://www.marktechpost.com/2025/10/27/how-to-build-an-agentic-decision-tree-rag-system-with-intelligent-query-routing-self-checking-and-iterative-refinement/

## Asif Razzaq

https://www.marktechpost.com/2025/10/27/how-to-build-an-agentic-decision-tree-rag-system-with-intelligent-query-routing-self-checking-and-iterative-refinement/

-   October 27, 2025
https://www.marktechpost.com/2025/10/27/how-to-build-an-agentic-decision-tree-rag-system-with-intelligent-query-routing-self-checking-and-iterative-refinement/#respond

In this tutorial, we build an advanced Agentic Retrieval-Augmented Generation (RAG) system that goes beyond simple question answering. We design it to intelligently route...

Meet ‘kvcached’: A Machine Learning Library to Enable Virtualized, Elastic KV Cache for LLM...

https://www.marktechpost.com/2025/10/26/meet-kvcached-a-machine-learning-library-to-enable-virtualized-elastic-kv-cache-for-llm-serving-on-shared-gpus/

## Asif Razzaq

https://www.marktechpost.com/2025/10/26/meet-kvcached-a-machine-learning-library-to-enable-virtualized-elastic-kv-cache-for-llm-serving-on-shared-gpus/

-   October 26, 2025
https://www.marktechpost.com/2025/10/26/meet-kvcached-a-machine-learning-library-to-enable-virtualized-elastic-kv-cache-for-llm-serving-on-shared-gpus/#respond

Large language model serving often wastes GPU memory because engines pre-reserve large static KV cache regions per model, even when requests are bursty or...

5 Common LLM Parameters Explained with Examples

https://www.marktechpost.com/2025/10/26/5-common-llm-parameters-explained-with-examples/

## Arham Islam

https://www.marktechpost.com/2025/10/26/5-common-llm-parameters-explained-with-examples/

-   October 26, 2025
https://www.marktechpost.com/2025/10/26/5-common-llm-parameters-explained-with-examples/#respond

Large language models (LLMs) offer several parameters that let you fine-tune their behavior and control how they generate responses. If a model isn’t producing...

How to Build, Train, and Compare Multiple Reinforcement Learning Agents in a Custom Trading...

https://www.marktechpost.com/2025/10/26/how-to-build-train-and-compare-multiple-reinforcement-learning-agents-in-a-custom-trading-environment-using-stable-baselines3/

## Asif Razzaq

https://www.marktechpost.com/2025/10/26/how-to-build-train-and-compare-multiple-reinforcement-learning-agents-in-a-custom-trading-environment-using-stable-baselines3/

-   October 26, 2025
https://www.marktechpost.com/2025/10/26/how-to-build-train-and-compare-multiple-reinforcement-learning-agents-in-a-custom-trading-environment-using-stable-baselines3/#respond

In this tutorial, we explore advanced applications of Stable-Baselines3 in reinforcement learning. We design a fully functional, custom trading environment, integrate multiple algorithms such...

A New AI Research from Anthropic and Thinking Machines Lab Stress Tests Model Specs...

https://www.marktechpost.com/2025/10/25/a-new-ai-research-from-anthropic-and-thinking-machines-lab-stress-tests-model-specs-and-reveal-character-differences-among-language-models/

## Asif Razzaq

https://www.marktechpost.com/2025/10/25/a-new-ai-research-from-anthropic-and-thinking-machines-lab-stress-tests-model-specs-and-reveal-character-differences-among-language-models/

-   October 25, 2025
https://www.marktechpost.com/2025/10/25/a-new-ai-research-from-anthropic-and-thinking-machines-lab-stress-tests-model-specs-and-reveal-character-differences-among-language-models/#respond

AI companies use model specifications to define target behaviors during training and evaluation. Do current specs state the intended behaviors with enough precision, and...

https://www.marktechpost.com/2025/10/20/the-local-ai-revolution-expanding-generative-ai-with-gpt-oss-20b-and-the-nvidia-rtx-ai-pc/

https://www.marktechpost.com/2025/10/20/the-local-ai-revolution-expanding-generative-ai-with-gpt-oss-20b-and-the-nvidia-rtx-ai-pc/

https://www.marktechpost.com/2025/10/20/the-local-ai-revolution-expanding-generative-ai-with-gpt-oss-20b-and-the-nvidia-rtx-ai-pc/

https://www.marktechpost.com/2025/10/20/the-local-ai-revolution-expanding-generative-ai-with-gpt-oss-20b-and-the-nvidia-rtx-ai-pc/

miniCON Event 2025

https://www.marktechpost.com/2025/10/20/the-local-ai-revolution-expanding-generative-ai-with-gpt-oss-20b-and-the-nvidia-rtx-ai-pc/

https://www.marktechpost.com/2025/10/20/the-local-ai-revolution-expanding-generative-ai-with-gpt-oss-20b-and-the-nvidia-rtx-ai-pc/

AI Magazine/Report

https://www.marktechpost.com/2025/10/20/the-local-ai-revolution-expanding-generative-ai-with-gpt-oss-20b-and-the-nvidia-rtx-ai-pc/

Privacy & TC

https://www.marktechpost.com/2025/10/20/the-local-ai-revolution-expanding-generative-ai-with-gpt-oss-20b-and-the-nvidia-rtx-ai-pc/

## Cookie Policy

https://www.marktechpost.com/2025/10/20/the-local-ai-revolution-expanding-generative-ai-with-gpt-oss-20b-and-the-nvidia-rtx-ai-pc/

🐝 Partnership and Promotion

https://www.marktechpost.com/2025/10/20/the-local-ai-revolution-expanding-generative-ai-with-gpt-oss-20b-and-the-nvidia-rtx-ai-pc/

© Copyright Reserved @2025 Marktechpost AI Media Inc

Loading Comments...

