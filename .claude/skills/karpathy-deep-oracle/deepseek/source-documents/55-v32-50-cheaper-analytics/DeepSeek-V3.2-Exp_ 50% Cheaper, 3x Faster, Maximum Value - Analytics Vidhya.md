---
sourceFile: "DeepSeek-V3.2-Exp: 50% Cheaper, 3x Faster, Maximum Value - Analytics Vidhya"
exportedBy: "Kortex"
exportDate: "2025-10-29T02:37:31.323Z"
---

# DeepSeek-V3.2-Exp: 50% Cheaper, 3x Faster, Maximum Value - Analytics Vidhya

209496aa-43d8-4fcd-b6be-a4dee6975722

DeepSeek-V3.2-Exp: 50% Cheaper, 3x Faster, Maximum Value - Analytics Vidhya

4e825b1b-01f4-4a94-acf2-efd9ac39880a

https://www.analyticsvidhya.com/blog/2025/09/deepseek-v3-2-exp/

Master Generative AI with 10+ Real-world Projects in 2025!

## Free Courses

## Learning Paths

## GenAI Pinnacle Plus

## Agentic AI Pioneer

## Switch Mode

## Interview Prep

## Prompt Engg

## Machine Learning

## Deep Learning

## GenAI Tools

## AIML Projects

## Reading list

What is Deep Learning?

## DL vs ML vs AI

Why Deep Learning is So Popular?

## Applications of Deep Learning

## Multi Layer Perceptron

## Visualizing the Neural Network

## Understanding Decision Boundary

## Forward and Backward Propagation Intuition

## Gradient Descent Algorithm

## Variants of Gradient Descent Algorithm

## Introduction to Loss Function

## Binary and Categorical Cross Entropy

Why Do We Need Activation Functions?

## Linear Activation Function

## Sigmoid and Tanh Function

How to Select Right Activation Function?

## Weights in Perceptron

## Introduction to Artificial Neural Network

## Understanding Forward Propagation Mathematically

## Understand Backward Propagation Mathematically

## Implementing Neural Networks in Python

## Problems with Gradient Descent

## Gradient Descent with Momentum

## Adagrad and Adadelta

## Introduction to Learning Rate Schedulers

## Overview of Deep Learning Frameworks

## Implementing Neural Networks using Keras

## Functional API in Keras

## Implementing Neural Networks using Keras

## Hyperparameter Tuning of MLP in Keras

## Understanding Early stopping

## Understanding Dropout

## Vanishing and Exploding Gradients

## Weights Initialization Techniques

## Implementing Weight Initializing Techniques

## Batch Normalization

## Image Augmentation Techniques

## Image Generator and Fit Generator

## Model Checkpointing

## Implementing Model Checkpointing

## Dealing with Class Imbalance

## Ensemble Deep Learning

## Introduction to GPU and TPU

## Introduction to Unsupervised Learning

How to Solve Unsupervised Learning Problems?

## Introduction to Autoencoders

## Implementing Autoencoders

## A Beginners Guide to Codeless Deep Learning

## TensorFlow Serving

## Build Deep Learning Models for Android

## Introduction to PyTorch and Tensors

## Mathematical and Matrix Operations in PyTorch

## Important PyTorch Modules

## Implement CNN in PyTorch

## Transfer Learning in PyTorch

## Working with Text Data in PyTorch

## Building a RNN model in PyTorch

## Autoencoders in PyTorch

DeepSeek-V3.2-Exp: 50% Cheaper, 3x Faster, Maximum Value

## Nitika Sharma

Last Updated : 07 Oct, 2025   7    min read

When it comes to building better AI, the usual strategy is to make models bigger. But this approach has a major problem: it becomes incredibly expensive.  But, DeepSeek-V3.2-Exp took a different path…

Instead of just adding more power, they focused on working smarter. The result is a new kind of model that delivers top-tier performance for a fraction of the cost. By introducing their “sparse attention” mechanism, DeepSeek isn’t just tweaking the engine; it’s redesigning the fuel injection system for unprecedented efficiency.

Let’s break down exactly how they did it.

## Table of contents

## Highlights of the Update

DeepSeek Sparse Attention (DSA)

The Training Pipeline: A Two-Stage Tune-Up

The Hardware Secret Sauce: Optimized Kernels

Performance & Cost – A New Balance

## Cost Reduction

## Better Performance

Deepseek-V3.1-Terminus vs DeepSeek-V3.2-Exp

Let’s Try the New DeepSeek-V3.2-Exp

Task 1: Travel Plan

Task 2: Coding Agent

## Highlights of the Update

## Adding Sparse Attention

: The only architectural difference between the new V3.2 model and its predecessor (V3.1) is the introduction of DSA. This shows they focused all their effort on solving the efficiency problem.

A “Lightning Indexer”

: DSA works by using a fast, lightweight component called a lightning indexer. This indexer quickly scans the text and picks out only the most important words for the model to focus on, ignoring the rest.

## A Massive Complexity Reduction

: DSA changes the core computational problem from an exponentially difficult one  O(L²)  to a much simpler, linear one  O(Lk) . This is the mathematical secret behind the huge speed and cost improvements.

## Built for Real Hardware

: The success of DSA relies on highly optimized software designed to run perfectly on modern AI chips (like H800 GPUs). This tight integration between the smart algorithm and the hardware is what delivers the final, dramatic gains.

Read about the Previous update here:

Deepseek-V3.1-Terminus

https://www.analyticsvidhya.com/blog/2025/09/deepseek-v3-1-terminus/

DeepSeek Sparse Attention (DSA)

At the heart of every LLM is the “attention” mechanism: the system that determines how important each word in a sentence is to every other word.

The problem?

Traditional “dense” attention is wildly inefficient. Its computational cost scales quadratically ( O(L²) ), meaning that doubling the text length quadruples the computation and cost.

DeepSeek Sparse Attention (DSA) is the solution to this bloat. It doesn’t look at everything; it smartly selects what to focus on. The system is composed of two key parts:

The Lightning Indexer:

This is a lightweight, high-speed scanner. For any given word (a “query token”), it rapidly scores all the preceding words to determine their relevance. Crucially, this indexer is designed for speed: it uses a small number of heads and can run in FP8 precision, making its computational footprint remarkably small.

Fine-Grained Token Selection:

Once the indexer has scored everything, DSA doesn’t just grab blocks of text. It performs a precise, “fine-grained” selection, plucking only the top-K most relevant tokens from across the entire document. The main attention mechanism then only processes this carefully selected, sparse set.

The Result:

DSA reduces the core attention complexity from  O(L²)  to  O(Lk) , where  k  is a fixed number of selected tokens. This is the mathematical foundation for the massive efficiency gains. While the lightning indexer itself still has  O(L²)  complexity, it’s so lightweight that the net effect is still a dramatic reduction in total computation.

The Training Pipeline: A Two-Stage Tune-Up

You can’t just slap a new attention mechanism onto a billion-parameter model and hope it works. DeepSeek employed a meticulous, two-stage training process to integrate DSA seamlessly.

Stage 1: Continued Pre-Training (The Warm-Up)

Dense Warm-up (2.1B tokens):

Starting from the V3.1-Terminus checkpoint, DeepSeek first “warmed up” the new lightning indexer. They kept the main model frozen and ran a short training stage where the indexer learned to predict the output of the full, dense attention mechanism. This aligned the new indexer with the model’s existing knowledge.

Sparse Training (943.7B tokens):

This is where the real magic happened. After the warm-up, DeepSeek switched on the full sparse attention, selecting the top 2048 key-value tokens for each query. For the first time, the entire model was trained to operate with this new, selective vision, learning to rely on the sparse selections rather than the dense whole.

Stage 2: Post-Training (The Finishing School)

To ensure a fair comparison, DeepSeek used the

exact same post-training pipeline as V3.1-Terminus

. This rigorous approach proves that any performance differences are due to DSA, not changes in training data.

Specialist Distillation:

They created five powerhouse specialist models (for Math, Coding, Reasoning, Agentic Coding, and Agentic Search) using heavy-duty Reinforcement Learning. The knowledge from these experts was then distilled into the final V3.2 model.

Mixed RL with GRPO:

Instead of a multi-stage process, they used

Group Relative Policy Optimization (GRPO)

in a single, blended stage. The reward function was carefully engineered to balance key trade-offs:

Length vs. Accuracy:

Penalizing unnecessarily long answers.

Language Consistency vs. Accuracy:

Ensuring responses remained coherent and human-like.

Rule-Based & Rubric-Based Rewards:

Using automated checks for reasoning/agent tasks and tailored rubrics for general tasks.

The Hardware Secret Sauce: Optimized Kernels

GitHub/DeepSeek

https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf

A brilliant algorithm is useless if it runs slowly on actual hardware. DeepSeek’s commitment to efficiency shines here with deeply optimized, open-source code.

The model leverages specialized kernels like FlashMLA, which are custom-built to run the complex MLA and DSA operations with extreme efficiency on modern Hopper GPUs (like the H800). These optimizations are publicly available in pull requests to repositories like DeepGEMM, FlashMLA, and tilelang, allowing the model to achieve near-theoretical peak memory bandwidth (up to 3000 GB/s) and compute performance. This hardware-aware design is what transforms the theoretical efficiency of DSA into tangible, real-world speed.

Performance & Cost – A New Balance

So, what’s the final outcome of this engineering marvel? The data reveals a clear and compelling story.

## Cost Reduction

The most immediate impact is on the bottom line. DeepSeek announced a >50% reduction in API pricing. The technical benchmarks are even more striking:

Inference Speed:

2–3x faster on long contexts.

Memory Usage:

30–40% lower.

Training Efficiency:

50% faster.

The real-world inference cost for decoding a 128K context window plummets to an estimated $0.25, compared to

for dense attention, making it 10x cheaper.

## Better Performance

On aggregate, V3.2-Exp maintains performance parity with its predecessor. However, a closer look reveals a logical trade-off:

## The model shows

significant gains in coding (Codeforces) and agentic tasks (BrowseComp)

. This makes perfect sense: code and tool-use often contain redundant information, and DSA’s ability to filter noise is a direct advantage.

The Trade-Offs:

There are minor regressions in a few ultra-complex, abstract reasoning benchmarks (like GPQA Diamond and HMMT). The hypothesis is that these tasks rely on connecting very subtle, long-range dependencies that the current DSA mask might occasionally miss.

Deepseek-V3.1-Terminus vs DeepSeek-V3.2-Exp

Let’s Try the New DeepSeek-V3.2-Exp

The tasks I will be doing here will be same as we did in one of our previous articles on

Deepseek-V3.1-Terminus

https://www.analyticsvidhya.com/blog/2025/09/deepseek-v3-1-terminus/

. This will help in identifying how the new update is better.

Task 1: Travel Plan

I need to plan a 7-day trip to Kyoto, Japan, for mid-November. The itinerary should focus on traditional culture, including temples, gardens, and tea ceremonies. Find the best time to see the autumn leaves, a list of three must-visit temples for ‘Momiji’ (autumn leaves), and a highly-rated traditional tea house with English-friendly services. Also, find a well-reviewed ryokan (traditional Japanese inn) in the Gion district. Organize all the information into a clear, day-by-day itinerary.

You can view the full output here.

https://www.analyticsvidhya.com/blog/2025/09/deepseek-v3-1-terminus/

Deepseek-V3.1-Terminus vs DeepSeek-V3.2-Exp Output Comparison

Feature   DeepSeek-V3.1-Terminus   DeepSeek-V3.2-Exp   Winner

Structure & Organization

Single continuous paragraph   Well-structured with clear sections and bullet points   V3.2-Exp

## Command Presentation

Commands embedded in running text   Dedicated code blocks with proper formatting   V3.2-Exp

Completeness & Depth

Basic explanation only   Includes practical examples and use cases   V3.2-Exp

## Beginner Friendliness

Assumes prior knowledge   Explains concepts clearly for newcomers   V3.2-Exp

Task 2: Coding Agent

I need a Python script that scrapes a public list of the top 100 films of all time from a website (you can choose a reliable source like IMDb, Rotten Tomatoes, or a well-known magazine’s list). The script should then save the film titles, release years, and a brief description for each movie into a JSON file. Include error handling for network issues or changes in the website’s structure. Can you generate the full script and explain each step of the process?

class IMDbTop100Scraper: def __init__(self): self.base_url = "https://www.imdb.com/chart/top" self.headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36' } self.movies = [] def fetch_page(self, url: str) -> Optional[BeautifulSoup]: """ Fetch the webpage and return BeautifulSoup object. Handles network errors and retries. """ try: response = requests.get(url, headers=self.headers, timeout=10) response.raise_for_status() # Raises an HTTPError for bad responses # Check if we got a valid HTML response if 'text/html' not in response.headers.get('content-type', ''): raise ValueError("Received non-HTML response") return BeautifulSoup(response.content, 'html.parser') except requests.exceptions.RequestException as e: print(f"Network error occurred: {e}") return None except Exception as e: print(f"Unexpected error while fetching page: {e}") return None def parse_movie_list(self, soup: BeautifulSoup) -> List[Dict]: """ Parse the main movie list page to extract titles and years. """ movies = [] try: # IMDb's top chart structure - this selector might need updating movie_elements = soup.select('li.ipc-metadata-list-summary-item') if not movie_elements: # Alternative selector if the primary one fails movie_elements = soup.select('.cli-children') if not movie_elements: raise ValueError("Could not find movie elements on the page") for element in movie_elements[:100]: # Limit to top 100 movie_data = self.extract_movie_data(element) if movie_data: movies.append(movie_data) except Exception as e: print(f"Error parsing movie list: {e}") return movies

Find full code here.

Deepseek-V3.1-Terminus vs DeepSeek-V3.2-Exp Output Comparison

Feature   DeepSeek-V3.1-Terminus   DeepSeek-V3.2-Exp   Winner

Structure & Presentation

Single dense paragraph   Clear headings, bullet points, summary table   V3.2-Exp

Safety & User Guidance

## No safety warnings

## Bold warning

about unstaged changes loss   V3.2-Exp

Completeness & Context

Basic two methods only   Adds legacy `git checkout` method and summary table   V3.2-Exp

## Actionability

Commands embedded in text   Dedicated command blocks with explicit flag explanations   V3.2-Exp

Also Read:

Evolution of DeepSeek: How it Became a Global AI Game-Changer!

https://www.analyticsvidhya.com/blog/2025/01/evolution-of-deepseek/

DeepSeek-V3.2-Exp is more than a model; it’s a statement. It proves that the next great leap in AI won’t necessarily be a leap in raw power, but a leap in efficiency. By surgically attacking the computational waste in traditional transformers, DeepSeek has made long-context, high-volume AI applications financially viable for a much broader market.

The “Experimental” tag is a candid admission that this is a work in progress, particularly in balancing performance across all tasks. But for the vast majority of enterprise use cases, where processing entire codebases, legal documents, and datasets is the goal. DeepSeek hasn’t just released a new model; it has started a new race.

To know more about the model, checkout this

https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf

## Nitika Sharma

https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf

Hello, I am Nitika, a tech-savvy Content Creator and Marketer. Creativity and learning new things come naturally to me. I have expertise in creating result-driven content strategies. I am well versed in SEO Management, Keyword Operations, Web Content Writing, Communication, Content Strategy, Editing, and Writing.

## Generative AI

https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf

https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf

Login to continue reading and enjoy expert-curated content.

## Free Courses

Generative AI - A Way of Life

Explore Generative AI for beginners: create text and images, use top AI tools, learn practical skills, and ethics.

## Getting Started with Large Language Models

Master Large Language Models (LLMs) with this course, offering clear guidance in NLP and model training made simple.

## Building LLM Applications using Prompt Engineering

This free course guides you on building LLM apps, mastering prompt engineering, and developing chatbots with enterprise data.

Improving Real World RAG Systems: Key Challenges & Practical Solutions

Explore practical solutions, advanced retrieval strategies, and agentic RAG systems to improve context, relevance, and accuracy in AI-driven applications.

Microsoft Excel: Formulas & Functions

Master MS Excel for data analysis with key formulas, functions, and LookUp tools in this comprehensive course.

## Recommended Articles

GPT-4 vs. Llama 3.1 – Which Model is Better?

Llama-3.1-Storm-8B: The 8B LLM Powerhouse Surpa...

A Comprehensive Guide to Building Agentic RAG S...

Top 10 Machine Learning Algorithms in 2025

45 Questions to Test a Data Scientist on Basics...

90+ Python Interview Questions and Answers (202...

8 Easy Ways to Access ChatGPT for Free

Prompt Engineering: Definition, Examples, Tips ...

What is LangChain?

What is Retrieval-Augmented Generation (RAG)?

## Responses From Readers

## Cancel reply

https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf

## Become an Author

Share insights, grow your voice, and inspire the data community.

## Reach a Global Audience

## Share Your Expertise with the World

Build Your Brand & Audience

## Join a Thriving AI Community

## Level Up Your AI Game

## Expand Your Influence in Genrative AI

## Flagship Programs

## GenAI Pinnacle Program

https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf

## GenAI Pinnacle Plus Program

https://www.analyticsvidhya.com/pinnacleplus/?ref=blogflashstripfooter

AI/ML BlackBelt Program

https://www.analyticsvidhya.com/bbplus?ref=footer

## Agentic AI Pioneer Program

https://www.analyticsvidhya.com/agenticaipioneer?ref=footer

## Free Courses

## Generative AI

https://www.analyticsvidhya.com/agenticaipioneer?ref=footer

https://www.analyticsvidhya.com/courses/getting-started-with-deepseek/?ref=footer

## OpenAI Agent SDK

https://www.analyticsvidhya.com/courses/demystifying-openai-agents-sdk/?ref=footer

## LLM Applications using Prompt Engineering

https://www.analyticsvidhya.com/courses/building-llm-applications-using-prompt-engineering-free/?ref=footer

## DeepSeek from Scratch

https://www.analyticsvidhya.com/courses/deepseek-from-scratch/?ref=footer

Stability.AI

https://www.analyticsvidhya.com/courses/exploring-stability-ai/?ref=footer

SSM & MAMBA

https://www.analyticsvidhya.com/courses/building-smarter-llms-with-mamba-and-state-space-model/?ref=footer

## RAG Systems using LlamaIndex

https://www.analyticsvidhya.com/courses/building-first-rag-systems-using-llamaindex/?ref=footer

## Building LLMs for Code

https://www.analyticsvidhya.com/courses/building-large-language-models-for-code/?ref=footer

https://www.analyticsvidhya.com/courses/introduction-to-data-science/?ref=footer

## Microsoft Excel

https://www.analyticsvidhya.com/courses/microsoft-excel-formulas-functions/?ref=footer

## Machine Learning

https://www.analyticsvidhya.com/courses/machine-learning-certification-program-beginners/?ref=footer

## Deep Learning

https://www.analyticsvidhya.com/courses/getting-started-with-deep-learning/?ref=footer

## Mastering Multimodal RAG

https://www.analyticsvidhya.com/courses/mastering-multimodal-rag-and-embeddings-with-amazon-nova-and-bedrock/?ref=footer

## Introduction to Transformer Model

https://www.analyticsvidhya.com/courses/introduction-to-transformers-and-attention-mechanisms/?ref=footer

Bagging & Boosting

https://www.analyticsvidhya.com/courses/bagging-boosting-ML-Algorithms/?ref=footer

## Loan Prediction

https://www.analyticsvidhya.com/courses/loan-prediction-practice-problem-using-python/?ref=footer

## Time Series Forecasting

https://www.analyticsvidhya.com/courses/creating-time-series-forecast-using-python/?ref=footer

https://www.analyticsvidhya.com/courses/tableau-for-beginners/?ref=footer

## Business Analytics

https://www.analyticsvidhya.com/courses/introduction-to-analytics/?ref=footer

## Vibe Coding in Windsurf

https://www.analyticsvidhya.com/courses/guide-to-vibe-coding-in-windsurf/?ref=footer

## Model Deployment using FastAPI

https://www.analyticsvidhya.com/courses/model-deployment-using-fastapi/?ref=footer

## Building Data Analyst AI Agent

https://www.analyticsvidhya.com/courses/building-data-analyst-AI-agent/?ref=footer

Getting started with OpenAI o3-mini

https://www.analyticsvidhya.com/courses/getting-started-with-openai-o3-mini/?ref=footer

## Introduction to Transformers and Attention Mechanisms

https://www.analyticsvidhya.com/courses/introduction-to-transformers-and-attention-mechanisms/?ref=footer

## Popular Categories

https://www.analyticsvidhya.com/courses/introduction-to-transformers-and-attention-mechanisms/?ref=footer

## Generative AI

https://www.analyticsvidhya.com/blog/category/generative-ai/?ref=footer

## Prompt Engineering

https://www.analyticsvidhya.com/blog/category/prompt-engineering/?ref=footer

## Generative AI Application

https://www.analyticsvidhya.com/blog/category/generative-ai-application/?ref=footer

https://news.google.com/publications/CAAqBwgKMJiWzAswyLHjAw?hl=en-IN&gl=IN&ceid=IN%3Aen

## Technical Guides

https://www.analyticsvidhya.com/blog/category/guide/?ref=footer

https://www.analyticsvidhya.com/blog/category/ai-tools/?ref=footer

## Interview Preparation

https://www.analyticsvidhya.com/blog/category/interview-questions/?ref=footer

## Research Papers

https://www.analyticsvidhya.com/blog/category/research-paper/?ref=footer

## Success Stories

https://www.analyticsvidhya.com/blog/category/success-story/?ref=footer

https://www.analyticsvidhya.com/blog/category/quiz/?ref=footer

https://www.analyticsvidhya.com/blog/category/use-cases/?ref=footer

https://www.analyticsvidhya.com/blog/category/listicle/?ref=footer

## Generative AI Tools and Techniques

https://www.analyticsvidhya.com/blog/category/listicle/?ref=footer

https://www.analyticsvidhya.com/blog/2023/07/an-overview-of-variational-autoencoders/?ref=footer

## Transformers

https://www.analyticsvidhya.com/blog/2019/06/understanding-transformers-nlp-state-of-the-art-models?ref=footer

https://www.analyticsvidhya.com/blog/2021/05/stylegan-explained-in-less-than-five-minutes/?ref=footer

https://www.analyticsvidhya.com/blog/2023/10/pix2pix-unleashed-transforming-images-with-creative-superpower?ref=footer

## Autoencoders

https://www.analyticsvidhya.com/blog/2021/06/autoencoders-a-gentle-introduction?ref=footer

https://www.analyticsvidhya.com/blog/2022/10/generative-pre-training-gpt-for-natural-language-understanding/?ref=footer

https://www.analyticsvidhya.com/blog/2022/11/comprehensive-guide-to-bert/?ref=footer

https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/?ref=footer

https://www.analyticsvidhya.com/blog/2021/03/introduction-to-long-short-term-memory-lstm?ref=footer

## Attention Mechanisms

https://www.analyticsvidhya.com/blog/2019/11/comprehensive-guide-attention-mechanism-deep-learning/?ref=footer

## Diffusion Models

https://www.analyticsvidhya.com/blog/2024/09/what-are-diffusion-models/?ref=footer

https://www.analyticsvidhya.com/blog/2023/03/an-introduction-to-large-language-models-llms/?ref=footer

https://www.analyticsvidhya.com/blog/2024/05/what-are-small-language-models-slms/?ref=footer

## Encoder Decoder Models

https://www.analyticsvidhya.com/blog/2023/10/advanced-encoders-and-decoders-in-generative-ai/?ref=footer

## Prompt Engineering

https://www.analyticsvidhya.com/blog/2023/06/what-is-prompt-engineering/?ref=footer

https://www.analyticsvidhya.com/blog/2024/06/langchain-guide/?ref=footer

https://www.analyticsvidhya.com/blog/2023/10/rag-pipeline-with-the-llama-index/?ref=footer

https://www.analyticsvidhya.com/blog/2023/09/retrieval-augmented-generation-rag-in-ai/?ref=footer

Fine-tuning

https://www.analyticsvidhya.com/blog/2023/08/fine-tuning-large-language-models/?ref=footer

## LangChain AI Agent

https://www.analyticsvidhya.com/blog/2024/07/langchains-agent-framework/?ref=footer

## Multimodal Models

https://www.analyticsvidhya.com/blog/2023/12/what-are-multimodal-models/?ref=footer

https://www.analyticsvidhya.com/blog/2022/03/a-brief-overview-of-recurrent-neural-networks-rnn/?ref=footer

https://www.analyticsvidhya.com/blog/2021/07/deep-convolutional-generative-adversarial-network-dcgan-for-beginners/?ref=footer

https://www.analyticsvidhya.com/blog/2021/05/progressive-growing-gan-progan/?ref=footer

Text-to-Image Models

https://www.analyticsvidhya.com/blog/2024/02/llm-driven-text-to-image-with-diffusiongpt/?ref=footer

https://www.analyticsvidhya.com/blog/2024/08/different-components-of-diffusion-models/?ref=footer

## Document Question Answering

https://www.analyticsvidhya.com/blog/2024/04/a-hands-on-guide-to-creating-a-pdf-based-qa-assistant-with-llama-and-llamaindex/?ref=footer

https://www.analyticsvidhya.com/blog/2024/09/google-imagen-3/?ref=footer

T5 (Text-to-Text Transfer Transformer)

https://www.analyticsvidhya.com/blog/2024/05/text-summarization-using-googles-t5-base/?ref=footer

Seq2seq Models

https://www.analyticsvidhya.com/blog/2020/08/a-simple-introduction-to-sequence-to-sequence-models/?ref=footer

https://www.analyticsvidhya.com/blog/2020/01/how-to-perform-automatic-music-generation/?ref=footer

Attention Is All You Need (Transformer Architecture)

https://www.analyticsvidhya.com/blog/2019/11/comprehensive-guide-attention-mechanism-deep-learning/?ref=footer

https://www.analyticsvidhya.com/blog/2024/11/windsurf-editor/?ref=footer

https://www.analyticsvidhya.com/blog/2025/03/vibe-coding-with-cursor-ai/?ref=footer

## Popular GenAI Models

https://www.analyticsvidhya.com/blog/2025/03/vibe-coding-with-cursor-ai/?ref=footer

https://www.analyticsvidhya.com/blog/2024/07/meta-llama-3-1/?ref=footer

https://www.analyticsvidhya.com/blog/2025/02/openai-gpt-4-5/?ref=footer

https://www.analyticsvidhya.com/blog/2025/04/open-ai-gpt-4-1/?ref=footer

https://www.analyticsvidhya.com/blog/2025/03/updated-gpt-4o/?ref=footer

https://www.analyticsvidhya.com/blog/2025/02/openai-o3-mini/?ref=footer

https://www.analyticsvidhya.com/blog/2024/12/openai-sora/?ref=footer

DeepSeek R1

https://www.analyticsvidhya.com/blog/2025/01/deepseek-r1/?ref=footer

DeepSeek V3

https://www.analyticsvidhya.com/blog/2025/01/ai-application-with-deepseek-v3/?ref=footer

https://www.analyticsvidhya.com/blog/2025/01/deepseek-janus-pro-7b/?ref=footer

https://www.analyticsvidhya.com/blog/2024/12/googles-veo-2/?ref=footer

Gemini 2.5 Pro

https://www.analyticsvidhya.com/blog/2025/03/gemini-2-5-pro-experimental/?ref=footer

https://www.analyticsvidhya.com/blog/2025/02/gemini-2-0-everything-you-need-to-know-about-googles-latest-llms/?ref=footer

https://www.analyticsvidhya.com/blog/2025/03/gemma-3/?ref=footer

Claude Sonnet 3.7

https://www.analyticsvidhya.com/blog/2025/02/claude-sonnet-3-7/?ref=footer

Claude 3.5 Sonnet

https://www.analyticsvidhya.com/blog/2024/06/claude-3-5-sonnet/?ref=footer

https://www.analyticsvidhya.com/blog/2025/02/microsoft-phi-4-multimodal/?ref=footer

https://www.analyticsvidhya.com/blog/2024/09/phi-3-5-slms/?ref=footer

Mistral Small 3.1

https://www.analyticsvidhya.com/blog/2025/03/mistral-small-3-1/?ref=footer

## Mistral NeMo

https://www.analyticsvidhya.com/blog/2024/08/mistral-nemo/?ref=footer

https://www.analyticsvidhya.com/blog/2024/01/making-the-most-of-mistral-7b-with-finetuning/?ref=footer

https://www.analyticsvidhya.com/blog/2024/02/building-end-to-end-generative-ai-models-with-aws-bedrock/?ref=footer

https://www.analyticsvidhya.com/blog/2024/02/build-deploy-and-manage-ml-models-with-google-vertex-ai/?ref=footer

Qwen QwQ 32B

https://www.analyticsvidhya.com/blog/2025/03/qwens-qwq-32b/?ref=footer

https://www.analyticsvidhya.com/blog/2024/06/qwen2/?ref=footer

Qwen 2.5 VL

https://www.analyticsvidhya.com/blog/2025/01/qwen2-5-vl-vision-model/?ref=footer

https://www.analyticsvidhya.com/blog/2025/03/qwen-chat/?ref=footer

https://www.analyticsvidhya.com/blog/2025/02/grok-3/?ref=footer

## AI Development Frameworks

https://www.analyticsvidhya.com/blog/2025/02/grok-3/?ref=footer

https://www.analyticsvidhya.com/blog/2024/06/langchain-guide/?ref=footer

https://www.analyticsvidhya.com/blog/2025/03/open-ai-responses-api/?ref=footer

A2A by Google

https://www.analyticsvidhya.com/blog/2025/04/agent-to-agent-protocol/?ref=footer

https://www.analyticsvidhya.com/blog/2025/01/smolagents/?ref=footer

https://www.analyticsvidhya.com/blog/2024/07/langgraph-revolutionizing-ai-agent/?ref=footer

https://www.analyticsvidhya.com/blog/2024/01/building-collaborative-ai-agents-with-crewai/?ref=footer

https://www.analyticsvidhya.com/blog/2025/03/agno-framework/?ref=footer

https://www.analyticsvidhya.com/blog/2023/06/langflow-ui-for-langchain-to-develop-applications-with-llms/?ref=footer

https://www.analyticsvidhya.com/blog/2023/11/launching-into-autogen-exploring-the-basics-of-a-multi-agent-framework/?ref=footer

https://www.analyticsvidhya.com/blog/2024/08/implementing-ai-agents-using-llamaindex/?ref=footer

https://www.analyticsvidhya.com/blog/2024/12/managing-multi-agent-systems-with-openai-swarm/?ref=footer

https://www.analyticsvidhya.com/blog/2023/05/learn-everything-about-autogpt/?ref=footer

## Data Science Tools and Techniques

https://www.analyticsvidhya.com/blog/2023/05/learn-everything-about-autogpt/?ref=footer

https://www.analyticsvidhya.com/blog/2016/02/complete-tutorial-learn-data-science-scratch/?ref=footer

https://www.analyticsvidhya.com/blog/2022/01/learning-sql-from-basics-to-advance/?ref=footer

## Jupyter Notebooks

https://www.analyticsvidhya.com/blog/2018/05/starters-guide-jupyter-notebook/?ref=footer

https://www.analyticsvidhya.com/blog/2021/11/tensorflow-for-beginners-with-examples-and-python-implementation/?ref=footer

Scikit-learn

https://www.analyticsvidhya.com/blog/2021/08/complete-guide-on-how-to-learn-scikit-learn-for-data-science/?ref=footer

https://www.analyticsvidhya.com/blog/2018/02/pytorch-tutorial/?ref=footer

https://www.analyticsvidhya.com/blog/2021/09/a-complete-guide-to-tableau-for-beginners-in-data-visualization/?ref=footer

## Apache Spark

https://www.analyticsvidhya.com/blog/2022/08/introduction-to-on-apache-spark-and-its-datasets/?ref=footer

https://www.analyticsvidhya.com/blog/2021/10/introduction-to-matplotlib-using-python-for-beginners/?ref=footer

https://www.analyticsvidhya.com/blog/2021/02/a-beginners-guide-to-seaborn-the-simplest-way-to-learn/?ref=footer

https://www.analyticsvidhya.com/blog/2021/03/pandas-functions-for-data-analysis-and-manipulation/?ref=footer

https://www.analyticsvidhya.com/blog/2022/05/an-introduction-to-hadoop-ecosystem-for-big-data/?ref=footer

https://www.analyticsvidhya.com/blog/2021/10/end-to-end-guide-to-docker-for-aspiring-data-engineers/?ref=footer

https://www.analyticsvidhya.com/blog/2021/09/git-and-github-tutorial-for-beginners/?ref=footer

https://www.analyticsvidhya.com/blog/2016/10/tutorial-optimizing-neural-networks-using-keras-with-image-recognition-case-study/?ref=footer

## Apache Kafka

https://www.analyticsvidhya.com/blog/2022/12/introduction-to-apache-kafka-fundamentals-and-working/?ref=footer

https://www.analyticsvidhya.com/blog/2020/09/what-is-aws-amazon-web-services-data-science/?ref=footer

https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/?ref=footer

## Random Forest

https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/?ref=footer

## Computer Vision

https://www.analyticsvidhya.com/blog/2020/01/computer-vision-learning-path/?ref=footer

## Data Visualization

https://www.analyticsvidhya.com/blog/2021/04/a-complete-beginners-guide-to-data-visualization/?ref=footer

## Data Exploration

https://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/?ref=footer

https://www.analyticsvidhya.com/blog/2021/05/what-is-big-data-introduction-uses-and-applications/?ref=footer

## Common Machine Learning Algorithms

https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/?ref=footer

## Machine Learning

https://www.analyticsvidhya.com/blog/category/Machine-Learning/?ref=footer

## Google Data Science Agent

https://www.analyticsvidhya.com/blog/2025/03/gemini-data-science-agent/?ref=footer

## Continue your learning for FREE

Forgot your password?

https://www.analyticsvidhya.com/blog/2025/03/gemini-data-science-agent/?ref=footer

## Enter email address to continue

## Enter OTP sent to

Wrong OTP.

## Enter the OTP

Resend OTP in  45s

