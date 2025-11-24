---
sourceFile: "deepseek-ai/ESFT: Expert Specialized Fine-Tuning - GitHub"
exportedBy: "Kortex"
exportDate: "2025-10-29T02:37:56.625Z"
---

# deepseek-ai/ESFT: Expert Specialized Fine-Tuning - GitHub

a136df79-3c90-4d81-a234-9c9e3107431b

deepseek-ai/ESFT: Expert Specialized Fine-Tuning - GitHub

774f7b09-b179-44b6-b853-0a46f4144ec1

https://github.com/deepseek-ai/ESFT

## Skip to content

## Navigation Menu

## Appearance settings

## GitHub Copilot   Write better code with AI

## GitHub Spark   New   Build and deploy intelligent apps

## GitHub Models   New   Manage and compare prompts

## GitHub Advanced Security   Find and fix vulnerabilities

## Actions   Automate any workflow

## Codespaces   Instant dev environments

## Issues   Plan and track work

## Code Review   Manage code changes

## Discussions   Collaborate outside of code

Code Search   Find more, search less

## Why GitHub

## Documentation

## GitHub Skills

## Integrations

## GitHub Marketplace

## MCP Registry

## View all features

## By company size

## Enterprises

## Small and medium teams

## Nonprofits

## By use case

## App Modernization

## DevSecOps

## View all use cases

## By industry

## Healthcare

## Financial services

## Manufacturing

## Government

## View all industries

## View all solutions

## Software Development

## Learning Pathways

Events & Webinars

Ebooks & Whitepapers

## Customer Stories

## Executive Insights

## GitHub Sponsors   Fund open source developers

## The ReadME Project   GitHub community articles

## Repositories

## Collections

Enterprise platform   AI-powered developer platform

Available add-ons

GitHub Advanced Security   Enterprise-grade security features

Copilot for business   Enterprise-grade AI features

Premium Support   Enterprise-grade 24/7 support

Search code, repositories, users, issues, pull requests...

## Search syntax tips

## Provide feedback

We read every piece of feedback, and take your input very seriously.

## Saved searches

## Use saved searches to filter your results more quickly

To see all available qualifiers, see our

documentation

https://docs.github.com/search-github/github-code-search/understanding-github-code-search-syntax

https://docs.github.com/search-github/github-code-search/understanding-github-code-search-syntax

https://docs.github.com/search-github/github-code-search/understanding-github-code-search-syntax

Appearance settings   You signed in with another tab or window.

https://github.com

to refresh your session.   You signed out in another tab or window.

https://github.com

to refresh your session.   You switched accounts on another tab or window.

https://github.com

to refresh your session.   Dismiss alert   {{ message }}

deepseek-ai

https://github.com/deepseek-ai

Couldn't load subscription status.

There was an error while loading.

## Please reload this page

https://github.com

Fork  260

https://github.com

Star  708

https://github.com

Expert Specialized Fine-Tuning

MIT, Unknown licenses found

## Licenses found

MIT   LICENSE-CODE

https://github.com

Unknown   LICENSE-MODEL

https://github.com

708  stars

https://github.com

260  forks

https://github.com

https://github.com

https://github.com

https://github.com

https://github.com

Couldn't load subscription status.

There was an error while loading.

## Please reload this page

https://github.com

## Additional navigation options

https://github.com

https://github.com

## Pull requests

https://github.com

https://github.com

https://github.com

https://github.com

https://github.com

deepseek-ai/ESFT

https://github.com

https://github.com

## Open more actions menu

## Folders and files

## Name Name Last commit message Last commit date

## Latest commit

https://github.com

https://github.com

https://github.com

https://github.com

https://github.com

https://github.com

https://github.com

https://github.com

https://github.com

https://github.com

https://github.com

https://github.com

https://github.com

LICENSE-CODE

https://github.com

LICENSE-CODE

https://github.com

LICENSE-MODEL

https://github.com

LICENSE-MODEL

https://github.com

https://github.com

https://github.com

__init__.py

https://github.com

__init__.py

https://github.com

benchmarks.py

https://github.com

benchmarks.py

https://github.com

https://github.com

https://github.com

eval_multigpu.py

https://github.com

eval_multigpu.py

https://github.com

https://github.com

https://github.com

train_ep.py

https://github.com

train_ep.py

https://github.com

https://github.com

https://github.com

## Repository files navigation

Expert-Specialized Fine-Tuning

## Official Repo for paper

Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning for Sparse Architectural Large Language Models

https://arxiv.org/abs/2407.01906

https://zihanwang314.github.io

https://victorchen96.github.io/chendeli.io/

https://scholar.google.com.hk/citations?user=8b-ysf0NWVoC&hl=zh-CN

https://runxinxu.github.io/aboutme/

http://www.idi.zju.edu.cn/member/3053.html

and Y. Wu.

aims to efficiently customize Large Language Models (LLMs) with Mixture-of-Experts (MoE) architecture by adjusting only task-relevant parts, improving efficiency and performance while using fewer resources and storage.

## Glad to announce that ESFT has been accepted to the

EMNLP 2024 Main Conference

## We now release the

## ESFT training code

! ✨ You can now try it with your own models and dataset!

🚀 Quick Start

## Installation and Setup

git clone https://github.com/deepseek-ai/ESFT.git  cd  esft

## Install required dependencies

pip install transformers torch safetensors accelerate

## Download necessary adapters

bash scripts/download_adapters.sh

🔧Key Scripts

eval_multigpu.py

This script evaluates the performance of the model on various datasets. See

scripts/eval.sh

for detailed configs and explanations.

python eval_multigpu.py \ --eval_dataset=translation \ --base_model_path=deepseek-ai/ESFT-vanilla-lite \ --adapter_dir=all_models/adapters/token/translation \ --output_path=results/completions/token/translation.jsonl \ --openai_api_key=YOUR_OPENAI_API_KEY

get_expert_scores.py

This script calculates the scores for each expert based on the evaluation datasets.

export  PYTHONPATH= $PYTHONPATH : $( pwd )  python scripts/expert/get_expert_scores.py \ --eval_dataset=intent \ --base_model_path=deepseek-ai/ESFT-vanilla-lite \ --output_dir=results/expert_scores/intent \ --n_sample_tokens=131072 \ --world_size=4 \ --gpus_per_rank=2  #  for N gpus, world_size should be N / gpus_per_rank

generate_expert_config.py

This script generates the configuration to convert a MoE model with only task-relevant tasks trained based on evaluation scores.

export  PYTHONPATH= $PYTHONPATH : $( pwd )  python scripts/expert/generate_expert_config.py \ --eval_dataset=intent \ --expert_scores_dir=results/expert_scores/intent \ --output_path=results/expert_configs/intent.json \ --score_function=token \ --top_p=0.2  #  the scoring function and top_p are hyperparameters

train_ep.py

This script trains the model with the expert configuration generated by the previous script. The train_ep.py file uses expert parallel and has been optimized for multi-GPU training.

python train.py \ --base_model_path=deepseek-ai/ESFT-vanilla-lite \ --expert_config=results/expert_configs/intent.json \ --train_dataset=intent \ --train_config=configs/base.yaml \ --output_dir=results/checkpoints/intent torchrun --nproc-per-node=8 train_ep.py \ --base_model_path=deepseek-ai/ESFT-vanilla-lite \ --expert_config=results/expert_configs/intent.json \ --train_dataset=intent \ --save_opt_states \ --train_config=configs/base.yaml \ --output_dir=results/checkpoints/test/eval_intent

## Contact and Support

For bug reports, feature requests, and general inquiries, please open an issue on our GitHub Issues page. Make sure to include as much detail as possible to help us address your issue quickly.

🌟Todo list

☑️ 📝 Update models, evaluation scripts, and expert selection scripts

☑️ 🔧 Update training scripts

🔲 🚀 More...

If you find our code or paper useful, please cite:

@article{wang2024letexpertsticklast, title={Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning  for  Sparse Architectural Large Language Models}, author={Zihan Wang and Deli Chen and Damai Dai and Runxin Xu and Zhuoshu Li and Y. Wu}, year={2024}, eprint={2407.01906}, archivePrefix={arXiv}, primaryClass={cs.CL}, url={https://arxiv.org/abs/2407.01906}, }

Expert Specialized Fine-Tuning

MIT, Unknown licenses found

## Licenses found

MIT   LICENSE-CODE

Unknown   LICENSE-MODEL

There was an error while loading.

## Please reload this page

https://github.com

https://github.com

## Custom properties

https://github.com

https://github.com

https://github.com

https://github.com

## Report repository

https://github.com

https://github.com

## No releases published

Packages  0

https://github.com

## No packages published

Contributors  2

https://github.com

ZihanWang314

## Zihan Wang

https://github.com

## GeeeekExplorer

## Xingkai Yu

https://github.com

Python   97.7%

https://github.com

Shell   2.3%

https://github.com

You can’t perform that action at this time.

