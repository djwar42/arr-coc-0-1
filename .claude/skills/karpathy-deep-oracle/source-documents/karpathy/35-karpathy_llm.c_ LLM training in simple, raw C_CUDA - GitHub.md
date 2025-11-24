---
sourceFile: "karpathy/llm.c: LLM training in simple, raw C/CUDA - GitHub"
exportedBy: "Kortex"
exportDate: "2025-10-28T19:01:56.502Z"
---

# karpathy/llm.c: LLM training in simple, raw C/CUDA - GitHub

3b90430d-c1b8-4f94-b717-29d68c79c56f

karpathy/llm.c: LLM training in simple, raw C/CUDA - GitHub

7c6723cc-60c4-4251-8ab6-746f9201ec02

https://github.com/karpathy/llm.c

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

https://github.com/karpathy

Couldn't load subscription status.

There was an error while loading.

## Please reload this page

https://github.com

Fork  3.3k

https://github.com

Star  28k

https://github.com

LLM training in simple, raw C/CUDA

## MIT license

https://github.com

28k  stars

https://github.com

3.3k  forks

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

## Discussions

https://github.com

https://github.com

https://github.com

https://github.com

https://github.com

karpathy/llm.c

https://github.com

https://github.com

## Open more actions menu

## Folders and files

## Name Name Last commit message Last commit date

## Latest commit

1,536 Commits

https://github.com

.github/ workflows

https://github.com

.github/ workflows

https://github.com

https://github.com

https://github.com

doc/ layernorm

https://github.com

doc/ layernorm

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

profile_gpt2.cu

https://github.com

profile_gpt2.cu

https://github.com

profile_gpt2cu.py

https://github.com

profile_gpt2cu.py

https://github.com

requirements.txt

https://github.com

requirements.txt

https://github.com

test_gpt2.c

https://github.com

test_gpt2.c

https://github.com

test_gpt2.cu

https://github.com

test_gpt2.cu

https://github.com

test_gpt2_fp32.cu

https://github.com

test_gpt2_fp32.cu

https://github.com

train_gpt2.c

https://github.com

train_gpt2.c

https://github.com

train_gpt2.cu

https://github.com

train_gpt2.cu

https://github.com

train_gpt2.py

https://github.com

train_gpt2.py

https://github.com

train_gpt2_fp32.cu

https://github.com

train_gpt2_fp32.cu

https://github.com

train_llama3.py

https://github.com

train_llama3.py

https://github.com

## Repository files navigation

LLMs in simple, pure C/CUDA with no need for 245MB of PyTorch or 107MB of cPython. Current focus is on pretraining, in particular reproducing the

https://github.com/openai/gpt-2

https://arxiv.org/abs/2005.14165

miniseries, along with a parallel PyTorch reference implementation in

train_gpt2.py

https://github.com/karpathy/llm.c/blob/master/train_gpt2.py

. You'll recognize this file as a slightly tweaked

https://github.com/karpathy/nanoGPT

, an earlier project of mine. Currently, llm.c is a bit faster than PyTorch Nightly (by about 7%). In addition to the bleeding edge mainline code in

train_gpt2.cu

https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu

, we have a simple reference CPU fp32 implementation in ~1,000 lines of clean code in one file

train_gpt2.c

https://github.com/karpathy/llm.c/blob/master/train_gpt2.c

. I'd like this repo to only maintain C and CUDA code. Ports to other languages or repos are very welcome, but should be done in separate repos, and I am happy to link to them below in the "notable forks" section. Developer coordination happens in the

## Discussions

https://github.com/karpathy/llm.c/discussions

and on Discord, either the  #llmc  channel on the

## Zero to Hero

https://discord.gg/3zy8kqD9Cp

channel, or on  #llmdotc  on

https://discord.gg/gpumode

quick start

The best introduction to the llm.c repo today is reproducing the GPT-2 (124M) model.

Discussion #481

https://github.com/karpathy/llm.c/discussions/481

steps through this in detail. We can reproduce other models from the GPT-2 and GPT-3 series in both llm.c and in the parallel implementation of PyTorch. Have a look at the

scripts README

https://github.com/karpathy/llm.c/blob/master/scripts/README.md

debugging tip: when you run the  make  command to build the binary, modify it by replacing  -O3  with  -g  so you can step through the code in your favorite IDE (e.g. vscode).

quick start (1 GPU, fp32 only)

If you won't be training on multiple nodes, aren't interested in mixed precision, and are interested in learning CUDA, the fp32 (legacy) files might be of interest to you. These are files that were "checkpointed" early in the history of llm.c and frozen in time. They are simpler, more portable, and possibly easier to understand. Run the 1 GPU, fp32 code like this:

chmod u+x ./dev/download_starter_pack.sh ./dev/download_starter_pack.sh make train_gpt2fp32cu ./train_gpt2fp32cu

The download_starter_pack.sh script is a quick & easy way to get started and it downloads a bunch of .bin files that help get you off the ground. These contain: 1) the GPT-2 124M model saved in fp32, in bfloat16, 2) a "debug state" used in unit testing (a small batch of data, and target activations and gradients), 3) the GPT-2 tokenizer, and 3) the tokenized

tinyshakespeare

https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

dataset. Alternatively, instead of running the .sh script, you can re-create these artifacts manually as follows:

pip install -r requirements.txt python dev/data/tinyshakespeare.py python train_gpt2.py

quick start (CPU)

The "I am so GPU poor that I don't even have one GPU" section. You can still enjoy seeing llm.c train! But you won't go too far. Just like the fp32 version above, the CPU version is an even earlier checkpoint in the history of llm.c, back when it was just a simple reference implementation in C. For example, instead of training from scratch, you can finetune a GPT-2 small (124M) to output Shakespeare-like text, as an example:

chmod u+x ./dev/download_starter_pack.sh ./dev/download_starter_pack.sh make train_gpt2 OMP_NUM_THREADS=8 ./train_gpt2

If you'd prefer to avoid running the starter pack script, then as mentioned in the previous section you can reproduce the exact same .bin files and artifacts by running  python dev/data/tinyshakespeare.py  and then  python train_gpt2.py .

The above lines (1) download an already tokenized

tinyshakespeare

https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

dataset and download the GPT-2 (124M) weights, (3) init from them in C and train for 40 steps on tineshakespeare with AdamW (using batch size 4, context length only 64), evaluate validation loss, and sample some text. Honestly, unless you have a beefy CPU (and can crank up the number of OMP threads in the launch command), you're not going to get that far on CPU training LLMs, but it might be a good demo/reference. The output looks like this on my MacBook Pro (Apple Silicon M3 Max):

[GPT-2] max_seq_len: 1024 vocab_size: 50257 num_layers: 12 num_heads: 12 channels: 768 num_parameters: 124439808 train dataset num_batches: 1192 val dataset num_batches: 128 num_activations: 73323776 val loss 5.252026 step 0: train loss 5.356189 (took 1452.121000 ms) step 1: train loss 4.301069 (took 1288.673000 ms) step 2: train loss 4.623322 (took 1369.394000 ms) step 3: train loss 4.600470 (took 1290.761000 ms) ... (trunctated) ... step 39: train loss 3.970751 (took 1323.779000 ms) val loss 4.107781 generating: --- Come Running Away, Greater conquer With the Imperial blood the heaviest host of the gods into this wondrous world beyond. I will not back thee, for how sweet after birth Netflix against repounder, will not flourish against the earlocks of Allay ---

The data files inside  /dev/data/(dataset).py  are responsible for downloading, tokenizing and saving the tokens to .bin files, readable easily from C. So for example when you run:

python dev/data/tinyshakespeare.py

## We download and tokenize the

tinyshakespeare

https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

dataset. The output of this looks like this:

writing 32,768 tokens to ./dev/data/tinyshakespeare/tiny_shakespeare_val.bin writing 305,260 tokens to ./dev/data/tinyshakespeare/tiny_shakespeare_train.bin

The .bin files contain a short header (1024 bytes) and then a stream of tokens in uint16, indicating the token ids with the GPT-2 tokenizer. More datasets are available in  /dev/data .

I am also attaching a simple unit test for making sure our C code agrees with the PyTorch code. On the CPU as an example, compile and run with:

make test_gpt2 ./test_gpt2

This now loads the  gpt2_124M_debug_state.bin  file that gets written by train_gpt2.py, runs a forward pass, compares the logits and loss with the PyTorch reference implementation, then it does 10 iterations of training with Adam and makes sure the losses match PyTorch. To test the GPU version we run:

#  fp32 test (cudnn not supported)  make test_gpt2cu PRECISION=FP32  &&  ./test_gpt2cu  #  mixed precision cudnn test  make test_gpt2cu USE_CUDNN=1  &&  ./test_gpt2cu

This tests both the fp32 path and the mixed precision path. The test should pass and print  overall okay: 1 .

I attached a very small tutorial here, in

doc/layernorm/layernorm.md

https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md

. It's a simple, step-by-step guide to implementing a single layer of the GPT-2 model, the layernorm layer. This is a good starting point to understand how the layers are implemented in C.

flash attention

. As of May 1, 2024 we use the Flash Attention from cuDNN. Because cuDNN bloats the compile time from a few seconds to ~minute and this code path is right now very new, this is disabled by default. You can enable it by compiling like this:

make train_gpt2cu USE_CUDNN=1

This will try to compile with cudnn and run it. You have to have cuDNN installed on your system. The

cuDNN installation instructions

https://developer.nvidia.com/cudnn

with apt-get will grab the default set of cuDNN packages. For a minimal setup, the cuDNN dev package is sufficient, e.g. on Ubuntu 22.04 for CUDA 12.x:

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb sudo dpkg -i cuda-keyring_1.1-1_all.deb sudo apt-get update sudo apt-get -y install libcudnn9-dev-cuda-12

## On top of this you need the

cuDNN frontend

https://github.com/NVIDIA/cudnn-frontend/tree/main

, but this is just header files. Simply clone the repo to your disk. The Makefile currently looks for it in either your home directory or the current directory. If you have put it elsewhere, add  CUDNN_FRONTEND_PATH=/path/to/your/cudnn-frontend/include  to the  make  command-line.

multi-GPU training

Make sure you install MPI and NCCL, e.g. on Linux:

sudo apt install openmpi-bin openmpi-doc libopenmpi-dev

## For NCCL follow the instructions from the

official website

https://developer.nvidia.com/nccl/nccl-download

(e.g. network installer)

make train_gpt2cu mpirun -np  < number of GPUs >  ./train_gpt2cu

or simply run one of our scripts under  ./scripts/ .

multi-node training

Make sure you've installed  NCCL  following instructions from

https://github.com#multi-gpu-training

There are 3 ways we currently support that allow you to run multi-node training:

Use OpenMPI to exchange nccl id and initialize NCCL. See e.g.  ./scripts/multi_node/run_gpt2_124M_mpi.sh  script for details.

Use shared file system to init NCCL. See  ./scripts/multi_node/run_gpt2_124M_fs.sbatch  script for details.

Use TCP sockets to init NCCL. See  ./scripts/multi_node/run_gpt2_124M_tcp.sbatch  script for details.

If you're running in a slurm environment and your slurm doesn't support PMIx (which we assume will be a common situation given that  slurm-wlm  dropped PMIx support) you will have to use FS (2) or TCP (3) approach. To test whether your slurm supports PMIx run:  srun --mpi=list  and see whether you get  pmix  in the output.

If you don't have slurm set up, you can kick off a multi-node run using  mpirun  - MPI (1).

None of these 3 methods is superior, we just offer you options so that you can run in your specific environment.

experiments / sweeps

Just as an example process to sweep learning rates on a machine with 4 GPUs on TinyStories. Run a shell script  sweep.sh  (after you of course  chmod u+x sweep.sh ):

#! /bin/bash  learning_rates=(3e-5 1e-4 3e-4 1e-3)  for   i   in  {0..3} ;   do   export  CUDA_VISIBLE_DEVICES= $i  screen -dmS  " tr $i "  bash -c  " ./train_gpt2cu -i data/TinyStories -v 250 -s 250 -g 144 -l  ${learning_rates[$i]}  -o stories $i .log "   done   #  you can bring these down with   #  screen -ls | grep -E "tr[0-3]" | cut -d. -f1 | xargs -I {} screen -X -S {} quit

This example opens up 4 screen sessions and runs the four commands with different LRs. This writes the log files  stories$i.log  with all the losses, which you can plot as you wish in Python. A quick example of how to parse and plot these logfiles is in

dev/vislog.ipynb

https://github.com/karpathy/llm.c/blob/master/dev/vislog.ipynb

A few more words on what I want this repo to be:

First, I want  llm.c  to be a place for education. E.g. our  dev/cuda  folder is a place for a library of kernels for all the layers that are manually hand-written and very well documented, starting from very simple kernels all the way to more complex / faster kernels. If you have a new kernel with various different tradeoffs, please feel free to contribute it here.

That said, I also want  llm.c  to be very fast too, even practically useful to train networks. E.g. to start, we should be able to reproduce the big GPT-2 (1.6B) training run. This requires that we incorporate whatever fastest kernels there are, including the use of libraries such as cuBLAS, cuBLASLt, CUTLASS, cuDNN, etc. I also think doing so serves an educational purpose to establish an expert upper bound, and a unit of measurement, e.g. you could say that your manually written kernels are 80% of cuBLAS speed, etc. Then you can choose to do a super fast run, or you can choose to "drag and drop" whatever manual kernels you wish to use, and run with those.

However, as a constraint, I want to keep the mainline  llm.c  in the root folder simple and readable. If there is a PR that e.g. improves performance by 2% but it "costs" 500 lines of complex C code, and maybe an exotic 3rd party dependency, I may reject the PR because the complexity is not worth it. As a concrete example - making cuBLAS for matmuls the default in the root training loop is a no-brainer: it makes the mainline code much faster, it is a single line of interpretable code, and it is a very common dependency. On the side of this, we can have manual implementations that can compete with cuBLAS in  dev/cuda .

Lastly, I will be a lot more sensitive to complexity in the root folder of the project, which contains the main / default files of the project. In comparison, the  dev/  folder is a bit more of a scratch space for us to develop a library of kernels or classes and share useful or related or educational code, and some of this code could be ok to be (locally) complex.

notable forks

## AMD support

https://github.com/karpathy/llm.c/blob/master/dev/vislog.ipynb

https://github.com/anthonix

: support for AMD devices, such as the 7900 XTX

https://github.com/anthonix

https://github.com/azret

: a C# port of this project

https://github.com/azret

https://github.com/nietras

: a C# port of this project with focus on easy to get started on any platform. Clone and run ✅

https://github.com/nietras

gevtushenko

https://github.com/gevtushenko

: a port of this project using the

CUDA C++ Core Libraries

https://github.com/NVIDIA/cccl

## A presentation this fork was covered in

this lecture

https://www.youtube.com/watch?v=WiB_3Csfj_Q

## GPU MODE Discord Server

https://discord.gg/cudamode

https://discord.gg/cudamode

https://github.com/zhangpiu

: a port of this project using the

https://gitlab.com/libeigen/eigen

, supporting CPU/CUDA.

https://gitlab.com/libeigen/eigen

austinvhuang

https://github.com/austinvhuang

: a library for portable GPU compute in C++ using native WebGPU. Aims to be a general-purpose library, but also porting llm.c kernels to WGSL.

https://github.com/austinvhuang

https://github.com/GaoYusong

: a port of this project featuring a C++ single-header

tinytorch.hpp

https://github.com/GaoYusong/llm.cpp/blob/main/tinytorch.hpp

https://github.com/GaoYusong/llm.cpp/blob/main/tinytorch.hpp

https://github.com/joshcarp

: a Go port of this project

https://github.com/joshcarp

harryjackson

https://github.com/harryjackson

: a Java port of this project

https://github.com/harryjackson

regrettable-username

https://github.com/regrettable-username

: LLM training in simple, raw C/Metal Shading Language

https://github.com/regrettable-username

https://github.com/dorjeduck

: a Mojo port of this project

https://github.com/dorjeduck

krrishnarraj

https://github.com/krrishnarraj

: an OpenCL port of this project

https://github.com/krrishnarraj

https://github.com/yijunyu

: a Rust rewrite with the aim to have same performance

https://github.com/yijunyu

https://github.com/ToJen

: a Rust port of this project

https://github.com/ToJen

https://github.com/otabuzzman

: a Swift port of this project

https://github.com/otabuzzman

https://github.com/Saimirbaci

: a Zig port of this project

Habana Gaudi2

https://github.com/Saimirbaci

abhilash1910

https://github.com/abhilash1910

: a Habana Gaudi2 port of this project

https://github.com/abhilash1910

https://github.com/Vindaar

: a Nim port of this project

discussions

Ways of organizing development:

Experiencing a concrete issue with the repo? Use

https://github.com/karpathy/llm.c/issues

Have some code to contribute? Open a

https://github.com/karpathy/llm.c/pulls

Chat about the repo, ask questions, etc.? Look at

## Discussions

https://github.com/karpathy/llm.c/discussions

Something faster? I created a new  #llmc  channel on my

## Zero to Hero Discord channel

https://discord.gg/3zy8kqD9Cp

LLM training in simple, raw C/CUDA

https://discord.gg/3zy8kqD9Cp

## MIT license

https://discord.gg/3zy8kqD9Cp

There was an error while loading.

## Please reload this page

https://github.com

https://github.com

https://github.com

https://github.com

https://github.com

## Report repository

https://github.com

There was an error while loading.

## Please reload this page

https://github.com

Contributors  64

https://github.com

+ 50 contributors

https://github.com

Cuda   66.2%

https://github.com

Python   13.5%

https://github.com

C   12.0%

https://github.com

C++   3.9%

https://github.com

Shell   2.2%

https://github.com

Makefile   1.7%

https://github.com

Jupyter Notebook   0.5%

https://github.com

You can’t perform that action at this time.

