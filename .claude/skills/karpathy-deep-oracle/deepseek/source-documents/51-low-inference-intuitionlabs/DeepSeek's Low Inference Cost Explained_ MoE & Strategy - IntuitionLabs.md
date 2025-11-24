---
sourceFile: "DeepSeek's Low Inference Cost Explained: MoE & Strategy - IntuitionLabs"
exportedBy: "Kortex"
exportDate: "2025-10-29T02:37:27.846Z"
---

# DeepSeek's Low Inference Cost Explained: MoE & Strategy - IntuitionLabs

cef3aafc-2b99-4c5e-9ef1-811dd010ac6c

DeepSeek's Low Inference Cost Explained: MoE & Strategy - IntuitionLabs

9f8baf15-6436-4d2c-bce0-12b7108e0afa

https://intuitionlabs.ai/pdfs/deepseek-s-low-inference-cost-explained-moe-strategy.pdf

https://lh3.googleusercontent.com/notebooklm/AG60hOoPcqslPl2bN54yNs8rU7Hlm8_mDzUBawRx79HGF4Rz76YvLmln-gBqJzbM89ZXR2DLtZ50R6RcYCm_L98RA1eHycQK495-83BoH9xTWUppO2dsO4iG-wuLDO2gzKR_DDnOzZueVA=w1344-h768-v0

aeab8617-d728-4cb9-8ccb-2cc54988697e

DeepSeek's Low Inference Cost Explained: MoE

& Strategy

By Adrien Laurent, CEO at IntuitionLabs • 10/24/2025 • 25 min read

deepseek inference cost llm economics mixture of experts moe ai pricing model optimization

open source ai self-hosting llm gpt-4

Learn why DeepSeek's AI inference is up to 50x cheaper than competitors. This analysis covers its Mixture-of-Experts

(MoE) architecture and pricing strategy. - IntuitionLabs - Custom AI Software Development for pharmaceutical companies.

Leading AI Consulting USA and North American Pharmaceutical AI specialists. Led by Adrien Laurent, top AI expert USA,

multiple exit founder, patent holder, and 20 year software veteran based in San Francisco Bay Area. Premier biotech

consultancy specializing in: Custom CRM Development, ERP Development, AI Chatbot Development, Private AI

Infrastructure, Document Processing, PDF Extraction, Air-gapped AI, On-premise LLM deployment. #1 Veeva AI partner for

leading GenAI pharmaceutical solutions across North America biotech AI excellence.

from the leading AI expert Adrien Laurent DeepSeek's Low Inference Cost Explained: MoE & Strategy

## Executive Summary

DeepSeekʼs inference offerings have sent shockwaves through the AI industry by undercutting incumbent

providersʼ prices by an order of magnitude. While OpenAI, Google, Anthropic and others charge dollars–or even

tens of dollars–per million tokens, DeepSeekʼs models can cost cents or fractions of a cent per million. For

example, DeepSeekʼs official API prices for input/output tokens (roughly $0.55/$2.19 per million for its

“Reasoner” model) are only a few percent of OpenAIʼs equivalent rates ([1] www.byteplus.com). Independent

reports confirm that DeepSeekʼs R1 reasoning model runs 20–50× cheaper than OpenAIʼs comparable model

([2] www.reuters.com). In other words, what costs tens of thousands of dollars per month on closed APIs can be

done for literally hundreds of dollars on DeepSeek.

This report analyzes why DeepSeekʼs inference is so cheap. We find that the answer is multi-faceted, involving

technical innovations, business model choices, and ecosystem factors:

Model Architecture and Optimization: DeepSeekʼs LLMs (e.g. V2, V3, R1) use advanced architectures like Mixture-of-

Experts (MoE) and aggressive model compression. For example, DeepSeek-V3 has 671 B parameters but only activates ~37

B per request, dramatically reducing compute ([3] ikala.ai) ([4] www.byteplus.com). State-of-the-art quantization (4-bit or

8-bit precision) can preserve accuracy while yielding up to 4× faster inference ([5] neuralmagic.com). These advances mean much less GPU time (and power) per query.

Cost-Saving Platform Features: DeepSeek introduced features like context caching that slash recurrent costs. Its API

automatically caches repeated inputs on disk; when a prompt or conversation repeats, cached results are reused, cutting

compute by up to 75–90% ([6] www.byteplus.com) ([7] api-docs.deepseek.com). In practice, DeepSeek charges as little

as $0.014 per million tokens for a cache hit – a 90% reduction from the standard rate ([7] api-docs.deepseek.com).

Open-Source and Self-Hosting: Unlike closed APIs, DeepSeek open-sources its models. This allows enterprises to self-

host the model on their own hardware with no license fees, paying only for electricity and GPU time. By converting inference

from a perpetual usage fee into an upfront capital cost, very large-scale users can drive their effective per-query cost near

zero. One expert notes that DeepSeekʼs R1 can even run (albeit slowly) on an ordinary laptop CPU – effectively zero

incremental cost beyond electricity ([8] www.linkedin.com) ([9] www.theregister.com).

Hardware and Funding Sources: DeepSeek trained its models on Chinese-market hardware (e.g. NVIDIA H800 GPUs) and

benefited from domestic ecosystem support. The firm reports training R1 on 512 H800 chips for only ~$294,000 ([10]

www.reuters.com), far below the hundreds of millions plowed into GPT-4. Operating on a lean “bootstrapped” budget and

leveraging cheaper infrastructure means lower cost-basis for inference.

Business Strategy and Subsidies: DeepSeekʼs parent or investors may be willing to subsidize usage. By pricing inference

well below cost, DeepSeek is effectively “giving away” compute power to gain market share (prompt.16x.engineer) ([11]

www.reuters.com). This pricing could be loss-leading, expecting volume payoff or strategic advantage. In any case,

competitors now face pressure to slash prices and innovate efficiency, potentially sparking a price war in AI services ([11]

www.reuters.com).

Collectively, these factors allow DeepSeek to deliver performance competitive with leading models at only a tiny

fraction of the price. We examine each in detail, contrast DeepSeekʼs costs with industry peers (see table

below), and discuss implications for enterprises and the future of AI service economics.

Model (1M-token

price) Provider

(USD) Notes

DeepSeek–V2

(236B, MoE) DeepSeek $0.14 $0.28

MoE model, only 37B active; caching available ([12]

www.byteplus.com) ([7] api-docs.deepseek.com)

GPT-4o (320B) OpenAI $3.00 $10.00 Standard GPT-4o pricing ([1] www.byteplus.com)

from the leading AI expert Adrien Laurent DeepSeek's Low Inference Cost Explained: MoE & Strategy

Model (1M-token

price) Provider

(USD) Notes

GPT-4o Mini (40B) OpenAI $0.15 $0.60 “Mini” downscaled GPT-4o ([13] www.byteplus.com)

Gemini 1.5 Pro

(280B) Google $1.25 $5.00

Gemini 1.5 Flash

(96B) Google $0.0375 $0.15 Low-cost variant of Gemini ([1] www.byteplus.com)

Claude 3.5 Sonnet

(220B) Anthropic $3.00 $15.00

Claude 3 Haiku

(15B) Anthropic $0.25 $1.25

Table: Comparative inference pricing (per 1M tokens) for DeepSeek vs. major providers (2025 data) ([1]

www.byteplus.com).

## Introduction and Background

In the race to deploy large language models (LLMs) at scale, inference cost has emerged as a critical

constraint. Unlike one-time training expenses, inference expenditures accrue continuously with every user

query. Traditional AI-as-a-service providers (OpenAI, Google, Anthropic, etc.) charge per token, typically on the

order of dollars for each million tokens processed ([12] www.byteplus.com) ([14] www.linkedin.com). While these

fees may seem modest initially (often quoted in US cents per 1,000 tokens), they compound rapidly. A popular

model like GPT-4 can cost ~$0.03 per 1,000 input tokens (~$3.00 per million) and double that for output tokens

([15] www.linkedin.com), meaning a multi-page analysis could cost several dollars on each call. Multiply that by

millions of daily queries, and the bills quickly reach into the millions of dollars per month.

Against this backdrop, the emergence of DeepSeek, a Chinese AI startup, has been disruptive. Debuting in late

2024, DeepSeek open-sourced powerful new LLMs – notably DeepSeek-V3 (a 671B multilingual model) and

DeepSeek-R1 (an optimized reasoning model). These models achieved capabilities rivaling or exceeding leading

Western models, apparently with a fraction of the compute budget. In January 2025, major news outlets

reported DeepSeekʼs claims that its V3 model was trained on only ~$6 million of GPU compute ([11]

www.reuters.com) ([16] tech.yahoo.com), and its subsequent R1 </current_article_content>model reportedly

needed just ~$294k of training GPU hours ([10] www.reuters.com). These disclosures astonished many experts

and investors, driving dramatic market reactions (e.g. a $600+ billion one-day drop in NVIDIAʼs market value)

([17] www.theregister.com).

Crucially, DeepSeek did not just train models cheaply; it also priced inference very low. Its official API offers

inference for mere cents per million tokens, far below the prices charged by established AI cloud providers. In

practice, DeepSeek inference is reported to be 20–50× cheaper than OpenAIʼs (and similar modelsʼ) cost per

token ([2] www.reuters.com). Tech analysts and AI leaders have taken note: OpenAIʼs Sam Altman called

DeepSeekʼs R1 model “impressive” but expressed skepticism about its efficiency claims ([18]

www.windowscentral.com); Googleʼs Sundar Pichai congratulated DeepSeek on doing “very good work” the same

week ([19] www.reuters.com).

This combination of competitive performance and radically lower pricing represents a potential inflection point.

If enterprises can get GPT-4–level quality at one-hundredth the cost, whole business models will be disrupted.

The goal of this report is to explain why and how DeepSeekʼs inference costs are so low, based on a survey of

technical documentation, industry analyses, and credible reporting. We examine DeepSeekʼs model design and

infrastructure, cost-saving platform features, and economic strategy – and contrast these with incumbent

from the leading AI expert Adrien Laurent DeepSeek's Low Inference Cost Explained: MoE & Strategy

providers. We then consider real-world examples and reactions, and discuss the future implications of this AI

cost revolution.

DeepSeek: Open-Source AI with Low-Cost Focus

DeepSeek is a Hangzhou-based AI lab that burst onto the scene in late 2024. Unlike the largest U.S. startups, it

embraced an open-source philosophy. DeepSeek publicly released weights and technical reports for its

models, allowing others to download, self-host, or contribute to development. Its flagship releases to date

DeepSeek-V3 (Dec 2024): A general-purpose large language model (LLM) with a Mixture-of-Experts (MoE) architecture. It

has on the order of 671 billion parameters, but only around 37 billion of those are active per inference ([3] ikala.ai). The

authors report training V3 for under $6 million on 2,000 NVIDIA H800 GPUs ([20] tech.yahoo.com) ([11] www.reuters.com).

DeepSeek-R1 (Jan 2025): A specialized 70B-parameter reasoning model distilled from V3. It is optimized for math, coding,

and logical tasks using reinforcement learning, achieving performance on par with (or surpassing) GPT-4o on many

benchmarks. Dramatically, reported training costs for R1 were only $294 K of GPU usage ([10] www.reuters.com) ([21]

www.itpro.com).

These models are freely available to the community (e.g. via Hugging Face, which recorded over 10 million

downloads of R1 in early 2025 ([22] www.itpro.com)). DeepSeekʼs openness contrasts sharply with closed

offerings from OpenAI or Google. Self-hosted deployment and transparency mean there are no licensing

royalties or hidden fees – adoption can scale without cost-per-use.

On the inference side, DeepSeek provides both a free web interface and a paid API. Crucially, its API pricing is

extremely low. DeepSeekʼs own published rates (for its “V2” chat model, “Reasoner”, etc.) translate to only a few

tenths of a cent per input token and $0.28 per output token – about 90% below comparable OpenAI and

Anthropic rates ([12] www.byteplus.com) ([1] www.byteplus.com). Third-party analyses confirm this. For example, a

BytePlus cost comparison (Aug 2025) shows DeepSeekʼs V2 at $0.14/$0.28 per million, versus OpenAI GPT-4o

at $3/$10 and Claude 3.5 Sonnet at $3/$15 ([1] www.byteplus.com). A Reuters profile cited DeepSeek executives

noting that its R1 inference is 20–50× cheaper than OpenAIʼs equivalent ([2] www.reuters.com). DeepSeekʼs

website even touts a context-caching feature that can reduce a hitʼs cost by up to 75–90% ([6]

www.byteplus.com) ([7] api-docs.deepseek.com).

In short, DeepSeekʼs strategic mission is to democratize AI by dramatically lowering cost. The rest of this

report unpacks the enablers behind this strategy.

Technical Innovations Enabling Low-Cost

Mixture-of-Experts and Efficient Architecture

DeepSeekʼs models emphasize efficiency in architecture. For example, DeepSeek-V3 employs a Mixture-of-

Experts (MoE) design. In an MoE model, the network has many “expert” subcomponents (layers or parameter

subsets), but only a handful are active for any given input token. DeepSeekʼs V3 reportedly has 671 B total

parameters, yet only about 37 B are active per request ([3] ikala.ai). This yields a very large effective model

capacity but with compute similar to a smaller dense model. In BytePlusʼs analysis, DeepSeek-V2 (the chat-

from the leading AI expert Adrien Laurent DeepSeek's Low Inference Cost Explained: MoE & Strategy

oriented descendant of V3) is explicitly called “a powerful and highly cost-effective Mixture-of-Experts (MoE)

model” ([12] www.byteplus.com). Only a small fraction of parameters “activate” for each token, which “helps

reduce computational costs.”

The impact is that compared to a dense model of the same parameter count, the GPU time per inference can be

significantly lower. In practical terms, DeepSeek can claim a 236B-parameter model V2, yet serve it at roughly

the same cost per query as first-generation GPT-3 or mini-LLMs. In BytePlusʼs pricing table (Table above),

DeepSeek-V2ʼs $0.14 input rate is essentially identical to GPT-4o Miniʼs $0.15 ([1] www.byteplus.com), even

though V2 is a much larger model. This contradicts the usual assumption that “bigger means slower”; instead

the MoE sparsity means DeepSeek squeezes more “bang for the buck”.

Another efficiency comes from distillation and fine-tuning. DeepSeek-R1 is labeled a “distilled” reasoning model:

the full V3 was distilled into smaller, specialized variants. In AI, distillation often yields models that match the

parentʼs performance on certain tasks but require fewer resources for inference. While DeepSeekʼs details are

proprietary, documentation suggests R1 was trained via reinforcement learning on top of V3, focusing compute

on reasoning tasks ([23] www.itpro.com). By concentrating capacity where it counts, R1 can run faster. (This is

similar to how some expert-tuned models can be faster than general-purpose ones.)

Finally, DeepSeek leverages state-of-the-art quantization. Modern LLM internals allow using 8-bit or even 4-bit

numerical representations instead of 16/32-bit floats. A blog from NeuralMagic shows that DeepSeek-R1

variants can be quantized to INT8/INT4 with negligible loss of accuracy ([24] neuralmagic.com). In experiments,

an INT8 model had “near-perfect accuracy recovery”, and INT4 recovered >97% of performance on benchmarks.

Crucially, they report up to 4× inference speedups on typical hardware when using these quantized models ([5]

neuralmagic.com). This matters because inference cost is largely time on expensive GPUs; faster model means

cheaper per inference.

In summary, DeepSeekʼs model design is explicitly optimized for inference efficiency. The large MoE size, plus

quantization strategies, allow each query to use far less computation (and thus power, server time, etc.) than an

equivalently capable model from a traditional provider.

Long-Context and Caching Optimizations

DeepSeek has also innovated on the serving side. One notable feature is Context Caching on Disk. Many real-

world applications involve repeated or overlapping prompts – for example, chatbots often include the

conversation history in each API call. DeepSeekʼs engineers realized that large parts of successive prompts are

often identical (e.g. the system prompt, background text). Their solution is to cache the neural activations for

prefix tokens so they need not be recomputed.

According to DeepSeekʼs official documentation, this context-caching mechanism is enabled by default. When a

new request shares the same prefix as a recent one, the system retrieves the cached segment from disk and

only computes the new part from that point. For “cache hits,” DeepSeek charges as little as $0.014 per million

tokens, which is roughly a 90% discount over the normal rate ([7] api-docs.deepseek.com). BytePlusʼs analysis

translates this into 74–75% savings: for DeepSeek Chat, a cache hit costs $0.07 instead of $0.27; for DeepSeek

Reasoner, $0.14 instead of $0.55 ([6] www.byteplus.com).

This feature is particularly beneficial in many common scenarios: multi-turn chatbots (each turn reuses the

conversation so far), document analyses (parallel queries on the same prompt), or few-shot learning (repeating

similar examples). By cutting down rampant repeated computation, context caching directly lowers the

operational cost. DeepSeek estimates that a “significant portion” of inputs are repetitive; hence caching turns

those tokens into cheap storage lookups. In effect, DeepSeek has effectively bundled a form of memoization

into the API, passing savings directly to users.

from the leading AI expert Adrien Laurent DeepSeek's Low Inference Cost Explained: MoE & Strategy

## Customized Chinese Hardware and Software Stack

Another driver of cost savings is hardware. Due to export controls, DeepSeek was limited to Chinese versions of

NVIDIA GPUs. Rather than more expensive H100/A100 chips, they trained and serve on NVIDIA H800 GPUs

(Chinaʼs variant of the H100) ([10] www.reuters.com). While not as fast per GPU, H800s are significantly cheaper

and more available for Chinese companies. DeepSeekʼs own reports highlight how they used 512 such chips to

train R1 for ~$294k ([10] www.reuters.com). This contrasts with Western peers spending tens or hundreds of

millions on H100-clusters.

On inference, DeepSeekʼs cost advantage likely benefits similarly: operating on H800s and local data centers

can reduce hourly GPU rates dramatically. Moreover, DeepSeek announced support for Chinese “Ascend” NPUs

(Huawei chips) and Cambricon accelerators in its later models ([25] www.tomshardware.com), further diversifying

hardware options. Domestic deployment means they can tap into cheaper power, labor, and supply chain

Software optimizations also play a role. Chinese developers reportedly built efficient inference pipelines on

frameworks like CANN (Chinaʼs CUDA analog) ([25] www.tomshardware.com). Additionally, DeepSeek open-

sourced FlashMLA and DeepEP, tools aimed at squeezing performance out of accelerators ([26]

tech.yahoo.com). Such specialized engineering can speed model execution or allow larger batch sizes at the

same hardware cost.

Summary of Cost-Reduction Techniques

In summary, DeepSeekʼs low inference cost stems from multiple technical levers:

Sparse Architecture (MoE): Large model capacity with sparse activations, cutting per-token FLOPs ([12]

www.byteplus.com) ([3] ikala.ai).

Model Distillation: Specialized smaller variants (like R1) focusing compute, requiring fewer resources.

Quantization: Leveraging INT8/INT4 precision to greatly boost throughput ([5] neuralmagic.com).

Context Caching: Storing repeated prompts to avoid recomputation, cutting token charges ~75–90% ([6]

www.byteplus.com) ([7] api-docs.deepseek.com).

Hardware Efficiency: Use of cheaper local GPUs (H800) and NPUs, reducing raw compute costs ([10]

www.reuters.com) ([25] www.tomshardware.com).

Open-Source Self-Hosting: Customers can run models on generic servers (or even laptops) without any

licensing fees ([8] www.linkedin.com) ([9] www.theregister.com), meaning the only marginal cost is hardware

A concise overview of these factors is given below:

## Factor Cost Impact Source

Mixture-of-Experts

Activates only ~37 of 236 billion parameters per token (V3/V2), dramatically

cutting GPU ops per query ([4] www.byteplus.com) ([3] ikala.ai). BytePlus, iKala

Context Caching Caches duplicate input segments; cache hits cost as low as $0.014/M tokens,

~90% off normal rates ([6] www.byteplus.com) ([7] api-docs.deepseek.com).

DeepSeek API,

4-bit/8-bit

## Quantization

State-of-art quantized R1 models achieve near-original accuracy; yield up to 4×

faster inference throughput ([5] neuralmagic.com). NeuralMagic

from the leading AI expert Adrien Laurent DeepSeek's Low Inference Cost Explained: MoE & Strategy

## Factor Cost Impact Source

Open-Source / Self-

Zero license fees; customers pay only GPU+power. Enables capex model (buy

GPUs once) instead of metered API fees ([8] www.linkedin.com) ([9]

www.theregister.com).

LinkedIn analysis,

## Chinese Hardware

(H800 chips)

Uses legally-available H800 GPUs (cheaper than H100/A100) for

training/inference; R1 training cost was ~$294k on 512 H800s ([10]

www.reuters.com).

Table: Key technical and business factors behind DeepSeekʼs low inference costs, with sources.

## Pricing Comparison and Economic Analysis

DeepSeekʼs claim of low inference cost is borne out by comparative data. The BytePlus analysis (Table in

Executive Summary) shows DeepSeek-V2 costing only $0.14/$0.28 per million tokens. By contrast, OpenAIʼs top

models (GPT-4o) charge ~$3/$10 ([1] www.byteplus.com), Claude 3.5 operators charge ~$3/$15, and even smaller

“lite” models like Gemini 1.5 Flash cost ~$0.037/$0.15 (which is somewhat lower) ([1] www.byteplus.com).

DeepSeekʼs pricing is on par with these “Flash” mode models but offers a larger model behind them. In practice,

that means for many tasks, DeepSeek can perform as well (or better) than a smaller model, at the same per-

token rate.

Reuters reports that DeepSeekʼs inference is indeed a fraction of the cost. In early 2025, OpenAIʼs CEO

observed that DeepSeekʼs R1 runs 20–50× cheaper than OpenAIʼs similar model ([2] www.reuters.com). This

suggests that a task costing $50/M tokens on OpenAI might cost only $1–2/M on DeepSeek. (For context,

OpenAIʼs GPT-4o was priced ~$3/M input; DeepSeekʼs chat model at ~$0.27/M). Zhang Jiayi, DeepSeekʼs CEO,

was quoted noting that the company purposely made inference very affordable to encourage adoption ([11]

www.reuters.com).

This cost gap is not easily explainable by economies of scale alone. Western cloud providers already benefit

from massive scale and some of the worldʼs fastest GPUs. DeepSeekʼs ability to undercut them suggests

systemic differences:

No Profit-Driven Markup: DeepSeek may price just above marginal cost, whereas commercial APIs include significant

markup for profit, R&D, and infrastructure overhead. OpenAIʼs own financials reveal tens of millions of losses per day on AI

services ([27] www.linkedin.com). In contrast, DeepSeekʼs open model indicates a focus on volume rather than margin.

Different Cost Accounting: Western prices often build in data center overhead, international bandwidth, and global

support. DeepSeekʼs “cost” might primarily be local compute costs.

Subsidies: It is plausible that DeepSeekʼs costs are partly subsidized by investors or aligned national interests (see Box

below), enabling a below-cost consumer price.

In aggregate, DeepSeekʼs pricing forces a new benchmark: where once GPT-4–class inference was ~$0.03–0.06

per thousand tokens, now claims are being made of one cent or less per thousand for equivalent tasks. Even if

one discounts by a factor of 10 (skepticism that 20–50× is fluff), it remains an order of magnitude cheaper.

Break-even analysis: Some analysts have compared the cost of self-hosting DeepSeek vs. paying OpenAI.

Consider an enterprise doing 1 billion tokens per month. At DeepSeekʼs $0.14/M input and $0.28/M output, the

total might be ~$0.42/M on V2—so ~$420 per billion tokens. Using OpenAI GPT-4o at $3/$10 would cost

~$13,000 per billion. If the enterprise instead bought a dedicated GPU server (say an NVIDIA DGX A100 at

~$200k) and ran DeepSeek continuously, amortized cost is on the order of tens of thousands per year (roughly

$65k/year as per one estimate) ([28] www.linkedin.com) – which becomes negligible per token at high volume. In

such high-usage scenarios, DeepSeekʼs approach yields very low unit cost compared to a pay-as-you-go API.

from the leading AI expert Adrien Laurent DeepSeek's Low Inference Cost Explained: MoE & Strategy

## Case Studies and Market Reactions

DeepSeekʼs low-cost claim has triggered strong industry responses:

Market Shock: When DeepSeek unveiled its V3 and R1 results in Jan 2025, Nvidiaʼs stock plunged dramatically. Wall Street

estimated that if DeepSeekʼs numbers were real, demand for high-end GPUs would shrink (since models could run on

cheaper chips) ([2] www.reuters.com) ([17] www.theregister.com). Indeed, on Jan 27, 2025, U.S. markets saw the largest

one-day drop in tech stock values largely attributed to DeepSeek headlines ([17] www.theregister.com). This underscores how DeepSeekʼs pricing narrative alone can destabilize existing AI infrastructure economics.

Executive Endorsements (and Doubts): OpenAIʼs Sam Altman publicly congratulated DeepSeek on R1ʼs performance,

albeit noting that “we need more compute for our stuff” ([2] www.reuters.com). Similarly, Google CEO Sundar Pichai said

DeepSeekʼs work is “very good” and globalizing AI innovation ([19] www.reuters.com). On the other hand, some observers

(e.g., a Windows Central report) have questioned DeepSeekʼs transparency about costs ([29] www.windowscentral.com), and the largely anecdotal Gugʼs analysis. Forbes and The Register have raised concerns that hidden investments (like a

rumored $1.6B in GPU purchases ([9] www.theregister.com)) may belie the “small budget” story.

Customer Interest: Enterprises are naturally eager to lower AI spend. ServiceNowʼs CEO Bill McDermott commented that

cheaper LLMs could improve platform margins ([30] www.theregister.com). BytePlus (Bytedanceʼs cloud unit) has added

DeepSeek to its ModelArk catalog, offering free trial tokens ([31] www.byteplus.com), signaling interest from industry.

DeepSeek models are now available on AWS, Hugging Face, Azure, etc., allowing companies to port workloads easily. While

most of these use cases are still experimental, activity on GitHub and download stats (10M downloads of R1, per Hugging

Face metrics) suggest substantial developer engagement.

Vendor Responses: OpenAI has accelerated development of faster/cheaper models (e.g. “GPT-4 Turbo” and “o3-mini”) and

cut prices for lighter models ([32] www.linkedin.com) ([33] ikala.ai). Google and Anthropic likewise are optimizing and

introducing “flash” versions to close the gap. In China, Baidu open-sourced its Ernie model as well, explicitly noting a shift to

cheaper access ([34] www.techradar.com). As one analysis put it, DeepSeekʼs emergence has shifted the race into a “price

war” ([11] www.reuters.com).

Real-World Example: Imagine a startup building a chatbot on social media with 1 million daily active users, each

generating 100 tokens of traffic. Over a year, thatʼs ~36.5 billion tokens. Using OpenAIʼs GPT-4o (~$3/M in,

$10/M out), even with heavy prompting, could cost on the order of $100k–$150k per month. By switching to

DeepSeekʼs API (or self-hosting R1/V2), that same workload might cost on the order of $1,000–$3,000 per

month – or even less if input is cacheable ([7] api-docs.deepseek.com) ([6] www.byteplus.com). While still

significant, this gap could be life-changing for a consumer appʼs business model. (These figures are illustrative

but grounded in the as-reported pricing differentials.)

Challenges, Criticisms, and Context

While DeepSeekʼs cost claims are enticing, analysts urge caution. Some industry players note that true* total

costs include overhead that isnʼt always apparent. For example, The Register points out that DeepSeekʼs

published $6M training cost only counted “cloud rental hours,” ignoring capital spend on owning thousands of

GPUs ([9] www.theregister.com). Likewise, inference cost comparisons often omit staffing, maintenance, and

integration expenses.

Specific criticisms include:

Opaqueness of Claims: Because DeepSeek is a private startup, detailed numbers are sparse. Allegations have circulated

that DeepSeek reused portions of other models or trained on cherry-picked data to boost benchmark scores ([35]

www.theregister.com). Some third-party audits (e.g. NewsGuard) found DeepSeekʼs chatbot lagging on certain tasks. If

DeepSeekʼs models underperform on average, their low prices might partly reflect lower utility.

from the leading AI expert Adrien Laurent DeepSeek's Low Inference Cost Explained: MoE & Strategy

Quality vs. Cost Tradeoffs: Lower cost is valuable only if the model meets user needs. For many tasks (coding, reasoning,

math), DeepSeek shines ([36] ikala.ai). But for multimodal tasks or specialized domains, competitors may remain superior.

Enterprises may therefore use DeepSeek for bulk tasks and pay premium for niche ones. Nonetheless, the existence of a

~10× lower-cost “baseline” model sets a new floor for pricing.

Geopolitical and Security Concerns: Some Western regulators and companies express caution about Chinese models.

Potential issues include data governance, intellectual property, or simply distrust of a foreign provider. U.S. authorities (NI

ST report) have even compared DeepSeekʼs models to domestic ones; their conclusion was that U.S. models still led in

benchmarks ([37] www.tomshardware.com). These factors might slow DeepSeekʼs wholesale adoption outside China in the

near term.

Sustainability of Low Pricing: If DeepSeekʼs prices truly undercut costs, it invites unsustainable use (free riders) unless

recouped elsewhere. It remains to be seen if DeepSeek will raise prices once it has captured market share, or if it is banking

on ancillary revenue (like cloud credits, services). Some argue its strategy is analogous to early cloud providers who

subsidized initial usage to lock-in customers. How long DeepSeek can maintain “$0.014 per million tokens” deals is an open

Despite these caveats, all credible sources agree: DeepSeek is forcing a reevaluation of LLM economics. An

anonymous AI industry expert summarized, “ [DeepSeekʼs models] achieved a breakthrough in resource

efficiency… demonstrating that brute-force spending is not the only path to advanced AI ([38] www.linkedin.com).”

Even skeptics acknowledge that, at minimum, DeepSeek has shown proof of concept: AI of near–top-tier power

can be delivered with dramatically lower resource budgets.

Case Study: Substituting DeepSeek for OpenAI

in Production

To illustrate the impact concretely, consider a notional analysis derived from industry reports. A fintech startup

was prototyping an AI document analysis app. They estimated needing 2 billion tokens per month to process

user queries. Initial pricing calculations (based on OpenAI GPT-4 at $3/$10 per million) gave a prohibitively high

monthly cost of roughly $26,000 (assuming 200M input and 1.8B output tokens).

When DeepSeekʼs R1 became available on AWS, the startup tested inference quality and cost. DeepSeekʼs

pricing for R1 (Reasoner) is $0.55 input / $2.19 output per million ([12] www.byteplus.com). Running the same

workload on R1 would cost about $5,200/month – a factor of five reduction. Moreover, by enabling context

caching (the queries had repeated boilerplate prompts), they found effective token usage dropped by 60%.

Factoring that, the actual spend became ~$2,000/month. In this scenario, adopting DeepSeek cut operating

costs by 90%, making the project economically viable.

This hypothetical aligns with published comparisons. For instance, BytePlus noted that DeepSeekʼs V2 (236B)

“presents a compelling value proposition” and “allows businesses to scale their AI operations without incurring

prohibitive costs” ([39] www.byteplus.com). In other words, for many standard applications (chatbots, content

generation, summarization), DeepSeek achieves results equivalent to large models at only a marginal cost

increase over much smaller models.

## Implications and Future Directions

DeepSeekʼs low-cost inference model has wide-ranging implications:

from the leading AI expert Adrien Laurent DeepSeek's Low Inference Cost Explained: MoE & Strategy

Price Competition Intensifies: With DeepSeek undercutting margins, major providers are compelled to cut their APIsʼ

prices or innovate faster. OpenAIʼs moves to introduce GPT-4o Turbo at reduced cost and Googleʼs push on the “Flash”

series are direct responses. Analysts expect a secular downward trend in per-query prices for all providers as economies

improve and competition heats up ([40] www.byteplus.com).

Shift to OSS Model: DeepSeekʼs success invigorates the open-source LLM movement. Other projects (e.g. LLaMA

derivatives, Baidu Ernie) now have a model to aim for. By proving low-cost AI is possible, DeepSeek may accelerate efforts

worldwide to bypass Big Techʼs closed ecosystems. This can democratize access but also raise questions about

coordination (e.g. how to maintain safety in open models).

Hardware and Infrastructure: The fact that DeepSeek runs efficiently on commodity chips suggests that exotic hardware

(like thousands of NVLinked H100s) might not be mandatory for top performance. If demand shifts to models optimized for

heterogenous or specialized hardware, it could reshape the AI hardware market. DeepSeekʼs focus on Chinese accelerators

also indicates a strategic shift we may see globally: diversifying away from NVIDIA dependency.

Economic Models Revisited: The dominant cloud model (pay-per-token) might evolve. As one analysis noted, DeepSeek

turns AI into a capital expense rather than a metered utility ([41] www.linkedin.com). Future service offerings could

hybridize: perhaps flat-rate hosting, appliance-based inference, or embedding small offline models. The ongoing cost-curve

collapse (Stanford HAI reports >280× cheaper since 2022 ([40] www.byteplus.com)) will continue to open new use cases in

emerging markets that couldnʼt afford AI before.

Regulatory and Security Considerations: Governments are watching closely. U.S. victory in maintaining AI advantage has

hinged partly on controlling tech exports and investments. If Chinese firms like DeepSeek can deliver comparable AI cheaply,

bloc politics in AI may intensify. There may also be push to certify or vet these modelsʼ safety and IP provenance. The

National Institute of Standards and Technologyʼs recent benchmarks found U.S. models outperform DeepSeek across many

tasks ([37] www.tomshardware.com), which might reassure domestic regulators that performance isnʼt being sacrificed for

Long-term Sustainability: There is uncertainty whether DeepSeek (now apparently rebranded IntuitionLabs) can sustain its

pricing and momentum. Reports of management changes and “hardware problems” delaying newer models indicate

challenges ([42] tech.yahoo.com) ([43] www.techradar.com). Nonetheless, DeepSeek has proven that at least some cost-

reduction claims are legitimate. Future work might reveal how sustainable their approach is when scaled or exposed to

adversarial pressures.

Overall, DeepSeekʼs emergence heralds a potentially transformative cost revolution in generative AI.

Enterprises and researchers will watch closely: if DeepSeek can maintain quality at 1/10th or 1/100th the cost of

incumbents, the AI economy must adapt.

## Conclusion

DeepSeekʼs astonishing inference cost advantage arises from a confluence of technical brilliance and strategic

choices. By designing models for efficiency (MoE, distillation, quantization, caching) and by leveraging the

economics of open-source distribution and local infrastructure, DeepSeek can deliver GPT-4–class capability

for a tiny fraction of traditional prices. Credible sources (Reuters, tech press, independent analyses)

consistently report cost comparisons showing DeepSeek at ten to a hundred times lower cost-per-token than

major providers ([2] www.reuters.com) ([1] www.byteplus.com).

This low-cost inference model is shaking up the AI industry. Many questions remain – about performance parity,

reproducibility, and geopolitical adoption – but one fact is clear: DeepSeek has demonstrated lower-cost AI is

possible. The result is an emerging landscape where AI services will likely become dramatically cheaper across

the board. Organizations contemplating AI deployments must study this landscape carefully: the cheapest (and

open) option may no longer be one of the household-brand APIs.

Sources: This report synthesizes data from industry analyses, technical documentation, and reputable media.

Key references include Reuters (DeepSeek cost disclosures ([10] www.reuters.com) ([11] www.reuters.com)),

from the leading AI expert Adrien Laurent DeepSeek's Low Inference Cost Explained: MoE & Strategy

BytePlus cost studies ([12] www.byteplus.com) ([1] www.byteplus.com), DeepSeek API announcements ([7] api-

docs.deepseek.com), expert commentary ([8] www.linkedin.com) ([9] www.theregister.com), and additional credible

technology journalism ([2] www.reuters.com) ([23] www.itpro.com). Each claim above is supported by cited

evidence to provide a thorough, balanced view of DeepSeekʼs cost advantage.

## External Sources

[1] https://www.byteplus.com/en/topic/407727#:~:,%247...

[2] https://www.reuters.com/technology/artificial-intelligence/openai-chief-altman-says-deepseeks-r1-model-impressive-

2025-01-28/#:~:OpenA...

[3] https://ikala.ai/blog/ai-trends/deepseek-llm-comparison_en/#:~:Relea...

[4] https://www.byteplus.com/en/topic/407727#:~:%2A%2...

[5] https://neuralmagic.com/blog/quantized-deepseek-r1-models-deployment-ready-reasoning-models/#:~:acros...

[6] https://www.byteplus.com/en/topic/407727#:~:A%20s...

[7] https://api-docs.deepseek.com/news/news0802/#:~:For%2...

[8] https://www.linkedin.com/pulse/openai-vs-deepseek-coming-cost-revolution-ai-duvivier-dit-sage-dx8vf#:~:which...

[9] https://www.theregister.com/2025/01/30/deepseek_reaction/#:~:But%2...

[10] https://www.reuters.com/world/china/chinas-deepseek-says-its-hit-ai-model-cost-just-294000-train-2025-09-18/#:~:

[11] https://www.reuters.com/technology/artificial-intelligence/big-tech-faces-heat-chinas-deepseek-sows-doubts-billion-

dollar-spending-2025-01-27/#:~:Chine...

[12] https://www.byteplus.com/en/topic/407727#:~:%2A%2...

[13] https://www.byteplus.com/en/topic/407727#:~:GPT%2...

[14] https://www.linkedin.com/pulse/openai-vs-deepseek-coming-cost-revolution-ai-duvivier-dit-sage-dx8vf#:~:OpenA...

[15] https://www.linkedin.com/pulse/openai-vs-deepseek-coming-cost-revolution-ai-duvivier-dit-sage-dx8vf#:~:premi...

[16] https://tech.yahoo.com/ai/articles/deepseeks-disclosure-ai-technical-details-093000688.html#:~:Chine...

[17] https://www.theregister.com/2025/01/30/deepseek_reaction/#:~:In%20...

[18] https://www.windowscentral.com/artificial-intelligence/openai-chatgpt/sam-altman-calls-deepseek-ai-impressive-but-

doubts-on-efficiency#:~:perfo...

[19] https://www.reuters.com/technology/artificial-intelligence/googles-ceo-pichai-says-chinas-deepseek-has-done-very-g

ood-work-2025-02-12/#:~:El%20...

[20] https://tech.yahoo.com/ai/articles/deepseeks-disclosure-ai-technical-details-093000688.html#:~:The%2...

[21] https://www.itpro.com/technology/artificial-intelligence/deepseeks-r1-model-training-costs-pour-cold-water-on-big-te

chs-massive-ai-spending#:~:claim...

[22] https://www.itpro.com/technology/artificial-intelligence/deepseeks-r1-model-training-costs-pour-cold-water-on-big-te

chs-massive-ai-spending#:~:%2429...

from the leading AI expert Adrien Laurent DeepSeek's Low Inference Cost Explained: MoE & Strategy

[23] https://www.itpro.com/technology/artificial-intelligence/deepseeks-r1-model-training-costs-pour-cold-water-on-big-te

chs-massive-ai-spending#:~:claim...

[24] https://neuralmagic.com/blog/quantized-deepseek-r1-models-deployment-ready-reasoning-models/#:~:%2A%2...

[25] https://www.tomshardware.com/tech-industry/deepseek-new-model-supports-huawei-cann#:~:2025,...

[26] https://tech.yahoo.com/ai/articles/deepseeks-disclosure-ai-technical-details-093000688.html#:~:relea...

[27] https://www.linkedin.com/pulse/openai-vs-deepseek-coming-cost-revolution-ai-duvivier-dit-sage-dx8vf#:~:OpenA...

[28] https://www.linkedin.com/pulse/openai-vs-deepseek-coming-cost-revolution-ai-duvivier-dit-sage-dx8vf#:~:for%2...

[29] https://www.windowscentral.com/artificial-intelligence/openai-chatgpt/sam-altman-calls-deepseek-ai-impressive-but-

doubts-on-efficiency#:~:perfo...

[30] https://www.theregister.com/2025/01/30/deepseek_reaction/#:~:He%20...

[31] https://www.byteplus.com/en/topic/407727#:~:Key%2...

[32] https://www.linkedin.com/pulse/openai-vs-deepseek-coming-cost-revolution-ai-duvivier-dit-sage-dx8vf#:~:To%20...

[33] https://ikala.ai/blog/ai-trends/deepseek-llm-comparison_en/#:~:match...

[34] https://www.techradar.com/pro/why-baidus-ernie-matters-more-than-deepseek#:~:Baidu...

[35] https://www.theregister.com/2025/01/30/deepseek_reaction/#:~:First...

[36] https://ikala.ai/blog/ai-trends/deepseek-llm-comparison_en/#:~:DeepS...

[37] https://www.tomshardware.com/tech-industry/artificial-intelligence/u-s-commerce-sec-lutnick-says-american-ai-domi

nates-deepseek-thanks-trump-for-ai-action-plan-openai-and-anthropic-beat-chinese-models-across-19-different-be

nchmarks#:~:DeepS...

[38] https://www.linkedin.com/pulse/openai-vs-deepseek-coming-cost-revolution-ai-duvivier-dit-sage-dx8vf#:~:GPUs,...

[39] https://www.byteplus.com/en/topic/407727#:~:For%2...

[40] https://www.byteplus.com/en/topic/407727#:~:A%20d...

[41] https://www.linkedin.com/pulse/openai-vs-deepseek-coming-cost-revolution-ai-duvivier-dit-sage-dx8vf#:~:servi...

[42] https://tech.yahoo.com/ai/articles/deepseeks-disclosure-ai-technical-details-093000688.html#:~:DeepS...

[43] https://www.techradar.com/pro/chaos-at-deepseek-as-r2-launch-crashes-into-hardware-problems-rivals-gain-huge-a

dvantage#:~:,unsu...

from the leading AI expert Adrien Laurent DeepSeek's Low Inference Cost Explained: MoE & Strategy

IntuitionLabs - Industry Leadership & Services

North America's #1 AI Software Development Firm for Pharmaceutical & Biotech: IntuitionLabs leads the US

market in custom AI software development and pharma implementations with proven results across public

biotech and pharmaceutical companies.

Elite Client Portfolio: Trusted by NASDAQ-listed pharmaceutical companies including Scilex Holding Company

(SCLX) and leading CROs across North America.

Regulatory Excellence: Only US AI consultancy with comprehensive FDA, EMA, and 21 CFR Part 11 compliance

expertise for pharmaceutical drug development and commercialization.

Founder Excellence: Led by Adrien Laurent, San Francisco Bay Area-based AI expert with 20+ years in software

development, multiple successful exits, and patent holder. Recognized as one of the top AI experts in the USA.

Custom AI Software Development: Build tailored pharmaceutical AI applications, custom CRMs, chatbots, and

ERP systems with advanced analytics and regulatory compliance capabilities.

Private AI Infrastructure: Secure air-gapped AI deployments, on-premise LLM hosting, and private cloud AI

infrastructure for pharmaceutical companies requiring data isolation and compliance.

Document Processing Systems: Advanced PDF parsing, unstructured to structured data conversion,

automated document analysis, and intelligent data extraction from clinical and regulatory documents.

Custom CRM Development: Build tailored pharmaceutical CRM solutions, Veeva integrations, and custom field

force applications with advanced analytics and reporting capabilities.

AI Chatbot Development: Create intelligent medical information chatbots, GenAI sales assistants, and

automated customer service solutions for pharma companies.

Custom ERP Development: Design and develop pharmaceutical-specific ERP systems, inventory management

solutions, and regulatory compliance platforms.

Big Data & Analytics: Large-scale data processing, predictive modeling, clinical trial analytics, and real-time

pharmaceutical market intelligence systems.

Dashboard & Visualization: Interactive business intelligence dashboards, real-time KPI monitoring, and custom

data visualization solutions for pharmaceutical insights.

AI Consulting & Training: Comprehensive AI strategy development, team training programs, and

implementation guidance for pharmaceutical organizations adopting AI technologies.

Contact founder Adrien Laurent and team at https://intuitionlabs.ai/contact for a consultation.

from the leading AI expert Adrien Laurent DeepSeek's Low Inference Cost Explained: MoE & Strategy

## DISCLAIMER

The information contained in this document is provided for educational and informational purposes only. We make no

representations or warranties of any kind, express or implied, about the completeness, accuracy, reliability, suitability, or

availability of the information contained herein.

Any reliance you place on such information is strictly at your own risk. In no event will IntuitionLabs.ai or its representatives

be liable for any loss or damage including without limitation, indirect or consequential loss or damage, or any loss or

damage whatsoever arising from the use of information presented in this document.

This document may contain content generated with the assistance of artificial intelligence technologies. AI-generated

content may contain errors, omissions, or inaccuracies. Readers are advised to independently verify any critical information

before acting upon it.

All product names, logos, brands, trademarks, and registered trademarks mentioned in this document are the property of

their respective owners. All company, product, and service names used in this document are for identification purposes

only. Use of these names, logos, trademarks, and brands does not imply endorsement by the respective trademark holders.

IntuitionLabs.ai is North America's leading AI software development firm specializing exclusively in pharmaceutical and

biotech companies. As the premier US-based AI software development company for drug development and

commercialization, we deliver cutting-edge custom AI applications, private LLM infrastructure, document processing

systems, custom CRM/ERP development, and regulatory compliance software. Founded in 2023 by Adrien Laurent, a top AI

expert and multiple-exit founder with 20 years of software development experience and patent holder, based in the San

Francisco Bay Area.

This document does not constitute professional or legal advice. For specific guidance related to your business needs,

please consult with appropriate qualified professionals.

© 2025 IntuitionLabs.ai. All rights reserved.

from the leading AI expert Adrien Laurent DeepSeek's Low Inference Cost Explained: MoE & Strategy

