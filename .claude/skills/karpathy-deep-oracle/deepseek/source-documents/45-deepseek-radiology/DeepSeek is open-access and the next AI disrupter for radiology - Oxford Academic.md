---
sourceFile: "DeepSeek is open-access and the next AI disrupter for radiology - Oxford Academic"
exportedBy: "Kortex"
exportDate: "2025-10-29T02:37:26.188Z"
---

# DeepSeek is open-access and the next AI disrupter for radiology - Oxford Academic

08b4f38c-30c9-4991-9795-903c18cab027

DeepSeek is open-access and the next AI disrupter for radiology - Oxford Academic

03d66e87-6ffc-4006-9491-ce5bf3db483e

https://academic.oup.com/radadv/article-pdf/2/1/umaf009/61945067/umaf009.pdf

https://lh3.googleusercontent.com/notebooklm/AG60hOrAmE1gPbcxCPpERoNex8zugyvLUlZE3rYGQ-baGKJoFhdPjUgdcO1oAVA1L56QQSwp-HAmU3Wl8xh0kKrgTr1lQvsQlswA0NW9TQTkivdy6-AzbGR6ngrbRuRQSXdFQeDKqx8vZA=w411-h208-v0

2c6abb1f-47ec-419b-86f8-eced2b7d60d7

DeepSeek is open-access and the next AI disrupter for radiology Yifan Peng , PhD�,1, Qingyu Chen, PhD2, George Shih, MD3

1Department of Population Health Sciences, Weill Cornell Medicine, New York, NY 10022, United States 2Department of Biomedical Informatics and Data Science, Yale School of Medicine, New Haven, CT 06510, United States 3Department of Radiology, Weill Cornell Medicine, New York, NY 10065, United States �Corresponding author: Yifan Peng, PhD, Department of Population Health Sciences, Weill Cornell Medicine, 575 Lexington Ave RM 651, New York, NY 10022, United States (yip4002@med.cornell.edu).

Keywords: DeepSeek, open-weight, radiology

Introduction Recent years have seen the rapid integration of artificial intel-ligence (AI) into the field of radiology. The most recent po-tential one is DeepSeek, which is redefining the benchmarks for open-weight, computational efficiency, and problem- solving capabilities.1 The emergence of DeepSeek has ignited enthusiasm as well as consternation, causing stock market volatility with its economic implications.2 More importantly, its technical capabilities have generated extensive discussions in the engineering and biomedical domains.3 Here, we briefly discuss what sets DeepSeek apart and why it could be the next tool in the AI transformation of radiology.

Innovations brought by DeepSeek DeepSeek is an AI startup based in Hangzhou, China. It is known for releasing various models, including DeepSeek-V3 and DeepSeek-R1. DeepSeek-V3 is targeted to compete di-rectly with OpenAI’s GPT-4o, whereas DeepSeek-R1 is posi-tioned against OpenAI's o1, as shown in Table 1.

One notable distinction of DeepSeek models is their open na-ture, which includes the models and weights, although not the training data and training code.1,4 It allows everyone to examine their training processes in detail. From a technological stand-point, DeepSeek-R1 employed the multistage training approach. It began with a “cold start” phase, focusing on fine-tuning with a small set of carefully crafted examples to enhance clarity and readability. Subsequently, the models underwent additional rein-forcement learning and refinement steps, including rejection of low-quality outputs based on human preference and verifiable rewards. This resulted in models that reason effectively while de-livering polished and consistent answers.

To reduce the training cost (only $5.576 M1) the DeepSeek team implemented FP8 training4 and Mixture of Experts. FP8 is a progression from 16-bit data formats.5

Here, DeepSeek models used a mixed precision framework for training, where most compute-density operations are con-ducted in FP8. In contrast, a select few key operations are strategically retained in their original data formats to balance training efficiency and numerical stability. The Mixture of Experts technique utilizes several expert networks to divide the problem space into homogeneous regions, optimizing

problem-solving. Consequently, DeepSeek requires less com-puting power than previous models. It has been reported that a single server equipped with eight H200 GPUs can effec-tively run the full version of DeepSeek-R1.6

In addition, DeepSeek introduces Multi-head Latent Attention, which reduces overhead by transforming the key- value cache required by standard Multi-head Attention into a latent vector. This approach significantly improves inference efficiency and scales well across small and large Mixture of Experts models.

How could radiology benefit from DeepSeek? One notable feature of DeepSeek-R1 is “chain-of-thought” reasoning (CoT). This technique guides large language mod-els (LLMs) to follow a reasoning process when dealing with complex problems. Although CoT can be applied with other LLMs, they often require users to carefully construct CoT prompts (eg, explain the reasoning), posing a barrier to lay end users.

DeepSeek uses large-scale reinforcement learning, reward modeling, and distillation to enhance reasoning performance. When we chatted with DeepSeek-R1, we did not immediately get a response. The model first uses CoT reasoning to con-sider the problem. Only once it finishes thinking does it start outputting the answer. We also observed that DeepSeek-R1 is capable of self-reflection, criticism, and correction. This has potential for radiology applications, where synthesizing dif-ferent data modalities and effective reasoning are essential.

To assess the capabilities of DeepSeek and other LLMs, we challenged them with a multiple-choice disease classification problem designed to identify specific abnormal findings from radiology reports. In this setup, the input is a radiology re-port and a list of candidate answers representing potential di-agnoses (detailed in Supplementary Materials). Our case examples were crafted using synthetic data created by radiol-ogists, ensuring that no patient information was used.

These samples demonstrate that DeepSeek-R1 outper-forms Llama-3.3-70b; however, we did not find significant advantages over GPT-4o. The open-weight nature of DeepSeek-R1 and its implementation locally suggest that de-veloping reasonably sized LLM models is feasible without

Received: February 10, 2025; Accepted: February 12, 2025. © The Author(s) 2025. Published by Oxford University Press on behalf of the Radiological Society of North America.  This is an Open Access article distributed under the terms of the Creative Commons Attribution License (https://creativecommons.org/licenses/by/4.0/), which

https://doi.org/10.1093/radadv/umaf009 Advance access publication: February 18, 2025 Editorial

nloaded from  https://academ

ic.oup.com /radadv/article/2/1/um

af009/8020802 by guest on 29 O ctober 2025

sacrificing high functionality. This observation is expected and may reflect similar performance in other tasks.

It is important to note that our evaluation encompassed only a limited number of examples. A more extensive com-parison across a broader range of tasks could yield more pro-found insights into DeepSeek's true capabilities.

Future challenges Using DeepSeek to analyze radiology reports presents several challenges and ethical considerations. As with all deep learn-ing platforms, data privacy is a primary concern. Currently, several governments have banned DeepSeek from govern-ment devices (e.g., New York State,7 Texas8) because of per-ceived security risks. However, unlike the other LLMs, DeepSeek’s open-weight nature and low resource require-ments make it feasible for enterprises like large health care organizations to run the model locally for in-house AI train-ing and implementation. Consequently, DeepSeek-V3 and DeepSeek-R1 are expected to foster collaborative environ-ments and accelerate AI innovation. However, the release of DeepSeek-R1 raises important questions, particularly regard-ing the curation of datasets9 and the absence of training code, which are critical for transparency and responsible use.

In addition, while DeepSeek emphasizes reasoning, we have observed that its responses tend to be excessively ver-bose, with many details that can quickly overwhelm users tasked with reading, reviewing, or approving them. This raises an important question: Is the detailed reasoning in responses truly beneficial, and to what extent does it add value? This remains an open area of inquiry.

Third, LLMs, including DeepSeek, still require comprehensive evaluation across several dimensions, such as Quality of Information, Understanding and Reasoning, Expression Style and Persona, Safety and Harm, and Trust and Confidence.10 To date, no publicly reported studies specifically explore DeepSeek's capabilities in analyzing clinical text. This underscores the need

for further investigation to understand and validate its potential in these areas fully.

Conclusion DeepSeek has disrupted both the AI industry and academia and is expected to act as a catalyst for further development in the field. These advanced capabilities will definitely promote the successful integration of AI in radiology in the near future.

As DeepSeek and other LLMs, both proprietary and open, gain prominence, there is an increasing demand for a collabora-tive co-design approach to integrating LLMs into clinical domains. This approach should encompass diverse stakehold-ers, including technology developers, ethicists, radiology do-main experts, and end-users, to ensure these models effectively address real-world human needs and reflect societal values.

Acknowledgments The authors thank Yishu Wei and Zihan Xu for preparing the data.

Supplementary material Supplementary material is available at Radiology Advances online.

Funding This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

Conflicts of interest Please see ICMJE form(s) for author conflicts of interest. These have been provided as supplementary materials. All authors declare that they have no conflicts of interest.

Table 1. Comparisons between GPT-4o, Open AI o1, Llama 3.3, DeepSeek-v3, and DeepSeek-R1.

GPT-4o OpenAI o1 Llama 3.3 DeepSeek-V3 DeepSeek-R1

Date of Release May 13, 2024 December 5, 2024 December 6, 2024 December 26, 2024 January 20, 2025 Design Number of parameters Not disclosed Not disclosed 70B 671B 671B Input modality Text, Image, Audio Text, Image Text Text Text Training dData Not disclosed Not disclosed A new mix of publicly

available online data. Not disclosed Not disclosed

Training strategy Not disclosed Not disclosed SFT, RL SFT, RL, FP8 SFT, RL, Multistage training

Context window 128k 200k 128k 128k 128k GPU hours for training Not disclosed Not disclosed 7.0M 2.79M Not disclosed Evaluation MMLU 88.7 92.3 86 88.5 90.8 Cost Input $1.25/1M cached $7.50/1M cached Free Free; API: $0.014/

1M cached Free; API: $0.14/ 1M cached

Output $10.00/1M $60.00/1M Free Free; API: $0.28/1M Free; API: $2.19/1M Usage Availability Private (OpenAI) Private (OpenAI) Open Open Open Run locally? API API Yes Yes Yes License Proprietary Proprietary Meta Llama

## Community License Agreement

## MIT license MIT license

Abbreviations: API ¼ application programming interface, MMLU ¼massive multitask language understanding, SFT ¼ supervised fine-tuning, RL ¼ reinforcement learning, FP8 ¼ 8-bit floating point.

ow nloaded from

https://academ ic.oup.com

/radadv/article/2/1/um af009/8020802 by guest on 29 O

ctober 2025

References 01. DeepSeek AI, Guo D, Yang D, et al. DeepSeek-R1: incentivizing

reasoning capability in LLMs via reinforcement learning. arXiv [csCL]. Published online January 22, 2025. http://arxiv.org/abs/ 2501.12948, preprint: not peer reviewed.

######### 02. Ng K, Drenon B, Gerken T, Cieslak M. What is DeepSeek—and why is everyone talking about it? BBC. January 27, 2025. Accessed February 10, 2025. https://www.bbc.com/news/articles/c5yv5976z9po

########## 03. Wang G. DeepSeek has more to offer beyond efficiency: explain-able AI. Forbes. January 30, 2025. Accessed January 31, 2025. https://www.forbes.com/sites/geruiwang/2025/01/30/deepseek- redefines-ai-with-explainable-reasoning-and-open-innovation/

########## 04. DeepSeek AI, Liu A, Feng B, et al. DeepSeek-V3 technical report. arXiv [csCL]. Published online December 26, 2024. http://arxiv. org/abs/2412.19437, preprint: not peer reviewed.

########## 05. Peng H, Wu K, Wei Y, et al. FP8-LM: training FP8 large language models. arXiv [csLG]. Published online October 27, 2023. http:// arxiv.org/abs/2310.18313, preprint: not peer reviewed.

########## 06. Pounds E. DeepSeek-R1 now live with NVIDIA NIM. NVIDIA Blog. January 30, 2025. Accessed February 4, 2025. https://blogs. nvidia.com/blog/deepseek-r1-nim-microservice/

########## 07. Katersky A. DeepSeek banned from government devices in New York state. ABC News. February 10, 2025. Accessed February 10, 2025.https://abcnews.go.com/US/deepseek-banned-government- devices-new-york-state/story?id=118653885

######### 08. Lathan N. Texas governor orders ban on DeepSeek, RedNote for government devices. AP News. January 31, 2025. Accessed February 4, 2025. https://apnews.com/article/texas-deepseek- apps-ban-3828a4743e9919398dfac0ba9d4a5c25

########## 09. Collier K, Cui J. OpenAI says DeepSeek may have “inappropriately” used its data. NBC News. January 29, 2025. Accessed January 31, 2025. https://www.nbcnews.com/tech/tech-news/openai-says-deep seek-may-inapproriately-used-data-rcna189872

######### 10. Tam TYC, Sivarajkumar S, Kapoor S, et al. A framework for hu-man evaluation of large language models in healthcare derived from literature review. NPJ Digit Med. 2024;7(1):258. https://doi. org/10.1038/s41746-024-01258-7

© The Author(s) 2025. Published by Oxford University Press on behalf of the Radiological Society of North America. This is an Open Access article distributed under the terms of the Creative Commons Attribution License (https://creativecommons.org/licenses/by/4.0/), which permits unrestricted reuse, distribution, and reproduction in any medium, provided the original work is properly cited. Radiology Advances, 2025, 2, 1–3 https://doi.org/10.1093/radadv/umaf009 Editorial

nloaded from  https://academ

ic.oup.com /radadv/article/2/1/um

af009/8020802 by guest on 29 O ctober 2025

