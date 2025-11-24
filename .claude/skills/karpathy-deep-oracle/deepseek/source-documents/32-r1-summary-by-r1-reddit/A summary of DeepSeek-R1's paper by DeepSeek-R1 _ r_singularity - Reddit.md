---
sourceFile: "A summary of DeepSeek-R1's paper by DeepSeek-R1 : r/singularity - Reddit"
exportedBy: "Kortex"
exportDate: "2025-10-29T02:37:19.232Z"
---

# A summary of DeepSeek-R1's paper by DeepSeek-R1 : r/singularity - Reddit

e9156f62-6ed6-4e22-ae21-7dcd4316e4d1

A summary of DeepSeek-R1's paper by DeepSeek-R1 : r/singularity - Reddit

844486f3-22f1-4a60-b33c-be1d3fa0ce97

https://www.reddit.com/r/singularity/comments/1i5uc3t/a_summary_of_deepseekr1s_paper_by_deepseekr1/

## Skip to main content

## Open navigation

## Go to Reddit Home

https://www.reddit.com/login/

## Log in to Reddit   Open settings menu

Log In / Sign Up

https://www.reddit.com/login/

## Advertise on Reddit

https://www.reddit.com/login/

## Cookie Preferences

## Try Reddit Pro   BETA

https://www.reddit.com/login/

Reddit, Inc. © 2025. All rights reserved.

https://www.reddit.com/login/

## Copy link

## Copy link

## Go to singularity

https://www.reddit.com/login/

r/singularity

https://www.reddit.com/login/

r/singularity

https://www.reddit.com/login/

Everything pertaining to the technological singularity and related topics, e.g. AI, human enhancement, etc.

Members   •

pigeon57434

https://www.reddit.com/user/pigeon57434/

https://www.reddit.com/user/pigeon57434/

https://www.reddit.com/user/pigeon57434/

https://www.reddit.com/user/pigeon57434/

A summary of DeepSeek-R1's paper by DeepSeek-R1

https://www.reddit.com/user/pigeon57434/

Aha moments emerged naturally in RL: Self-correction behaviors like "Wait, let’s reevaluate..." arose without SFT.

Cold-start SFT fixed readability: ~1k structured examples resolved language mixing.

GRPO cut RL costs by 30%: Group-wise reward normalization outperformed PPO.

RL increased CoT length autonomously: Reasoning steps grew from 100→1k tokens without penalties.

Distillation beat direct RL in small models: SFT on R1 data outperformed RL-trained base models.

Process rewards failed; outcome rewards worked better: Rule-based final-answer checks stabilized training.

XML tags reduced hallucinations 15%: Structured <think>/<answer> improved reward clarity.

Language mixing fixed via consistency rewards: Penalized code-switching in multilingual outputs.

I find it funny that ive seen multiple AI youtubers explain papers and they just go to another AI to help them in the video but hey it does a good job

https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf

https://www.reddit.com/user/pigeon57434/

Archived post. New comments cannot be posted and votes cannot be cast.

https://www.reddit.com/r/singularity/comments/1i5uc3t/a_summary_of_deepseekr1s_paper_by_deepseekr1/

Sort by:  Best   Open comment sort options   Best

## Controversial

pigeon57434

https://www.reddit.com/r/singularity/comments/1i5uc3t/a_summary_of_deepseekr1s_paper_by_deepseekr1/

https://www.reddit.com/r/singularity/comments/1i5uc3t/comment/m86stzp/

•    Edited  9mo ago
DeepSeek-R1 is roughly 4.41x cheaper than o1 which means for the same price as a singular o1 query give or take you could run a consensus voting tree-of-agents system with 4 separate instances of R1 which could outperform o1 for the same price if not still cheaper

ohHesRightAgain

https://www.reddit.com/r/singularity/comments/1i5uc3t/comment/m86stzp/

https://www.reddit.com/r/singularity/comments/1i5uc3t/comment/m876rt6/

I think it was 30x cheaper

_thispageleftblank

https://www.reddit.com/r/singularity/comments/1i5uc3t/comment/m876rt6/

https://www.reddit.com/r/singularity/comments/1i5uc3t/comment/m8a6x75/

I looked up the API costs of the two models and here's the formula for the cost ratio. I'm assuming the same inputs and outputs, but potential differences in CoT lengths.

""" OpenAI o1 - 1m input tokens: $15 - 1m output tokens: $60 DeepSeek R1 - 1m input tokens: $0.55 - 1m output tokens: $2.19 """ def cost_r1(n_input_tokens, n_cot_tokens_r1, n_answer_tokens): return 0.55 * (n_input_tokens / 1000000) + 2.19 * ((n_cot_tokens_r1 + n_answer_tokens) / 1000000) def cost_o1(n_input_tokens, n_cot_tokens_o1, n_answer_tokens): return 15 * (n_input_tokens / 1000000) + 60 * ((n_cot_tokens_o1 + n_answer_tokens) / 1000000) def cost_ratio(n_input_tokens, n_cot_tokens_r1, n_cot_tokens_o1, n_answer_tokens): return cost_r1(n_input_tokens, n_cot_tokens_r1, n_answer_tokens) / cost_o1(n_input_tokens, n_cot_tokens_o1, n_answer_tokens) # same number of CoT tokens print(cost_ratio(300, 3000, 3000, 500)) # 0.0365 # r1 has twice as many CoT tokens print(cost_ratio(300, 3000, 1500, 500)) # 0.0629

_thispageleftblank

https://www.reddit.com/user/_thispageleftblank/

https://www.reddit.com/r/singularity/comments/1i5uc3t/comment/m8a1q26/

This depends on the ratio of R1 and o1 CoT lengths. 1m tokens of R1 output cost about $2. 1m tokens of o1 output cost about $60. If o1‘s CoT is only half as long, the cost difference is 15x in favor of R1. If it‘s the other way round, it would be 60x.

pigeon57434

https://www.reddit.com/r/singularity/comments/1i5uc3t/comment/m8a1q26/

https://www.reddit.com/r/singularity/comments/1i5uc3t/comment/m8a3dos/

their chains of thought seem to be similar size that it wont make a negligible difference

_thispageleftblank

https://www.reddit.com/r/singularity/comments/1i5uc3t/comment/m8a3dos/

https://www.reddit.com/r/singularity/comments/1i5uc3t/comment/m8a43w9/

Do you have some experimental data to back this up? I've seen R1 perform way too many unnecessary sanity checks today. If o1 doesn't do that, it could easily have a CoT < 1/2 the size. But only experiments can tell.

pigeon57434

https://www.reddit.com/r/singularity/comments/1i5uc3t/comment/m8a43w9/

https://www.reddit.com/r/singularity/comments/1i5uc3t/comment/m8asxuc/

here i just did an official calculation detailed here including exact sources and calculations

https://www.reddit.com/r/LocalLLaMA/comments/1i6axmv/i_calculated_the_effective_cost_of_r1_vs_o1_and/

https://www.reddit.com/r/singularity/comments/1i5uc3t/comment/m8asxuc/

more reply

https://www.reddit.com/r/singularity/comments/1i5uc3t/comment/m8asxuc/

## More replies

https://www.reddit.com/r/singularity/comments/1i5uc3t/comment/m8asxuc/

## More replies

https://www.reddit.com/r/singularity/comments/1i5uc3t/comment/m8asxuc/

## More replies

https://www.reddit.com/r/singularity/comments/1i5uc3t/comment/m8asxuc/

## More replies

https://www.reddit.com/r/singularity/comments/1i5uc3t/comment/m8asxuc/

## More replies

https://www.reddit.com/r/singularity/comments/1i5uc3t/comment/m8asxuc/

Immediate_Simple_217

https://www.reddit.com/r/singularity/comments/1i5uc3t/comment/m8asxuc/

https://www.reddit.com/r/singularity/comments/1i5uc3t/comment/m86r1fj/

The new benchmarks for R1 zero are INSANE!!!

Immediate_Simple_217

https://www.reddit.com/r/singularity/comments/1i5uc3t/comment/m86r1fj/

https://www.reddit.com/r/singularity/comments/1i5uc3t/comment/m86r2u8/

pigeon57434

https://www.reddit.com/r/singularity/comments/1i5uc3t/comment/m86r2u8/

https://www.reddit.com/r/singularity/comments/1i5uc3t/comment/m86rtzb/

its not r1-zero its just R1 Zero is the one only trained on sft normal R1 is even better

WearOk4875

https://www.reddit.com/r/singularity/comments/1i5uc3t/comment/m86rtzb/

https://www.reddit.com/r/singularity/comments/1i5uc3t/comment/m9c3f20/

I'm relatively new to AI and just learning. Everyone talks about "benchmarks". How are the benchmarks maintained? What is the governance around them and who keeps them updated?

## More replies

https://www.reddit.com/r/singularity/comments/1i5uc3t/comment/m9c3f20/

## More replies

https://www.reddit.com/r/singularity/comments/1i5uc3t/comment/m9c3f20/

## More posts you may like

DeepSeek R1 Research Paper Audio Summary

https://www.reddit.com/r/singularity/comments/1i5uc3t/comment/m9c3f20/

r/singularity

https://www.reddit.com/r/singularity/comments/1i5uc3t/comment/m9c3f20/

r/singularity

https://www.reddit.com/r/singularity/

Everything pertaining to the technological singularity and related topics, e.g. AI, human enhancement, etc.

DeepSeek R1 Research Paper Audio Summary

upvotes  ·    comments

So what happened with Deepseek R2?

https://www.reddit.com/r/singularity/comments/1iadqq3/deepseek_r1_research_paper_audio_summary/

r/singularity

https://www.reddit.com/r/singularity/comments/1iadqq3/deepseek_r1_research_paper_audio_summary/

r/singularity

https://www.reddit.com/r/singularity/

Everything pertaining to the technological singularity and related topics, e.g. AI, human enhancement, etc.

So what happened with Deepseek R2?

upvotes  ·    comments

DeepSeek-R1-0528

https://www.reddit.com/r/singularity/comments/1kpidw9/so_what_happened_with_deepseek_r2/

r/singularity

https://www.reddit.com/r/singularity/comments/1kpidw9/so_what_happened_with_deepseek_r2/

r/singularity

https://www.reddit.com/r/singularity/

Everything pertaining to the technological singularity and related topics, e.g. AI, human enhancement, etc.

DeepSeek-R1-0528

upvotes  ·    comments

[OC] Authors of DeepSeek paper, past research collaborators and their affiliations

https://www.reddit.com/r/singularity/comments/1kxnsv4/deepseekr10528/

r/dataisbeautiful

https://www.reddit.com/r/singularity/comments/1kxnsv4/deepseekr10528/

r/dataisbeautiful

https://www.reddit.com/r/dataisbeautiful/

DataIsBeautiful is for visualizations that effectively convey information. Aesthetics are an important part of information visualization, but pretty pictures are not the sole aim of this subreddit.

[OC] Authors of DeepSeek paper, past research collaborators and their affiliations

upvotes  ·    comments

It just happened! DeepSeek-R1 is here!

https://www.reddit.com/r/dataisbeautiful/comments/1icuolp/oc_authors_of_deepseek_paper_past_research/

r/singularity

https://www.reddit.com/r/dataisbeautiful/comments/1icuolp/oc_authors_of_deepseek_paper_past_research/

r/singularity

https://www.reddit.com/r/singularity/

Everything pertaining to the technological singularity and related topics, e.g. AI, human enhancement, etc.

It just happened! DeepSeek-R1 is here!

x    upvotes  ·    comments

DeepSeek AI - Researchers, the past co-authors, and their affiliations

https://www.reddit.com/r/singularity/comments/1i5pqi6/it_just_happened_deepseekr1_is_here/

r/Infographics

https://www.reddit.com/r/singularity/comments/1i5pqi6/it_just_happened_deepseekr1_is_here/

r/Infographics

https://www.reddit.com/r/Infographics/

DeepSeek AI - Researchers, the past co-authors, and their affiliations

upvotes  ·    comments

Deep Seek OCR Condenses Charts and Code and Reduces Tokens Per Image by 20X

https://www.reddit.com/r/Infographics/comments/1iculw2/deepseek_ai_researchers_the_past_coauthors_and/

https://www.reddit.com/r/Infographics/comments/1iculw2/deepseek_ai_researchers_the_past_coauthors_and/

https://www.reddit.com/r/agi/

Artificial general intelligence (AGI) is the intelligence of a machine that could successfully perform any intellectual task that a human being can. It is a primary goal of artificial intelligence research and an important topic for science fiction writers and futurists. Artificial general intelligence is also referred to as "strong AI", "full AI" or as the ability of a machine to perform "general intelligent action". /r/neuralnetworks /r/artificial /r/machinelearning /r/OpenCog /r/causality

Deep Seek OCR Condenses Charts and Code and Reduces Tokens Per Image by 20X

nextbigfuture    upvotes  ·    comment

No Hype DeepSeek-R1 [R]eading List

https://www.reddit.com/r/agi/comments/1ogmjq2/deep_seek_ocr_condenses_charts_and_code_and/

r/MachineLearning

https://www.reddit.com/r/agi/comments/1ogmjq2/deep_seek_ocr_condenses_charts_and_code_and/

r/MachineLearning

https://www.reddit.com/r/MachineLearning/

Beginners -> /r/mlquestions or /r/learnmachinelearning , AGI -> /r/singularity, career advices -> /r/cscareerquestions, datasets -> r/datasets

No Hype DeepSeek-R1 [R]eading List

upvotes  ·    comments

DeepSeek R1 just got a 2x speed boost. The crazy part? The code for the boost was written by R1 itself. Self-improving AI is here.

https://www.reddit.com/r/MachineLearning/comments/1ideupn/no_hype_deepseekr1_reading_list/

r/artificial

https://www.reddit.com/r/MachineLearning/comments/1ideupn/no_hype_deepseekr1_reading_list/

r/artificial

https://www.reddit.com/r/artificial/

Reddit’s home for Artificial Intelligence (AI)

DeepSeek R1 just got a 2x speed boost. The crazy part? The code for the boost was written by R1 itself. Self-improving AI is here.

upvotes  ·    comments

Academic Paper using DeepSeek R1

https://www.reddit.com/r/artificial/comments/1id3pht/deepseek_r1_just_got_a_2x_speed_boost_the_crazy/

r/singularity

https://www.reddit.com/r/artificial/comments/1id3pht/deepseek_r1_just_got_a_2x_speed_boost_the_crazy/

r/singularity

https://www.reddit.com/r/singularity/

Everything pertaining to the technological singularity and related topics, e.g. AI, human enhancement, etc.

Academic Paper using DeepSeek R1

upvotes  ·    comments

[D] How exactly did Deepseek R1 achieve massive training cost reductions, most posts I read are about its performance, RL, chain of thought, etc, but it’s not clear how the cost of training of the model was brought down so drastically

https://www.reddit.com/r/singularity/comments/1i8c6le/academic_paper_using_deepseek_r1/

r/MachineLearning

https://www.reddit.com/r/singularity/comments/1i8c6le/academic_paper_using_deepseek_r1/

r/MachineLearning

https://www.reddit.com/r/MachineLearning/

Beginners -> /r/mlquestions or /r/learnmachinelearning , AGI -> /r/singularity, career advices -> /r/cscareerquestions, datasets -> r/datasets

[D] How exactly did Deepseek R1 achieve massive training cost reductions, most posts I read are about its performance, RL, chain of thought, etc, but it’s not clear how the cost of training of the model was brought down so drastically

upvotes  ·    comments

deepseek r1's author list - they brought the whole squad

https://www.reddit.com/r/MachineLearning/comments/1ibijhg/d_how_exactly_did_deepseek_r1_achieve_massive/

r/artificial

https://www.reddit.com/r/MachineLearning/comments/1ibijhg/d_how_exactly_did_deepseek_r1_achieve_massive/

r/artificial

https://www.reddit.com/r/artificial/

Reddit’s home for Artificial Intelligence (AI)

deepseek r1's author list - they brought the whole squad

upvotes  ·    comments

You can now run DeepSeek-R1-0528 on your local device! (20GB RAM min.)

https://www.reddit.com/r/artificial/comments/1i9ljdy/deepseek_r1s_author_list_they_brought_the_whole/

r/singularity

https://www.reddit.com/r/artificial/comments/1i9ljdy/deepseek_r1s_author_list_they_brought_the_whole/

r/singularity

https://www.reddit.com/r/singularity/

Everything pertaining to the technological singularity and related topics, e.g. AI, human enhancement, etc.

You can now run DeepSeek-R1-0528 on your local device! (20GB RAM min.)

upvotes  ·    comments

Deepseek 3.1 benchmarks released

https://www.reddit.com/r/singularity/comments/1kz6qku/you_can_now_run_deepseekr10528_on_your_local/

r/singularity

https://www.reddit.com/r/singularity/comments/1kz6qku/you_can_now_run_deepseekr10528_on_your_local/

r/singularity

https://www.reddit.com/r/singularity/

Everything pertaining to the technological singularity and related topics, e.g. AI, human enhancement, etc.

Deepseek 3.1 benchmarks released

3    upvotes  ·    comments

[R] Learn How to Run DeepSeek-R1 Locally, a Free Alternative to OpenAI’s $200/Month o1 model

https://www.reddit.com/r/singularity/comments/1mw3jha/deepseek_31_benchmarks_released/

r/MachineLearning

https://www.reddit.com/r/singularity/comments/1mw3jha/deepseek_31_benchmarks_released/

r/MachineLearning

https://www.reddit.com/r/MachineLearning/

Beginners -> /r/mlquestions or /r/learnmachinelearning , AGI -> /r/singularity, career advices -> /r/cscareerquestions, datasets -> r/datasets

[R] Learn How to Run DeepSeek-R1 Locally, a Free Alternative to OpenAI’s $200/Month o1 model

upvotes  ·    comments

You can now run DeepSeek-R1 on your own local device!

https://www.reddit.com/r/MachineLearning/comments/1i9xwbr/r_learn_how_to_run_deepseekr1_locally_a_free/

r/singularity

https://www.reddit.com/r/MachineLearning/comments/1i9xwbr/r_learn_how_to_run_deepseekr1_locally_a_free/

r/singularity

https://www.reddit.com/r/singularity/

Everything pertaining to the technological singularity and related topics, e.g. AI, human enhancement, etc.

You can now run DeepSeek-R1 on your own local device!

upvotes  ·    comments

DeepSeek-R1 Scored 100% on a 2023 A Levels Mathematics (Advanced PAPER 1: Pure Mathematics 1)

https://www.reddit.com/r/singularity/comments/1ic9x8z/you_can_now_run_deepseekr1_on_your_own_local/

r/singularity

https://www.reddit.com/r/singularity/comments/1ic9x8z/you_can_now_run_deepseekr1_on_your_own_local/

r/singularity

https://www.reddit.com/r/singularity/

Everything pertaining to the technological singularity and related topics, e.g. AI, human enhancement, etc.

DeepSeek-R1 Scored 100% on a 2023 A Levels Mathematics (Advanced PAPER 1: Pure Mathematics 1)

upvotes  ·    comments

DeepSeek’s efficiency has major implications for the industry

https://www.reddit.com/r/singularity/comments/1i5r85h/deepseekr1_scored_100_on_a_2023_a_levels/

r/singularity

https://www.reddit.com/r/singularity/comments/1i5r85h/deepseekr1_scored_100_on_a_2023_a_levels/

r/singularity

https://www.reddit.com/r/singularity/

Everything pertaining to the technological singularity and related topics, e.g. AI, human enhancement, etc.

DeepSeek’s efficiency has major implications for the industry

upvotes  ·    comments

DeepSeek R2 delayed

https://www.reddit.com/r/singularity/comments/1ia63px/deepseeks_efficiency_has_major_implications_for/

r/singularity

https://www.reddit.com/r/singularity/comments/1ia63px/deepseeks_efficiency_has_major_implications_for/

r/singularity

https://www.reddit.com/r/singularity/

Everything pertaining to the technological singularity and related topics, e.g. AI, human enhancement, etc.

DeepSeek R2 delayed

upvotes  ·    comments

You can now train your own DeepSeek-R1 model on your local device!

https://www.reddit.com/r/singularity/comments/1ll6j7g/deepseek_r2_delayed/

r/singularity

https://www.reddit.com/r/singularity/comments/1ll6j7g/deepseek_r2_delayed/

r/singularity

https://www.reddit.com/r/singularity/

Everything pertaining to the technological singularity and related topics, e.g. AI, human enhancement, etc.

You can now train your own DeepSeek-R1 model on your local device!

upvotes  ·    comments

In rare disclosure, DeepSeek claims R1 model training cost just $294K

https://www.reddit.com/r/singularity/comments/1ik2zf6/you_can_now_train_your_own_deepseekr1_model_on/

r/technology

https://www.reddit.com/r/singularity/comments/1ik2zf6/you_can_now_train_your_own_deepseekr1_model_on/

r/technology

https://www.reddit.com/r/technology/

Subreddit dedicated to the news and discussions about the creation and use of technology and its surrounding issues.

In rare disclosure, DeepSeek claims R1 model training cost just $294K

techspot    upvotes  ·    comments

DeepSeek delays new AI model amid Huawei chip issues- FT

https://www.reddit.com/r/technology/comments/1nms4ms/in_rare_disclosure_deepseek_claims_r1_model/

r/singularity

https://www.reddit.com/r/technology/comments/1nms4ms/in_rare_disclosure_deepseek_claims_r1_model/

r/singularity

https://www.reddit.com/r/singularity/

Everything pertaining to the technological singularity and related topics, e.g. AI, human enhancement, etc.

DeepSeek delays new AI model amid Huawei chip issues- FT

upvotes  ·    comments

DeepSeek-R1: How Did They Make an OpenAI-Level Reasoning Model So Damn Efficient?

https://www.reddit.com/r/singularity/comments/1mptk91/deepseek_delays_new_ai_model_amid_huawei_chip/

r/singularity

https://www.reddit.com/r/singularity/comments/1mptk91/deepseek_delays_new_ai_model_amid_huawei_chip/

r/singularity

https://www.reddit.com/r/singularity/

Everything pertaining to the technological singularity and related topics, e.g. AI, human enhancement, etc.

DeepSeek-R1: How Did They Make an OpenAI-Level Reasoning Model So Damn Efficient?

upvotes  ·    comments

DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning

https://www.reddit.com/r/singularity/comments/1i9lkbh/deepseekr1_how_did_they_make_an_openailevel/

r/singularity

https://www.reddit.com/r/singularity/comments/1i9lkbh/deepseekr1_how_did_they_make_an_openailevel/

r/singularity

https://www.reddit.com/r/singularity/

Everything pertaining to the technological singularity and related topics, e.g. AI, human enhancement, etc.

DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning

nature    upvotes  ·    comments

DeepSeek-R1-Lite-Preview seems to beat DeepSeek V3 on multiple benchmarks, so why is V3 getting so much more hype?

https://www.reddit.com/r/singularity/comments/1nk43b1/deepseekr1_incentivizes_reasoning_in_llms_through/

r/LocalLLaMA

https://www.reddit.com/r/singularity/comments/1nk43b1/deepseekr1_incentivizes_reasoning_in_llms_through/

r/LocalLLaMA

https://www.reddit.com/r/LocalLLaMA/

Subreddit to discuss AI & Llama, the large language model created by Meta AI.

DeepSeek-R1-Lite-Preview seems to beat DeepSeek V3 on multiple benchmarks, so why is V3 getting so much more hype?

upvotes  ·    comments

## View Post in

https://www.reddit.com/r/LocalLLaMA/comments/1howm2w/deepseekr1litepreview_seems_to_beat_deepseek_v3/

Tiếng Việt

https://www.reddit.com/r/LocalLLaMA/comments/1howm2w/deepseekr1litepreview_seems_to_beat_deepseek_v3/

https://www.reddit.com/r/LocalLLaMA/comments/1howm2w/deepseekr1litepreview_seems_to_beat_deepseek_v3/

https://www.reddit.com/r/LocalLLaMA/comments/1howm2w/deepseekr1litepreview_seems_to_beat_deepseek_v3/

## Top Posts

Reddit  reReddit: Top posts of January 20, 2025

https://www.reddit.com/r/LocalLLaMA/comments/1howm2w/deepseekr1litepreview_seems_to_beat_deepseek_v3/

Reddit  reReddit: Top posts of January 2025

https://www.reddit.com/r/LocalLLaMA/comments/1howm2w/deepseekr1litepreview_seems_to_beat_deepseek_v3/

Reddit  reReddit: Top posts of 2025

https://www.reddit.com/r/LocalLLaMA/comments/1howm2w/deepseekr1litepreview_seems_to_beat_deepseek_v3/

## Reddit Rules

https://www.reddit.com/r/LocalLLaMA/comments/1howm2w/deepseekr1litepreview_seems_to_beat_deepseek_v3/

## Privacy Policy

https://www.reddit.com/r/LocalLLaMA/comments/1howm2w/deepseekr1litepreview_seems_to_beat_deepseek_v3/

## User Agreement

https://www.reddit.com/r/LocalLLaMA/comments/1howm2w/deepseekr1litepreview_seems_to_beat_deepseek_v3/

## Accessibility

https://www.reddit.com/r/LocalLLaMA/comments/1howm2w/deepseekr1litepreview_seems_to_beat_deepseek_v3/

Reddit, Inc. © 2025. All rights reserved.

https://www.reddit.com/r/LocalLLaMA/comments/1howm2w/deepseekr1litepreview_seems_to_beat_deepseek_v3/

Log In / Sign Up

https://www.reddit.com/r/LocalLLaMA/comments/1howm2w/deepseekr1litepreview_seems_to_beat_deepseek_v3/

## Advertise on Reddit

https://www.reddit.com/r/LocalLLaMA/comments/1howm2w/deepseekr1litepreview_seems_to_beat_deepseek_v3/

## Cookie Preferences

## Try Reddit Pro   BETA

https://www.reddit.com/r/LocalLLaMA/comments/1howm2w/deepseekr1litepreview_seems_to_beat_deepseek_v3/

