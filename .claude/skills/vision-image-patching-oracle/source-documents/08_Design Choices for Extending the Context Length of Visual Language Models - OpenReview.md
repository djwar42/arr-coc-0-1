---
sourceFile: "Design Choices for Extending the Context Length of Visual Language Models - OpenReview"
exportedBy: "Kortex"
exportDate: "2025-10-28T18:42:18.076Z"
---

# Design Choices for Extending the Context Length of Visual Language Models - OpenReview

3c2110d6-859c-4872-917e-a74f64ef1d40

Design Choices for Extending the Context Length of Visual Language Models - OpenReview

51e1c193-11b4-4af3-b9d4-c05b38dbcceb

https://openreview.net/pdf/aa9e3e53fb762010568e6f123e9948dac8620a0f.pdf

## Design Choices for Extending the Context Length of Visual Language Models

## Anonymous ACL submission

Visual Language Models (VLMs) demonstrate001 impressive capabilities in processing multi-002 modal inputs, yet applications such as visual003 agents, which require handling multiple images004 and high-resolution videos, demand enhanced005 long-range modeling. Moreover, existing open-006 source VLMs lack systematic exploration into007 extending their context length, and commer-008 cial models often provide limited details. To009 tackle this, we aim to establish an effective so-010 lution that enhances long context performance011 of VLMs while preserving their capacities in012 short context scenarios. Towards this goal, we013 make the best design choice through exten-014 sive experiment settings from data curation to015 context window extending and utilizing: (1)016 we analyze data sources and length distribu-017 tions to construct ETVLM - a data recipe to018 balance the performance across scenarios; (2)019 we examine existing position extending meth-020 ods, identify their limitations and propose M-021 RoPE++ as an enhanced approach; we also022 choose to solely instruction-tune the backbone023 with mixed-source data; (3) we discuss how024 to better utilize extended context windows and025 propose hybrid-resolution training. Built on026 the Qwen-VL series model, we propose GI-027 RAFFE, which is effectively extended to 128K028 lengths. Evaluated on extensive long context029 VLM benchmarks such as VideoMME and030 Viusal Haystacks, our GIRAFFE achieves state-031 of-the-art performance among similarly sized032 open-source long VLMs and is competitive033 with commercial model GPT-4V.1034

1 Introduction035

Visual Language Models (VLMs) (OpenAI, 2023;036

Gemini Team, 2024) integrate visual and textual037

information, which are pivotal in understanding the038

multimodal world and excel in various applications,039

such as visual question answering and video under-040

standing (Liu et al., 2023c; Li et al., 2022). How-041

1We will open-source the code, data, and models.

ever, more advanced scenarios involve multi-image 042

and long video comprehension, which challenge 043

the long-range modeling capabilities of VLMs. For 044

instance, a 2K context length can only digest less 045

than a few frames (Liu et al., 2023c,b; Li et al., 046

2023a), limiting the upper bound of long video 047

understanding. Consequently, there is a pressing 048

need for methods to extend the context window of 049

VLMs and improve their performance in long con- 050

text scenarios. This would benefit next-generation 051

VLMs in performing long history visual agents or 052

serving as world models (Liu et al., 2024a). 053

Recent efforts for longer context VLMs focus on 054

extending base Large Language Models (LLMs), 055

along with visual alignment or efficient architec- 056

tures. LongVA (Zhang et al., 2024a) seeks to trans- 057

fer long context ability from language models to 058

vision by modifying position embeddings in the 059

LLM backbone (PI, Chen et al. 2023b; NTK, Lo- 060

calLLaMA 2023). LongVILA (Xue et al., 2024) 061

and LongLLaVA (Wang et al., 2024b) accommo- 062

date longer sequences using multi-stage alignment 063

and instruction tuning (Peng et al., 2023; Fu et al., 064

2024c) with additional infrastructure and architec- 065

ture. Despite these initial explorations, they have 066

not investigated the feasibility of directly extending 067

the context window of existing VLMs or system- 068

atically explored the design space in the extending 069

pipeline. To bridge this gap, we decompose the 070

challenge of extending context windows of existing 071

VLMs into three fundamental research questions: 072

(1) How to effectively organize and curate training 073

data? (2) How to efficiently train longer VLMs? 074

(3) How to leverage the extended context window? 075

In our work, our goal is to answer the three 076

research questions and find a solution in prac- 077

tice. To validate our design choices, we imple- 078

ment thorough experiments based on Qwen-VL 079

series model (Bai et al., 2023; Wang et al., 2024a) 080

and conduct comprehensive evaluations on single 081

image understanding, image interleave, and video 082

tasks (§2.1). For data curation, we prepare a diverse083

dataset comprising long context instruction data,084

multimodal instruction data, multimodal interleave085

data, and video instruction data (§2.2). We analyze086

the impact of different data compositions, ratios,087

and lengths on model performance (§2.3) and find088

that (1) short multimodal instruction data is crucial089

for both extending long context capability and re-090

taining short context performance; (2) a balanced091

data ratio contributes to balanced performance on092

downstream tasks. For the second research ques-093

tion on extending training, we examine the effec-094

tive context length of previous position embedding095

extending alternatives such as PI and NTK, discov-096

ering that, akin to LLM studies (Gao et al., 2024a;097

An et al., 2024b), the effective length is shorter than098

the training length (§3.1). We propose M-RoPE++099

(§3.2) to extend position embedding on spatial and100

temporal dimensions. Validation experiments re-101

veal that our method achieves better downstream102

task performance and longer effective length under103

the same training length (§3.2). Different from104

LongVA (Zhang et al., 2024a) that first extend105

LLM base or LongLLaVA (Wang et al., 2024b)106

and LongVILA (Xue et al., 2024) that adopt multi-107

stage training with visual alignment and instruction108

tuning, we find that directly training VLMs by only109

updating LLM backbone’s parameters achieves op-110

timal results (§3.3). To figure out how to use long111

context well in VLM, the third research question,112

we examine the trade-off between single-frame res-113

olution and frame numbers regarding task perfor-114

mance (§3.4). We consequently propose hybrid-115

resolution training, which further improves the uti-116

lization of a fixed context length (§3.5).117

Based on our findings from the three research118

questions, we carefully select data recipes and train-119

ing methods to extend Qwen-VL and Qwen2-VL120

to GIRAFFE-QwenVL and GIRAFFE with 128K121

length. Our final models are evaluated on both122

short context tasks such as single image understand-123

ing and long context tasks with multi-image and124

long videos. Experimental results demonstrate that125

our GIRAFFE achieves state-of-the-art performance126

among long VLMs and there is a significant im-127

provement for our GIRAFFE-QwenVL compared128

with Qwen-VL base (§4.2). Summarized contribu-129

### 1. We investigate different design choices to ex-131

tend the context window of existing VLMs to132

128K while maintaining comparable perfor-133

mance on short visual tasks. 134

### 2. Technically, M-RoPE++ and hybrid- 135

resolution training methods are newly 136

proposed by us to enhance model perfor- 137

mance during training and inference. 138

### 3. On existing long VLM benchmarks, GIRAFFE 139

achieves state-of-the-art performance among 140

similar scale open-sourced long VLMs and is 141

competitive to commercial models. 142

2 How to Curate Extending Data 143

Developing an effective recipe for extending the 144

context window of VLMs is crucial. To systemati- 145

cally evaluate such recipes, we construct a compre- 146

hensive metric suite encompassing single-image, 147

multi-image, and video tasks (§2.1), enabling a 148

thorough assessment of model performance across 149

diverse scenarios. This section focuses on the se- 150

lection and preprocessing of training data (§2.2), 151

with an emphasis on understanding how data com- 152

positions, ratios, and lengths influence the model’s 153

capabilities (§2.3). 154

2.1 Evaluation Tasks 155

We evaluate both long and short-context multi- 156

modal tasks, as it is essential for VLMs to sus- 157

tain performance on short-context tasks after ex- 158

tended training. For short-context evaluation, we 159

utilize widely adopted benchmarks such as single- 160

image MME (Fu et al., 2023) and MMBench (Liu 161

et al., 2024b), which capture the diverse capabili- 162

ties of VLMs. For multi-image tasks, we incorpo- 163

rate Mantis-Eval (Jiang et al., 2024), QBench (Wu 164

et al., 2024b), and BLINK (Fu et al., 2024b), in line 165

with LLaVA-Interleave (Li et al., 2024a). Given 166

the temporal nature of videos, which naturally rep- 167

resent long-context multimodal tasks, we evalu- 168

ate on LongVideoBench (Wu et al., 2024a) and 169

VideoMME (Fu et al., 2024a). Additionally, we 170

include the Visual Haystack Single Needle Chal- 171

lenge (Wu et al., 2024c), which requires locat- 172

ing specific visual information within a long se- 173

quence of images, providing a robust measure of 174

the model’s effective context length. 175

2.2 Extending Data Curation 176

To construct our extending training dataset, 177

ETVLM, we incorporate four primary types of 178

data with varying lengths: (i) Long-context in- 179

struction data, sourced primarily from LongAlign- 180

10K (Bai et al., 2024) and LongAlpaca (Chen et al., 181

https://lh3.googleusercontent.com/notebooklm/AG60hOqeK5Qb-0R1L2qb8QaYuYdRjD7tZKUAWYn9c69NU0rKztkfP1XNF6sBV48Zh9lh5k9h06XkMmjMg1v8WH6jp3XGbUP_x645qRzi9QYcwxXE11auUyiut4-gA16jjVuRPYQSZoDOGg=w133-h133-v0

bac072bd-4a1d-4607-a8d3-2d19d8a8ebe5

https://lh3.googleusercontent.com/notebooklm/AG60hOqvg9jTQSl9z2OR9mrpFhjRGhO0HWLdJ9aSmXvnFeqWabC9be2PbTamsckG4-CfOW6arRL5a8tQmKnbabNA7FHohcfxfwA4kItME2xW8MuNE-66dRHMNwa1F1DZH7CYr2w2BeV6zw=w137-h137-v0

eb85a2b2-273f-42fc-aaec-342714f78f32

https://lh3.googleusercontent.com/notebooklm/AG60hOr9xVz3YBKxgPF22UKkv0r5nubLAyCEPFLcczKfqK0rKbB8mhfWEab4BHdRxlaxQqTZfdN9VOBvOHuqeXlotXK6iO4P1y37d4QxtBY-KGmzu98Kit4nD6gOKaD728lc-RzDxUkwZw=w155-h155-v0

ce178463-54bb-4432-ba01-f2b768703dbc

https://lh3.googleusercontent.com/notebooklm/AG60hOpiEdcwkQ0HWXVNdEZKclQJfhnuiRw0TeV9G3PZDyKx9glfw0KfO4W1qzU-wNFp-p-TVNdqFhXxJUfmzi1MQSX8ROymJVB5kAV7ggpvYRbGERqrVisGXPgc9YFkseBatvtj86JqDw=w762-h261-v0

cf1d6a78-6516-4849-b89d-df7916005b24

https://lh3.googleusercontent.com/notebooklm/AG60hOqIP4rqCQAUiUocX9452NHm10jAfMXwEzWTp6XwdxkKY2oNgl6MQe4T7V4j0iLJQCFWzi_t1zrg6YZzycW83ILLXfFY1y2RoP2Lt1ZTBSao9jb4kfAQmDJv7vOe--ePJNgAwm_e=w781-h261-v0

c9cf4698-430b-4b63-895b-41909a14e46f

https://lh3.googleusercontent.com/notebooklm/AG60hOorBNW7GJDqe-cdKHuQBj7N5aNebk6SIrMBo5NP7SHqLmXknynkRpQcNOc5UM_1Sqa9OnLaHepIL6GjzENkSqdiC6vv7tXDbgh08JkcTYX4TcHnijPsgEtmsx8oGgT0CnNnG3KgCw=w1472-h261-v0

5144987f-f6fc-4288-9cb9-f8968fc40356

https://lh3.googleusercontent.com/notebooklm/AG60hOr8k6lNm2pZZT3-2eCfrvc9zSKgDWOpYFugtA3aV_A5B1zlNECQMwr4s3ZkESiUVpcMRpIH8fZRG6Rct7OvJZZPHfdDMpXLdVcK_mYZDRMXLZqS8H5wyBcFnKP6KgjTeipl6cQEuw=w702-h261-v0

e3a0ba3a-81bc-4e44-9274-37e1f70f2b9d

https://lh3.googleusercontent.com/notebooklm/AG60hOr22_EFxtTFDGb8mn9ouhIoh-qnKFhcS-Tlm6eraOu-mpGQLOM1IylTf6LsPPj-9s3Gh4PcHzvIpPAAznNsidxIqlTAeUR_Zb1IR4dRgXumD1zdN-pO-uNEh7lT0sxM_ricTTtTXg=w704-h261-v0

16b85328-50fe-4171-b954-a6904aaf8c8e

https://lh3.googleusercontent.com/notebooklm/AG60hOpBt_g_Y4_TJJHVKK2Jg2PyGWDbt_OLbdsdGa23Q-zG1u-HJt1E4W2GiTE0cD0fFN-1xAhHHXYTddtygDlDqAeLUcQStokg05ys-YZ7OpaYW_mNYfxdrrh55qx2Rug8LgPGpnC1dg=w1471-h261-v0

12e42a0b-6b0a-4e3a-ba73-a82e8ddad4d9

https://lh3.googleusercontent.com/notebooklm/AG60hOryq5ytj82-BzUHhdJRM_bF8HYzq-iVOwoUYXsVfRfSCjHCqHTGXCHxte31BxAW6gc4Ro6_OT9_GUmsXW9xi0sIHDRcWNewNNrcnwY9mIT8TrIho9NB6XeKBRESIfID6oIa6FiCAQ=w112-h112-v0

04f2cf79-a8df-403b-8465-35bec259f83d

## Long Context Instructions

## Short Visual Instructions Image Interleave Data

## Video Instructions

## ETVLM DATA  Construction

Mix & Concat

tim e Extending context

M-RoPE++: Multi-dimensional Interpolation

𝑅! 𝜃, 𝑖", 𝑖#, 𝑖$ = 𝐴!:#$

Pre-trained Range Extending Range

𝑅( - rotary matrix 𝐴) - rotary block

Hybrid-resolution Training

16k 64k 128k2k 4k 128k32k

Temporal information: directly extrapolation

Height: interpolation weighted by relative position

Width information: fully interpolation

3 frames, 12 tokens

1 high-res frame + 2 low-res frames

Figure 1: Pipeline of extending visual language models. We collect data from text, text-image pairs, and videos. We propose M-RoPE++ in extending training and hybrid-resolution inference to enhance the model performance.

2023c), with typical lengths ranging from 10K182

to 100K tokens. (ii) Short multimodal instruc-183

tion data, drawn mainly from LLaVA-Instruct (Liu184

et al., 2023c) and M3IT (Li et al., 2023b). While185

the original datasets are generally under 10K to-186

kens, we concatenate samples to achieve lengths187

between 10K and 100K tokens. (iii) Interleaved188

multimodal pre-training data, comprising multi-189

ple images with typical lengths of 1K–10K to-190

kens, sourced from MMDU (Liu et al., 2024c)191

and Mantis (Jiang et al., 2024). We also process192

interleaved image data from arXiv following the193

arXivQA protocol (Li et al., 2024c). (iv) Long194

multimodal instruction data, created by sampling195

multiple frames from video datasets, primarily196

sourced from ShareGPT4V (Chen et al., 2023a) and197

ShareGPT4O (Chen et al., 2024). To address the198

scarcity of long video instruction data, we sample199

videos longer than 5 minutes from MLVU (Zhou200

et al., 2024), ensuring MLVU is excluded from201

our test set to maintain fair evaluation. The data202

composition details are summarized in Appendix A203

Table 7. The data processing details are shown in204

Appendix B.205

2.3 Data Recipe Exploration206

We investigate the impact of different data ratio and207

data length on downstream task performance and208

provide recommendations for optimal data recipes.209

Using the same number of training tokens across210

all datasets, we conduct experiments with Qwen-211

VL (Bai et al., 2023) as the base model.212

Data Ratio To further investigate the impact of213

data composition on model performance, we con-214

duct experiments by varying the proportion of a215

single data type from 10% to 90% while keeping216

the total training volume consistent. The results217

presented in Figure 2 reveal that increasing the pro-218

portion of long video data improves long video219

10 30 50 70 90 Ratio (%)

## Long Text Data

## VideoMME MMbench

10 30 50 70 90 Ratio (%)

65 Short MM Data

## VideoMME MMbench

10 30 50 70 90 Ratio (%)

65 Interleave Data

## VideoMME MMbench

10 30 50 70 90 Ratio (%)

65 Video Data

## VideoMME MMbench

Figure 2: Performance of extending Qwen-VL with different data composition ratios.

comprehension but compromises performance on 220

other tasks. Similarly, increasing the ratio of any 221

specific data type predominantly enhances its as- 222

sociated downstream task performance. Based on 223

these findings, we determine the final data composi- 224

tion strategy, as shown in Table 7, which modestly 225

increases the proportion of video data while reduc- 226

ing the share of pure text data. This adjusted recipe 227

achieves a well-balanced performance across di- 228

verse task types. 229

0 20 40 60 80 100 Long-Short Data Ratio (%)

0 20 40 60 80 100 Long-Short Data Ratio (%)

Figure 3: Performance on Qwen-VL trained with differ-ent composition ratio of long (>8K) and short data.

Data Length We categorize data into long data 230

and short data based on whether their length ex- 231

ceeds 8K tokens. We investigate how different 232

ratios of long and short data affect downstream per- 233

formance on both long-context and short-context 234

tasks. As shown in Figure 3, increasing the propor- 235

tion of long data leads to improved performance 236

on long-context tasks, with performance plateauing 237

after the long data ratio reaches 60%. However, for 238

short-context tasks, when the proportion of long 239

data exceeds 60%, there is a notable decline in per- 240

formance. Based on these observations, we adopt241

a 60% long data ratio for our extending training to242

achieve an optimal balance between long and short243

task performance.244

Findings 1

Short multimodal instruction data is crucial for both extending long context capability and retaining short context performance. A balanced data ratio contributes to balanced performance on downstream tasks.

3 How to Extend Context Length246

In this section, we test the effective length of exist-247

ing length-extending methods, address their limita-248

tions (§3.1), and introduce our position embedding249

technique M-ROPE++ (§3.2). We find that for ex-250

tending VLMs, it is sufficient to tune the LLM251

base of VLMs without requiring multi-stage train-252

ing (§3.3). We propose hybrid-resolution training253

to further leverage the fixed context length (§3.5).254

3.1 Effective Length of VLMs255

To evaluate the effective context length of VLMs,256

we draw inspiration from recent studies on LLMs,257

which suggest that their effective lengths are often258

only about half of their training lengths (An et al.,259

2024b; Gao et al., 2024a). We adopt the single260

needle setting from Visual Haystack (Wu et al.,261

2024c), where models process varying numbers262

of input images and are tasked with identifying263

specific images and answering questions such as,264

"For the image with the anchor object, is there a265

target object?" This setup enables the assessment266

of performance across different context lengths,267

with random guessing yielding a 50% success rate.268

All tests are conducted using native image reso-269

lutions consistent with the original configuration.270

As shown in Figure 4, retrieval success rates de-271

crease as the number of input images grows. We272

define an accuracy threshold of 60% to determine273

the effective length. The base Qwen2-VL model274

achieves effectiveness up to 15 images, correspond-275

ing to an effective length to approximately 10K276

tokens. After extending the training length to 128K277

tokens using existing length-extending methods278

like PI and NTK, the effective length increases279

to around 50 images, equivalent to approximately280

40K tokens—still less than one-third of the training281

length. These findings highlight that the extended282

1235 25 50 100 150 Number of Images

Qwen2-VL Qwen2-VL PI Qwen2-VL NTK Qwen2-VL M-RoPE++ LongViLA Gemini-1.5-Pro GPT-4V

Figure 4: Results on visual haystack. The x-axis shows the number of input images, and the y-axis shows the retrieval success rate. The dashed line indicates the 60% threshold for effective length.

VLMs, similar to LLMs, exhibit the falls short phe- 283

nomenon (An et al., 2024b), where effective length 284

falls short of the training length. These findings 285

highlight the need for a novel position-extending 286

method to enhance the effective length of models. 287

The effective length in VLMs, includ-ing models that utilize existing position-extending methods, is smaller than the train-ing length.

3.2 Position Extending on VLM 289

In this subsection, we briefly introduce M-RoPE, 290

discuss potential issues associated with existing 291

position extending methods, and then present our 292

proposed M-RoPE++ along with experimental re- 293

sults validating its effectiveness. 294

M-RoPE Multimodal Rotary Position Embed- 295

ding (M-RoPE) proposed in Qwen2-VL (Wang 296

et al., 2024a) extends the RoPE (Su et al., 2024) 297

to effectively model positional information with 298

multi-dimensions. M-RoPE deconstructs the origi- 299

nal rotary embedding into three components: tem- 300

poral, height, and width. The formal definition of 301

M-RoPE and RoPE can be found in Appendix C. 302

For a 16x-dimensional M-RoPE matrix, the di- 303

mensions are allocated in a 2:3:3 ratio for temporal, 304

height, and width components respectively. This 305

can be represented as: 306

RM (θ, it, ih, iw) =

 A1 0 · · · 0 0 A2 · · · 0 ...

... 0 0 · · · A8x

 , (1) 307

where each Ai ∈ R2×2 is a rotary block and it, iw, 308

ih are position indices. θ represents the rotary base. 309

The blocks are allocated as follows: 310

A1 to A2x represent the temporal dimension;311

A2x+1 to A5x represent the height dimension;312

A5x+1 to A8x represent the width dimension.313

Each rotary block Ai is defined as:314

[ cos(ixθd) − sin(ixθd) sin(ixθd) cos(ixθd)

] , (2)315

where ix represents it, ih, or iw depending on316

which dimension the block belongs to. The fre-317

quency basis θ is shared across all dimensions.318

Position extending on M-RoPE In M-RoPE, the319

temporal index are allocated to the lower dimen-320

sions of the rotary embedding, which correspond to321

high-frequency information. Preserving this infor-322

mation is crucial for maintaining the model’s ability323

to discern temporal order. Position extending meth-324

ods such as position interpolation (PI; Chen et al.325

2023b) or modifying the RoPE base (NTK; Local-326

LLaMA 2023) tend to compress high-frequency327

signals indiscriminately, potentially confusing the328

model’s perception of order of close-by frames.329

Conversely, the height and width dimensions oc-330

cupy higher-dimensional spaces in the rotary em-331

bedding, indicating that they may not have fully332

covered the rotational domain during pre-training.333

This necessitates the application of interpolation334

to these dimensions. To address this, we propose335

M-RoPE++ that applies extrapolation exclusively336

to the temporal index and apply interpolation on337

height and weight index.338

M-RoPE++ We begin by defining key parame-339

ters following YaRN ((Peng et al., 2023) :340

LV , (3)341

where s is the ratio between the extended context342

length L′ and the original visual context length LV .343

We define λd as the wavelength of the RoPE344

embedding at the d-th hidden dimension:345

2d |D| , (4)346

and introduce the ratio r:347

λ . (5)348

For M-RoPE, the index range is divided into349

three segments: temporal (t), height (h), and width350

(w). Temporal information is predominantly in 351

high-frequency, which has been covered during 352

pre-training stage. Therefore, we maintain extrap- 353

olation for this segment. For the height and width 354

segments, where λ > L′, indicating insufficient 355

rotational domain training, we employ interpola- 356

tion to preserve their performance. This design is 357

illustrated in Figure 1 right part. 358

We propose the following piecewise function to 359

obtain the updated θ′d for M-RoPE++: 360

 θd if 0 < d ≤ 2x,

( 1 s + (1− 1

s ) · d−r5x

r2x−r5x ) · θd if 2x < d ≤ 5x,

if 5x < d ≤ 8x.

Experiment Validation We conduct a compara- 362

tive analysis of various methods for extending the 363

context length of VLMs, focusing on their perfor- 364

mance on the VideoMME long context task and 365

Single Needle Visual Haystacks in Table 1. 366

Method VideoMME Long Score (Frames) VH(Images)

64 128 256 512 768 100

Direct extrapolation 52.5 54.3 56.0 55.4 55.6 51.3 PI training 52.1 54.6 56.7 56.0 55.1 57.8 NTK-aware 53.8 54.8 55.8 56.2 56.0 56.7 M-RoPE++ 53.4 55.9 57.5 58.5 58.5 61.3

Table 1: Comparison of position embedding extension methods on VideoMME long video task and visual haystack on Qwen2-VL.

Our results demonstrate that M-RoPE++ con- 367

sistently surpasses other methods, showing con- 368

tinued improvement as the number of frames in- 369

creases in VideoMME Long tasks. This indicates 370

that M-RoPE++ effectively captures long-range 371

dependencies in video data. While direct extrapo- 372

lation shows some potential for context extension, 373

increasing the frame count without additional train- 374

ing does not lead to further performance gains. The 375

PI method, due to significant interpolation of high- 376

frequency information, exhibits slight performance 377

degradation on shorter tasks. The NTK-aware ap- 378

proach achieves better results than the base model 379

but still falls short of M-RoPE++ when handling 380

higher frame counts, emphasizing the importance 381

of preserving the original RoPE base in temporal 382

dimensions. In the Visual Haystack test with 100 383

images, M-RoPE++ outperforms all baseline meth- 384

ods, demonstrating its ability to further enhance 385

the effective length of VLMs. These findings high- 386

light the effectiveness of M-RoPE++ in extending 387

context length in VLMs. 388

Findings 3

The effective lengths achieved by existing position-extending methods remain insuf-ficiently long. M-RoPE++ achieves better downstream task performance and longer effective length in the same training length.

3.3 Multi-Stage Training390

We investigate whether multi-stage training strate-391

gies commonly used in VLM training are neces-392

sary for extending context length. Previous works393

on long-context VLMs, typically training from an394

LLM base, often employ multiple stages, including395

extending the text-based model’s context length,396

multimodal alignment, and multimodal instruction397

tuning. For extending existing VLMs like Qwen2-398

VL, we explore three approaches: (1) train VLM399

with mixed instruction data while only updating400

LLM backbone, (2) extending the LLM base with401

additional pure text data (Wiki-103) followed by402

multimodal instruction data, like LongVA (Zhang403

et al., 2024a), and (3) multimodal alignment us-404

ing image-text pairs (Sampled from LAION-5B)405

followed by instruction tuning (Xue et al., 2024;406

Wang et al., 2024b). As shown in Table 2, our

## Training Strategy MMBench BLINK VideoMME

One-stage MM Instruction 82.8 54.6 58.5 Two-stage Text Extending + MM Instruction 79.8 52.9 58.1 Two-stage MM Alignment + MM Instruction 80.5 51.2 57.8

Table 2: Comparison of different training strategies for extending Qwen2-VL context length.

407 experiments indicate that pre-extending the text-408

based model with pure text data provides no sig-409

nificant advantage. This is likely because train-410

ing with long-context multimodal data already ad-411

dresses diverse length distributions, rendering pure412

text extension redundant. Moreover, performing413

multimodal alignment before instruction tuning de-414

grades performance on short-context tasks. This415

could be attributed to Qwen2-VL already under-416

going instruction tuning before extending training;417

further tuning of MLP and ViT layers with align-418

ment objectives may disrupt the model’s learned419

distributions. With fixed training steps, this disrup-420

tion negatively impacts short-context performance421

without yielding improvements for long-context422

multimodal tasks.423

Findings 4

Directly train VLM with mixed instruction data while only updating LLM backbone’s parameters achieves optimal results.

3.4 Trade-off in Fixed Context Length 425

When encoding videos with a fixed total number of 426

visual tokens, there exists an inherent balance be- 427

tween the resolution of each frame and the number 428

of frames included. To investigate this balance on 429

video tasks, we test various combinations of frame 430

counts and resolutions, adjusting one in response 431

to changes in the other. Table 3 summarizes the re- 432

sults of GIRAFFE on VideoMME medium and long 433

sub-tasks under these configurations, highlighting 434

the impact of different frame-resolution trade-offs. 435

## Frame Image Token VideoMME VideoMME Count Count Medium Long

128 960 62.5 55.6 256 480 63.9 57.3 512 240 64.6 58.2 768 160 64.8 58.5 768 120 64.3 58.3 1024 120 64.7 58.5

Table 3: Performance of different frame counts and resolutions on VideoMME tasks for GIRAFFE.

From the perspective of frame count, perfor- 436

mance on medium-length tasks tends to plateau 437

at 512 frames, with little to no substantial improve- 438

ment beyond this threshold. For longer tasks, how- 439

ever, increasing the frame count continues to yield 440

performance gains, despite a corresponding reduc- 441

tion in the resolution of each frame. Notably, when 442

the frame count is high but individual frame reso- 443

lution is already low, further compression of res- 444

olution negatively impacts performance. These 445

findings highlight the importance of a strategy that 446

preserves high resolution for critical frames while 447

accommodating longer sequences. 448

3.5 Hybrid-resolution Training 449

To address this, we propose hybrid-resolution train- 450

ing, inspired by SlowFast (Feichtenhofer et al., 451

2019), which reduces token usage while maintain- 452

ing performance in long-form video understanding 453

tasks. We partition the video frames into N groups, 454

each containing L frames. For each group, we pro- 455

cess the first frame using a high-resolution image 456

that occupies m visual tokens. The subsequent 457

L− 1 frames within the group are processed that458

occupy m s tokens, where s is the compression ratio.459

This approach significantly reduces the token usage460

from L ∗N ∗m tokens to (1 + L−1 s ) ∗N ∗m to-461

kens. The high-resolution frames at the beginning462

of each group provide detailed visual information,463

while the low-resolution frames maintain temporal464

continuity and context at a reduced computational465

cost. This design is illustrated in Figure 1.466

Frames (L,m,s) Avg. Image VideoMME VideoMME Count Tokens Medium Long

512 (1,240,1) 240 64.2 57.9 512 (4,240,3) 120 64.0 57.6

1024 (1,120,1) 120 64.7 58.5 1024 (4,240,3) 120 66.2 60.4

Table 4: Performance comparison of hybrid-resolution training settings on VideoMME tasks.

The results in Table 4 demonstrate the effective-467

ness of hybrid-resolution training. Comparing the468

first two rows, we observe that reducing the res-469

olution of low-res frames using hybrid resolution470

only marginally affects downstream task perfor-471

mance while halving visual token usage. Further-472

more, the bottom two rows reveal that under equiv-473

alent visual token constraints, hybrid-resolution474

inference enables increased resolution for high-res475

frames and successfully enhances downstream task476

performance. These findings suggest that hybrid-477

resolution inference offers a promising approach to478

optimize the trade-off between computational effi-479

ciency and model performance in long-form video480

understanding tasks. We use (L,m,s)=(4,240,3) by481

default for other evaluations.482

Findings 5

Hybrid-resolution training can further im-prove the performance of VLM in a fixed context length.

4 Extended VLMs484

In this section, we first present the experimental485

setup and the relevant models, followed by an486

analysis of their performance across various down-487

stream tasks. For infrastructure and engineering488

details, please refer to Appendix F.489

4.1 Models490

We assess the following models: Qwen-VL-Chat-491

7B (Bai et al., 2023) A visual language model based492

on the Qwen language model, incorporating visual 493

capabilities through cross-attention and learnable 494

query embeddings. VideoLLaVA-7B (Lin et al., 495

2024) A video-language model that extends LLaVA 496

to handle video inputs, capable of processing up 497

to 8 frames. VideoChat2-Mistral-7B (Li et al., 498

2024b) An advanced VLM built on the Mistral-7B, 499

designed to process up to 16 frames. LongVA- 500

7B (Zhang et al., 2024a) A long context VLM 501

based on Qwen-2 language model, utilizing a two- 502

stage alignment process to handle up to 128 frames. 503

LongVILA-8B (Xue et al., 2024) A long context 504

VLM based on VILA language model, capable of 505

processing up to 256 frames. Qwen2-VL (Wang 506

et al., 2024a) A foundational VLM that employs dy- 507

namic image tokenization and M-RoPE, with pre- 508

trained 16K context length. We select Qwen2-VL 509

(for GIRAFFE), Qwen-VL (for GIRAFFE-QwenVL) 510

as the base model with the best extending training 511

setting shown in §2 and §3. 512

4.2 Video Task Results 513

Our extended models, GIRAFFE-QwenVL and GI- 514

RAFFE, demonstrate substantial improvements in 515

video understanding across various temporal scales 516

while specifically maintaining competitive perfor- 517

mance on short videos. Table 5 shows that GI- 518

RAFFE-QwenVL significantly outperforms its base 519

model Qwen-VL-Chat, enabling better understand- 520

ing of video content. Notably, GIRAFFE, based on 521

an improved base model and capable of processing 522

1024 frames, achieves state-of-the-art performance 523

among open-source models in both VideoMME 524

and LongVideoBench, even surpassing GPT-4V 525

in several categories. These results provide com- 526

pelling evidence that our approach successfully 527

extends the context window of VLMs, particularly 528

benefiting long context video understanding tasks 529

while reserving original short context capacities. 530

4.3 Image Task Results 531

The results from Table 6 demonstrate that our 532

GIRAFFE maintains competitive performance on 533

short-form multimodal tasks. This balanced ca- 534

pability can be attributed to our training strategy, 535

which incorporates a mix of short instruction data 536

alongside long context video inputs. Incorporating 537

LLaVA-Instruct and M3IT in our training process 538

ensures the model retains its capacity in single- 539

image understanding. For multi-image task results, 540

please refer to Appendix G. 541

Methods Frames VideoMME Frames LongVideoBench Avg Short Medium Long Overall (8, 15) (15, 60) (180, 600) (900, 3600)

Close-source VLMs

GPT-4V (turbo) 10 70.5 55.8 53.5 59.9 256 66.4 71.1 61.7 54.5 59.1 GPT-4o 384 80.0 70.3 65.3 71.9 256 71.6 76.8 66.7 61.6 66.7 Gemini-1.5-Pro 1/0.5fps 81.7 74.3 67.4 75.0 256 68.3 73.2 63.1 56.3 62.7

Open-source VLMs

VideoLLaVA-7B 8 45.3 38.0 36.2 39.9 8 43.1 44.6 36.4 34.4 39.1 VideoChat2-Mistral-7B 16 48.3 37.0 33.2 39.5 16 49.3 49.3 39.0 37.5 39.3 VideoLLaMA2-7B 16 56.0 45.4 42.1 47.9 - - - - - -LLaVA-NeXT-Qwen2-7B 32 58.0 47.0 43.4 49.5 - - - - - -LongVA-7B 128 61.1 50.4 46.2 52.6 - - - - - -LongVILA-8B 256 61.8 49.7 39.7 50.5 - - - - - -

Qwen-VL-Chat-7B 4 46.9 38.7 37.8 41.1 - - - - - -GIRAFFE-QwenVL 128 55.4 51.2 46.9 51.2 - - - - - -Qwen2-VL-7B 256 71.2 62.5 56.0 63.2 256 67.8 70.4 56.6 51.3 61.5 GIRAFFE 768 71.1 64.8 58.5 64.8 768 67.4 70.6 59.1 55.9 63.3

w/ Hybrid-res train&inf 1024 71.1 66.2 60.5 65.9 1024 67.4 71.0 60.8 58.1 64.3

Table 5: Performance comparison across VLMs on VideoMME and LongVideoBench tasks. We bold the best results for both close-source and open-source VLMs. We choose the best frames from our experiments in §3.4 and only use Hybrid-res inference on tasks above 512 frames.

Model MMEp MMEc MMBench(en)

GPT-4V 1590.5 573.2 82.8 Qwen-VL 1487.6 360.7 60.9 GIRAFFE-QwenVL 1489.7 372.9 61.5 Qwen2-VL 1695.3 1630.4 82.8 GIRAFFE 1692.9 1635.4 82.1

Table 6: VLM performance on the single-image sce-nario: MME and MMBench tasks. We bold the best results and underline the second best.

5 Related Work542

5.1 Long Context Language Models543

The main solution for long context scenery ad-544

dresses the out-of-distribution issue with position-545

embedding and enhancing model extrapola-546

tion capabilities. Training-free methods like547

streamingLLM (Xiao et al., 2024b), InfLLM (Xiao548

et al., 2024a) and ChunkLLaMA (An et al., 2024a)549

offer cost-effective ways to scale context window550

size. Additionally, further training using modified551

RoPE (Su et al., 2024) base frequency is intro-552

duced in NTK (LocalLLaMA, 2023), PI (Chen553

et al., 2023b) and YaRN (Peng et al., 2023), a ef-554

fective practice adopted by models such as CodeL-555

lama (Rozière et al., 2024) and LLaMA 3.1 (Team,556

2024). Moreover, efforts have also been made on557

data curation for long context training (Bai et al.,558

2024; Gao et al., 2024b; Fu et al., 2024c). However,559

corresponding comprehensive studies on extending560

context for open-source VLMs remain limited.561

5.2 Long Visual Language Models 562

For long context VLMs, recent LongVA (Zhang 563

et al., 2024a) are first extending an LLM base 564

model to 128K token lengths and then developing 565

it into a VLM. Concurrent work LongVILA (Xue 566

et al., 2024) also involves multi-stage training start- 567

ing from an LLM backbone and employs an im- 568

proved sequence parallel technique for efficient 569

training, while LongLLaVA (Wang et al., 2024b) 570

combines Mamba and Transformer blocks to re- 571

duce memory usage. In contrast, our model GI- 572

RAFFE optimizes various data recipes and position 573

extending designs, establishing itself as the state- 574

of-the-art among open-source long VLMs. 575

6 Conclusion and Future Work 576

We develop an effective solution to extend the con- 577

text length of VLMs while preserving their per- 578

formance on shorter contexts. Our comprehensive 579

experiments led to the introduction of the ETVLM 580

dataset for extended training and M-RoPE++ for 581

improved position embedding learning. We use 582

Hybrid-res training to better use long context win- 583

dow. Our extended model, GIRAFFE, achieves 584

state-of-the-art performance for long context tasks. 585

In the future, we aim to apply GIRAFFE to more 586

complex scenarios, such as long-term history multi- 587

modal chats and visual agents in real-world appli- 588

cations. 589

Limitations590

Our study has several limitations that warrant con-591

sideration. (i) Due to limited computational re-592

sources, we were unable to conduct a more compre-593

hensive exploration of optimal data ratios through594

additional experiments. This limitation may have595

prevented us from determining a more precise and596

effective data composition for training. (ii) The597

current implementation of M-RoPE++ is restricted598

to models pre-trained with M-RoPE. Adapting this599

technique to other model architectures remains a600

subject for future investigation. (iii) Our evalua-601

tion primarily focused on question-answering tasks602

due to the scarcity of diverse long context video603

datasets. This constraint limits our ability to assess604

the model’s performance in more realistic applica-605

tion scenarios, such as embodied agents or long-606

term visual agents. Addressing these limitations in607

future work could potentially yield more robust and608

generalizable long context visual language models.609

Ethical Considerations610

The ethical considerations for our study encompass611

several key aspects: (i) Data sourcing: All data612

utilized in our research was obtained from publicly613

shared sources, adhering strictly to their respective614

open-source licenses. (ii) Model development: Our615

further training on the Qwen model complies fully616

with Qwen’s licensing agreements. (iii) Evaluation617

methodology: We exclusively employed automated618

evaluation tools for assessment, avoiding the need619

for human annotators. (iv) Potential misuse: While620

we have focused on benign applications, we ac-621

knowledge the potential for misuse of advanced622

visual language models and encourage ongoing623

discussions on responsible AI development and624

deployment.625

References626

Chenxin An, Fei Huang, Jun Zhang, Shansan Gong,627 Xipeng Qiu, Chang Zhou, and Lingpeng Kong.628 2024a. Training-free long-context scaling of large629 language models.630

Chenxin An, Jun Zhang, Ming Zhong, Lei Li, Shansan631 Gong, Yao Luo, Jingjing Xu, and Lingpeng Kong.632 2024b. Why does the effective context length of llms633 fall short?634

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang,635 Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou,636

and Jingren Zhou. 2023. Qwen-vl: A versatile vision- 637 language model for understanding, localization, text 638 reading, and beyond. 639

Yushi Bai, Xin Lv, Jiajie Zhang, Yuze He, Ji Qi, Lei 640 Hou, Jie Tang, Yuxiao Dong, and Juanzi Li. 2024. 641 Longalign: A recipe for long context alignment of 642 large language models. 643

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Con- 644 ghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. 645 2023a. Sharegpt4v: Improving large multi-modal 646 models with better captions. 647

Shouyuan Chen, Sherman Wong, Liangjian Chen, and 648 Yuandong Tian. 2023b. Extending context window 649 of large language models via positional interpolation. 650

Yukang Chen, Shaozuo Yu, Shengju Qian, Hao- 651 tian Tang, Xin Lai, Zhijian Liu, Song Han, and 652 Jiaya Jia. 2023c. Long alpaca: Long-context 653 instruction-following models. https://github. 654 com/dvlab-research/LongLoRA. 655

Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, 656 Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi 657 Hu, Jiapeng Luo, Zheng Ma, et al. 2024. How far 658 are we to gpt-4v? closing the gap to commercial 659 multimodal models with open-source suites. arXiv 660 preprint arXiv:2404.16821. 661

Tri Dao. 2024. FlashAttention-2: Faster attention with 662 better parallelism and work partitioning. In Inter- 663 national Conference on Learning Representations 664 (ICLR). 665

Tri Dao, Daniel Y Fu, Stefano Ermon, Atri Rudra, 666 and Christopher Re. 2022. Flashattention: Fast and 667 memory-efficient exact attention with IO-awareness. 668 In Advances in Neural Information Processing Sys- 669 tems. 670

Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, 671 and Kaiming He. 2019. Slowfast networks for video 672 recognition. 673

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, 674 Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, 675 Ke Li, Xing Sun, et al. 2023. Mme: A comprehensive 676 evaluation benchmark for multimodal large language 677 models. arXiv preprint arXiv:2306.13394. 678

Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai 679 Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yun- 680 hang Shen, Mengdan Zhang, et al. 2024a. Video- 681 mme: The first-ever comprehensive evaluation bench- 682 mark of multi-modal llms in video analysis. arXiv 683 preprint arXiv:2405.21075. 684

Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu 685 Wang, Xudong Lin, Dan Roth, Noah A. Smith, Wei- 686 Chiu Ma, and Ranjay Krishna. 2024b. Blink: Mul- 687 timodal large language models can see but not per- 688 ceive. 689

Yao Fu, Rameswar Panda, Xinyao Niu, Xiang Yue, Han-690 naneh Hajishirzi, Yoon Kim, and Hao Peng. 2024c.691 Data engineering for scaling language models to 128k692 context. In Forty-first International Conference on693 Machine Learning.694

Tianyu Gao, Alexander Wettig, Howard Yen, and Danqi695 Chen. 2024a. How to train long-context language696 models (effectively).697

Tianyu Gao, Alexander Wettig, Howard Yen, and698 Danqi Chen. 2024b. How to train long-context699 language models (effectively). arXiv preprint700 arXiv:2410.02660.701

Gemini Team. 2024. Gemini: A family of highly capa-702 ble multimodal models.703

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan704 Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and705 Weizhu Chen. 2022. Lora: Low-rank adaptation of706 large language models. In The Tenth International707 Conference on Learning Representations, ICLR 2022,708 Virtual Event, April 25-29, 2022. OpenReview.net.709

Dongfu Jiang, Xuan He, Huaye Zeng, Cong Wei,710 Max W.F. Ku, Qian Liu, and Wenhu Chen. 2024.711 Mantis: Interleaved multi-image instruction tuning.712 arXiv2405.01483.713

Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang,714 Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. 2024a.715 Llava-next-interleave: Tackling multi-image, video,716 and 3d in large multimodal models.717

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi.718 2023a. Blip-2: Bootstrapping language-image pre-719 training with frozen image encoders and large lan-720 guage models. ArXiv preprint, abs/2301.12597.721

Junnan Li, Dongxu Li, Caiming Xiong, and Steven C. H.722 Hoi. 2022. BLIP: bootstrapping language-image pre-723 training for unified vision-language understanding724 and generation. In International Conference on Ma-725 chine Learning, ICML 2022, 17-23 July 2022, Balti-726 more, Maryland, USA, volume 162 of Proceedings727 of Machine Learning Research, pages 12888–12900.728

KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wen-729 hai Wang, Ping Luo, Yali Wang, Limin Wang, and730 Yu Qiao. 2024b. Videochat: Chat-centric video un-731 derstanding.732

Lei Li, Yuqi Wang, Runxin Xu, Peiyi Wang, Xiachong733 Feng, Lingpeng Kong, and Qi Liu. 2024c. Mul-734 timodal ArXiv: A dataset for improving scientific735 comprehension of large vision-language models. In736 Proceedings of the 62nd Annual Meeting of the As-737 sociation for Computational Linguistics (Volume 1:738 Long Papers), pages 14369–14387, Bangkok, Thai-739 land. Association for Computational Linguistics.740

Lei Li, Yuwei Yin, Shicheng Li, Liang Chen, Peiyi741 Wang, Shuhuai Ren, Mukai Li, Yazheng Yang,742 Jingjing Xu, Xu Sun, Lingpeng Kong, and Qi Liu.743 2023b. M3IT: A large-scale dataset towards744

multi-modal multilingual instruction tuning. ArXiv 745 preprint, abs/2306.04387. 746

Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, 747 Peng Jin, and Li Yuan. 2024. Video-llava: Learn- 748 ing united visual representation by alignment before 749 projection. 750

Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. 751 2024a. World model on million-length video and 752 language with ringattention. arXiv preprint. 753

Hao Liu, Matei Zaharia, and Pieter Abbeel. 2023a. 754 Ring attention with blockwise transformers for near- 755 infinite context. ArXiv, abs/2310.01889. 756

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae 757 Lee. 2023b. Improved baselines with visual instruc- 758 tion tuning. arXiv preprint arXiv:2310.03744. 759

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae 760 Lee. 2023c. Visual instruction tuning. ArXiv 761 preprint, abs/2304.08485. 762

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, 763 Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi 764 Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua 765 Lin. 2024b. Mmbench: Is your multi-modal model 766 an all-around player? 767

Ziyu Liu, Tao Chu, Yuhang Zang, Xilin Wei, Xiaoyi 768 Dong, Pan Zhang, Zijian Liang, Yuanjun Xiong, 769 Yu Qiao, Dahua Lin, et al. 2024c. Mmdu: A 770 multi-turn multi-image dialog understanding bench- 771 mark and instruction-tuning dataset for lvlms. arXiv 772 preprint arXiv:2406.11833. 773

LocalLLaMA. 2023. Ntk-aware scaled rope allows 774 llama models to have extended (8k+) context size 775 without any fine-tuning and minimal perplexity degra- 776 dation. 777

OpenAI. 2023. Gpt-4v(ision) system card. OpenAI 778 Research. 779

OpenAI. 2024. Chatml documents. 780

Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and En- 781 rico Shippole. 2023. Yarn: Efficient context window 782 extension of large language models. 783

Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, 784 and Yuxiong He. 2020. Zero: Memory optimizations 785 toward training trillion parameter models. In SC20: 786 International Conference for High Performance Com- 787 puting, Networking, Storage and Analysis, pages 1– 788 16. IEEE. 789

Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten 790 Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, 791 Jingyu Liu, Romain Sauvestre, Tal Remez, Jérémy 792 Rapin, Artyom Kozhevnikov, Ivan Evtimov, Joanna 793 Bitton, Manish Bhatt, Cristian Canton Ferrer, Aaron 794 Grattafiori, Wenhan Xiong, Alexandre Défossez, 795 Jade Copet, Faisal Azhar, Hugo Touvron, Louis Mar- 796 tin, Nicolas Usunier, Thomas Scialom, and Gabriel 797 Synnaeve. 2024. Code llama: Open foundation mod- 798 els for code. 799

Jianlin Su. 2023. Extending llm context window beyond800 2048 tokens.801

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan,802 Wen Bo, and Yunfeng Liu. 2024. Roformer: En-803 hanced transformer with rotary position embedding.804 Neurocomput., 568(C).805

Yutao Sun, Li Dong, Barun Patra, Shuming Ma, Shao-806 han Huang, Alon Benhaim, Vishrav Chaudhary, Xia807 Song, and Furu Wei. 2022. A length-extrapolatable808 transformer.809

Llama Team. 2024. The llama 3 herd of models.810

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhi-811 hao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin812 Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei813 Du, Xuancheng Ren, Rui Men, Dayiheng Liu,814 Chang Zhou, Jingren Zhou, and Junyang Lin. 2024a.815 Qwen2-vl: Enhancing vision-language model’s per-816 ception of the world at any resolution.817

Xidong Wang, Dingjie Song, Shunian Chen, Chen818 Zhang, and Benyou Wang. 2024b. Longllava: Scal-819 ing multi-modal llms to 1000 images efficiently via a820 hybrid architecture.821

Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li.822 2024a. Longvideobench: A benchmark for long-823 context interleaved video-language understanding.824

Haoning Wu, Zicheng Zhang, Erli Zhang, Chaofeng825 Chen, Liang Liao, Annan Wang, Chunyi Li, Wenxiu826 Sun, Qiong Yan, Guangtao Zhai, and Weisi Lin.827 2024b. Q-bench: A benchmark for general-purpose828 foundation models on low-level vision. In ICLR.829

Tsung-Han Wu, Giscard Biamby, Jerome Quenum,830 Ritwik Gupta, Joseph E. Gonzalez, Trevor Darrell,831 and David M. Chan. 2024c. Visual haystacks: A832 vision-centric needle-in-a-haystack benchmark.833

Chaojun Xiao, Pengle Zhang, Xu Han, Guangxuan Xiao,834 Yankai Lin, Zhengyan Zhang, Zhiyuan Liu, Song835 Han, and Maosong Sun. 2024a. Infllm: Unveiling the836 intrinsic capacity of llms for understanding extremely837 long sequences with training-free memory. arXiv.838

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song839 Han, and Mike Lewis. 2024b. Efficient streaming840 language models with attention sinks. In The Twelfth841 International Conference on Learning Representa-842 tions.843

Fuzhao Xue, Yukang Chen, Dacheng Li, Qinghao Hu,844 Ligeng Zhu, Xiuyu Li, Yunhao Fang, Haotian Tang,845 Shang Yang, Zhijian Liu, Ethan He, Hongxu Yin,846 Pavlo Molchanov, Jan Kautz, Linxi Fan, Yuke Zhu,847 Yao Lu, and Song Han. 2024. Longvila: Scaling848 long-context visual language models for long videos.849

Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng,850 Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Hao-851 ran Tan, Chunyuan Li, and Ziwei Liu. 2024a. Long852 context transfer from language to vision. arXiv853 preprint arXiv:2406.16852.854

Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun 855 Ma, Ziwei Liu, and Chunyuan Li. 2024b. Video 856 instruction tuning with synthetic data. 857

Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, 858 Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, 859 and Zheng Liu. 2024. Mlvu: A comprehensive 860 benchmark for multi-task long video understanding. 861 arXiv preprint arXiv:2406.04264. 862

Zhilin Zhu. 2023. Ring flash attention. 863 https://github.com/zhuzilin/ 864 ring-flash-attention. 865

A Data Composition866

Table 7 shows the composition details of our867

ETVLM data.868

B Data Processing869

All data are processed into a dialogue format con-870

sistent with ChatML style (OpenAI, 2024). Data871

are maintained in their original length and as con-872

catenated multi-turn dialogues. For original-length873

text instruction data, we filter out special tokens.874

For short visual instruction and interleaved data,875

we adjust formatting and remove unnecessary sym-876

bols. Video data are sampled at 2 fps to reduce877

computational overhead. During data concatena-878

tion, we aim to match the target context length (e.g.,879

32K, 128K) as closely as possible without truncat-880

ing content, ensuring a balance between efficiency881

and context preservation.882

C RoPE and M-RoPE883

Attention is defined over C embeddings X =884

[x1, x2, . . . , xC ] T ∈ RC×d where d is the model885

dimension. Learned weight matrices Wv ∈ Rd×dk ,886

Wq ∈ Rd×dk , and Wk ∈ Rd×dk are used to trans-887

form these inputs where dk is the projected hidden888

dimension. The attention mechanism itself com-889

putes the attention matrix and applies it to produce890

a weighted sum of the value vectors:891

Attention(Q,K, V ) = AV = softmax ( QKT

) V. (7)892

Basic attention was originally defined with: Q =893

XWq, K = XWk, V = XWv. However, this894

approach does not directly encode the relative posi-895

tion of keys and values.896

Rotary Position Embeddings (RoPE) (Sun et al.,897

2022) encode positional information by applying898

a phase rotation to each element of the embedding899

vectors. Formally, we define a transformation f :900

fW (xi, θ) = R(θ, i)W Txi (8)901

Here xi ∈ Rdk is an embedding for position902

i, W is a projection matrix, and θ ∈ Rdk/2 is a903

frequency basis. The function is defined based on904

the rotary position matrix:905

 cos iθ1 − sin iθ1 ··· 0 0 sin iθ1 cos iθ1 ··· 0 0

0 0 ··· cos iθdk/2 − sin iθdk/2

0 0 ··· sin iθdk/2 cos iθdk/2

 (9)906

Due to the arrangement of frequencies, this 907

matrix has the property that R(θ, n − m) = 908

R(θ,m)TR(θ, n) by Ptolemy’s identity. We rede- 909

fine the query-key product between two positions 910

m and n as, 911

qTmkn = fWq(xm, θ)T fWk (xn, θ) (10) 912

Multimodal Rotary Position Embedding (M- 913

RoPE) extends the concept of RoPE to effectively 914

model positional information of multimodal inputs. 915

M-RoPE deconstructs the original rotary embed- 916

ding into three components: temporal, height, and 917

width. For text inputs, these components utilize 918

identical position IDs, making M-RoPE function- 919

ally equivalent to 1D-RoPE. For image inputs, the 920

temporal IDs remain constant, while distinct IDs 921

are assigned to the height and width components 922

based on the token’s position in the image. For 923

video inputs, the temporal ID increments for each 924

frame, while the height and width components fol- 925

low the same ID assignment pattern as images. 926

Formally, we define the M-RoPE transformation 927

function fM as: 928

fM (xi, θt, θw, θh) = [Rt(θt, it)W T t xit; 929

Rw(θw, iw)W T w xiw; (11) 930

Rh(θh, ih)W T h xih] 931

where xi is the embedding vector, θt, θw, θh are 932

frequency bases, it, iw, ih are position indices, and 933

Wt, Ww, Wh are projection matrices for temporal, 934

width, and height dimensions respectively. 935

The query-key product for M-RoPE is then rede- 936

fined as: 937

qTmkn = fM (xm, θt, θw, θh) T fM (xn, θt, θw, θh) (12) 938

For a 16x-dimensional M-RoPE matrix, the di- 939

mensions are allocated in a 2:3:3 ratio for temporal, 940

height, and width components respectively. This 941

can be represented as: 942

RM (θ, it, ih, iw) =

 A1 0 · · · 0 0 A2 · · · 0 ...

... 0 0 · · · A8x

 (13) 943

where each Ai ∈ R2×2 is a rotary block. The 944

blocks are allocated as follows: 945

A1 to A2x represent the temporal dimension 946

Categories Task types Data sources %Part

Text Long context instructions LongAlign (Bai et al., 2024), LongAlpaca (Chen et al., 2023c) 20%

Image Short visual instruction data LLaVA-Instruct (Liu et al., 2023c), M3IT (Li et al., 2023b) 25%

Image interleave data MMDU (Liu et al., 2024c), Mantis (Jiang et al., 2024), ArXivQA-interleave* 25%

Video Video QA ShareGPT4O (Chen et al., 2024), MLVU (Zhou et al., 2024), LLaVA-Video (Zhang et al., 2024b) 30%Video Summary ShareGPT4V (Chen et al., 2023a)

Table 7: Overview of our ETVLM training dataset. This dataset encompasses a wide range of modalities and is concatenated to target context length. * indicates that we reconstruct this data by our own.

A2x+1 to A5x represent the height dimension947

A5x+1 to A8x represent the width dimension948

Each rotary block Ai is defined as:949

[ cos(ixθd) − sin(ixθd) sin(ixθd) cos(ixθd)

where ix represents it, ih, or iw depending on951

which dimension the block belongs to. The fre-952

quency basis θ is shared across all dimensions.953

This formulation allows M-RoPE to effectively954

model multimodal inputs while maintaining the955

rotary structure for each dimension.956

D Impact of RoPE Base957

We investigated the effect of different RoPE bases958

on the performance of Qwen-VL. Our findings in-959

dicate that the optimal performance was achieved960

by following the recommendations from Su’s blog,961

specifically using a RoPE base of 500,000 for a962

context length of 128k. Increasing the base beyond963

this point did not yield significant improvements964

while keeping the default base of 10,000 resulted in965

a notable performance drop. Table 8 summarizes966

our results.967

## RoPE Base VideoMME Long VideoMME Avg MME Sum MMBench

10,000 (default) 39.5 41.1 1848.29 60.9 500,000 (optimal) 43.2 51.2 1862.62 61.5 1,000,000 43.1 51.1 1862.20 61.4

Table 8: Performance comparison of different RoPE bases across various benchmarks.

These results underscore the significance of968

meticulously adjusting the RoPE base when ex-969

panding the context window of visual language970

models. Our findings corroborate the conclusions971

presented in Su’s blog (Su, 2023), which posits that972

for models with a context length of 128k, an opti-973

mal RoPE base of 4.9× 106 is recommended. This974

value closely approximates our selected base of975

5× 105, which consistently demonstrates superior976

performance compared to the default configuration 977

across all evaluated metrics. 978

Interestingly, further increasing the base beyond 979

this point does not yield significant performance 980

improvements. This observation is consistent with 981

the approaches taken by models like LLaMA 2 and 982

Qwen, which have opted for even larger base val- 983

ues. Such choices may provide additional flexibil- 984

ity for future extensions of model context lengths. 985

The effectiveness of the optimized RoPE base in 986

capturing long-range dependencies in multimodal 987

data underscores the critical role of position em- 988

bedding strategies in enhancing the performance of 989

extended visual language models. 990

E Progressive Extending 991

To ensure more stable training, we adopted a 992

progressive extending strategy. For GIRAFFE- 993

QwenVL, we set multiple incrementally increasing 994

context lengths: 8K, 32K, 64K, and 128K. We con- 995

catenate and chunk ETVLM data according to these 996

different context lengths. For GIRAFFE-QwenVL, 997

we investigate the optimal RoPE base setting, as 998

detailed in Appendix D. Following Su (2023), we 999

experiment with bases of 5×104, 1×106, 2.5×106, 1000

and 5×106. For GIRAFFE, we employ M-RoPE++, 1001

training up to 64K before extending to 128K. This 1002

approach allows the model to gradually adapt to 1003

longer sequences while maintaining performance 1004

on shorter contexts. 1005

Ablation of progressive extending We conduct 1006

comparative experiments on Qwen-VL to evalu- 1007

ate two methods for extending the model’s context 1008

length: a single-stage approach and a progressive 1009

multi-stage approach. Both methods are using the 1010

same number of training steps. The results are sum- 1011

marized in Table 9. Our experiments demonstrate 1012

that the progressive extending approach consis- 1013

tently outperforms the single-stage method across 1014

different evaluated tasks. This suggests that grad- 1015

ually increasing the context length during train- 1016

ing allows the model to better adapt to longer se- 1017

## Method MMEP MMEc VideoMME

Single-step (2k→128k) 1462.58 350.71 48.9 Progressive 1487.58 360.71 51.2

Table 9: Comparison of single-stage and progressive extension methods on Qwen-VL.

quences, resulting in improved performance on var-1018

ious tasks.1019

F Infrastructure and Engineering1020

We employ the NTK method for Qwen-VL and M-1021

RoPE++ for GIRAFFE to extend the model’s win-1022

dow length. Training long VLMs results in substan-1023

tial memory demands, thus we employ several opti-1024

mization strategies to perform training on such long1025

sequences. These include FlashAttention-2 (Dao1026

et al., 2022; Dao, 2024), Ring Attention (Liu et al.,1027

2023a), ZERO (Rajbhandari et al., 2020) (including1028

activation checkpointing, and parameter offload).1029

To balance the load across 8 80G H100 GPUs, we1030

shard the sequence in a zigzag way (Zhu, 2023).1031

We use LoRA (Hu et al., 2022) to reduce the GPU1032

memory usage to train longer VLMs. We train the1033

model for an average of 80 H100 hours.1034

G Multi Image Task Results1035

Model Mantis-Eval QBench BLINK

LLaVA-v1.5-7B 31.3 49.3 37.1 GPT-4V 62.7 76.5 51.1 Qwen-VL 39.2 45.9 31.1 GIRAFFE-QwenVL 48.3 57.4 41.2 Qwen2-VL 63.4 76.9 53.3 GIRAFFE 63.9 76.8 54.5

Table 10: VLMs results on multi-image scenario: Mantis-Eval, QBench and BLINK. We bold the best results and underline the second best.

In the multi-image evaluation presented in Ta-1036

ble 10, GIRAFFE-QwenVL exhibits substantial1037

improvements, whereas GIRAFFE also demon-1038

strates enhancements, validating the efficacy of our1039

pipeline. In multi-image scenarios, context length1040

is less critical than in long video tasks. Qwen-VL’s1041

superior performance stems from capacities trained1042

on the ETVLM dataset, compared to its initial 2K1043

context length. In contrast, Qwen2-VL has already1044

undergone substantial pre-training in 16K contexts.1045

Additionally, Qwen2-VL benefits from a broader1046

range of training data compared to Qwen-VL, ren-1047

dering the incremental advantages from ETVLM 1048

data relatively modest. 1049

