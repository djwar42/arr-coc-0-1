---
sourceFile: "When Tokens Talk Too Much: A Survey of Multimodal Long-Context Token Compression across Images, Videos, and Audios - arXiv"
exportedBy: "Kortex"
exportDate: "2025-10-28T18:42:20.954Z"
---

# When Tokens Talk Too Much: A Survey of Multimodal Long-Context Token Compression across Images, Videos, and Audios - arXiv

168397c9-f84c-4433-8290-adf484dbb7d8

When Tokens Talk Too Much: A Survey of Multimodal Long-Context Token Compression across Images, Videos, and Audios - arXiv

37df0567-c81e-4e48-bf6e-33bbf1738c02

https://arxiv.org/html/2507.20198v3

When Tokens Talk Too Much: A Survey of Multimodal Long-Context Token Compression across Images, Videos, and Audios

, Kejia Zhang

, Sicheng Feng

, Yuzhang Shang

,  Haoxuan You

, Huan Wang

Zhejiang University,

Westlake University,

Xiamen University,

National University of Singapore,

University of Wisconsin-Madison,

University of Central Florida,

Columbia University,

Salesforce AI Research,

Rice University  ∗ * ∗ : Equal Contribution.  † {\dagger} † : Corresponding Author (wanghuan@westlake.edu.cn).

Multimodal large language models (MLLMs) have made remarkable strides, largely driven by their ability to process increasingly  long and complex  contexts, such as high-resolution images, extended video sequences, and lengthy audio input. While this ability significantly enhances MLLM capabilities, it introduces substantial computational challenges, primarily due to the quadratic complexity of self-attention mechanisms with numerous input tokens. To mitigate these bottlenecks, token compression has emerged as an auspicious and critical approach, efficiently reducing the number of tokens during both training and inference. In this paper, we present the first systematic survey and synthesis of the burgeoning field of multimodal long context token compression. Recognizing that effective compression strategies are deeply tied to the unique characteristics and redundancies of each modality, we categorize existing approaches by their primary data focus, enabling researchers to quickly access and learn methods tailored to their specific area of interest:  (1) image-centric compression , which addresses spatial redundancy in visual data;  (2) video-centric compression , which tackles spatio-temporal redundancy in dynamic sequences; and  (3) audio-centric compression , which handles temporal and spectral redundancy in acoustic signals. Beyond this modality-driven categorization, we further dissect methods based on their underlying mechanisms, including  transformation-based ,  similarity-based ,  attention-based ,  and  query-based  approaches. By providing a comprehensive and structured overview, this survey aims to consolidate current progress, identify key challenges, and inspire future research directions in this rapidly evolving domain. For ongoing updates and to track the latest advances in this promising area, we maintain a public repository:

https://github.com/cokeshao/Awesome-Multimodal-Token-Compression

https://github.com/cokeshao/Awesome-Multimodal-Token-Compression

Index Terms:

Multimodal Large Language Models, Token Compression, Long Context, Efficient AI

1  Introduction

Multimodal large language models (MLLMs)  [

https://arxiv.org/html/2507.20198v3#bib.bib1

https://arxiv.org/html/2507.20198v3#bib.bib2

https://arxiv.org/html/2507.20198v3#bib.bib3

https://arxiv.org/html/2507.20198v3#bib.bib4

https://arxiv.org/html/2507.20198v3#bib.bib5

https://arxiv.org/html/2507.20198v3#bib.bib6

https://arxiv.org/html/2507.20198v3#bib.bib7

https://arxiv.org/html/2507.20198v3#bib.bib8

https://arxiv.org/html/2507.20198v3#bib.bib9

https://arxiv.org/html/2507.20198v3#bib.bib10

https://arxiv.org/html/2507.20198v3#bib.bib11

]  have demonstrated exceptional performance on complex tasks, including visual question answering (VQA), automatic speech recognition (ASR) and multimodal content generation, by extending the architectural principles of large language models (LLMs)  [

https://arxiv.org/html/2507.20198v3#bib.bib12

https://arxiv.org/html/2507.20198v3#bib.bib13

https://arxiv.org/html/2507.20198v3#bib.bib14

https://arxiv.org/html/2507.20198v3#bib.bib15

] . These powerful models derive their strength from processing very long and diverse contexts, such as high-resolution images, extended video sequences, and long audio input with the transformer architectures.

Figure 1:  Up:  Image, video, and audio data types can scale in their representation dimensions, leading to a corresponding increase in the number of tokens.  Down:  Top-performing MLLMs cannot address real-world demands, as the number of tokens for multimodal input, especially video, vastly exceeds that of text, and most visual tokens are redundant. Therefore, token compression is crucial to address this limitation.

Achieving this capability, however, face a significant challenge: the quadratic complexity of the self-attention mechanism. As the number of tokens increases, this complexity leads to substantial computational and memory demands. This problem is particularly pronounced in MLLMs, where the tokenization of visual and audio data can generate sequences of orders of magnitude longer than text  [

https://arxiv.org/html/2507.20198v3#bib.bib16

https://arxiv.org/html/2507.20198v3#bib.bib17

https://arxiv.org/html/2507.20198v3#bib.bib18

Figure 2:  Representative Architecture of MLLMs.  Within MLLM reasoning processes, token sequences comprise concatenated system tokens, multimodal tokens, text tokens. Multimodal tokens usually constitute the majority of the sequence tokens.

For instance, as illustrated in Figure

https://arxiv.org/html/2507.20198v3#S1.F1

, the number of image tokens is directly proportional to resolution, while the number of video tokens scales with both resolution and duration, and audio tokens are proportional to duration. A single content-rich video can produce tens of millions of tokens, dramatically exacerbating computational inefficiencies and leading to severe inference latency (90 minutes video will be converted into 54M tokens)

1 90  min × 60  s / min × 10  frames / s × 1000  tokens / frame . 90\,\text{min}\times 60\,\text{s}/\text{min}\times 10\,\text{frames}/\text{s}\times 1000\,\text{tokens}/\text{frame}. 90 min × 60 s / min × 10 frames / s × 1000 tokens / frame . . Consequently, addressing this computational bottleneck is critical for unlocking the full potential of MLLMs in real-world applications.

To address the challenges posed by the long context,

token compression

has emerged as a critical research focus for enhancing the inference efficiency and practical deployment of MLLMs. This approach is highly effective because multimodal inputs, like those processed by vision transformers (ViT), contain significant redundancy  [

https://arxiv.org/html/2507.20198v3#bib.bib19

https://arxiv.org/html/2507.20198v3#bib.bib20

https://arxiv.org/html/2507.20198v3#bib.bib21

https://arxiv.org/html/2507.20198v3#bib.bib22

https://arxiv.org/html/2507.20198v3#bib.bib23

https://arxiv.org/html/2507.20198v3#bib.bib24

https://arxiv.org/html/2507.20198v3#bib.bib25

] . Extensive research, for example, demonstrates that more than  50 % 50\% 50 %  of tokens in a typical MLLM sequence receive minimal attention during inference  [

https://arxiv.org/html/2507.20198v3#bib.bib26

https://arxiv.org/html/2507.20198v3#bib.bib27

https://arxiv.org/html/2507.20198v3#bib.bib17

https://arxiv.org/html/2507.20198v3#bib.bib16

https://arxiv.org/html/2507.20198v3#bib.bib28

https://arxiv.org/html/2507.20198v3#bib.bib29

] . While some advanced techniques integrate compression directly into a model’s architecture or training framework  [

https://arxiv.org/html/2507.20198v3#bib.bib30

https://arxiv.org/html/2507.20198v3#bib.bib31

https://arxiv.org/html/2507.20198v3#bib.bib32

https://arxiv.org/html/2507.20198v3#bib.bib33

https://arxiv.org/html/2507.20198v3#bib.bib2

https://arxiv.org/html/2507.20198v3#bib.bib34

https://arxiv.org/html/2507.20198v3#bib.bib35

https://arxiv.org/html/2507.20198v3#bib.bib36

https://arxiv.org/html/2507.20198v3#bib.bib37

https://arxiv.org/html/2507.20198v3#bib.bib38

https://arxiv.org/html/2507.20198v3#bib.bib39

https://arxiv.org/html/2507.20198v3#bib.bib40

] , a major advantage of token compression is its ability to be applied as a post-optimization technique without requiring expensive retraining. These methods typically operate by first establishing a specialized metric to evaluate token importance, then performing a corresponding pruning or compression. By significantly accelerating inference and reducing memory consumption, these techniques enable the practical deployment of MLLMs in real-world applications  [

https://arxiv.org/html/2507.20198v3#bib.bib41

https://arxiv.org/html/2507.20198v3#bib.bib38

https://arxiv.org/html/2507.20198v3#bib.bib39

https://arxiv.org/html/2507.20198v3#bib.bib42

https://arxiv.org/html/2507.20198v3#bib.bib43

Recent extensive research demonstrates that token compression substantially enhances inference efficiency, driving the continuous development of diverse compression strategies and sophisticated methodologies  [

https://arxiv.org/html/2507.20198v3#bib.bib44

https://arxiv.org/html/2507.20198v3#bib.bib45

https://arxiv.org/html/2507.20198v3#bib.bib28

https://arxiv.org/html/2507.20198v3#bib.bib27

https://arxiv.org/html/2507.20198v3#bib.bib46

https://arxiv.org/html/2507.20198v3#bib.bib18

https://arxiv.org/html/2507.20198v3#bib.bib29

https://arxiv.org/html/2507.20198v3#bib.bib47

https://arxiv.org/html/2507.20198v3#bib.bib48

https://arxiv.org/html/2507.20198v3#bib.bib49

https://arxiv.org/html/2507.20198v3#bib.bib26

https://arxiv.org/html/2507.20198v3#bib.bib50

https://arxiv.org/html/2507.20198v3#bib.bib51

https://arxiv.org/html/2507.20198v3#bib.bib52

https://arxiv.org/html/2507.20198v3#bib.bib25

] . However, the inherent heterogeneity of multimodal data means that redundancy manifests differently across modalities. Unlike textual prompts, where redundancy is primarily in syntactic or semantic, visual and auditory data exhibit unique structural properties. For instance, high-resolution images contain strong local correlations, while video streams feature extensive spatiotemporal redundancy across frames, and audio signals often contain extended segments of silence or stationary noise. Consequently, most existing methods focus on compressing one or two specific modalities.

Significant strides have been made in compressing tokens in text LLMs. For instance,  [

https://arxiv.org/html/2507.20198v3#bib.bib53

]  has thoroughly explored prompt compression for text LLMs, highlighting advancements in this domain. In MLLMs, position paper  [

https://arxiv.org/html/2507.20198v3#bib.bib54

]  has begun to broaden our understanding, emphasizing that token compression offers benefits beyond mere efficiency. Furthermore, some researchers argue that the focus of research for efficient AI is shifting from model-centric compression to data-centric compression  [

https://arxiv.org/html/2507.20198v3#bib.bib55

] . However, there has not yet been a systematic classification of token compression methods specifically for MLLMs, leaving an opportunity for a comprehensive survey in this area.

Motivated by the critical need for efficiency in MLLMs and a desire to address this current research fragmentation, this work presents the first comprehensive, structured survey of long-context token compression techniques. We systematically categorize existing approaches according to their primary modality focus:

Image-centric  token compression addresses inherent spatial redundancy, leveraging the fact that neighboring patches usually represent similar textures or colors;

Video-centric  token compression targets spatiotemporal redundancy, mitigating the significant inter-frame correlation where consecutive frames typically share extensive background elements and limited motion;

Audio-centric  token compression mitigates temporal and spectral redundancy, as salient information often concentrates within sparse, brief segments and specific frequency bands amidst long silent pauses or stationary background noise.

Importantly, while acknowledging modality-specific influences on redundancy patterns and optimal compression strategies, we observe that fundamental algorithmic principles frequently transcend individual modalities. Effective compression fundamentally centers on  three  core computational objectives:  importance identification ,  redundancy quantification , and  token merging or pruning . These objectives manifest similarly across visual, temporal, and auditory domains despite distinct structural constraints. Consequently, we further categorize methodologies according to their underlying mechanisms: transform-based, similarity-based, attention-based, and query-based approaches.

This work presents the first structured survey of token compression techniques for MLLMs, a critical step in navigating their inherent computational complexities. By consolidating current progress, this survey identifies key challenges and illuminates promising future research directions, providing a foundational resource for both researchers and developers.

The remaining sections of the article are organized as follows: we will first discuss the architecture of MLLMs in the background section (Section

https://arxiv.org/html/2507.20198v3#S2.SS1

), followed by an examination of how token compression has been utilized in prior methods for large language models (LLMs, Section

https://arxiv.org/html/2507.20198v3#S2.SS2

) and vision transformers (ViTs, Section

https://arxiv.org/html/2507.20198v3#S2.SS3

) Subsequent sections will be dedicated to token compression methods for specific modalities: Section

https://arxiv.org/html/2507.20198v3#S3

for image LLMs, Section

https://arxiv.org/html/2507.20198v3#S4

for video LLMs, and Section

https://arxiv.org/html/2507.20198v3#S5

for audio LLMs. Following this, Section

https://arxiv.org/html/2507.20198v3#S6

will provide insights into token compression research. Finally, Section

https://arxiv.org/html/2507.20198v3#S7

will introduce the broad application space of token compression, followed by the concluding Section

https://arxiv.org/html/2507.20198v3#S8

forked edges, for tree= grow=east, reversed=true, anchor=base west, parent anchor=east, child anchor=west, base=left, font= , rectangle, draw=hidden-black, rounded corners, align=left, minimum width=4em, edge+=darkgray, line width=1pt, s sep=3pt, inner xsep=2pt, inner ysep=4pt, line width=1.1pt, ver/.style=rotate=90, child anchor=north, parent anchor=south, anchor=center,  , where level=1text width=10.5em,font=,, where level=2text width=11.5em,font=,, where level=3text width=12em,font=,, where level=4text width=50em,font=,, [ Multimodal Token Compression , ver, [    Image  (§

https://arxiv.org/html/2507.20198v3#S3

)   ,ver [  Transformation-based        (§

https://arxiv.org/html/2507.20198v3#S3.SS1

)    [ InternVL1.5  [

https://arxiv.org/html/2507.20198v3#bib.bib30

] , NVLM  [

https://arxiv.org/html/2507.20198v3#bib.bib31

] , Qwen2-VL  [

https://arxiv.org/html/2507.20198v3#bib.bib32

] , Qwen2.5-VL  [

https://arxiv.org/html/2507.20198v3#bib.bib33

] , LaCo  [

https://arxiv.org/html/2507.20198v3#bib.bib56

] , LLaVA-OneVision  [

https://arxiv.org/html/2507.20198v3#bib.bib2

] ,  LLaVA-Video  [

https://arxiv.org/html/2507.20198v3#bib.bib34

] , Seed1.5-VL  [

https://arxiv.org/html/2507.20198v3#bib.bib57

https://arxiv.org/html/2507.20198v3#bib.bib35

] , DeCo  [

https://arxiv.org/html/2507.20198v3#bib.bib36

] , SlowFast-LLaVA  [

https://arxiv.org/html/2507.20198v3#bib.bib58

] , PruMerge+  [

https://arxiv.org/html/2507.20198v3#bib.bib29

] ,  C-Abstractor  [

https://arxiv.org/html/2507.20198v3#bib.bib37

] , MobileVLM  [

https://arxiv.org/html/2507.20198v3#bib.bib38

] , MobileVLM V2  [

https://arxiv.org/html/2507.20198v3#bib.bib39

] , LLaMA-VID  [

https://arxiv.org/html/2507.20198v3#bib.bib40

] , Dynamic-VLM  [

https://arxiv.org/html/2507.20198v3#bib.bib59

]  , leaf, text width=44em] ] [    Similarity-based         (§

https://arxiv.org/html/2507.20198v3#S3.SS2

)    [ ToMe  [

https://arxiv.org/html/2507.20198v3#bib.bib60

] , FOLDER  [

https://arxiv.org/html/2507.20198v3#bib.bib61

] , DivPrune  [

https://arxiv.org/html/2507.20198v3#bib.bib28

] , AuroraCap  [

https://arxiv.org/html/2507.20198v3#bib.bib45

] , TopV  [

https://arxiv.org/html/2507.20198v3#bib.bib49

] , Skip-Vision  [

https://arxiv.org/html/2507.20198v3#bib.bib62

] , DyMU  [

https://arxiv.org/html/2507.20198v3#bib.bib63

] ,  Dynamic-VLM  [

https://arxiv.org/html/2507.20198v3#bib.bib59

] , STTM  [

https://arxiv.org/html/2507.20198v3#bib.bib64

] , VisionZip  [

https://arxiv.org/html/2507.20198v3#bib.bib18

] , VisPruner  [

https://arxiv.org/html/2507.20198v3#bib.bib65

] , PuMer  [

https://arxiv.org/html/2507.20198v3#bib.bib48

] , iLLaVA  [

https://arxiv.org/html/2507.20198v3#bib.bib66

] ,  G-Prune  [

https://arxiv.org/html/2507.20198v3#bib.bib67

https://arxiv.org/html/2507.20198v3#bib.bib68

]  , leaf, text width=44em] ] [    Attention-based         (§

https://arxiv.org/html/2507.20198v3#S3.SS3

)    [  Attention in Encoder:  PruMerge  [

https://arxiv.org/html/2507.20198v3#bib.bib29

] , VisionZip  [

https://arxiv.org/html/2507.20198v3#bib.bib18

] , VisPruner  [

https://arxiv.org/html/2507.20198v3#bib.bib65

] , GlobalCom

https://arxiv.org/html/2507.20198v3#bib.bib69

] ,  MustDrop  [

https://arxiv.org/html/2507.20198v3#bib.bib52

] , VScan  [

https://arxiv.org/html/2507.20198v3#bib.bib47

] , HiRED  [

https://arxiv.org/html/2507.20198v3#bib.bib70

] , FiCoCo  [

https://arxiv.org/html/2507.20198v3#bib.bib71

]   Attention in Decoder:  FastV  [

https://arxiv.org/html/2507.20198v3#bib.bib26

] , PyramidDrop  [

https://arxiv.org/html/2507.20198v3#bib.bib72

https://arxiv.org/html/2507.20198v3#bib.bib73

] , FitPrune  [

https://arxiv.org/html/2507.20198v3#bib.bib74

https://arxiv.org/html/2507.20198v3#bib.bib75

] ,  ATP-LLaVA  [

https://arxiv.org/html/2507.20198v3#bib.bib76

] , CoreMatching  [

https://arxiv.org/html/2507.20198v3#bib.bib77

] , ZipVL  [

https://arxiv.org/html/2507.20198v3#bib.bib46

] , HoliTom  [

https://arxiv.org/html/2507.20198v3#bib.bib16

] , TokenCarve  [

https://arxiv.org/html/2507.20198v3#bib.bib78

] , SparseVLM  [

https://arxiv.org/html/2507.20198v3#bib.bib51

] ,  MustDrop  [

https://arxiv.org/html/2507.20198v3#bib.bib52

] , VScan  [

https://arxiv.org/html/2507.20198v3#bib.bib47

] , FrameFusion  [

https://arxiv.org/html/2507.20198v3#bib.bib79

] , iLLaVA  [

https://arxiv.org/html/2507.20198v3#bib.bib66

] , G-Search  [

https://arxiv.org/html/2507.20198v3#bib.bib80

] , FiCoCo  [

https://arxiv.org/html/2507.20198v3#bib.bib71

https://arxiv.org/html/2507.20198v3#bib.bib68

]  , leaf, text width=44em] ] [    Query-based          (§

https://arxiv.org/html/2507.20198v3#S3.SS4

)    [  Token Distillation:  InstructBLIP  [

https://arxiv.org/html/2507.20198v3#bib.bib1

] , BLIP-2  [

https://arxiv.org/html/2507.20198v3#bib.bib81

] , mPLUG-Owl  [

https://arxiv.org/html/2507.20198v3#bib.bib82

] , Minigpt-4  [

https://arxiv.org/html/2507.20198v3#bib.bib83

] , Flamingo  [

https://arxiv.org/html/2507.20198v3#bib.bib84

] ,  Qwen-VL  [

https://arxiv.org/html/2507.20198v3#bib.bib4

] , LLaMA-VID  [

https://arxiv.org/html/2507.20198v3#bib.bib40

] , LLaVA-Mini  [

https://arxiv.org/html/2507.20198v3#bib.bib85

] , VoCo-LLaMA  [

https://arxiv.org/html/2507.20198v3#bib.bib86

] , Victor  [

https://arxiv.org/html/2507.20198v3#bib.bib87

]   Cross-Modal Selection:  SparseVLM  [

https://arxiv.org/html/2507.20198v3#bib.bib51

] , AdaFV  [

https://arxiv.org/html/2507.20198v3#bib.bib88

] , TRIM  [

https://arxiv.org/html/2507.20198v3#bib.bib89

]  , leaf, text width=44em] ] ] [    Video  (§

https://arxiv.org/html/2507.20198v3#S4

)   , ver [  Transformation-based        (§

https://arxiv.org/html/2507.20198v3#S4.SS1

)    [ PLLaVA  [

https://arxiv.org/html/2507.20198v3#bib.bib3

] , Video-ChatGPT  [

https://arxiv.org/html/2507.20198v3#bib.bib90

] , LongVLM  [

https://arxiv.org/html/2507.20198v3#bib.bib91

] , VideoLLaMA 2  [

https://arxiv.org/html/2507.20198v3#bib.bib10

] , SlowFast-LLaVA-1.5  [

https://arxiv.org/html/2507.20198v3#bib.bib92

] ,  VideoChat-Flash  [

https://arxiv.org/html/2507.20198v3#bib.bib93

] , MobileVLM  [

https://arxiv.org/html/2507.20198v3#bib.bib38

]  , leaf3, text width=44em] ] [    Similarity-based         (§

https://arxiv.org/html/2507.20198v3#S4.SS2

)    [ Chat-UniVi  [

https://arxiv.org/html/2507.20198v3#bib.bib94

] , PruneVid  [

https://arxiv.org/html/2507.20198v3#bib.bib27

] , FastVID  [

https://arxiv.org/html/2507.20198v3#bib.bib44

] , HoliTom  [

https://arxiv.org/html/2507.20198v3#bib.bib16

] , DyCoke  [

https://arxiv.org/html/2507.20198v3#bib.bib17

] , FrameFusion  [

https://arxiv.org/html/2507.20198v3#bib.bib79

] ,  LongVU  [

https://arxiv.org/html/2507.20198v3#bib.bib95

https://arxiv.org/html/2507.20198v3#bib.bib96

] , DynTok  [

https://arxiv.org/html/2507.20198v3#bib.bib97

] , MovieChat  [

https://arxiv.org/html/2507.20198v3#bib.bib98

] , VideoChat-Flash  [

https://arxiv.org/html/2507.20198v3#bib.bib93

] , Video-XL  [

https://arxiv.org/html/2507.20198v3#bib.bib99

] ,  TimeChat-Online  [

https://arxiv.org/html/2507.20198v3#bib.bib100

https://arxiv.org/html/2507.20198v3#bib.bib101

] , LLaVA-Scissor  [

https://arxiv.org/html/2507.20198v3#bib.bib102

https://arxiv.org/html/2507.20198v3#bib.bib103

]  , leaf3, text width=44em] ] [    Attention-based         (§

https://arxiv.org/html/2507.20198v3#S4.SS3

)    [  Attention in Encoder:  PruMerge  [

https://arxiv.org/html/2507.20198v3#bib.bib29

] , VisionZip  [

https://arxiv.org/html/2507.20198v3#bib.bib18

] , VisPruner  [

https://arxiv.org/html/2507.20198v3#bib.bib65

] , MustDrop  [

https://arxiv.org/html/2507.20198v3#bib.bib52

] , FiCoCo  [

https://arxiv.org/html/2507.20198v3#bib.bib71

]   Attention in Decoder:  FastV  [

https://arxiv.org/html/2507.20198v3#bib.bib26

] , PyramidDrop  [

https://arxiv.org/html/2507.20198v3#bib.bib72

https://arxiv.org/html/2507.20198v3#bib.bib73

] , CoreMatching  [

https://arxiv.org/html/2507.20198v3#bib.bib77

] , ZipVL  [

https://arxiv.org/html/2507.20198v3#bib.bib46

] ,  SparseVLM  [

https://arxiv.org/html/2507.20198v3#bib.bib51

] , FastVID  [

https://arxiv.org/html/2507.20198v3#bib.bib44

] , MustDrop  [

https://arxiv.org/html/2507.20198v3#bib.bib52

] , HoliTom  [

https://arxiv.org/html/2507.20198v3#bib.bib16

] , VScan  [

https://arxiv.org/html/2507.20198v3#bib.bib47

] , FrameFusion  [

https://arxiv.org/html/2507.20198v3#bib.bib79

] ,  iLLaVA  [

https://arxiv.org/html/2507.20198v3#bib.bib66

] , G-Search  [

https://arxiv.org/html/2507.20198v3#bib.bib80

] , FiCoCo  [

https://arxiv.org/html/2507.20198v3#bib.bib71

https://arxiv.org/html/2507.20198v3#bib.bib103

]  , leaf3, text width=44em] ] [    Query-based          (§

https://arxiv.org/html/2507.20198v3#S4.SS4

)    [  Token Distillation:  Token Turing Machines  [

https://arxiv.org/html/2507.20198v3#bib.bib104

] , BLIP-3-Video  [

https://arxiv.org/html/2507.20198v3#bib.bib105

] , Long-VMNet  [

https://arxiv.org/html/2507.20198v3#bib.bib106

] ,  STORM  [

https://arxiv.org/html/2507.20198v3#bib.bib107

] , LinVT  [

https://arxiv.org/html/2507.20198v3#bib.bib108

]  LLaMA-VID  [

https://arxiv.org/html/2507.20198v3#bib.bib40

] , LLaVA-Mini  [

https://arxiv.org/html/2507.20198v3#bib.bib85

] , VideoChat-Flash  [

https://arxiv.org/html/2507.20198v3#bib.bib93

]   Cross-Modal Selection:  LongVU  [

https://arxiv.org/html/2507.20198v3#bib.bib95

]  , leaf3, text width=44em] ] ] [    Audio  (§

https://arxiv.org/html/2507.20198v3#S5

)   , ver [  Transformation-based        (§

https://arxiv.org/html/2507.20198v3#S5.SS1

)    [ HTS-AT  [

https://arxiv.org/html/2507.20198v3#bib.bib109

] , SLAM-ASR  [

https://arxiv.org/html/2507.20198v3#bib.bib110

] , LLaMA-Omni  [

https://arxiv.org/html/2507.20198v3#bib.bib111

] , SpeechVerse  [

https://arxiv.org/html/2507.20198v3#bib.bib112

] , Qwen2-audio  [

https://arxiv.org/html/2507.20198v3#bib.bib113

] ,  Qwen2.5-Omni  [

https://arxiv.org/html/2507.20198v3#bib.bib5

] , Baichuan-Audio  [

https://arxiv.org/html/2507.20198v3#bib.bib114

] , Llama-AVSR  [

https://arxiv.org/html/2507.20198v3#bib.bib115

] , Llama-MTSK  [

https://arxiv.org/html/2507.20198v3#bib.bib116

] , OSUM  [

https://arxiv.org/html/2507.20198v3#bib.bib117

] ,  LUCY  [

https://arxiv.org/html/2507.20198v3#bib.bib118

]  , leaf5, text width=44em] ] [    Similarity-based         (§

https://arxiv.org/html/2507.20198v3#S5.SS2

)    [ A-ToMe  [

https://arxiv.org/html/2507.20198v3#bib.bib119

]  , leaf5, text width=44em] ] [    Attention-based         (§

https://arxiv.org/html/2507.20198v3#S5.SS3

)    [  Attention in Encoder:  Top-K  [

https://arxiv.org/html/2507.20198v3#bib.bib120

]   Attention in Decoder:  SpeechPrune  [

https://arxiv.org/html/2507.20198v3#bib.bib121

]  , leaf5, text width=44em] ] [    Query-based          (§

https://arxiv.org/html/2507.20198v3#S5.SS4

)    [  Token Distillation:  Video-LLaMA  [

https://arxiv.org/html/2507.20198v3#bib.bib7

] , SALMONN  [

https://arxiv.org/html/2507.20198v3#bib.bib122

] , video-SALMONN  [

https://arxiv.org/html/2507.20198v3#bib.bib123

] , Typhoon 2  [

https://arxiv.org/html/2507.20198v3#bib.bib124

] ,  MMCE-Qformer  [

https://arxiv.org/html/2507.20198v3#bib.bib125

] , MMS-LLaMA  [

https://arxiv.org/html/2507.20198v3#bib.bib126

]   Cross-Modal Selection:  SpeechPrune  [

https://arxiv.org/html/2507.20198v3#bib.bib121

]  , leaf5, text width=44em] ] ] ]

Figure 3:  Taxonomy of Multimodal Token Compression.  Our classification organizes existing methods by their dominant data modality, accounting for inherent differences in redundancy across modalities. This is further refined by a dissection of their underlying mechanisms, enabling researchers to quickly pinpoint methods tailored to specific research domains.

2  Background

2.1  Multimodal Architecture

The general multimodal large language model (MLLM) framework (see Figure

https://arxiv.org/html/2507.20198v3#S1.F2

), consists of three core components: (1) a modality-specific encoder ( g g italic_g ), (2) a projector module ( P P italic_P ), and (3) a pre-trained large language model (LLM).

The process begins with the modality encoder,  g g italic_g , which is responsible for processing a given input, such as a visual or audio signal. This encoder compresses the high-dimensional raw data into a sequence of compact and semantically meaningful patch embeddings. For an input image  X v X_{v} italic_X start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT  and an audio  X a X_{a} italic_X start_POSTSUBSCRIPT italic_a end_POSTSUBSCRIPT , this can be expressed as:

Z v = g  ( X v ) , Z a = g  ( X a ) . Z_{v}=g(X_{v}),\quad Z_{a}=g(X_{a}). italic_Z start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT = italic_g ( italic_X start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT ) , italic_Z start_POSTSUBSCRIPT italic_a end_POSTSUBSCRIPT = italic_g ( italic_X start_POSTSUBSCRIPT italic_a end_POSTSUBSCRIPT ) .   (1)

The encoding function  g g italic_g  is a flexible component that can be specialized for various modalities, including vision, audio, sensor data, etc. Widely adopted encoders implementing this function include:

Vision encoders:  CLIP  [

https://arxiv.org/html/2507.20198v3#bib.bib127

] , SigLIP  [

https://arxiv.org/html/2507.20198v3#bib.bib128

] , DINO  [

https://arxiv.org/html/2507.20198v3#bib.bib129

https://arxiv.org/html/2507.20198v3#bib.bib130

] , and ViT  [

https://arxiv.org/html/2507.20198v3#bib.bib33

Audio encoders:  Whisper  [

https://arxiv.org/html/2507.20198v3#bib.bib131

]  and Audio-CLIP  [

https://arxiv.org/html/2507.20198v3#bib.bib132

Subsequently, the encoded embeddings ( Z v Z_{v} italic_Z start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT  or  Z a Z_{a} italic_Z start_POSTSUBSCRIPT italic_a end_POSTSUBSCRIPT ) are transformed by the projector module,  P P italic_P . The primary role of this module is to bridge the modality gap by mapping the embeddings into the same latent space as the text embeddings of LLM.

H v = P  ( Z v ) , H a = P  ( Z a ) . H_{v}=P(Z_{v}),\quad H_{a}=P(Z_{a}). italic_H start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT = italic_P ( italic_Z start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT ) , italic_H start_POSTSUBSCRIPT italic_a end_POSTSUBSCRIPT = italic_P ( italic_Z start_POSTSUBSCRIPT italic_a end_POSTSUBSCRIPT ) .   (2)

The output of the projector, a sequence of projected embeddings, can then be seamlessly concatenated with the text prompts and fed into the LLM.

The pre-trained LLM  [

https://arxiv.org/html/2507.20198v3#bib.bib12

https://arxiv.org/html/2507.20198v3#bib.bib13

https://arxiv.org/html/2507.20198v3#bib.bib14

]  forms the core of the framework, with its large-scale parameters providing emergent capabilities such as zero-shot generalization and in-context learning. The LLM receives a composite input sequence formed by concatenating the projected multimodal embeddings  H v H_{v} italic_H start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT  and  H a H_{a} italic_H start_POSTSUBSCRIPT italic_a end_POSTSUBSCRIPT , as well as the textual prompt embeddings  H q H_{q} italic_H start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT . The textual prompt  X q X_{q} italic_X start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT  is first converted into embeddings  H q H_{q} italic_H start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT  by an integrated tokenizer.

The LLM then generates a response sequence  Y a Y_{a} italic_Y start_POSTSUBSCRIPT italic_a end_POSTSUBSCRIPT  through autoregressive decoding:

p  ( Y a | H v , H a , H q ) = ∏ i = 1 L p  ( y i | H v , H a , H q , y < i ) , p(Y_{a}|H_{v},H_{a},H_{q})=\prod_{i=1}^{L}p(y_{i}|H_{v},H_{a},H_{q},y_{<i}), italic_p ( italic_Y start_POSTSUBSCRIPT italic_a end_POSTSUBSCRIPT | italic_H start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT , italic_H start_POSTSUBSCRIPT italic_a end_POSTSUBSCRIPT , italic_H start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT ) = ∏ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_L end_POSTSUPERSCRIPT italic_p ( italic_y start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT | italic_H start_POSTSUBSCRIPT italic_v end_POSTSUBSCRIPT , italic_H start_POSTSUBSCRIPT italic_a end_POSTSUBSCRIPT , italic_H start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT , italic_y start_POSTSUBSCRIPT < italic_i end_POSTSUBSCRIPT ) ,   (3)

where  L L italic_L  signifies the output sequence length.

The high dimensionality of multimodal data and the resulting computational demands remain a challenge. As shown in Figure

https://arxiv.org/html/2507.20198v3#S1.F2

, the token sequence processed comprises a mix of system prompt, multimodal context, textual instruction. In most reasoning tasks, multimodal tokens can account for over  80 % 80\% 80 %  of the total sequence length, thereby forming the primary computational bottleneck. This bottleneck is a critical obstacle to scaling MLLMs and achieving efficient inference. Consequently, a key strategy for optimizing computational efficiency involves employing specialized projector architectures. These projectors are designed to reduce the number of multimodal tokens while preserving their semantic fidelity, thus mitigating the computational burden.

While the MLLM architecture presents unique challenges, token compression research has been explored for both encoders and LLMs independently. Therefore, the subsequent sections will first dive into token compression techniques relevant to these individual components, paving the way for more efficient multimodal models. Specifically, Section

https://arxiv.org/html/2507.20198v3#S2.SS2

will focus on token compression methods for large language models (LLMs), and Section

https://arxiv.org/html/2507.20198v3#S2.SS3

will explore techniques for vision transformers (ViTs).

2.2  Large Language Model Token Compression

The backbone of modern MLLMs is often built upon and fine-tuned from powerful text-based LLMs. As a foundational component, a solid understanding of token compression techniques developed for text LLMs is crucial, as they offer an accurate and lightweight solution for handling real-world long-context scenarios, such as understanding an entire book or a code repository. Within the domain of large language models, these methods are frequently termed prompt compression  [

https://arxiv.org/html/2507.20198v3#bib.bib53

AutoCompressor  [

https://arxiv.org/html/2507.20198v3#bib.bib133

]  condenses context into summary vectors as soft prompts. Extensible Tokenization  [

https://arxiv.org/html/2507.20198v3#bib.bib134

]  employs intermediate modules to compress embeddings, while SentenceVAE  [

https://arxiv.org/html/2507.20198v3#bib.bib135

]  represents sentences with single tokens. Selective Context  [

https://arxiv.org/html/2507.20198v3#bib.bib136

]  employs self-information metrics to eliminate low-information tokens. LLMLingua  [

https://arxiv.org/html/2507.20198v3#bib.bib137

https://arxiv.org/html/2507.20198v3#bib.bib138

https://arxiv.org/html/2507.20198v3#bib.bib139

]  series utilizes hierarchical token pruning with instruction tuning and further introduces LongLLMLingua  [

https://arxiv.org/html/2507.20198v3#bib.bib138

]  to mitigate position decay through semantic density ranking. In parallel, query-guided methods like QUITO  [

https://arxiv.org/html/2507.20198v3#bib.bib140

]  and QUITO-X  [

https://arxiv.org/html/2507.20198v3#bib.bib141

]  leverage attention scores or information bottleneck theory for relevance-based filtering. AdaComp  [

https://arxiv.org/html/2507.20198v3#bib.bib142

]  implements adaptive extraction governed by query complexity predictors. Concept Distillation  [

https://arxiv.org/html/2507.20198v3#bib.bib143

]  employs Abstract Meaning Representation (AMR) graphs to distill key concepts, whereas xRAG  [

https://arxiv.org/html/2507.20198v3#bib.bib144

]  collapses documents into single-token representations. ICAE  [

https://arxiv.org/html/2507.20198v3#bib.bib145

]  encodes context into discrete memory slots. Recursive frameworks including RCC  [

https://arxiv.org/html/2507.20198v3#bib.bib146

]  and XL3M  [

https://arxiv.org/html/2507.20198v3#bib.bib147

]  generate piecewise summaries through relevant fusion. Additionally, SoftPromptComp  [

https://arxiv.org/html/2507.20198v3#bib.bib148

]  fuses natural language prompts with dynamic embeddings, while PromptIntern  [

https://arxiv.org/html/2507.20198v3#bib.bib149

]  internalizes task instructions into model parameters via phased training.

While these text-centric token compression techniques have demonstrated notable efficacy, their direct application to MLLMs faces fundamental challenges. The inherent heterogeneity of multimodal data introduces distinct redundancy patterns that are not present in unimodal text. These include, but are not limited to, spatial correlations in high-resolution images, spatiotemporal continuity in video sequences, and spectral-temporal locality in audio streams. These specialized redundancies necessitate the development of dedicated compression strategies. Consequently, this survey systematically reviews emerging token compression methodologies designed specifically for MLLMs that effectively reduce token redundancy while preserving task performance.

2.3  Vision Transformer Token Compression

Visual token compression, originally pioneered in vision transformers (ViTs)  [

https://arxiv.org/html/2507.20198v3#bib.bib24

https://arxiv.org/html/2507.20198v3#bib.bib150

https://arxiv.org/html/2507.20198v3#bib.bib151

https://arxiv.org/html/2507.20198v3#bib.bib152

https://arxiv.org/html/2507.20198v3#bib.bib153

https://arxiv.org/html/2507.20198v3#bib.bib154

https://arxiv.org/html/2507.20198v3#bib.bib155

https://arxiv.org/html/2507.20198v3#bib.bib156

] , offers insights for addressing analogous challenges in MLLMs.

Spatial redundancy manifests in ViTs through adjacent image patches, where not all tokens contribute equally to classification outcomes, compounded by semantic imbalance: foreground objects demand disproportionate computational resources compared to homogeneous backgrounds. To mitigate these issues, visual token compression techniques are employed to reduce computational overhead while maintaining model accuracy substantially.

Foundational approaches, including DynamicViT  [

https://arxiv.org/html/2507.20198v3#bib.bib19

]  and EViT  [

https://arxiv.org/html/2507.20198v3#bib.bib20

] , quantify token relevance through attention scores, dynamically pruning low-saliency tokens. Complementary techniques like ToMe  [

https://arxiv.org/html/2507.20198v3#bib.bib21

]  and TokenLearner  [

https://arxiv.org/html/2507.20198v3#bib.bib22

]  either merge semantically similar tokens using similarity metrics or generate compact token sets via learned spatial attention mechanisms. DeiT  [

https://arxiv.org/html/2507.20198v3#bib.bib23

]  employs lightweight ‘student’ heads to predict categorical labels from compressed token subsets. Furthermore, methods such as MADTP  [

https://arxiv.org/html/2507.20198v3#bib.bib157

]  leverage cross-modal alignment to filter tokens.

The preceding analysis demonstrates that ViT token compression methodologies offer substantive inspiration for token reduction in MLLMs. However, MLLMs possess not only multimodal tokens encoding low-level features but also text tokens conveying high-level abstractions, coupled with significantly longer token sequences. Consequently, token compression in MLLMs presents greater challenges than in ViT while being increasingly critical for computational efficiency. Therefore, this survey analyzes the evolution and future directions of token compression techniques for MLLMs operating in long-context multimodal environments.

TABLE I:  Four Categories of Methods Based on Intrinsic Mechanisms: Diagram, Summary, and Pros & Cons.   Method   Transformation-based   Similarity-based   Attention-based   Query-based   Diagram   Summary   Transform tokens into a more compact form   Compress by merging or grouping similar tokens   Remove less attentive tokens via attention sparsity   Use external queries to guide token compression   Pros   Preserve the structural representation of information well   Simplify processing, flexibility in choosing where to compress   Dynamically prune tokens by relevance; tie to original computation, boosting interpretability   Suitable for specific and video tasks, as compressed information is more relevant and concise   Cons   Limited by the transformation method, compression rate isn’t flexible enough   May lose fine-grained info if tokens over-generalized; poor structural feature retention if token differentiation is low   Explicit attention score calculation might be incompatible with mainstream acceleration libraries   Not user-friendly for multi-turn conversations; requires re-condensing information

3  Image-centric Token Compression

Multimodal long context token compression methods generally fall into  four  categories based on their underlying mechanisms:  transformation-based  approaches directly transform the cross-modality information to compress tokens by altering their scale or representation;  similarity-based  techniques reduce tokens by leveraging the inherent resemblances between them;  attention-based  strategies exploit the sparsity of attention within the multimodal data to guide compression; and  query-based  methods selectively refine multimodal information, guided by prompts, to distill the most irrelevant tokens. Each of these methods has its own set of advantages and disadvantages, which are summarized in Table

https://arxiv.org/html/2507.20198v3#S2.T1

. Representative image-centric token compression methods are further compared in Table

https://arxiv.org/html/2507.20198v3#S3.T2

3.1  Transformation-based Image-centric Compression

Transformation-based image-centric compression methods leverage the spatial redundancy inherent in 2D image representations. Some image token compression techniques are derived from image downsampling operations (

, pooling, bilinear interpolation). Based on the specific transformation method, these can be broadly categorized as follows:

3.1.1  Pixel Unshuffle

Pixel unshuffle is the inverse operation of pixel shuffle. It transforms a feature map from a high spatial resolution with a small number of channels into a lower-resolution feature map with a larger number of channels. This effectively reduces the number of tokens. The transformation can be mathematically expressed as:

Pixel Unshuffle:   H × W × D → H r × W r × ( D ⋅ r 2 ) , \displaystyle H\times W\times D\to\frac{H}{r}\times\frac{W}{r}\times(D\cdot r^{2}), italic_H × italic_W × italic_D → divide start_ARG italic_H end_ARG start_ARG italic_r end_ARG × divide start_ARG italic_W end_ARG start_ARG italic_r end_ARG × ( italic_D ⋅ italic_r start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT ) ,   (4)

where  H H italic_H ,  W W italic_W  denote the height and width of the token grid,  D D italic_D  is the hidden dimension of each token, and  r r italic_r  is the downsampling ratio.

Recent works like InternVL series  [

https://arxiv.org/html/2507.20198v3#bib.bib30

https://arxiv.org/html/2507.20198v3#bib.bib158

https://arxiv.org/html/2507.20198v3#bib.bib159

https://arxiv.org/html/2507.20198v3#bib.bib160

] , Qwen2 series  [

https://arxiv.org/html/2507.20198v3#bib.bib32

https://arxiv.org/html/2507.20198v3#bib.bib33

] , and NVLM  [

https://arxiv.org/html/2507.20198v3#bib.bib31

]  utilize pixel unshuffle to reduce the number of tokens generated by the vision tower by a factor of one-quarter. Subsequently, an MLP is employed to align the visual dimension with the text dimension, addressing the mismatch in the hidden dimension.

3.1.2  Spatial Pooling / Interpolation

Unlike pixel unshuffle, pooling and interpolation directly perform 2D downsampling on tokens, without altering the hidden dimension. This process can be defined as:

Pooling / Interpolation:   H × W × D → H S × W S × D , \displaystyle H\times W\times D\to\frac{H}{S}\times\frac{W}{S}\times D, italic_H × italic_W × italic_D → divide start_ARG italic_H end_ARG start_ARG italic_S end_ARG × divide start_ARG italic_W end_ARG start_ARG italic_S end_ARG × italic_D ,   (5)

where  S S italic_S  is the downsampling factor.

LLaVA-OneVision  [

https://arxiv.org/html/2507.20198v3#bib.bib2

]  employs bilinear interpolation for 2D downsampling of aligned tokens, while LLaVA-Video  [

https://arxiv.org/html/2507.20198v3#bib.bib34

]  uses average pooling for downsampling. M

https://arxiv.org/html/2507.20198v3#bib.bib35

]  utilizes a simple pooling operation to learn an inherently multi-granular representation during training. This allows the model to achieve comparable performance with fewer tokens during inference, effectively addressing efficiency concerns. DeCo  [

https://arxiv.org/html/2507.20198v3#bib.bib36

]  argues that the Q-former  [

https://arxiv.org/html/2507.20198v3#bib.bib1

https://arxiv.org/html/2507.20198v3#bib.bib81

]  is an inefficient visual compressor and similarly achieves token compression through a straightforward average pooling approach, leading to improved convergence efficiency and performance.

3.1.3  Spatial Convolution

Convolutional operations offer a more sophisticated approach to token compression compared to simple pooling or interpolation, by learning to abstract local information while reducing spatial dimensions. The transformation can be expressed as:

Convolution:   H × W × D i  n → H S × W S × D o  u  t , \displaystyle H\times W\times D_{in}\to\frac{H}{S}\times\frac{W}{S}\times D_{out}, italic_H × italic_W × italic_D start_POSTSUBSCRIPT italic_i italic_n end_POSTSUBSCRIPT → divide start_ARG italic_H end_ARG start_ARG italic_S end_ARG × divide start_ARG italic_W end_ARG start_ARG italic_S end_ARG × italic_D start_POSTSUBSCRIPT italic_o italic_u italic_t end_POSTSUBSCRIPT ,   (6)

where S is the stride, which determines the downsampling factor, and  D i  n D_{in} italic_D start_POSTSUBSCRIPT italic_i italic_n end_POSTSUBSCRIPT ,  D o  u  t D_{out} italic_D start_POSTSUBSCRIPT italic_o italic_u italic_t end_POSTSUBSCRIPT  represent the input and output channel dimensions, respectively.

Honeybee  [

https://arxiv.org/html/2507.20198v3#bib.bib37

]  proposes the C-Abstractor, which uses convolution to extract and compress token information while preserving locality. MobileVLM  [

https://arxiv.org/html/2507.20198v3#bib.bib38

] , on the other hand, employs an LDP module that utilizes depth-wise convolution to reduce the number of tokens by  75 % 75\% 75 % .

3.1.4  Comparative Analysis of Transformation Methods

These transformation-based image-centric compression methods effectively utilize all image tokens while consciously preserving the spatial local information of 2D features. Pixel unshuffle, pooling, and interpolation are inherently parameter-free, thus introducing no additional weight overhead, which is a key advantage. In contrast, methods that use convolution introduce trainable weights to learn a more sophisticated local abstraction.

Another notable difference lies in how these methods handle feature dimensions: pixel unshuffle typically alters the hidden dimension, necessitating a subsequent trained MLP to align with the text dimension. Conversely, pooling and interpolation can be implemented in a training-free manner as they operate directly on the aligned token dimension.

By extracting more condensed information, they achieve a superior balance between performance and efficiency. However, due to the inherent characteristics of 2D downsampling, their token compression ratios are typically limited to a few specific magnitudes, with a  25 % 25\% 25 %  compression rate being the most common.

3.2  Similarity-based Image-centric Compression

Similarity-based image-centric compression methods reduce the number of visual tokens by identifying and merging similar tokens based on their distance or similarity in an implicit space. This typically involves selecting representative cluster-center tokens to encapsulate visual information.

Early works in this area include ToMe  [

https://arxiv.org/html/2507.20198v3#bib.bib60

] , an acceleration method for ViTs. ToMe introduces a token merge module between the attention and MLP blocks, calculating token similarity and merging similar tokens via bipartite soft matching. In the context of MLLMs, FOLDER  [

https://arxiv.org/html/2507.20198v3#bib.bib61

]  employs a similar approach, inserting a token merge module within the last attention block of the vision encoder. This reduces the number of tokens that were subsequently passed to the LLM decoder. DivPrune  [

https://arxiv.org/html/2507.20198v3#bib.bib28

]  reframes the token compression problem as a Max-Min diversity problem  [

https://arxiv.org/html/2507.20198v3#bib.bib161

] , aiming to select a subset of tokens with maximal internal differences. AuroraCap  [

https://arxiv.org/html/2507.20198v3#bib.bib45

]  adopts a strategy consistent with ToMe, performing token merging within each attention and MLP block of the vision tower. This progressively reduces the number of tokens throughout the ViT model. While the aforementioned methods primarily leverage similarity-based clustering of tokens within the ViT, TopV  [

https://arxiv.org/html/2507.20198v3#bib.bib49

]  extends this principle to compress tokens within the LLM layers. TopV comprehensively considers both the similarity and distance functions between features to guide the token compression process, operating directly within the multimodal representation space of the LLM.

3.3  Attention-based Image-centric Compression

Attention-based token compression methods leverage the inherent sparsity of visual feature attention to guide token pruning. Tokens with low attention scores can often be considered removable without significantly impacting the original computation. In vision language models, both the vision encoder and the LLM decoder incorporate transformers. Consequently, attention-based compression strategies can be broadly categorized into those applied within the encoder and those within the decoder.

TABLE II:  Quantitative comparison of training-free token compression methods of MLLMs in image understanding task. We select the most representative results of each presentation method. For all values, higher is better.   Method   #Vision Tokens   Res.   Benchmarks   VQA

GQA   VisWiz   SciQA   VQA T {}^{\text{T}} start_FLOATSUPERSCRIPT T end_FLOATSUPERSCRIPT   POPE   MME   MMB   SEED   LLaVA W {}^{\text{W}} start_FLOATSUPERSCRIPT W end_FLOATSUPERSCRIPT   MM–Vet   BLIP-2  [

https://arxiv.org/html/2507.20198v3#bib.bib81

]   32   224   65.0   41.0   19.6   61.0   42.5   85.3   1293.8   –   46.4   38.1   22.4   IDEFICS-9B  [

https://arxiv.org/html/2507.20198v3#bib.bib162

]   64   224   50.9   38.4   35.5   –   25.9   –   –   48.2   –   –   –   MobileVLM-3B  [

https://arxiv.org/html/2507.20198v3#bib.bib39

]   144   336   –   59.0   –   61.0   47.5   84.9   1288.9   59.6   –   –   –   mPLUG-Owl2  [

https://arxiv.org/html/2507.20198v3#bib.bib82

]   1024   448   79.4   56.1   54.5   68.7   54.3   –   1450.2   64.5   57.8   –   36.2   Video-LLaVA  [

https://arxiv.org/html/2507.20198v3#bib.bib6

]   256   224   74.7   60.3   48.1   66.4   51.8   84.4   –   60.9   –   73.1   32.0   Qwen-VL  [

https://arxiv.org/html/2507.20198v3#bib.bib32

]   256   448   78.8   59.3   35.2   67.1   63.8   –   –   38.2   56.3   –   –   LLaVA-v1.5  [

https://arxiv.org/html/2507.20198v3#bib.bib1

]   576   336   78.5   62.0   50.0   66.8   58.2   85.9   1510.7   64.3   58.6   63.4   30.5   LLaVA-v1.5-7B w/ Token Compression Methods (Training Free)   ToMe  [

https://arxiv.org/html/2507.20198v3#bib.bib60

]   192   336   68.0   54.3   –   –   52.1   –   1563.0   60.5   –   –   –   FastV  [

https://arxiv.org/html/2507.20198v3#bib.bib26

]   192   336   67.1   52.7   –   –   52.5   64.8   1612.0   61.2   57.1   –   27.7   SparseVLM  [

https://arxiv.org/html/2507.20198v3#bib.bib51

]   192   336   75.6   57.6   –   –   56.1   83.6   1721.0   62.5   55.8   –   31.5   MustDrop  [

https://arxiv.org/html/2507.20198v3#bib.bib52

]   192   336   76.0   58.2   51.4   –   56.5   –   1787.0   62.3   –   –   –   PruMerge+  [

https://arxiv.org/html/2507.20198v3#bib.bib29

]   144   336   76.8   –   –   68.3   57.1   84.0   1462.4   64.9   –   –   –   ATP-LLaVA  [

https://arxiv.org/html/2507.20198v3#bib.bib76

]   144   336   76.4   59.5   –   –   –   84.2   1473.9   66.0   57.3   –   31.5   VisionZip++  [

https://arxiv.org/html/2507.20198v3#bib.bib18

]   128   336   76.6   58.9   –   –   57.0   83.7   1823.0   –   55.8   –   32.9   VisPruner  [

https://arxiv.org/html/2507.20198v3#bib.bib65

]   128   336   75.8   58.2   52.7   –   57.0   84.6   1461.4   62.7   –   –   33.7   VisionZip++  [

https://arxiv.org/html/2507.20198v3#bib.bib18

]   64   336   74.2   57.0   –   –   56.0   80.9   1756.0   –   53.4   –   30.2   TokenCarve  [

https://arxiv.org/html/2507.20198v3#bib.bib78

]   64   336   74.8   –   –   –   57.0   79.9   1754.0   62.0   –   –   29.3   VisPruner  [

https://arxiv.org/html/2507.20198v3#bib.bib65

]   32   336   67.7   52.2   53.0   –   53.9   72.7   1271.0   58.4   –   –   28.8   MLLMs w/ Token Compression   LLaMA-VID  [

https://arxiv.org/html/2507.20198v3#bib.bib40

]   2   336   –   55.5   –   68.8   49.0   83.1   –   –   –   –   –   LLaVA-Mini  [

https://arxiv.org/html/2507.20198v3#bib.bib85

]   1   336   77.6   60.9   56.2   70.4   57.0   84.7   1466.0   65.6   58.5   68.9   36.6   VoCo-LLAMA  [

https://arxiv.org/html/2507.20198v3#bib.bib86

]   1   336   72.3   57.0   –   65.4   –   81.4   1323.3   58.8   53.7   –   –

3.3.1  Attention in Encoder

Methods focusing on the vision encoder primarily select visual tokens based on attention scores within a single image or crops, relying on the capabilities of the vision transformer (ViT). This reduces the number of visual tokens before they’re passed to the LLM. To achieve this, the set of retained tokens,  𝒯 encoder \mathcal{T}_{\mathrm{encoder}} caligraphic_T start_POSTSUBSCRIPT roman_encoder end_POSTSUBSCRIPT , is determined by selecting the top  k k italic_k  tokens based on their attention scores relative to the [CLS] token:

𝒯 encoder = TopK k  ( { Attention  ( 𝐯 i , 𝐯 cls ) ∣ 𝐯 i ∈ 𝒱 } ) , \displaystyle\mathcal{T}_{\mathrm{encoder}}=\mathrm{TopK}_{k}\left(\{\mathrm{Attention}\left(\mathbf{v}_{i},\mathbf{v}_{\mathrm{cls}}\right)\mid\mathbf{v}_{i}\in\mathcal{V}\}\right), caligraphic_T start_POSTSUBSCRIPT roman_encoder end_POSTSUBSCRIPT = roman_TopK start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ( { roman_Attention ( bold_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , bold_v start_POSTSUBSCRIPT roman_cls end_POSTSUBSCRIPT ) ∣ bold_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∈ caligraphic_V } ) ,   (7)

where  𝒱 \mathcal{V} caligraphic_V  is the original set of visual tokens,  𝐯 i \mathbf{v}_{i} bold_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT  is the  i i italic_i -th visual token, and  𝐯 cls \mathbf{v}_{\mathrm{cls}} bold_v start_POSTSUBSCRIPT roman_cls end_POSTSUBSCRIPT  is the [CLS] token. This strategy ensures that only the most salient visual information, as highlighted by the [CLS] attention, is carried forward for further processing.

Prumerge  [

https://arxiv.org/html/2507.20198v3#bib.bib29

]  selects cluster centers for visual tokens based on [CLS] attention in the encoder. It then merges the remaining less attentive tokens using K-nearest neighbors (KNN) clustering and a weighted cluster center update mechanism. VisionZip  [

https://arxiv.org/html/2507.20198v3#bib.bib18

]  retains visual tokens with high attention scores and subsequently merges the remaining tokens through clustering. VisPruner  [

https://arxiv.org/html/2507.20198v3#bib.bib65

]  similarly preserves a subset of high-attention visual tokens. Then it progressively removes duplicates based on similarity in multiple rounds, ultimately retaining an additional set of diverse tokens. GlobalCom

https://arxiv.org/html/2507.20198v3#bib.bib69

]  employs a hierarchical strategy. It coordinates the attention scores of thumbnail tokens to guide the pruning of high-resolution crops, thereby achieving effective global context reduction.

3.3.2  Attention in Decoder

Unlike attention-based compression within the encoder, methods focusing on attention in the decoder leverage the capabilities of the LLMs to guide token compression. Here, attention is computed across all tokens within the LLM’s attention window, which includes not only visual tokens but also textual tokens. This allows the LLM to determine the importance of visual and textual information in a joint space, leading to more context-aware token pruning.

A common approach for compression in the decoder involves selecting the most salient visual tokens. The set of retained tokens,  𝒯 decoder \mathcal{T}_{\mathrm{decoder}} caligraphic_T start_POSTSUBSCRIPT roman_decoder end_POSTSUBSCRIPT , is typically determined by choosing the top  k k italic_k  visual tokens based on the average attention they receive from all other tokens in that layer’s attention window:

A ¯  ( 𝐯 i ) \displaystyle\bar{A}(\mathbf{v}_{i}) over¯ start_ARG italic_A end_ARG ( bold_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT )   = 1 | 𝒮 |  ∑ 𝐬 j ∈ 𝒮 Attention  ( 𝐯 i , 𝐬 j ) , \displaystyle=\frac{1}{|\mathcal{S}|}\sum_{\mathbf{s}_{j}\in\mathcal{S}}\mathrm{Attention}\left(\mathbf{v}_{i},\mathbf{s}_{j}\right), = divide start_ARG 1 end_ARG start_ARG | caligraphic_S | end_ARG ∑ start_POSTSUBSCRIPT bold_s start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ∈ caligraphic_S end_POSTSUBSCRIPT roman_Attention ( bold_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , bold_s start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) ,   (8)   𝒯 decoder \displaystyle\mathcal{T}_{\mathrm{decoder}} caligraphic_T start_POSTSUBSCRIPT roman_decoder end_POSTSUBSCRIPT   = TopK k  ( { A ¯  ( 𝐯 i ) ∣ 𝐯 i ∈ 𝒱 } ) , \displaystyle=\mathrm{TopK}_{k}\left(\{\bar{A}(\mathbf{v}_{i})\mid\mathbf{v}_{i}\in\mathcal{V}\}\right), = roman_TopK start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ( { over¯ start_ARG italic_A end_ARG ( bold_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) ∣ bold_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∈ caligraphic_V } ) ,   (9)

where  𝒱 \mathcal{V} caligraphic_V  denotes the set of visual tokens, and  𝒮 \mathcal{S} caligraphic_S  represents the entire set of tokens present in the current layer’s attention window (which may include visual, textual, or special tokens). This method allows the model to prioritize the visual information that is most relevant in the ongoing context.

https://arxiv.org/html/2507.20198v3#bib.bib26

]  is among the first to identify a significant inefficiency in large vision language models (LVLMs), namely the extremely low attention efficiency of visual tokens. For instance, in LLaVA-v1.5, visual tokens received only  0.21 % 0.21\% 0.21 %  of the attention obtained by system prompts after the second layer. FastV posits that this is due to an overabundance of visual signals, leading to specific features aggregating onto “anchor” tokens via shallow self-attention mechanisms. Consequently, pruning  50 % 50\% 50 %  of visual tokens based on attention scores after the second layer maintains maximal performance. PyramidDrop  [

https://arxiv.org/html/2507.20198v3#bib.bib72

]  structures the token compression process within the LLM into multiple stages. It employs progressive token compression to avoid excessive loss of visual information in shallower layers. VTW  [

https://arxiv.org/html/2507.20198v3#bib.bib73

]  takes a more aggressive pruning approach, arguing that visual tokens can be entirely removed after a certain layer within the LLM. The specific layer for visual token removal is determined using a calibration dataset. FitPrune  [

https://arxiv.org/html/2507.20198v3#bib.bib74

]  focuses on reducing the length of visual tokens per layer. It considers both the self-attention of visual tokens and their cross-attention with text tokens to guide compression. The goal is to find an optimal pruning “recipe” that minimizes the distributional gap before and after pruning. ST

https://arxiv.org/html/2507.20198v3#bib.bib75

]  dynamically reduces tokens during the generation process. It also progressively prunes inattentive visual tokens as the layer goes deeper. ATP-LLaVA  [

https://arxiv.org/html/2507.20198v3#bib.bib76

]  introduces an adaptive token pruning (ATP) module within the decoder layers. This module trains threshold heads to adaptively predict pruning thresholds for the current layer and instance, thereby removing redundant or text-irrelevant visual tokens. ZipVL  [

https://arxiv.org/html/2507.20198v3#bib.bib46

]  achieves progressive compression by determining the compression ratio for each layer based on its attention score distribution. This allows for a granular and adaptive reduction of visual tokens throughout the model.

3.3.3  Critical Challenge for Pruning in Decoder

While these methods leverage attention scores within the LLM decoder to offer sophisticated ways to compress visual tokens, they face a significant practical challenge: the explicit need to access attention scores. This direct access is often incompatible with highly optimized acceleration libraries like FlashAttention  [

https://arxiv.org/html/2507.20198v3#bib.bib163

https://arxiv.org/html/2507.20198v3#bib.bib164

] , which compute attention implicitly or in a fused manner for speed. This incompatibility can be mitigated by performing an additional, separate attention calculation solely for pruning purposes. However, for multi-layer pruning strategies such as FitPrune, ST3, and ZipVL, this additional computational overhead becomes significantly more pronounced, potentially negating the efficiency gains.

3.4  Query-based Image-centric Compression

Visual information often contains a substantial amount of features irrelevant to the given query. Query-based image-centric compression leverages the query prompt to guide the compression of visual tokens. These methods can be broadly categorized into two types: (1)  Token Distillation:  These methods compress visual tokens by distilling visual tokens into a specific, reduced number of tokens. (2)  Cross-Modal Selection:  These approaches compress tokens by matching between modality-aligned visual and text tokens.

3.4.1  Token Distillation

Token distillation originates from the early projector designs of MLLMs. The goal is to distill visual tokens to learn the most text-relevant visual representations, thereby reducing visual tokens while simultaneously aligning modalities.

The Q-Former series  [

https://arxiv.org/html/2507.20198v3#bib.bib1

https://arxiv.org/html/2507.20198v3#bib.bib81

] , a pioneering approach, uses learnable queries and cross-attention to extract pertinent visual cues from visual features. Similarly, mPLUG-Owl  [

https://arxiv.org/html/2507.20198v3#bib.bib82

] , MiniGPT-4  [

https://arxiv.org/html/2507.20198v3#bib.bib83

] , Flamingo  [

https://arxiv.org/html/2507.20198v3#bib.bib84

] , and Qwen-VL  [

https://arxiv.org/html/2507.20198v3#bib.bib4

]  all employ variations of learnable query-based architectures to condense visual information into a smaller fixed set of tokens that are then aligned with the language model. LLaMA-VID  [

https://arxiv.org/html/2507.20198v3#bib.bib40

]  employs a highly aggressive approach to visual token compression. For a single image or video frame, it utilizes context attention where the text query aggregates text-related visual cues from the visual embedding. Ultimately, it represents an entire image’s information using only two tokens. LLaVA-Mini  [

https://arxiv.org/html/2507.20198v3#bib.bib85

]  achieves comparable performance by pre-fusing visual information directly into text tokens, requiring just one visual token. While previous methods relied on external modules for visual token compression, VoCo-LLaMA  [

https://arxiv.org/html/2507.20198v3#bib.bib86

]  is notable as the first approach to use LLMs themselves for visual token compression. It distills the LLM’s understanding of visual tokens into the processing of VoCo tokens via attention distillation. Victor  [

https://arxiv.org/html/2507.20198v3#bib.bib87

]  introduces a small number of learnable “register tokens” after the visual tokens. It then uses the shallow layers of a large model to distill visual information into these registers, discarding all original visual tokens to significantly improve inference and training efficiency.

3.4.2  Cross-Modal Selection

Cross-modal selection aims to reduce the number of tokens in one modality by leveraging aligned tokens from another. This compression is achieved by identifying and retaining only the most relevant information across modalities, leading to more efficient and effective processing. Several notable approaches have been proposed to address this challenge:

SparseVLM  [

https://arxiv.org/html/2507.20198v3#bib.bib51

]  employs visual tokens to pre-select relevant text tokens. By leveraging the visual modality as an initial filter, SparseVLM efficiently narrows down the textual search space, focusing on information pertinent to the visual content. AdaFV  [

https://arxiv.org/html/2507.20198v3#bib.bib88

]  employs a dual-metric approach for selecting the most informative visual tokens. It calculates both text-to-image similarity and visual saliency extracted from the vision encoder. By combining these two indicators, AdaFV identifies visual tokens that are not only semantically aligned with the text but also visually prominent or significant. TRIM  [

https://arxiv.org/html/2507.20198v3#bib.bib89

]  introduces a unique method that begins by identifying outlier tokens based on the similarity between text and visual tokens; these outliers are deemed important. Subsequently, a clustering algorithm is utilized to merge the remaining, less critical tokens. This approach prioritizes distinct, highly relevant tokens before consolidating the rest.

TABLE III:  Quantitative comparison of training free token compression methods in video LLMs. We present the officially published evaluation results by using the token retained ratio as a criterion for fair comparison. We select the most representative results of each presentation method. For all values, higher is better. VideoChat-GPT comprises five subtasks: CI stands for correctness of information, DO stands for detail orientation, CU stands for contextual understanding, TU stands for temporal understanding, and CO stands for consistency.   Method   #Token Ratio   ActivityNet   Video-ChatGPT   Next-QA   EgoSchema   LongVideo   Bench   VideoMME   MVBench   Acc.   Score   CI   DO   CU   TU   CO   mc   ↑ \uparrow ↑   ↑ \uparrow ↑   ↑ \uparrow ↑   ↑ \uparrow ↑   LLaVA-OneVision  [

https://arxiv.org/html/2507.20198v3#bib.bib2

]   100%   48.09   3.47   3.37   3.78   3.52   3.02   2.63   81.33   60.4   56.4   58.6   58.3   50% Visual Token Retained Ratio   FastV  [

https://arxiv.org/html/2507.20198v3#bib.bib26

]   50%   47.95   3.47   3.36   3.77   3.50   2.99   2.57   81.1   58.0   –   57.5   –   DyCoke  [

https://arxiv.org/html/2507.20198v3#bib.bib17

]   50%   47.88   3.47   3.33   3.76   3.51   3.01   2.58   81.1   57.7   –   57.4   57.5   PLLaVA  [

https://arxiv.org/html/2507.20198v3#bib.bib3

]   50%   47.59   3.45   3.36   3.73   3.52   3.00   2.66   81.0   57.7   –   56.9   –   VisionZip  [

https://arxiv.org/html/2507.20198v3#bib.bib18

]   50%   45.42   3.47   3.16   3.63   3.34   2.75   2.61   78.5   53.57   –   54.2   –   LLaVA-Scissor  [

https://arxiv.org/html/2507.20198v3#bib.bib102

]   50%   47.89   3.47   3.37   3.76   3.47   3.00   2.65   81.12   57.6   –   57.4   –   25% – 35% Visual Token Retained Ratio   FastV  [

https://arxiv.org/html/2507.20198v3#bib.bib26

]   35%   47.83   3.46   3.32   3.74   3.47   2.97   2.61   80.5   57.8   –   56.0   61.3   DyCoke  [

https://arxiv.org/html/2507.20198v3#bib.bib17

]   35%   47.81   3.45   3.31   3.74   3.46   2.98   2.54   80.9   57.7   –   56.2   61.8   PLLaVA  [

https://arxiv.org/html/2507.20198v3#bib.bib3

]   35%   47.23   3.42   3.26   3.70   3.39   2.92   2.59   79.66   56.07   –   54.26   59.5   FastVID  [

https://arxiv.org/html/2507.20198v3#bib.bib44

]   25%   –   –   –   –   –   –   –   –   –   56.3   58.0   56.5   PruneVid  [

https://arxiv.org/html/2507.20198v3#bib.bib27

]   25%   –   –   –   –   –   –   –   –   59.9   55.7   57.4   57.4   VisionZip  [

https://arxiv.org/html/2507.20198v3#bib.bib18

]   25%   –   –   –   –   –   –   –   –   60.3   56.5   58.2   57.9   LLaVA-Scissor  [

https://arxiv.org/html/2507.20198v3#bib.bib102

]   25%   47.79   3.47   3.33   3.76   3.47   2.98   2.62   80.66   57.64   –   56.44   –   HoliTom  [

https://arxiv.org/html/2507.20198v3#bib.bib16

]   25%   –   –   –   –   –   –   –   –   61.2   56.7   58.9   58.4   5% – 15% Visual Token Retained Ratio   VisionZip  [

https://arxiv.org/html/2507.20198v3#bib.bib18

]   15%   –   –   –   –   –   –   –   –   59.8   54.4   56.1   56.5   PruneVid  [

https://arxiv.org/html/2507.20198v3#bib.bib27

]   10%   –   –   –   –   –   –   –   –   59.8   54.5   56.0   56.2   FastVID  [

https://arxiv.org/html/2507.20198v3#bib.bib44

]   10%   –   –   –   –   –   –   –   –   –   56.3   57.3   55.9   LLaVA-Scissor  [

https://arxiv.org/html/2507.20198v3#bib.bib102

]   10%   47.75   3.46   3.26   3.68   3.41   2.90   2.52   80.0   57.5   –   55.2   57.9   HoliTom  [

https://arxiv.org/html/2507.20198v3#bib.bib16

]   10%   –   –   –   –   –   –   –   –   61.2   56.3   56.8   57.3

4  Video-centric Token Compression

Processing long high-definition (HD) videos poses significant challenges for VLMs due to the immense number of tokens generated, far exceeding those from high-resolution images. Unlike image-centric compression, video inherently possesses an additional temporal redundancy. While capturing complete temporal information typically requires a frame rate of at least 24 frames per second (FPS), processing a 10-minute HD video at even 1 FPS still yields token sequences orders of magnitude larger than those from high-resolution images, rendering conventional transformer-based MLLMs impractical for real-world deployment over the videos.

To address this, current video LLMs commonly employ a 1 FPS sampling rate to reduce token counts. Furthermore, unlike the typical approach for single images, where the image is split into patches and fed into an encoder for detailed information, video processing often foregoes this detailed frame-level segmentation to keep token numbers manageable. Even with these strategies, the quantity of video tokens remains substantial. During model training and understanding,  transformation-based  methods, such as the pooling technique used in LLaVA-Video  [

https://arxiv.org/html/2507.20198v3#bib.bib34

] , are frequently employed to reduce tokens and aid the model’s comprehension of video content.

Beyond training-time optimizations, alternative approaches primarily focus on post-training optimization. Specifically,  similarity-based  and  attention-based  methods offer generic compression techniques for pre-trained video MLLMs. These methods process encoded token sequences without modifying model weights, enabling plug-and-play acceleration across diverse architectures. By dynamically identifying critical spatio-temporal regions and pruning redundant tokens, these techniques significantly enhance the practicality of video MLLMs for real-world applications.

To fully grasp token compression for video LLMs, it is highly recommended to first review Section

https://arxiv.org/html/2507.20198v3#S3

, which details spatial compression methods. Next, we will primarily discuss techniques addressing the temporal domain. Similar to image-centric methods, selected video-centric token compression methods are compared in Table

https://arxiv.org/html/2507.20198v3#S3.T3

4.1  Transformation-based Video-centric Compression

Like image LLMs, video LLMs use image encoders for visual tokens. Consequently, transformation-based video-centric compression methods fundamentally operate on the principles established in Section

https://arxiv.org/html/2507.20198v3#S3.SS1

, with the added capability of performing 3D transformations. A multitude of models showcase cross-modal applicability, performing effectively in both image and video inference tasks. Following the structure of Section

https://arxiv.org/html/2507.20198v3#S3.SS1

, we will now detail transformation-based video-centric compression methods.

4.1.1  2D/3D Pooling

In video LLMs, token pooling is a crucial strategy for managing the high dimensionality of video data. While 2D spatial pooling, as seen in LLaVA-Video  [

https://arxiv.org/html/2507.20198v3#bib.bib34

] , can effectively reduce the token count within individual frames, its efficacy alone may be limited for long-duration videos. A growing number of video LLMs, including PLLaVA  [

https://arxiv.org/html/2507.20198v3#bib.bib3

] , Video-ChatGPT  [

https://arxiv.org/html/2507.20198v3#bib.bib90

] , SlowFast-LLaVA  [

https://arxiv.org/html/2507.20198v3#bib.bib92

] , and LongVLM  [

https://arxiv.org/html/2507.20198v3#bib.bib91

] , consequently emphasize temporal pooling, which involves downsampling at the frame level.

Notably, PLLaVA’s experiments demonstrate that model performance exhibits greater sensitivity to temporal pooling than to spatial pooling, highlighting its critical role. For extremely long video sequences, LLaMA-VID  [

https://arxiv.org/html/2507.20198v3#bib.bib40

]  employs a more aggressive adaptive pooling approach. This method intelligently maintains original resolution for single-image inputs but compresses each video frame to a single token during extended sequence processing, achieving substantial data reduction while aiming to preserve essential information.

This dual focus on spatial and increasingly on temporal pooling underscores their combined importance in enabling efficient processing and comprehensive understanding of video content, particularly as video durations extend. SlowFast-LLaVA-1.5  [

https://arxiv.org/html/2507.20198v3#bib.bib92

]  incorporates a two-stream SlowFast projector into a LLaVA-style architecture, using a slow pathway to sample fewer, spatially rich frames and a fast pathway to sample more, spatially compressed frames, then concatenates both for the LLM—achieving efficient long-form video understanding with reduced token count while preserving spatiotemporal details.

4.1.2  2D/3D Convolution

Similar to pooling, convolution can also be employed for downsampling video tokens, but it does so in a parameterized manner. Instead of simply aggregating information like pooling, convolution layers learn filters to process and condense spatial and temporal features. VideoLLaMA 2  [

https://arxiv.org/html/2507.20198v3#bib.bib10

] , for instance, thoroughly investigated both 2D and 3D pooling and convolution approaches. Their experiments ultimately showed that 3D convolution yielded the best balance of performance and efficiency for video token downsampling. This suggests that learning intricate spatiotemporal relationships through convolutions is more effective for comprehensive video understanding compared to pooling alone.

4.2  Similarity-based Video-centric Compression

Given the temporal redundancy inherent in video, where adjacent frames often exhibit high visual similarity, temporal compression is frequently prioritized over, or integrated with spatial compression. To effectively handle this temporal redundancy, video frames are typically first clustered.

Chat-UniVi  [

https://arxiv.org/html/2507.20198v3#bib.bib94

]  initially pools each video frame into a single frame-level representation token. It then utilizes DPC-KNN  [

https://arxiv.org/html/2507.20198v3#bib.bib165

https://arxiv.org/html/2507.20198v3#bib.bib166

]  (density peak clustering based on K-nearest neighbors) to amalgamate non-essential frames based on these frame representation tokens. Within each resulting cluster, tokens from multiple frames are further clustered to obtain concise spatiotemporal visual representations. Similarly, FastVID  [

https://arxiv.org/html/2507.20198v3#bib.bib44

]  divides video frames solely based on the similarity of their adjacent frame representation tokens. It then employs DPC-KNN within these clustered frames to merge tokens, thereby reducing spatiotemporal redundancy. PruneVid  [

https://arxiv.org/html/2507.20198v3#bib.bib27

]  adopts the same frame clustering methodology as Chat-UniVi. The key distinction is that it performs an initial merging of temporally static tokens before executing the spatiotemporal token consolidation. HoliTom  [

https://arxiv.org/html/2507.20198v3#bib.bib17

]  argues that relying on a single frame-level representation token for video frame clustering can lead to suboptimal detail capture, and that the preliminary merging of static temporal tokens is disconnected from the original frame clustering method. HoliTom re-conceptualizes temporal redundancy compression as an optimization problem aimed at maximizing the compressible temporal redundant features within all clustered frames, thus addressing temporal compression more holistically. DyCoke  [

https://arxiv.org/html/2507.20198v3#bib.bib17

]  groups frames into sets of four, directly performing temporal pruning within each group.

While some methods do not explicitly cluster video frames, FrameFusion  [

https://arxiv.org/html/2507.20198v3#bib.bib79

] , for example, acts as a token compression technique for streaming video LLMs. It directly merges temporally redundant tokens exceeding a specific threshold in the shallow layers of the model.

4.3  Attention-based Video-centric Compression

Current attention-based token compression methods in video LLMs and image LLMs share significant similarities. When attention is applied within the encoder to guide token compression, videos are typically treated as a sequence of independent images fed into an image encoder. Consequently, attention computations largely disregard inter-frame relationships, making these approaches fundamentally image-centric token compression. For a more concise discussion of such attention-based methods, please refer to Section

https://arxiv.org/html/2507.20198v3#S3.SS3

In contrast, methods employing attention within the decoder process video frames sequentially, concatenating their tokens over time. For longer videos, particularly in the context of streaming video LLMs, windowed attention is commonly used to mitigate computational overhead by focusing on local temporal visual information. However, it’s notable that even these windowed attention-based methods within the decoder often share the same foundational principles as those discussed in Section

https://arxiv.org/html/2507.20198v3#S3.SS3

4.4  Query-based Video-centric Compression

4.4.1  Token Distillation

Token distillation in video LLMs commonly relies on specialized adaptor modules, such as the Q-former  [

https://arxiv.org/html/2507.20198v3#bib.bib1

https://arxiv.org/html/2507.20198v3#bib.bib81

]  or Token Turing Machines  [

https://arxiv.org/html/2507.20198v3#bib.bib104

] . These modules typically process video tokens with the learnable query tokens to be attended.

Token Turing Machines (TTMs)  [

https://arxiv.org/html/2507.20198v3#bib.bib104

]  maintain a compact external memory of summary tokens, sequentially compressing both new input tokens and memory at each timestep via a Transformer-based read/write mechanism, allowing scalable and efficient processing of long video sequences. BLIP-3-Video  [

https://arxiv.org/html/2507.20198v3#bib.bib105

]  introduces an explicit temporal encoder that abstracts hundreds of frame-level visual tokens into as few as 16–32 spatiotemporal tokens using learnable pooling and sequential models, enabling efficient video understanding with limited token usage. LinVT  [

https://arxiv.org/html/2507.20198v3#bib.bib108

]  proposes a plug-and-play Linear Video Tokenizer, which linearly aggregates frame-level visual tokens into a compact set of video tokens through spatio-temporal scoring, multi-scale pooling, and text-conditioned aggregation, enabling existing image-LLMs to efficiently process videos and dynamically extract question-relevant information. Long-VMNet  [

https://arxiv.org/html/2507.20198v3#bib.bib106

]  accelerates long-form video understanding by using a neural sampler to select discriminative visual tokens from clips and storing them in a fixed-size memory bank for each video; downstream queries are answered by processing only these memory tokens, greatly reducing computational cost while preserving key spatiotemporal information. STORM  [

https://arxiv.org/html/2507.20198v3#bib.bib107

]  inserts a Mamba-based  [

https://arxiv.org/html/2507.20198v3#bib.bib167

]  temporal encoder between the image encoder and LLM, using spatiotemporal scanning and pooling to inject temporal context into frame tokens and then aggressively compresses tokens by temporal and spatial pooling, enabling efficient long video understanding with minimal token loss. To understand more methods and applications of token distillation in video LLMs, please also refer to Section

https://arxiv.org/html/2507.20198v3#S3.SS4

for a detailed explanation.

4.4.2  Cross-Modal Selection

In video large language models (video LLMs), a query is commonly used to guide the selection of salient frames. In extreme cases, only a handful of frames are relevant to the posed question, allowing the tokens from the vast majority of remaining frames to be discarded. When dealing with an immense number of frames, finding query-relevant information can be akin to searching for a "needle in a haystack" for the LLM. Query-based token compression methods can pre-filter these query-relevant tokens, significantly alleviating the computational burden on the LLM.

https://arxiv.org/html/2507.20198v3#bib.bib95

]  exemplifies this approach. It calculates the relevance of each video frame to the query via cross-modal interaction. This relevance score then dictates a lower compression ratio for key frames, better preserving critical information, all while ensuring the total number of tokens remains within the maximum context length of the LLM.

5  Audio-centric Token Compression

For audio LLMs, the demand for longer context arises from the need to process higher sampling rates and extended durations of audio.

The extraction of information from the audio modality can be categorized according to the format of audio representation: (1)  continuous sequence modeling:  this approach utilizes a pre-trained audio encoder, typically models like Whisper  [

https://arxiv.org/html/2507.20198v3#bib.bib131

]  or Conformer  [

https://arxiv.org/html/2507.20198v3#bib.bib168

] , to produce continuous audio embeddings; (2)  discrete sequence modeling:  this method transforms the input audio signal into discrete audio tokens, usually via vector quantization, where continuous audio features are encoded into a learnable codebook. Mainstream methods include HuBERT  [

https://arxiv.org/html/2507.20198v3#bib.bib169

]  and EnCodec  [

https://arxiv.org/html/2507.20198v3#bib.bib170

https://arxiv.org/html/2507.20198v3#bib.bib171

The second category inherently reduces the number of tokens by optimizing the tokenizer structure and the design of the codebook. Nevertheless, detailed exploration of these specific design considerations falls outside the purview of this survey.

Audio, a 1D signal representing amplitude over time, must be transformed into a suitable format for deep learning models, especially when integrating with MLLMs. MLLMs often leverage architectures designed for 2D data (like images) or general sequences. While the raw waveform is the source, spectrograms (especially Mel-spectrograms) are frequently the preferred representation for audio in MLLMs. This preference arises because spectrograms allow the application of processing techniques similar to those used for images, thereby facilitating multimodal learning.

Consequently, much like the visual modality, we categorize audio token compression methods as follows:

5.1  Transformation-based audio-centric Compression

Following the categorization in the visual modality, we can classify methods based on their downsampling operations:

5.1.1  Token Stacking

Similar to the pixel unshuffle operation in 2D image processing, this approach for audio LLMs token compression involves stacking multiple consecutive tokens along the hidden dimension of the token. This effectively reduces the total number of tokens. Notably, HTS-AT  [

https://arxiv.org/html/2507.20198v3#bib.bib109

] , an early example of audio token stacking for classification tasks within audio transformers, utilized 2D pixel-unshuffling on the 2D features extracted from Mel spectrograms to reduce audio tokens. More recent methods such as SLAM-ASR  [

https://arxiv.org/html/2507.20198v3#bib.bib110

] , LLaMA-Omni  [

https://arxiv.org/html/2507.20198v3#bib.bib111

] , Llama-AVSR  [

https://arxiv.org/html/2507.20198v3#bib.bib115

]  and others  [

https://arxiv.org/html/2507.20198v3#bib.bib172

]  have adopted this technique. Since these token stacking operations alter the hidden dimension, a MLP is typically used to realign the dimension for compatibility with other modalities.

5.1.2  Pooling

Another common technique for reducing the number of audio tokens is pooling. Models like Qwen2-audio  [

https://arxiv.org/html/2507.20198v3#bib.bib113

]  and Qwen2.5-Omni  [

https://arxiv.org/html/2507.20198v3#bib.bib5

]  leverage pooling layers with a stride of 2 to directly decrease the length of the audio representation in a parameter-free manner. This effectively downsamples the audio features, leading to a more compact token sequence. Extending this concept, Llama-MTSK  [

https://arxiv.org/html/2507.20198v3#bib.bib116

]  employs a matryoshka-based training approach for flexible token compression. It trains the model with multi-scale audio and video information, achieved by applying average pooling or token stacking at various rates to the initial tokens. This enables Llama-MTSK to dynamically adjust the number of tokens processed during inference, balancing compression and performance within a single model.

5.1.3  Temporal Convolution

For audio tokens, 1D convolutions applied across the temporal dimension can reduce the number of tokens. This method simultaneously allows for the alignment of the hidden dimension for subsequent LLM. Approaches like SpeechVerse  [

https://arxiv.org/html/2507.20198v3#bib.bib112

] , Baichuan-Audio  [

https://arxiv.org/html/2507.20198v3#bib.bib114

] , OSUM  [

https://arxiv.org/html/2507.20198v3#bib.bib117

] , and LUCY  [

https://arxiv.org/html/2507.20198v3#bib.bib118

]  have employed this technique, often resulting in a downsampled audio representation with an effective sampling rate of 12.5 Hz.

These methods demonstrate how insights from image compression, particularly those involving transformations, can be effectively applied to the audio domain to achieve more efficient token representation for large models.

5.2  Similarity-based audio-centric Compression

Similarity-based compression methods aim for each audio token to carry unique information rather than being overly redundant. Similarly to the ToMe  [

https://arxiv.org/html/2507.20198v3#bib.bib21

]  method used in vision transformers (ViT), A-ToMe  [

https://arxiv.org/html/2507.20198v3#bib.bib119

]  inserts a token merge module between the multihead self-attention (MHSA) and feed-forward network (FFN) in each layer. This module merges adjacent audio tokens that have high cosine similarity.

5.3  Attention-based audio-centric Compression

For audio tasks, attention-based methods are also effectively utilized to compress tokens.

5.3.1  Attention in Encoder

https://arxiv.org/html/2507.20198v3#bib.bib120

]  operates within an audio spectrogram transformer block, directly retain only the top K audio tokens based on their attention scores. This prunes less attentive tokens, focusing on those with higher relevance as determined by the self-attention mechanism within the transformer.

5.3.2  Attention in Decoder

SpeechPrune  [

https://arxiv.org/html/2507.20198v3#bib.bib121

] , works in the LLM backbone. It prunes audio tokens based on attention scores provided by the first transformer layer. By utilizing the initial layer’s attention, SpeechPrune efficiently identifies and discards less crucial tokens early in the processing pipeline, aiming to reduce computational load and improve efficiency for subsequent layers without significant loss of information.

5.4  Query-based audio-centric Compression

Audio feature representations can also be compressed using other modalities or learned query mechanisms. Analogous to image LLMs, these methods can be broadly categorized into token distillation and cross-modal selection, based on whether learned queries are explicitly employed.

5.4.1  Token Distillation

This category leverages learnable query tokens to distill comprehensive audio information into a compact, fixed-length representation.

Video-LLaMA  [

https://arxiv.org/html/2507.20198v3#bib.bib7

]  and SALMONN series  [

https://arxiv.org/html/2507.20198v3#bib.bib122

https://arxiv.org/html/2507.20198v3#bib.bib123

]  employs an audio Q-former to transform variable-length audio inputs into a fixed-length sequence of learnable queries, thereby condensing audio information for the LLM. MMCE-Qformer  [

https://arxiv.org/html/2507.20198v3#bib.bib125

]  compresses acoustic information by utilizing learnable queries to extract global acoustic context from contextual audio embeddings. Concurrently, a cross-attention mechanism, guided by input text embeddings, captures local acoustic context relevant to each text token. This dual approach distills both broad and specific audio features into compact, text-relevant representations. MMS-LLaVA  [

https://arxiv.org/html/2507.20198v3#bib.bib126

]  reduces multimodal token length for efficient speech LLMs. It first halves the sequence length with an Early AV-Fusion Module, which combines visual and audio features. Subsequently, an AV Q-Former further compresses these fused features into a fixed number of queries, effectively capturing full speech context to bridge the token gap with text.

5.4.2  Cross-Modal Selection

Similar to the visual modality, audio token compression can also be guided by information from other modalities. Speechprune  [

https://arxiv.org/html/2507.20198v3#bib.bib121

] , for example, leverages audio-text correlation to identify semantically important audio segments. This is achieved by calculating a cross-modal similarity matrix based on cosine similarity, which then guides the compression of audio tokens. This approach ensures that the most relevant audio information is retained.

6  Discussions

6.1  Synergies and Distinctions with Other Compression Methods

Beyond token compression, the research community has seen the emergence of several other compression methods, including model quantization  [

https://arxiv.org/html/2507.20198v3#bib.bib173

https://arxiv.org/html/2507.20198v3#bib.bib174

https://arxiv.org/html/2507.20198v3#bib.bib175

https://arxiv.org/html/2507.20198v3#bib.bib176

https://arxiv.org/html/2507.20198v3#bib.bib177

https://arxiv.org/html/2507.20198v3#bib.bib178

] , network pruning  [

https://arxiv.org/html/2507.20198v3#bib.bib179

https://arxiv.org/html/2507.20198v3#bib.bib180

https://arxiv.org/html/2507.20198v3#bib.bib181

https://arxiv.org/html/2507.20198v3#bib.bib182

] , knowledge distillation  [

https://arxiv.org/html/2507.20198v3#bib.bib183

https://arxiv.org/html/2507.20198v3#bib.bib184

] , and low-rank factorization  [

https://arxiv.org/html/2507.20198v3#bib.bib185

https://arxiv.org/html/2507.20198v3#bib.bib186

https://arxiv.org/html/2507.20198v3#bib.bib187

https://arxiv.org/html/2507.20198v3#bib.bib188

https://arxiv.org/html/2507.20198v3#bib.bib189

] . These methods typically focus on directly compressing model weights to achieve efficiency.

For a Transformer-based model, the computational cost (FLOPs) is often dominated by matrix multiplications, particularly in the self-attention and feed-forward layers. A simplified representation of FLOPs can be given as:

FLOPs ∝ O  ( N ⋅ D 2 + N 2 ⋅ D ) , \displaystyle\text{FLOPs}\propto O(N\cdot D^{2}+N^{2}\cdot D), FLOPs ∝ italic_O ( italic_N ⋅ italic_D start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT + italic_N start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT ⋅ italic_D ) ,   (10)

where  N N italic_N  is the number of tokens,  D D italic_D  is the model dimension.

6.1.1  Weight-Focused Compression Methods

These methods mainly target the model dimension ( D D italic_D ) by reducing the effective size or complexity of the model weights.  Model Quantization  reduces weight precision, directly impacting the memory associated with  D D italic_D . A key limitation is that highly aggressive quantization (e.g., 4-bit) often compromises accuracy, meaning there’s no "free lunch" when it comes to achieving lossless performance. Furthermore, effectively accelerating these lower bit-rates often necessitates specialized hardware.  Network Pruning  removes redundant connections, effectively reducing the active parameters contributing to  D D italic_D . For LLMs, aggressive structured pruning (e.g., beyond  20 % 20\% 20 %  for downstream tasks) often leads to significant performance degradation or near-collapse due to the difficulty in preserving architectural integrity.  Knowledge Distillation  trains a smaller student model (with a smaller  D D italic_D ) to mimic a larger teacher  [

https://arxiv.org/html/2507.20198v3#bib.bib183

] . Its main limitation is the "knowledge gap", as the student may struggle to fully capture the teacher’s comprehensive knowledge, leading to performance disparities, especially on complex or out-of-distribution data.  Low-Rank Factorization  decomposes weight matrices into lower-rank approximations, thus reducing parameters related to  D D italic_D . The challenge lies in finding an optimal low-rank approximation for diverse tasks without performance loss, as this is often task-dependent and complex to apply consistently across deep networks.

6.1.2  Token Compression

In contrast, token compression directly targets the sequence length ( N N italic_N ) by reducing the number of tokens processed for long contexts. By reducing  N N italic_N , token compression significantly impacts FLOPs:

FLOPs ∝ O  ( M ⋅ D 2 + M 2 ⋅ D ) , \displaystyle\text{FLOPs}\propto O(M\cdot D^{2}+M^{2}\cdot D), FLOPs ∝ italic_O ( italic_M ⋅ italic_D start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT + italic_M start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT ⋅ italic_D ) ,   (11)

where  M ≪ N M\ll N italic_M ≪ italic_N  represents the reduced sequence length after token compression.

This approach offers benefits like improved efficiency for long context processing, breaking through context window limitations, and closer alignment with API cost reduction, as many LLM APIs charge by token count.

6.1.3  Complementary Nature and Synergistic Gains

The methods for compressing model weights and token compression are structurally orthogonal and can be effectively combined for superior results. For example: NVILA  [

https://arxiv.org/html/2507.20198v3#bib.bib190

]  pushes inference latency reduction and throughput maximization to the extreme by simultaneously applying quantization and token compression. CoreMatching  [

https://arxiv.org/html/2507.20198v3#bib.bib77

]  achieves synergistic acceleration by concurrently compressing both neurons (a form of pruning/weight reduction) and tokens.

This orthogonality means that combining these approaches holds the potential for compounded efficiency gains that are greater than applying either method in isolation.

6.2  Token Compression: Efficiency and Beyond

Token compression is often perceived solely as a training-free method to boost efficiency. However, its significance extends far beyond this, having been intrinsically incorporated into the design of MLLM, particularly within the modality transition modules (

, adapter). This integration not only facilitates superior modality alignment but also enhances the quality of information, leading to more efficient and stable training processes.

6.2.1  Enhanced Modality Alignment

Effectively aligning and comprehending information from disparate modalities remains a significant challenge. Traditional encoders segment and tokenize all multimodal information to align with linguistic representations. However, low-quality and low-density multimodal representations expand the alignment space, complicating the task of modality matching. Token compression addresses this by enabling a more precise correspondence between language representations and multimodal information.

A prime example is the Q-Former  [

https://arxiv.org/html/2507.20198v3#bib.bib1

https://arxiv.org/html/2507.20198v3#bib.bib81

] , which employs a trainable vector to distill visual tokens, achieving direct alignment of the modality simultaneously. Similarly, M

https://arxiv.org/html/2507.20198v3#bib.bib35

]  adopts a coarse-to-fine semantic granularity training approach, empowering MLLMs to align with and interpret visual representations at various levels.

6.2.2  Improved Information Representation

The sheer volume of multimodal information often leads to inefficient training and inference, with an overabundance of multimodal tokens that potentially degrade the capabilities of the text modality  [

https://arxiv.org/html/2507.20198v3#bib.bib191

] . This issue is compounded by inherent redundancies within multimodal data itself:  (1) Feature Redundancy  arises from similar backgrounds in visual data or silent segments in audio.  (2) Task-Irrelevant Redundancy  is evident in tasks like visual question answering (VQA), where a significant portion of multimodal representations may be entirely irrelevant to deriving the correct answer.  (3) Attention Computation Redundancy  emerges from two aspects: first, due to the nature of attention mechanisms, tokens positioned later in a sequence often receive disproportionately higher attention  [

https://arxiv.org/html/2507.20198v3#bib.bib192

] , suggesting potential computational redundancy for tokens not at the sequence’s end; and second, because multimodal information receives inherently less attention than textual data  [

https://arxiv.org/html/2507.20198v3#bib.bib26

https://arxiv.org/html/2507.20198v3#bib.bib89

] , an abundance of multimodal tokens can still introduce substantial computational redundancy.

Addressing these issues, the method classifications discussed earlier directly correspond to these types of data redundancy. Specifically, the transformation-based methods elaborated in Section

https://arxiv.org/html/2507.20198v3#S3.SS1

https://arxiv.org/html/2507.20198v3#S4.SS1

https://arxiv.org/html/2507.20198v3#S5.SS1

, along with similarity-based approaches in Section

https://arxiv.org/html/2507.20198v3#S3.SS2

https://arxiv.org/html/2507.20198v3#S4.SS2

https://arxiv.org/html/2507.20198v3#S5.SS2

, are effective in mitigating the feature redundancy. Furthermore, attention-based methods, as presented in Section

https://arxiv.org/html/2507.20198v3#S3.SS3

https://arxiv.org/html/2507.20198v3#S4.SS3

https://arxiv.org/html/2507.20198v3#S5.SS3

, play a crucial role in minimizing attention computation redundancy. Lastly, query-based methods, detailed in Section

https://arxiv.org/html/2507.20198v3#S3.SS4

https://arxiv.org/html/2507.20198v3#S4.SS4

https://arxiv.org/html/2507.20198v3#S5.SS4

, are designed to reduce task-irrelevant redundancy.

6.2.3  Enable One-Shot Long-Context Understanding

Limited by the inherent length of the context, MLLMs are unable to learn or comprehend real-world scenarios involving extremely long contexts, such as understanding entire code repositories or extended video and audio sequences  [

https://arxiv.org/html/2507.20198v3#bib.bib193

] . However, token compression significantly condenses and abstracts original information representations, making it possible for MLLMs to understand these long contexts in a single pass.

Traditional methods for handling long contexts in MLLMs, like FlashAttention  [

https://arxiv.org/html/2507.20198v3#bib.bib163

https://arxiv.org/html/2507.20198v3#bib.bib164

]  or RingAttention  [

https://arxiv.org/html/2507.20198v3#bib.bib194

] , involve architectural changes to the model’s attention mechanism to directly accommodate longer sequences. While effective, these require fundamental model modifications. Token compression offers a different, often simpler, route. Instead of redesigning the model to fit more tokens, it focuses on making each token more powerful. By creating information-dense tokens, we pack more meaning into fewer pieces of data. This lets existing MLLM architectures process significantly longer conceptual contexts without major overhauls. It’s a more efficient and accessible way to achieve that crucial one-shot understanding of vast, complex real-world information  [

https://arxiv.org/html/2507.20198v3#bib.bib195

6.3  Combining Different Token Compression Methods

## In Section

https://arxiv.org/html/2507.20198v3#S6.SS2.SSS2

, we explored three distinct types of redundancy and the corresponding methods to reduce them. Since prune different forms of redundancy and can be applied at various stages, they seem orthogonal in their function. This raises a natural question: can we combine multiple token compression methods to achieve a synergistic effect?

Surprisingly, the answer is often no. Research has shown that a simple combination can lead to a phenomenon where the total effect is less than the sum of its parts. For example, the MoB  [

https://arxiv.org/html/2507.20198v3#bib.bib196

]  paper analyzed this in the context of visual token pruning, finding that the combining methods do not exhibit superiority. This is probably attributed to the difference in each compression technique. When methods target different redundancies, their compression metrics can be inconsistent, leading to conflicting pruning decisions and a suboptimal outcome. Therefore, while each method excels at its specific task, their combined use does not necessarily guarantee a better result and may even degrade performance.

6.4  Current Challenges

6.4.1  Performance Degradation

While token compression can effectively condense and abstract multimodal features, it also introduces a risk of performance degradation. Current research on visual MLLMs, for example, has shown that for models like LLaVA-OV-7B  [

https://arxiv.org/html/2507.20198v3#bib.bib2

] , near-lossless performance can be achieved by retaining as few as  10 % 10\% 10 %  of the original tokens. However, performance declines sharply when the compression rate is pushed further. This challenge is even more pronounced for larger and more recent models such as Qwen2.5-VL  [

https://arxiv.org/html/2507.20198v3#bib.bib33

] , LLaVA-Video-7B  [

https://arxiv.org/html/2507.20198v3#bib.bib34

]  and LLaVA-OV-72B  [

https://arxiv.org/html/2507.20198v3#bib.bib2

] , where achieving lossless compression seems to be more difficult.

This increased difficulty may stem from the models’ enhanced representational capabilities. It has been suggested that less capable models are inherently less sensitive to information loss from aggressive compression, as their weaker understanding already struggles to fully process the complex, uncompressed data. In contrast, more sophisticated models, which possess a more nuanced and holistic comprehension of multimodal tokens, are more susceptible to the subtle degradation caused by compression. For these models, achieving high performance requires a far more delicate and precise approach to preserve the token.

6.4.2  Task-Specific Challenges

Token compression, while beneficial for efficiency, can be destructive to performance on tasks that demand high representational fidelity. For  optical character recognition (OCR) , which requires a high information density within local regions, compression often leads to the loss of critical details and a subsequent drop in performance. This is particularly evident on benchmarks like RefCOCO  [

https://arxiv.org/html/2507.20198v3#bib.bib197

] , where the model’s ability to ground objects based on fine-grained textual cues is compromised.

A similar challenge arises in preserving  temporal perception . Video and audio are fundamentally structured by fixed sampling rates  [

https://arxiv.org/html/2507.20198v3#bib.bib55

] . By merging adjacent frames or sequential tokens, compression methods disrupt this inherent temporal consistency, hindering the model’s ability to reason about motion, pace, and other crucial temporal dynamics that are essential for a complete understanding of the content.

TABLE IV:  Detail of the benchmarks can be used for performance evaluation of image–language and video–language tasks. CE-VQA: Closed-Ended Visual Question Answering. OE-VQA: Open-Ended Visual Question Answering. MC-VQA: Multiple-Choice Visual Question Answering.   Benchmark   Task   Metric   System Prompt   Image Task   GQA  [

https://arxiv.org/html/2507.20198v3#bib.bib198

]   CE-VQA   Exact Match   Answer the question using a single word or phrase.   MMB  [

https://arxiv.org/html/2507.20198v3#bib.bib199

]   MC-VQA   Accuracy   Answer with the option’s letter from the given choices directly.   MME  [

https://arxiv.org/html/2507.20198v3#bib.bib200

]   CE-VQA   Perception Score   Answer the question using a single word or phrase.   POPE  [

https://arxiv.org/html/2507.20198v3#bib.bib201

]   CE-VQA   F1 Score   Answer the question using a single word or phrase.   ScienceQA-Image  [

https://arxiv.org/html/2507.20198v3#bib.bib202

]   Visual reasoning   Exact Match   Answer with the option’s letter from the given choices directly.   SeedBench-Image  [

https://arxiv.org/html/2507.20198v3#bib.bib203

]   MC-VQA   Accuracy   Answer with the option’s letter from the given choices directly.   VizWiz  [

https://arxiv.org/html/2507.20198v3#bib.bib204

]   CE-VQA   Exact Match   When the provided information is insufficient, respond with “Unanswerable”. Answer the question using a single word or phrase.   VQA

https://arxiv.org/html/2507.20198v3#bib.bib205

]   CE-VQA   Exact Match   Answer the question using a single word or phrase.   MM-Vet  [

https://arxiv.org/html/2507.20198v3#bib.bib206

]   Visual reasoning   GPT-score   First please perform reasoning, and think step by step to provide the best answer to the following question:   LLaVA W {}^{\text{W}} start_FLOATSUPERSCRIPT W end_FLOATSUPERSCRIPT   [

https://arxiv.org/html/2507.20198v3#bib.bib1

]   Visual reasoning   GPT-score   A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human’s questions.   Video Task   ActivityNet  [

https://arxiv.org/html/2507.20198v3#bib.bib207

]   CE-VQA   Accuracy / GPT-score   Answer the question using a single word or phrase.   VideoChatGPT  [

https://arxiv.org/html/2507.20198v3#bib.bib208

]   OE-VQA   GPT-score   Evaluate the temporal accuracy of the prediction compared to the answer.*   NextQA  [

https://arxiv.org/html/2507.20198v3#bib.bib209

]   CE-VQA   WUPS   Answer a question using a short phrase or sentence.   EgoSchema  [

https://arxiv.org/html/2507.20198v3#bib.bib210

]   MC-VQA   Accuracy   Answer with the option’s letter from the given choices directly.   MVBench  [

https://arxiv.org/html/2507.20198v3#bib.bib8

]   MC-VQA   Accuracy   Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.   LongVideo Bench  [

https://arxiv.org/html/2507.20198v3#bib.bib211

]   MC-VQA   Accuracy   Answer with the option’s letter from the given choices directly.   VideoMME  [

https://arxiv.org/html/2507.20198v3#bib.bib212

]   MC-VQA   Accuracy   Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.   PerceptionTest  [

https://arxiv.org/html/2507.20198v3#bib.bib213

]   MC-VQA   Accuracy   Answer with the option’s letter from the given choices directly.   VideoDC  [

https://arxiv.org/html/2507.20198v3#bib.bib214

]   Video Caption   GPT-score   Please provide a detailed description of the video, focusing on the main subjects, their actions, and the background scenes   AuroraCap  [

https://arxiv.org/html/2507.20198v3#bib.bib45

]   Video Caption   VDCscore   Describe the video in detail.   Chardes-STA  [

https://arxiv.org/html/2507.20198v3#bib.bib215

]   Temporal Grounding   IoU   Please find the visual event described by a sentence in the video, determining its starting and ending times.

6.4.3  Deployment Hurdles

Despite their potential, many token compression methods face significant barriers to real-world deployment, stemming from a fundamental incompatibility with current large-scale model architectures and applications.

A major challenge lies in their integration with modern acceleration libraries  [

https://arxiv.org/html/2507.20198v3#bib.bib163

https://arxiv.org/html/2507.20198v3#bib.bib164

] . Methods that rely on explicit attention scores to prune tokens cannot be seamlessly integrated into  current optimized frameworks , as these libraries fuse matrix multiplication and softmax operations to maximize throughput and minimize memory usage, thereby making those scores inaccessible. This creates a critical gap, as these compression methods cannot leverage the performance gains of state-of-the-art deployment pipelines.

Furthermore, the design of certain techniques creates a fundamental disconnect with  multi-turn conversational tasks . Methods that perform token compression internally within the model’s backbone or rely on cross-modal fusion are not natively compatible with this type of application. They lack an efficient mechanism to carry over and update a compressed representation across turns, instead requiring a costly re-computation of the entire conversation history for each new query.

6.4.4  Evaluation Challenges

Rethinking Evaluation Metrics.  Current evaluation methods for token compression techniques face limitations, hindering accurate and comprehensive comparisons.

For methods requiring training, the interplay of various factors like training data and methodologies makes it challenging to isolate and directly compare the effectiveness of different approaches.

For training-free token compression methods, current evaluations often rely on metrics such as the number of compressed tokens and FLOPs. However, these metrics offer an incomplete picture. While the number of compressed tokens provides a preliminary classification, the location where compression is applied significantly impacts the downstream computational load; earlier pruning generally leads to greater reductions. Similarly, FLOPs, while useful for theoretical computational estimates, frequently do not accurately reflect actual inference speed. Therefore, for training-free methods, more practical metrics like Time To First Token (TTFT) and decoding latency per token are crucial for a more accurate assessment of real-world inference acceleration.

Evaluation Benchmarks Gap.  Current evaluation datasets for MLLM token compression, often relying on general multimodal benchmarks (Table

https://arxiv.org/html/2507.20198v3#S6.T4

), provide insufficient granularity. For example, in challenging long video understanding tasks, performance hinges more on sparse frame sampling capturing key frames than on the specific token compression method. This can obscure the true impact of token compression, making its efficacy appear negligible.

This reveals a critical gap: current datasets often fail to isolate and precisely measure the effect of token compression where it genuinely matters. Therefore, adopting evaluation methodologies similar to EffiVLM-Bench  [

https://arxiv.org/html/2507.20198v3#bib.bib216

] , which focuses on training-free acceleration evaluation, is crucial for accurately assessing the true efficacy and nuanced benefits of token compression methods.

6.5  Pruning Location and Trade-offs

Given the cascaded architecture of current MLLMs, the placement of the pruning operation directly influences the trade-off between computational efficiency and performance.

Pruning tokens at an early stage, such as within the encoder or projector, can dramatically shorten the sequence length. This significantly reduces the computational burden on the downstream LLM, leading to faster inference. However, this early compression carries a higher risk of discarding critical information, which can negatively impact overall model performance.

Conversely, token compression at a later stage, within the LLM’s internal modules, is more computationally demanding. However, it reduces the risk of erroneous judgment because the tokens have already undergone initial processing and feature extraction, thereby retaining more refined information. The optimal location for token compression within these architectures remains an open question, warranting further investigation.

6.6  Future Directions

6.6.1  Unified Multimodal Token Compression

Although different modalities exhibit unique redundancy patterns requiring specialized compression strategies, contemporary MLLMs predominantly focus on joint multimodal inference  [

https://arxiv.org/html/2507.20198v3#bib.bib10

] , whereas unimodal scenarios remain constrained by single-modality inputs. As established in Sections

https://arxiv.org/html/2507.20198v3#S3

https://arxiv.org/html/2507.20198v3#S4

https://arxiv.org/html/2507.20198v3#S5

, fundamental algorithmic principles (including transformation-based, similarity-based, attention-based, and query-based approaches) demonstrate transmodal applicability, indicating the viability of developing a unified multimodal token compression framework. Furthermore, leveraging cross-modal correlations represents a promising research direction for future token compression.

6.6.2  Improved Architecture

Current token compression methods are often employed as a remedial measure to efficiently process long contexts. However, a more valuable approach might involve designing model architectures that intrinsically account for data redundancy during their initial conception. By doing so, the number of tokens could be reduced during the abstraction of data features. This is particularly relevant for current architectures, especially those of video LLMs, where generated tokens still exhibit significant redundancy. Therefore, exploring architectural designs that inherently foster more condensed information abstraction from the outset represents a promising research direction.

7  Applications

The potential of multimodal token compression extends beyond technical enhancements, emerging as a universal efficiency engine for data-intensive AI systems. Multimodal models frequently process extreme-length token sequences exhibiting  > 70 % >70\% > 70 %  task-agnostic redundancy according to empirical analyses. Capitalizing on recent breakthroughs, we delineate four high-impact application domains:

7.1  GUI Agents and Human-Computer Interaction

Graphical user interface (GUI) agents perceive and interact with visual interfaces, interpret natural language instructions, analyze GUI states, and execute corresponding actions. These agents have to parse screen streams in real-time, producing extensive token sequences that often exceed computational limits  [

https://arxiv.org/html/2507.20198v3#bib.bib217

https://arxiv.org/html/2507.20198v3#bib.bib148

] . Multimodal token compression significantly enhances the efficiency of GUI agents. This approach mitigates context overflow in extended operation sequences by dynamically compressing redundant visual elements (e.g., extra white space or simple backgrounds). For some small but important control elements, it should also eliminate other irrelevant visual elements and highlight their importance. For instance, ShowUI  [

https://arxiv.org/html/2507.20198v3#bib.bib41

]  is the first model to apply token selection strategy to GUI agents. ShowUI segments GUI screenshots into connected components by clustering pixels with similar RGB values, significantly reducing the total number of discrete elements. During both training and inference phases, the system employs an adaptive token selection strategy that probabilistically prunes redundant tokens within these components, thereby optimizing computational efficiency while preserving functional semantics However, excessive compression risks inducing operational ambiguity, necessitating careful calibration.

7.2  Healthcare and Medical Imaging

Contemporary medical diagnosis and research critically depend on the interpretation and synthesis of multimodal medical imagery. MLLMs can integrate radiographic findings, medical histories, and ancillary diagnostic tests to generate differential diagnoses, which clinicians can correlate with patient records and physician notes to enhance diagnostic accuracy  [

https://arxiv.org/html/2507.20198v3#bib.bib218

] . Furthermore, MLLMs can automatically draft preliminary radiology reports, potentially reducing the workload of radiologists  [

https://arxiv.org/html/2507.20198v3#bib.bib219

https://arxiv.org/html/2507.20198v3#bib.bib220

https://arxiv.org/html/2507.20198v3#bib.bib221

] . However, high-dimensional modalities like PET-MRI exhibit prohibitively large data volumes while containing diagnostically critical yet sparse features  [

https://arxiv.org/html/2507.20198v3#bib.bib222

] . Token compression techniques enable pathology-aware local lossless compression. Through hierarchical compression of anatomical structure tokens and pathological feature representations, these methods significantly reduce computational costs while maintaining diagnostic confidence levels. Critical attention must be devoted to preserving sub-resolution pathological features, particularly micro-lesions that may indicate early-stage conditions.

7.3  Robotics and Autonomous Systems

Leveraging the significant capabilities of video LLMs in long-form video comprehension enables their deployment in robotics  [

https://arxiv.org/html/2507.20198v3#bib.bib42

]  and autonomous driving systems  [

https://arxiv.org/html/2507.20198v3#bib.bib43

https://arxiv.org/html/2507.20198v3#bib.bib223

] . However, the inherent computational complexity of long-duration video processing creates fundamental latency-efficiency tradeoffs that challenge real-time implementation. Token compression addresses this by prioritizing salient spatio-temporal dynamics (e.g., agent movements, action trajectories) and fine-grained per-frame details, enabling computationally efficient video understanding for these domains. VTS  [

https://arxiv.org/html/2507.20198v3#bib.bib43

]  proposes a token pruning strategy for autonomous driving scenarios. VTS employs a proposal model based on a lightweight convolutional neural network that is able to adaptively identify keyframes and pry less informative tokens (e.g., invariant backgrounds and stationary objects). StreamVLN  [

https://arxiv.org/html/2507.20198v3#bib.bib42

]  further enhances inference efficiency for real-time navigation by employing a voxel-based spatial pruning strategy at test time to reduce memory tokens. This approach makes real-time navigation feasible.

7.4  Efficient Reasoning

Token compression improves efficiency by removing redundant input tokens. However, in many cases, the main source of computational cost shifts from the input to the output, most notably in reasoning models  [

https://arxiv.org/html/2507.20198v3#bib.bib224

https://arxiv.org/html/2507.20198v3#bib.bib225

https://arxiv.org/html/2507.20198v3#bib.bib226

] , where lengthy generation chains are common. The “slow-thinking” paradigm improves reasoning ability but results in lengthy reasoning chains  [

https://arxiv.org/html/2507.20198v3#bib.bib227

https://arxiv.org/html/2507.20198v3#bib.bib228

https://arxiv.org/html/2507.20198v3#bib.bib229

https://arxiv.org/html/2507.20198v3#bib.bib230

] . Some efficient reasoning methods compress these chains using similar techniques (e.g., attention mechanisms, semantic importance)  [

https://arxiv.org/html/2507.20198v3#bib.bib231

https://arxiv.org/html/2507.20198v3#bib.bib232

https://arxiv.org/html/2507.20198v3#bib.bib233

https://arxiv.org/html/2507.20198v3#bib.bib234

] , typically requiring fine-tuning via Supervised Fine-Tuning (SFT) or Reinforcement Learning (RL). Beyond token compression, other approaches improve reasoning efficiency by compressing the model  [

https://arxiv.org/html/2507.20198v3#bib.bib235

https://arxiv.org/html/2507.20198v3#bib.bib236

https://arxiv.org/html/2507.20198v3#bib.bib237

https://arxiv.org/html/2507.20198v3#bib.bib238

]  or accelerating decoding  [

https://arxiv.org/html/2507.20198v3#bib.bib239

https://arxiv.org/html/2507.20198v3#bib.bib240

https://arxiv.org/html/2507.20198v3#bib.bib241

https://arxiv.org/html/2507.20198v3#bib.bib242

https://arxiv.org/html/2507.20198v3#bib.bib243

8  Conclusion

This paper presents the first comprehensive and structured survey of multimodal long-context token compression for Multimodal Large Language Models (MLLMs). This is a critical area, driven by the growing need to process extensive and intricate data across various modalities. We systematically categorized existing methods based on their primary data focus: image-centric, video-centric, and audio-centric compression, acknowledging that effective strategies are intrinsically linked to the unique characteristics and redundancies of each modality. Furthermore, we delved into the underlying mechanisms of these methods, including transformation-based, similarity-based, attention-based, and query-based approaches. By consolidating current progress and identifying key challenges, this survey aims to inspire future research and accelerate advancements in this rapidly evolving field. The insights presented here are designed to guide researchers in developing more efficient and scalable MLLMs, ultimately pushing the boundaries of what these powerful models can achieve with increasingly longer and more complex inputs.

[1]    H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” in

Proc. Adv. Neural Inform. Process. Syst.

[2]    B. Li, Y. Zhang, D. Guo, R. Zhang, F. Li, H. Zhang, K. Zhang, P. Zhang, Y. Li, Z. Liu, and C. Li, “Llava-onevision: Easy visual task transfer,”

Trans. Mach. Learn. Res.

[3]    L. Xu, Y. Zhao, D. Zhou, Z. Lin, S. K. Ng, and J. Feng, “Pllava: Parameter-free llava extension from images to videos for video dense captioning,”

arXiv preprint arXiv:2404.16994

[4]    J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang

, “Qwen technical report,”

arXiv preprint arXiv:2309.16609

[5]    J. Xu, Z. Guo, J. He, H. Hu, T. He, S. Bai, K. Chen, J. Wang, Y. Fan, K. Dang

, “Qwen2. 5-omni technical report,”

arXiv preprint arXiv:2503.20215

[6]    B. Lin, Y. Ye, B. Zhu, J. Cui, M. Ning, P. Jin, and L. Yuan, “Video-llava: Learning united visual representation by alignment before projection,” in

Proc. Conf. Empir. Methods Nat. Lang. Process.

[7]    H. Zhang, X. Li, and L. Bing, “Video-llama: An instruction-tuned audio-visual language model for video understanding,” in

Proc. Conf. Empir. Methods Nat. Lang. Process.

[8]    K. Li, Y. Wang, Y. He, Y. Li, Y. Wang, Y. Liu, Z. Wang, J. Xu, G. Chen, P. Luo

, “Mvbench: A comprehensive multi-modal video understanding benchmark,” in

Proc. IEEE Conf. Comput. Vis. Pattern Recognit.

[9]    K. Li, Y. He, Y. Wang, Y. Li, W. Wang, P. Luo, Y. Wang, L. Wang, and Y. Qiao, “Videochat: Chat-centric video understanding,”

arXiv preprint arXiv:2305.06355

[10]    Z. Cheng, S. Leng, H. Zhang, Y. Xin, X. Li, G. Chen, Y. Zhu, W. Zhang, Z. Luo, D. Zhao

, “Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llm,”

arXiv preprint arXiv:2406.07476

[11]    B. Zhang, K. Li, Z. Cheng, Z. Hu, Y. Yuan, G. Chen, S. Leng, Y. Jiang, H. Zhang, X. Li

, “Videollama 3: Frontier multimodal foundation models for image and video understanding,”

arXiv preprint arXiv:2501.13106

[12]    W.-L. Chiang, Z. Li, Z. Lin, Y. Sheng, Z. Wu, H. Zhang, L. Zheng, S. Zhuang, Y. Zhuang, J. E. Gonzalez, I. Stoica, and E. P. Xing, “Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality,” March 2023. [Online]. Available:

https://lmsys.org/blog/2023-03-30-vicuna/

https://lmsys.org/blog/2023-03-30-vicuna/

[13]    Q. Team, “Qwen2 technical report,”

arXiv preprint arXiv:2407.10671

[14]    AI@Meta, “Llama 3 model card,”

https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md

https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md

[15]    M. Abdin, J. Aneja, H. Behl, S. Bubeck, R. Eldan, S. Gunasekar, M. Harrison, R. J. Hewett, M. Javaheripi, P. Kauffmann

, “Phi-4 technical report,”

arXiv preprint arXiv:2412.08905

[16]    K. Shao, K. Tao, C. Qin, H. You, Y. Sui, and H. Wang, “Holitom: Holistic token merging for fast video large language models,”

arXiv preprint arXiv:2505.21334

[17]    K. Tao, C. Qin, H. You, Y. Sui, and H. Wang, “Dycoke: Dynamic compression of tokens for fast video large language models,” in

Proc. IEEE Conf. Comput. Vis. Pattern Recognit.

[18]    S. Yang, Y. Chen, Z. Tian, C. Wang, J. Li, B. Yu, and J. Jia, “Visionzip: Longer is better but not necessary in vision language models,” in

Proc. IEEE Conf. Comput. Vis. Pattern Recognit.

[19]    Y. Rao, W. Zhao, B. Liu, J. Lu, J. Zhou, and C.-J. Hsieh, “Dynamicvit: Efficient vision transformers with dynamic token sparsification,” in

Proc. Adv. Neural Inform. Process. Syst.

[20]    Y. Liang, C. Ge, Z. Tong, Y. Song, J. Wang, and P. Xie, “Not all patches are what you need: Expediting vision transformers via token reorganizations,” in

Proc. Int. Conf. Learn. Represent.

[21]    D. Bolya, C.-Y. Fu, X. Dai, P. Zhang, C. Feichtenhofer, and J. Hoffman, “Token merging: Your vit but faster,” in

Proc. Int. Conf. Learn. Represent.

[22]    M. S. Ryoo, A. Piergiovanni, A. Arnab, M. Dehghani, and A. Angelova, “Tokenlearner: What can 8 learned tokens do for images and videos?”

arXiv preprint arXiv:2106.11297

[23]    H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, and H. Jégou, “Training data-efficient image transformers & distillation through attention,” in

Proc. Int. Conf. Mach. Learn.

[24]    A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” in

Proc. Adv. Neural Inform. Process. Syst.

[25]    S. Yang, J. Li, X. Lai, B. Yu, H. Zhao, and J. Jia, “Visionthink: Smart and efficient vision language model via reinforcement learning,”

arXiv preprint arXiv:2507.13348

[26]    L. Chen, H. Zhao, T. Liu, S. Bai, J. Lin, C. Zhou, and B. Chang, “An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models,” in

Proc. Eur. Conf. Comput. Vis.

[27]    X. Huang, H. Zhou, and K. Han, “Prunevid: Visual token pruning for efficient video large language models,” in

Proc. Annu. Meet. Assoc. Comput. Linguist.

[28]    S. R. Alvar, G. Singh, M. Akbari, and Y. Zhang, “Divprune: Diversity-based visual token pruning for large multimodal models,” in

Proc. IEEE Conf. Comput. Vis. Pattern Recognit.

[29]    Y. Shang, M. Cai, B. Xu, Y. J. Lee, and Y. Yan, “Llava-prumerge: Adaptive token reduction for efficient large multimodal models,” in

Proc. IEEE Int. Conf. Comput. Vis.

[30]    Z. Chen, W. Wang, H. Tian, S. Ye, Z. Gao, E. Cui, W. Tong, K. Hu, J. Luo, Z. Ma

, “How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites,”

Sci. China Inf. Sci.

, vol. 67, no. 12, p. 220101, 2024.

[31]    W. Dai, N. Lee, B. Wang, Z. Yang, Z. Liu, J. Barker, T. Rintamaki, M. Shoeybi, B. Catanzaro, and W. Ping, “Nvlm: Open frontier-class multimodal llms,”

arXiv preprint arXiv:2409.11402

[32]    P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge

, “Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution,”

arXiv preprint arXiv:2409.12191

[33]    S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang

, “Qwen2. 5-vl technical report,”

arXiv preprint arXiv:2502.13923

[34]    Y. Zhang, J. Wu, W. Li, B. Li, Z. Ma, Z. Liu, and C. Li, “Video instruction tuning with synthetic data,”

arXiv preprint arXiv:2410.02713

[35]    M. Cai, J. Yang, J. Gao, and Y. J. Lee, “Matryoshka multimodal models,” in

## NeurIPS Workshop

[36]    L. Yao, L. Li, S. Ren, L. Wang, Y. Liu, X. Sun, and L. Hou, “Deco: Decoupling token compression from semantic abstraction in multimodal large language models,”

arXiv preprint arXiv:2405.20985

[37]    J. Cha, W. Kang, J. Mun, and B. Roh, “Honeybee: Locality-enhanced projector for multimodal llm,” in

Proc. IEEE Conf. Comput. Vis. Pattern Recognit.

[38]    X. Chu, L. Qiao, X. Lin, S. Xu, Y. Yang, Y. Hu, F. Wei, X. Zhang, B. Zhang, X. Wei

, “Mobilevlm: A fast, strong and open vision language assistant for mobile devices,”

arXiv preprint arXiv:2312.16886

[39]    X. Chu, L. Qiao, X. Zhang, S. Xu, F. Wei, Y. Yang, X. Sun, Y. Hu, X. Lin, B. Zhang

, “Mobilevlm v2: Faster and stronger baseline for vision language model,”

arXiv preprint arXiv:2402.03766

[40]    Y. Li, C. Wang, and J. Jia, “Llama-vid: An image is worth 2 tokens in large language models,” in

Proc. Eur. Conf. Comput. Vis.

[41]    K. Q. Lin, L. Li, D. Gao, Z. Yang, S. Wu, Z. Bai, S. W. Lei, L. Wang, and M. Z. Shou, “Showui: One vision-language-action model for gui visual agent,” in

Proc. IEEE Conf. Comput. Vis. Pattern Recognit.

, 2025, pp. 19 498–19 508.

[42]    M. Wei, C. Wan, X. Yu, T. Wang, Y. Yang, X. Mao, C. Zhu, W. Cai, H. Wang, Y. Chen

, “Streamvln: Streaming vision-and-language navigation via slowfast context modeling,”

arXiv preprint arXiv:2507.05240

[43]    Y. Ma, A. Abdelraouf, R. Gupta, Z. Wang, and K. Han, “Video token sparsification for efficient multimodal llms in autonomous driving,”

arXiv preprint arXiv:2409.11182

[44]    L. Shen, G. Gong, T. He, Y. Zhang, P. Liu, S. Zhao, and G. Ding, “Fastvid: Dynamic density pruning for fast video large language models,”

arXiv preprint arXiv:2503.11187

[45]    W. Chai, E. Song, Y. Du, C. Meng, V. Madhavan, O. Bar-Tal, J.-N. Hwang, S. Xie, and C. D. Manning, “Auroracap: Efficient, performant video detailed captioning and a new benchmark,” in

Proc. Int. Conf. Learn. Represent.

[46]    Y. He, F. Chen, J. Liu, W. Shao, H. Zhou, K. Zhang, and B. Zhuang, “Zipvl: Efficient large vision-language models with dynamic token sparsification and kv cache compression,” in

Proc. IEEE Int. Conf. Comput. Vis.

[47]    C. Zhang, K. Ma, T. Fang, W. Yu, H. Zhang, Z. Zhang, Y. Xie, K. Sycara, H. Mi, and D. Yu, “Vscan: Rethinking visual token reduction for efficient large vision-language models,”

arXiv preprint arXiv:2505.22654

[48]    Q. Cao, B. Paranjape, and H. Hajishirzi, “Pumer: Pruning and merging tokens for efficient vision language models,”

arXiv preprint arXiv:2305.17530

[49]    C. Yang, Y. Sui, J. Xiao, L. Huang, Y. Gong, C. Li, J. Yan, Y. Bai, P. Sadayappan, X. Hu

, “Topv: Compatible token pruning with inference time optimization for fast and low-memory multimodal vision language model,” in

Proc. IEEE Conf. Comput. Vis. Pattern Recognit.

[50]    K. Tao, H. You, Y. Sui, C. Qin, and H. Wang, “Plug-and-play 1. x-bit kv cache quantization for video large language models,”

arXiv preprint arXiv:2503.16257

[51]    Y. Zhang, C.-K. Fan, J. Ma, W. Zheng, T. Huang, K. Cheng, D. Gudovskiy, T. Okuno, Y. Nakata, K. Keutzer

, “Sparsevlm: Visual token sparsification for efficient vision-language model inference,”

arXiv preprint arXiv:2410.04417

[52]    T. Liu, L. Shi, R. Hong, Y. Hu, Q. Yin, and L. Zhang, “Multi-stage vision yoken dropping: Towards efficient multimodal large language model,”

arXiv preprint arXiv:2411.10803

[53]    Z. Li, Y. Liu, Y. Su, and N. Collier, “Prompt compression for large language models: A survey,” in

Proc. Conf. North American Chapter Assoc. Comput. Linguist.

[54]    Z. Kong, Y. Li, F. Zeng, L. Xin, S. Messica, X. Lin, P. Zhao, M. Kellis, H. Tang, and M. Zitnik, “Token reduction should go beyond efficiency in generative models–from vision, language to multimodality,”

arXiv preprint arXiv:2505.18227

[55]    X. Liu, Z. Wen, S. Wang, J. Chen, Z. Tao, Y. Wang, X. Jin, C. Zou, Y. Wang, C. Liao

, “Shifting ai efficiency from model-centric to data-centric compression,”

arXiv preprint arXiv:2505.19147

[56]    J. Liu, L. Niu, W. Chen, J. Zhou, and F. Meng, “Laco: Efficient layer-wise compression of visual tokens for multimodal large language models,”

arXiv preprint arXiv:2507.02279

[57]    D. Guo, F. Wu, F. Zhu, F. Leng, G. Shi, H. Chen, H. Fan, J. Wang, J. Jiang, J. Wang

, “Seed1. 5-vl technical report,”

arXiv preprint arXiv:2505.07062

[58]    M. Xu, M. Gao, Z. Gan, H.-Y. Chen, Z. Lai, H. Gang, K. Kang, and A. Dehghan, “Slowfast-llava: A strong training-free baseline for video large language models,”

arXiv preprint arXiv:2407.15841

[59]    H. Wang, Y. Nie, Y. Ye, D. GuanYu, Y. Wang, S. Li, H. Yu, J. Lu, and C. Huang, “Dynamic-vlm: Simple dynamic visual token compression for videollm,”

arXiv preprint arXiv:2412.09530

[60]    D. Bolya, C.-Y. Fu, X. Dai, P. Zhang, C. Feichtenhofer, and J. Hoffman, “Token merging: Your vit but faster,” in

Proc. Int. Conf. Learn. Represent.

[61]    H. Wang, Z. Yu, G. Spadaro, C. Ju, V. Quétu, S. Xiao, and E. Tartaglione, “Folder: Accelerating multi-modal large language models with enhanced performance,” in

Proc. IEEE Int. Conf. Comput. Vis.

[62]    W. Zeng, Z. Huang, K. Ji, and Y. Yan, “Skip-vision: Efficient and scalable acceleration of vision-language models via adaptive token skipping,” in

Proc. IEEE Int. Conf. Comput. Vis.

[63]    Z. Wang, S. Purushwalkam, C. Xiong, S. Savarese, H. Ji, and R. Xu, “Dymu: Dynamic merging and virtual unmerging for efficient vlms,”

arXiv preprint arXiv:2504.17040

[64]    J. Hyun, S. Hwang, S. H. Han, T. Kim, I. Lee, D. Wee, J.-Y. Lee, S. J. Kim, and M. Shim, “Multi-granular spatio-temporal token merging for training-free acceleration of video llms,”

arXiv preprint arXiv:2507.07990

[65]    Q. Zhang, A. Cheng, M. Lu, R. Zhang, Z. Zhuo, J. Cao, S. Guo, Q. She, and S. Zhang, “Beyond text-visual attention: Exploiting visual cues for effective token pruning in vlms,” in

Proc. IEEE Int. Conf. Comput. Vis.

[66]    L. Hu, F. Shang, L. Wan, and W. Feng, “Illava: An image is worth fewer than 1/3 input tokens in large multimodal models,”

arXiv preprint arXiv:2412.06263

[67]    Y. Jiang, Q. Wu, W. Lin, W. Yu, and Y. Zhou, “What kind of visual tokens do we need? training-free visual token pruning for multi-modal large language models from the perspective of graph,” in

Proc. AAAI Conf. Artif. Intell.

[68]    K. Li, X. Chen, C. Gao, Y. Li, and X. Chen, “Balanced token pruning: Accelerating vision language models beyond local optimization,”

arXiv preprint arXiv:2505.22038

[69]    X. Liu, Z. Wang, Y. Han, Y. Wang, J. Yuan, J. Song, B. Zheng, L. Zhang, S. Huang, and H. Chen, “Compression with global guidance: Towards training-free high-resolution mllms acceleration,”

arXiv preprint arXiv:2501.05179

[70]    K. H. I. Arif, J. Yoon, D. S. Nikolopoulos, H. Vandierendonck, D. John, and B. Ji, “Hired: Attention-guided token dropping for efficient inference of high-resolution vision-language models in resource-constrained environments,” in

Proc. AAAI Conf. Artif. Intell.

[71]    Y. Han, X. Liu, P. Ding, D. Wang, H. Chen, Q. Yan, and S. Huang, “Rethinking token reduction in mllms: Towards a unified paradigm for training-free acceleration,”

arXiv preprint arXiv:2411.17686

[72]    L. Xing, Q. Huang, X. Dong, J. Lu, P. Zhang, Y. Zang, Y. Cao, C. He, J. Wang, F. Wu

, “Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction,” in

Proc. IEEE Conf. Comput. Vis. Pattern Recognit.

[73]    Z. Lin, M. Lin, L. Lin, and R. Ji, “Boosting multimodal large language models with visual tokens withdrawal for rapid inference,” in

Proc. AAAI Conf. Artif. Intell.

[74]    W. Ye, Q. Wu, W. Lin, and Y. Zhou, “Fit and prune: Fast and training-free visual token pruning for multi-modal large language models,” in

Proc. AAAI Conf. Artif. Intell.

[75]    J. Zhuang, L. Lu, M. Dai, R. Hu, J. Chen, Q. Liu, and H. Hu, “St3: Accelerating multimodal large language model by spatial-temporal visual token trimming,” in

Proc. AAAI Conf. Artif. Intell.

[76]    X. Ye, Y. Gan, Y. Ge, X.-P. Zhang, and Y. Tang, “Atp-llava: Adaptive token pruning for large vision language models,” in

Proc. IEEE Conf. Comput. Vis. Pattern Recognit.

[77]    Q. Wang, H. Ye, M.-Y. Chung, Y. Liu, Y. Lin, M. Kuo, M. Ma, J. Zhang, and Y. Chen, “Corematching: A co-adaptive sparse inference framework with token and neuron pruning for comprehensive acceleration of vision-language models,” in

Proc. Int. Conf. Mach. Learn.

[78]    X. Tan, P. Ye, C. Tu, J. Cao, Y. Yang, L. Zhang, D. Zhou, and T. Chen, “Tokencarve: Information-preserving visual token compression in multimodal large language models,”

arXiv preprint arXiv:2503.10501

[79]    T. Fu, T. Liu, Q. Han, G. Dai, S. Yan, H. Yang, X. Ning, and Y. Wang, “Framefusion: Combining similarity and importance for video token reduction on large visual language models,” in

Proc. IEEE Int. Conf. Comput. Vis.

[80]    S. Zhao, Z. Wang, F. Juefei-Xu, X. Xia, M. Liu, X. Wang, M. Liang, N. Zhang, D. N. Metaxas, and L. Yu, “Accelerating multimodal large language models by searching optimal vision token reduction,” in

Proc. IEEE Conf. Comput. Vis. Pattern Recognit.

[81]    J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models,” in

Proc. Int. Conf. Mach. Learn.

[82]    Q. Ye, H. Xu, G. Xu, J. Ye, M. Yan, Y. Zhou, J. Wang, A. Hu, P. Shi, Y. Shi

, “Mplug-owl: Modularization empowers large language models with multimodality,”

arXiv preprint arXiv:2304.14178

[83]    D. Zhu, J. Chen, X. Shen, X. Li, and M. Elhoseiny, “Minigpt-4: Enhancing vision-language understanding with advanced large language models,” in

Proc. Int. Conf. Learn. Represent.

[84]    J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds

, “Flamingo: A visual language model for few-shot learning,”

Proc. Adv. Neural Inform. Process. Syst.

[85]    S. Zhang, Q. Fang, Z. Yang, and Y. Feng, “Llava-mini: Efficient image and video large multimodal models with one vision token,” in

Proc. Int. Conf. Learn. Represent.

[86]    X. Ye, Y. Gan, X. Huang, Y. Ge, and Y. Tang, “Voco-llama: Towards vision compression with large language models,” in

Proc. IEEE Conf. Comput. Vis. Pattern Recognit.

[87]    Y. Wen, Q. Cao, Q. Fu, S. Mehta, and M. Najibi, “Efficient vision-language models by summarizing visual tokens into compact registers,”

arXiv preprint arXiv:2410.14072

[88]    J. Han, L. Du, Y. Wu, X. Zhou, H. Du, and W. Zheng, “Adafv: Accelerating vlms with self-adaptive cross-modality attention mixture,”

arXiv preprint arXiv:2501.09532

[89]    D. Song, W. Wang, S. Chen, X. Wang, M. Guan, and B. Wang, “Less is more: A simple yet effective token reduction method for efficient multi-modal llms,” in

Proc. Int. Conf. Comput. Linguist.

[90]    M. Maaz, H. Rasheed, S. Khan, and F. Khan, “Video-chatgpt: Towards detailed video understanding via large vision and language models,” in

Proc. Annu. Meet. Assoc. Comput. Linguist.

[91]    Y. Weng, M. Han, H. He, X. Chang, and B. Zhuang, “Longvlm: Efficient long video understanding via large language models,” in

Proc. Eur. Conf. Comput. Vis.

[92]    M. Xu, M. Gao, S. Li, J. Lu, Z. Gan, Z. Lai, M. Cao, K. Kang, Y. Yang, and A. Dehghan, “Slowfast-llava-1.5: A family of token-efficient video large language models for long-form video understanding,”

arXiv preprint arXiv:2503.18943

[93]    X. Li, Y. Wang, J. Yu, X. Zeng, Y. Zhu, H. Huang, J. Gao, K. Li, Y. He, C. Wang

, “Videochat-flash: Hierarchical compression for long-context video modeling,”

arXiv preprint arXiv:2501.00574

[94]    P. Jin, R. Takanobu, W. Zhang, X. Cao, and L. Yuan, “Chat-univi: Unified visual representation empowers large language models with image and video understanding,” in

Proc. IEEE Conf. Comput. Vis. Pattern Recognit.

[95]    X. Shen, Y. Xiong, C. Zhao, L. Wu, J. Chen, C. Zhu, Z. Liu, F. Xiao, B. Varadarajan, F. Bordes

, “Longvu: Spatiotemporal adaptive compression for long video-language understanding,” in

Proc. Int. Conf. Mach. Learn.

[96]    X. Liu, Y. Wang, J. Ma, and L. Zhang, “Video compression commander: Plug-and-play inference acceleration for video large language models,”

arXiv preprint arXiv:2505.14454

[97]    H. Zhang, J. Zhang, X. Ji, Q. Wang, and F. Zhang, “Dyntok: Dynamic compression of visual tokens for efficient and effective video understanding,”

arXiv preprint arXiv:2506.03990

[98]    E. Song, W. Chai, G. Wang, Y. Zhang, H. Zhou, F. Wu, H. Chi, X. Guo, T. Ye, Y. Zhang

, “Moviechat: From dense token to sparse memory for long video understanding,” in

Proc. IEEE Conf. Comput. Vis. Pattern Recognit.

[99]    Y. Shu, Z. Liu, P. Zhang, M. Qin, J. Zhou, Z. Liang, T. Huang, and B. Zhao, “Video-xl: Extra-long vision language model for hour-scale video understanding,” in

Proc. IEEE Conf. Comput. Vis. Pattern Recognit.

[100]    L. Yao, Y. Li, Y. Wei, L. Li, S. Ren, Y. Liu, K. Ouyang, L. Wang, S. Li, S. Li, L. Kong, Q. Liu, Y. Zhang, and X. Sun, “Timechat-online: 80

[101]    H. Huang, F. Chen, W. Chai, C. Su, L. Xia, S. Jung, C. Yang, J. Hwang, M. Sun, and C. Kuo, “Zero-shot 3d question answering via voxel-based dynamic token compression,” in

Proc. IEEE Conf. Comput. Vis. Pattern Recognit.

[102]    B. Sun, J. Zhao, X. Wei, and Q. Hou, “Llava-scissor: Token compression with semantic connected components for video llms,”

arXiv preprint arXiv:2506.21862

[103]    Y. Zhong, Z. Liu, Y. Li, and L. Wang, “Aim: Adaptive inference of multi-modal llms via token merging and pruning,” in

Proc. IEEE Int. Conf. Comput. Vis.

[104]    M. S. Ryoo, K. Gopalakrishnan, K. Kahatapitiya, T. Xiao, K. Rao, A. Stone, Y. Lu, J. Ibarz, and A. Arnab, “Token turing machines,” in

Proc. IEEE Conf. Comput. Vis. Pattern Recognit.

[105]    M. S. Ryoo, H. Zhou, S. Kendre, C. Qin, L. Xue, M. Shu, J. Park, K. Ranasinghe, S. Savarese, R. Xu

, “Xgen-mm-vid (blip-3-video): You only need 32 tokens to represent a video even in vlms,”

arXiv preprint arXiv:2410.16267

[106]    S. Gurukar and A. Kadav, “Long-vmnet: Accelerating long-form video understanding via fixed memory,”

arXiv preprint arXiv:2503.13707

[107]    J. Jiang, X. Li, Z. Liu, M. Li, G. Chen, Z. Li, D.-A. Huang, G. Liu, Z. Yu, K. Keutzer

, “Token-efficient long video understanding for multimodal llms,”

arXiv preprint arXiv:2503.04130

[108]    L. Gao, Y. Zhong, Y. Zeng, H. Tan, D. Li, and Z. Zhao, “Linvt: Empower your image-level large language model to understand videos,”

arXiv preprint arXiv:2412.05185

[109]    K. Chen, X. Du, B. Zhu, Z. Ma, T. Berg-Kirkpatrick, and S. Dubnov, “Hts-at: A hierarchical token-semantic audio transformer for sound classification and detection,” in

Proc. IEEE Int. Conf. Acoust. Speech Signal Process.

[110]    Z. Ma, G. Yang, Y. Yang, Z. Gao, J. Wang, Z. Du, F. Yu, Q. Chen, S. Zheng, S. Zhang

, “An embarrassingly simple approach for llm with strong asr capacity,”

arXiv preprint arXiv:2402.08846

[111]    Q. Fang, S. Guo, Y. Zhou, Z. Ma, S. Zhang, and Y. Feng, “Llama-omni: Seamless speech interaction with large language models,”

arXiv preprint arXiv:2409.06666

[112]    N. Das, S. Dingliwal, S. Ronanki, R. Paturi, Z. Huang, P. Mathur, J. Yuan, D. Bekal, X. Niu, S. M. Jayanthi

, “Speechverse: A large-scale generalizable audio language model,”

arXiv preprint arXiv:2405.08295

[113]    Y. Chu, J. Xu, Q. Yang, H. Wei, X. Wei, Z. Guo, Y. Leng, Y. Lv, J. He, J. Lin

, “Qwen2-audio technical report,”

arXiv preprint arXiv:2407.10759

[114]    T. Li, J. Liu, T. Zhang, Y. Fang, D. Pan, M. Wang, Z. Liang, Z. Li, M. Lin, G. Dong

, “Baichuan-audio: A unified framework for end-to-end speech interaction,”

arXiv preprint arXiv:2502.17239

[115]    U. Cappellazzo, M. Kim, H. Chen, P. Ma, S. Petridis, D. Falavigna, A. Brutti, and M. Pantic, “Large language models are strong audio-visual speech recognition learners,” in

Proc. IEEE Int. Conf. Acoust. Speech Signal Process.

[116]    U. Cappellazzo, M. Kim, and S. Petridis, “Adaptive audio-visual speech recognition via matryoshka-based multimodal llms,”

arXiv preprint arXiv:2503.06362

[117]    X. Geng, K. Wei, Q. Shao, S. Liu, Z. Lin, Z. Zhao, G. Li, W. Tian, P. Chen, Y. Li

, “Osum: Advancing open speech understanding models with limited resources in academia,”

arXiv preprint arXiv:2501.13306

[118]    H. Gao, H. Shao, X. Wang, C. Qiu, Y. Shen, S. Cai, Y. Shi, Z. Xu, Z. Long, Y. Zhang

, “Lucy: Linguistic understanding and control yielding early stage of her,”

arXiv preprint arXiv:2501.16327

[119]    Y. Li, Y. Wu, J. Li, and S. Liu, “Accelerating transducers through adjacent token merging,” in

## Interspeech

[120]    T. Lee and H. Lee, “Token pruning in audio transformers: Optimizing performance and decoding patch importance,”

arXiv preprint arXiv:2504.01690

[121]    Y. Lin, Y. Fu, J. Zhang, Y. Liu, J. Zhang, J. Sun, H. Li, Y. Chen

, “Speechprune: Context-aware token pruning for speech information retrieval,” in

Proc. IEEE Int. Conf. Multimed. Expo

[122]    C. Tang, W. Yu, G. Sun, X. Chen, T. Tan, W. Li, L. Lu, Z. MA, and C. Zhang, “Salmonn: Towards generic hearing abilities for large language models,” in

Proc. Int. Conf. Learn. Represent.

[123]    G. Sun, W. Yu, C. Tang, X. Chen, T. Tan, W. Li, L. Lu, Z. MA, Y. Wang, and C. Zhang, “Video-salmonn: Speech-enhanced audio-visual large language models,” in

Proc. Int. Conf. Mach. Learn.

[124]    K. Pipatanakul, P. Manakul, N. Nitarach, W. Sirichotedumrong, S. Nonesung, T. Jaknamon, P. Pengpun, P. Taveekitworachai, A. Na-Thalang, S. Sripaisarnmongkol

, “Typhoon 2: A family of open text and multimodal thai large language models,”

arXiv preprint arXiv:2412.13702

[125]    J. Xue, Y. Deng, Y. Han, Y. Gao, and Y. Li, “Improving audio codec-based zero-shot text-to-speech synthesis with multi-modal context and large language model,” in

## Interspeech

[126]    J. H. Yeo, H. Rha, S. J. Park, and Y. M. Ro, “Mms-llama: Efficient llm-based audio-visual speech recognition with minimal multimodal speech tokens,”

arXiv preprint arXiv:2503.11315

[127]    A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark

, “Learning transferable visual models from natural language supervision,” in

Proc. Int. Conf. Mach. Learn.

[128]    X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, “Sigmoid loss for language image pre-training,” in

Proc. IEEE Conf. Comput. Vis. Pattern Recognit.

[129]    M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, and A. Joulin, “Emerging properties in self-supervised vision transformers,” in

Proc. IEEE Conf. Comput. Vis. Pattern Recognit.

[130]    M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby

, “Dinov2: Learning robust visual features without supervision,”

arXiv preprint arXiv:2304.07193

[131]    A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in

Proc. Int. Conf. Mach. Learn.

[132]    A. Guzhov, F. Raue, J. Hees, and A. Dengel, “Audioclip: Extending clip to image, text and audio,” in

Proc. IEEE Int. Conf. Acoust. Speech Signal Process.

[133]    A. Chevalier, A. Wettig, A. Ajith, and D. Chen, “Adapting language models to compress contexts,”

arXiv preprint arXiv:2305.14788

[134]    N. Shao, S. Xiao, Z. Liu, and P. Zhang, “Flexibly scaling large language models contexts through extensible tokenization,”

arXiv preprint arXiv:2401.07793

[135]    H. An, Y. Chen, Z. Sun, and X. Li, “Sentencevae: Enable next-sentence prediction for large language models with faster speed, higher accuracy and longer context,”

arXiv preprint arXiv:2408.00655

[136]    Y. Li, B. Dong, C. Lin, and F. Guerin, “Compressing context to enhance inference efficiency of large language models,” in

Proc. Conf. Empir. Methods Nat. Lang. Process.

[137]    H. Jiang, Q. Wu, C.-Y. Lin, Y. Yang, and L. Qiu, “Llmlingua: Compressing prompts for accelerated inference of large language models,” in

Proc. Conf. Empir. Methods Nat. Lang. Process.

[138]    H. Jiang, Q. Wu, X. Luo, D. Li, C.-Y. Lin, Y. Yang, and L. Qiu, “Longllmlingua: Accelerating and enhancing llms in long context scenarios via prompt compression,” in

Proc. Annu. Meet. Assoc. Comput. Linguist.

[139]    Z. Pan, Q. Wu, H. Jiang, M. Xia, X. Luo, J. Zhang, Q. Lin, V. Rühle, Y. Yang, C.-Y. Lin

, “Llmlingua-2: Data distillation for efficient and faithful task-agnostic prompt compression,” in

Findings Assoc. Comput. Linguist.

[140]    W. Wang, Y. Wang, Y. Fan, H. Liao, and J. Guo, “Quito: Accelerating long-context reasoning through query-guided context compression,” in

Proc. China Conf. Inf. Retr.

, 2024, pp. 136–148.

[141]    Y. Wang, X. Huang, B. Tian, Y. Su, L. Yu, H. Liao, Y. Fan, J. Guo, and X. Cheng, “Quito-x: A new perspective on context compression from the information bottleneck theory,”

arXiv preprint arXiv:2408.10497

[142]    Q. Zhang, H. Zhang, L. Pang, H. Zheng, and Z. Zheng, “Adacomp: Extractive context compression with adaptive predictor for retrieval-augmented large language models,”

arXiv preprint arXiv:2409.01579

[143]    K. Shi, X. Sun, Q. Li, and G. Xu, “Compressing long context for enhancing rag with amr-based concept distillation,”

arXiv preprint arXiv:2405.03085

[144]    X. Cheng, X. Wang, X. Zhang, T. Ge, S.-Q. Chen, F. Wei, H. Zhang, and D. Zhao, “Xrag: Extreme context compression for retrieval-augmented generation with one token,” in

Proc. Adv. Neural Inform. Process. Syst.

[145]    T. Ge, J. Hu, L. Wang, X. Wang, S.-Q. Chen, and F. Wei, “In-context autoencoder for context compression in a large language model,”

arXiv preprint arXiv:2307.06945

[146]    C. Huang, G. Zhu, X. Wang, Y. Luo, G. Ge, H. Chen, D. Yi, and J. Wang, “Recurrent context compression: Efficiently expanding the context window of llm,”

arXiv preprint arXiv:2406.06110

[147]    S. Wang, Y. Bai, L. Zhang, P. Zhou, S. Zhao, G. Zhang, S. Wang, R. Chen, H. Xu, and H. Sun, “Xl3m: A training-free framework for llm length extension based on segment-wise inference,”

arXiv preprint arXiv:2405.17755

[148]    C. Wang, Y. Yang, R. Li, D. Sun, R. Cai, Y. Zhang, and C. Fu, “Adapting llms for efficient context processing through soft prompt compression,” in

Proc. Int. Conf. Model., Nat. Lang. Process. Mach. Learn.

[149]    J. Zou, M. Zhou, T. Li, S. Han, and D. Zhang, “Promptintern: Saving inference costs by internalizing recurrent prompt during large language model fine-tuning,”

arXiv preprint arXiv:2407.02211

[150]    A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly

, “An image is worth 16x16 words: Transformers for image recognition at scale,” in

Proc. Int. Conf. Learn. Represent.

[151]    X. Dong, J. Bao, D. Chen, W. Zhang, N. Yu, L. Yuan, D. Chen, and B. Guo, “Cswin transformer: A general vision transformer backbone with cross-shaped windows,” in

Proc. IEEE Conf. Comput. Vis. Pattern Recognit.

[152]    Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin transformer: Hierarchical vision transformer using shifted windows,” in

Proc. IEEE Int. Conf. Comput. Vis.

[153]    H. Fan, B. Xiong, K. Mangalam, Y. Li, Z. Yan, J. Malik, and C. Feichtenhofer, “Multiscale vision transformers,” in

Proc. IEEE Int. Conf. Comput. Vis.

[154]    Y. Li, C.-Y. Wu, H. Fan, K. Mangalam, B. Xiong, J. Malik, and C. Feichtenhofer, “Mvitv2: Improved multiscale vision transformers for classification and detection,” in

Proc. IEEE Conf. Comput. Vis. Pattern Recognit.

[155]    B. Graham, A. El-Nouby, H. Touvron, P. Stock, A. Joulin, H. Jégou, and M. Douze, “Levit: A vision transformer in convnet’s clothing for faster inference,” in

Proc. IEEE Int. Conf. Comput. Vis.

[156]    H.-W. Huang, W. Chai, K.-M. Chen, C.-Y. Yang, and J.-N. Hwang, “Tosa: Token merging with spatial awareness,” in

IEEE Int. Conf. Intell. Rob. Syst.

[157]    J. Cao, P. Ye, S. Li, C. Yu, Y. Tang, J. Lu, and T. Chen, “Madtp: Multimodal alignment-guided dynamic token pruning for accelerating vision-language transformer,” in

Proc. IEEE Conf. Comput. Vis. Pattern Recognit.

[158]    Z. Chen, W. Wang, Y. Cao, Y. Liu, Z. Gao, E. Cui, J. Zhu, S. Ye, H. Tian, Z. Liu

, “Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling,”

arXiv preprint arXiv:2412.05271

[159]    Z. Gao, Z. Chen, E. Cui, Y. Ren, W. Wang, J. Zhu, H. Tian, S. Ye, J. He, X. Zhu

, “Mini-internvl: A flexible-transfer pocket multi-modal model with 5% parameters and 90% performance,”

Visual Intell.

, vol. 2, no. 1, pp. 1–17, 2024.

[160]    J. Zhu, W. Wang, Z. Chen, Z. Liu, S. Ye, L. Gu, H. Tian, Y. Duan, W. Su, J. Shao

, “Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models,”

arXiv preprint arXiv:2504.10479

[161]    D. C. Porumbel, J.-K. Hao, and F. Glover, “A simple and effective algorithm for the maxmin diversity problem,”

Ann. Oper. Res.

, vol. 186, pp. 275–293, 2011.

[162]    H. Laurençon, L. Saulnier, L. Tronchon, S. Bekman, A. Singh, A. Lozhkov, T. Wang, S. Karamcheti, A. Rush, D. Kiela

, “Obelics: An open web-scale filtered dataset of interleaved image-text documents,” in

Proc. Adv. Neural Inform. Process. Syst.

[163]    T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré, “Flashattention: Fast and memory-efficient exact attention with io-awareness,” in

Proc. Adv. Neural Inform. Process. Syst.

[164]    T. Dao, “Flashattention-2: Faster attention with better parallelism and work partitioning,” in

Proc. Int. Conf. Learn. Represent.

[165]    M. Du, S. Ding, and H. Jia, “Study on density peaks clustering based on k-nearest neighbors and principal component analysis,”

Knowl.-Based Syst.

, vol. 99, pp. 135–145, 2016.

[166]    A. Rodriguez and A. Laio, “Clustering by fast search and find of density peaks,”

, vol. 344, no. 6191, pp. 1492–1496, 2014.

[167]    A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” in

Proc. Conf. Lang. Model.

[168]    A. Gulati, J. Qin, C.-C. Chiu, N. Parmar, Y. Zhang, J. Yu, W. Han, S. Wang, Z. Zhang, Y. Wu

, “Conformer: Convolution-augmented transformer for speech recognition,” in

## Interspeech

[169]    W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, and A. Mohamed, “Hubert: Self-supervised speech representation learning by masked prediction of hidden units,”

IEEE/ACM Trans. Audio Speech Lang. Process.

, vol. 29, pp. 3451–3460, 2021.

[170]    A. Défossez, J. Copet, G. Synnaeve, and Y. Adi, “High fidelity neural audio compression,”

Trans. Mach. Learn. Res.

[171]    N. Zeghidour, A. Luebs, A. Omran, J. Skoglund, and M. Tagliasacchi, “Soundstream: An end-to-end neural audio codec,”

IEEE/ACM Trans. Audio Speech Lang. Process.

, vol. 30, pp. 495–507, 2021.

[172]    Y. Fathullah, C. Wu, E. Lakomkin, J. Jia, Y. Shangguan, K. Li, J. Guo, W. Xiong, J. Mahadeokar, O. Kalinli

, “Prompting large language models with speech recognition abilities,” in

Proc. IEEE Int. Conf. Acoust. Speech Signal Process.

[173]    J. Lin, J. Tang, H. Tang, S. Yang, W.-M. Chen, W.-C. Wang, G. Xiao, X. Dang, C. Gan, and S. Han, “Awq: Activation-aware weight quantization for on-device llm compression and acceleration,” in

Proc. Mach. Learn. Syst.

[174]    G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, and S. Han, “Smoothquant: Accurate and efficient post-training quantization for large language models,” in

Proc. Int. Conf. Mach. Learn.

[175]    E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh, “gptq: Accurate post-training compression for generative pretrained transformers,” in

Proc. Int. Conf. Learn. Represent.

[176]    Y. Shang, Z. Yuan, B. Xie, B. Wu, and Y. Yan, “Post-training quantization on diffusion models,” in

Proc. IEEE Conf. Comput. Vis. Pattern Recognit.

[177]    Y. Sui, Y. Li, A. Kag, Y. Idelbayev, J. Cao, J. Hu, D. Sagar, B. Yuan, S. Tulyakov, and J. Ren, “Bitsfusion: 1.99 bits weight quantization of diffusion model,”

## Advances in Neural Information Processing Systems

, vol. 37, pp. 76 775–76 818, 2024.

[178]    A. Gholami, S. Kim, Z. Dong, Z. Yao, M. W. Mahoney, and K. Keutzer, “A survey of quantization methods for efficient neural network inference,” in

Proc. of the Book: Low-Power Computer Vision

, 2022, pp. 291–326.

[179]    S. Han, H. Mao, and W. J. Dally, “Deep compression: Compressing deep neural network with pruning, trained quantization and huffman coding,” in

Proc. Int. Conf. Learn. Represent.

[180]    X. Ma, G. Fang, and X. Wang, “Llm-pruner: On the structural pruning of large language models,” in

Proc. Adv. Neural Inform. Process. Syst.

[181]    Y. Sui, M. Yin, Y. Xie, H. Phan, S. Aliari Zonouz, and B. Yuan, “Chip: Channel independence-based pruning for compact neural networks,” in

Proc. Adv. Neural Inform. Process. Syst.

[182]    H. Cheng, M. Zhang, and J. Q. Shi, “A survey on deep neural network pruning: Taxonomy, comparison, analysis, and recommendations,”

IEEE Trans. Pattern Anal. Mach. Intell.

[183]    G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,”

arXiv preprint arXiv:1503.02531

[184]    J. Gou, B. Yu, S. J. Maybank, and D. Tao, “Knowledge distillation: A survey,”

Int. J. Comput. Vis.

, vol. 129, no. 6, pp. 1789–1819, 2021.

[185]    X. Yu, T. Liu, X. Wang, and D. Tao, “On compressing deep models by low rank and sparse decomposition,” in

Proc. IEEE Int. Conf. Comput. Vis.

[186]    M. Yin, Y. Sui, S. Liao, and B. Yuan, “Towards efficient tensor decomposition-based dnn model compression with optimization framework,” in

Proc. IEEE Conf. Comput. Vis. Pattern Recognit.

[187]    J. Xiao, C. Zhang, Y. Gong, M. Yin, Y. Sui, L. Xiang, D. Tao, and B. Yuan, “Haloc: Hardware-aware automatic low-rank compression for compact neural networks,” in

Proc. AAAI Conf. Artif. Intell.

[188]    Y. Sui, M. Yin, Y. Gong, and B. Yuan, “Co-exploring structured sparsification and low-rank tensor decomposition for compact dnns,”

IEEE Trans. Neural Netw. Learn. Syst.

, vol. 36, no. 4, pp. 6642–6654, 2024.

[189]    C. Yang, Y. Sui, J. Xiao, L. Huang, Y. Gong, Y. Duan, W. Jia, M. Yin, Y. Cheng, and B. Yuan, “Moe-i

: Compressing mixture of experts models through inter-expert pruning and intra-expert low-rank decomposition,”

arXiv preprint arXiv:2411.01016

[190]    Z. Liu, L. Zhu, B. Shi, Z. Zhang, Y. Lou, S. Yang, H. Xi, S. Cao, Y. Gu, D. Li

, “Nvila: Efficient frontier visual language models,” in

Proc. IEEE Conf. Comput. Vis. Pattern Recognit.

[191]    J. Bellver-Soler, M. Rodriguez-Cantelar, R. Córdoba, and L. F. D’Haro, “Cutting through overload: Efficient token dropping for speech emotion recognition in multimodal large language models,” in

Proc. Annu. Meet. Assoc. Comput. Linguist.

[192]    Z. Wen, Y. Gao, W. Li, C. He, and L. Zhang, “Token pruning in multimodal large language models: Are we solving the right problem?”

arXiv preprint arXiv:2502.11501

[193]    T. Qu, L. Tang, B. Peng, S. Yang, B. Yu, and J. Jia, “Does your vision-language model get lost in the long video sampling dilemma?”

arXiv preprint arXiv:2503.12496

[194]    H. Liu, M. Zaharia, and P. Abbeel, “Ringattention with blockwise transformers for near-infinite context,” in

Proc. Int. Conf. Learn. Represent.

[195]    E. Song, W. Chai, W. Xu, J. Xie, Y. Liu, and G. Wang, “Video-mmlu: A massive multi-discipline lecture understanding benchmark,”

arXiv preprint arXiv:2504.14693

[196]    Y. Li, H. Zhan, T. Chen, Q. Liu, and Y. Lu, “Why 1+ 1< 1 in visual token pruning: Beyond naive integration via multi-objective balanced covering,”

arXiv preprint arXiv:2505.10118

[197]    L. Yu, P. Poirson, S. Yang, A. C. Berg, and T. L. Berg, “Modeling context in referring expressions,” in

Proc. Eur. Conf. Comput. Vis.

[198]    D. A. Hudson and C. D. Manning, “Gqa: A new dataset for real-world visual reasoning and compositional question answering,” in

Proc. IEEE Conf. Comput. Vis. Pattern Recognit.

[199]    Y. Liu, H. Duan, Y. Zhang, B. Li, S. Zhang, W. Zhao, Y. Yuan, J. Wang, C. He, Z. Liu

, “Mmbench: Is your multi-modal model an all-around player?” in

Proc. Eur. Conf. Comput. Vis.

[200]    S. Yin, C. Fu, S. Zhao, K. Li, X. Sun, T. Xu, and E. Chen, “A survey on multimodal large language models,”

## National Science Review

, vol. 11, no. 12, p. nwae403, 2024.

[201]    Y. Li, Y. Du, K. Zhou, J. Wang, W. X. Zhao, and J.-R. Wen, “Evaluating object hallucination in large vision-language models,” in

Proc. Conf. Empir. Methods Nat. Lang. Process.

[202]    P. Lu, S. Mishra, T. Xia, L. Qiu, K.-W. Chang, S.-C. Zhu, O. Tafjord, P. Clark, and A. Kalyan, “Learn to explain: Multimodal reasoning via thought chains for science question answering,” in

Proc. Adv. Neural Inform. Process. Syst.

[203]    B. Li, R. Wang, G. Wang, Y. Ge, Y. Ge, and Y. Shan, “Seed-bench: Benchmarking multimodal llms with generative comprehension,”

arXiv preprint arXiv:2307.16125

[204]    D. Gurari, Q. Li, A. J. Stangl, A. Guo, C. Lin, K. Grauman, J. Luo, and J. P. Bigham, “Vizwiz grand challenge: Answering visual questions from blind people,” in

Proc. IEEE Conf. Comput. Vis. Pattern Recognit.

[205]    Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, and D. Parikh, “Making the v in vqa matter: Elevating the role of image understanding in visual question answering,” in

Proc. IEEE Conf. Comput. Vis. Pattern Recognit.

[206]    W. Yu, Z. Yang, L. Li, J. Wang, K. Lin, Z. Liu, X. Wang, and L. Wang, “Mm-vet: Evaluating large multimodal models for integrated capabilities,” in

Proc. Int. Conf. Mach. Learn.

[207]    Z. Yu, D. Xu, J. Yu, T. Yu, Z. Zhao, Y. Zhuang, and D. Tao, “Activitynet-qa: A dataset for understanding complex web videos via question answering,” in

Proc. AAAI Conf. Artif. Intell.

[208]    M. Maaz, H. Rasheed, S. Khan, and F. S. Khan, “Video-chatgpt: Towards detailed video understanding via large vision and language models,” in

Proc. Annu. Meet. Assoc. Comput. Linguist.

[209]    J. Xiao, X. Shang, A. Yao, and T.-S. Chua, “Next-qa: Next phase of question-answering to explaining temporal actions,” in

Proc. IEEE Conf. Comput. Vis. Pattern Recognit.

[210]    K. Mangalam, R. Akshulakov, and J. Malik, “Egoschema: A diagnostic benchmark for very long-form video language understanding,” in

Proc. Adv. Neural Inform. Process. Syst.

[211]    H. Wu, D. Li, B. Chen, and J. Li, “Longvideobench: A benchmark for long-context interleaved video-language understanding,” in

Proc. Adv. Neural Inform. Process. Syst.

[212]    C. Fu, Y. Dai, Y. Luo, L. Li, S. Ren, R. Zhang, Z. Wang, C. Zhou, Y. Shen, M. Zhang

, “Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis,” in

Proc. IEEE Conf. Comput. Vis. Pattern Recognit.

[213]    V. Patraucean, L. Smaira, A. Gupta, A. Recasens, L. Markeeva, D. Banarse, S. Koppula, M. Malinowski, Y. Yang, C. Doersch

, “Perception test: A diagnostic benchmark for multimodal video models,” in

Proc. Adv. Neural Inform. Process. Syst.

[214]    LMMs-Lab, “Video detail caption,” Nov. 2024, accessed: 2024-11.

[215]    J. Gao, C. Sun, Z. Yang, and R. Nevatia, “Tall: Temporal activity localization via language query,” in

Proc. IEEE Int. Conf. Comput. Vis.

[216]    Z. Wang, M. Ma, Z. Wang, R. Mu, L. Shan, M. Liu, and B. Qin, “Effivlm-bench: A comprehensive benchmark for evaluating training-free acceleration in large vision-language models,”

Proc. Annu. Meet. Assoc. Comput. Linguist.

[217]    R. Zhang, Y. Lyu, R. Shao, G. Chen, W. Guan, and L. Nie, “Token-level correlation-guided compression for efficient multimodal document understanding,”

arXiv preprint arXiv:2407.14439

[218]    C. X. Liang, P. Tian, C. H. Yin, Y. Yua, W. An-Hou, L. Ming, T. Wang, Z. Bi, and M. Liu, “A comprehensive survey and guide to multimodal large language models in vision-language tasks,”

arXiv preprint arXiv:2411.06284

[219]    R. Beddiar and M. Oussalah, “Explainability in medical image captioning,”

Explainable Deep Learn. AI

, pp. 239–261, 2023.

[220]    Y. Bazi, M. M. A. Rahhal, L. Bashmal, and M. Zuair, “Vision-language model for visual question answering in medical imagery,”

## Bioengineering

, vol. 10, no. 3, p. 380, 2023.

[221]    X. He, Y. Zhang, L. Mou, E. Xing, and P. Xie, “Pathvqa: 30000+ questions for medical visual question answering,”

arXiv preprint arXiv:2003.10286

[222]    S.-C. Huang, M. Jensen, S. Yeung-Levy, M. P. Lungren, H. Poon, and A. S. Chaudhari, “Multimodal foundation models for medical imaging-a systematic review and implementation guidelines,”

[223]    H. Zhou, Z. Gao, M. Ye, Z. Chen, Q. Chen, T. Cao, and H. Qi, “Hints of prompt: Enhancing visual representation for multimodal llms in autonomous driving,”

arXiv preprint arXiv:2411.13076

[224]    K. Team, A. Du, B. Gao, B. Xing, C. Jiang, C. Chen, C. Li, C. Xiao, C. Du, C. Liao

, “Kimi k1. 5: Scaling reinforcement learning with llms,”

arXiv preprint arXiv:2501.12599

[225]    D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi

, “Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning,”

arXiv preprint arXiv:2501.12948

[226]    A. Jaech, A. Kalai, A. Lerer, A. Richardson, A. El-Kishky, A. Low, A. Helyar, A. Madry, A. Beutel, A. Carney

, “Openai o1 system card,”

arXiv preprint arXiv:2412.16720

[227]    S. Feng, G. Fang, X. Ma, and X. Wang, “Efficient reasoning models: A survey,”

arXiv preprint arXiv:2504.10903

[228]    Y. Sui, Y.-N. Chuang, G. Wang, J. Zhang, T. Zhang, J. Yuan, H. Liu, A. Wen, H. Chen, X. Hu

, “Stop overthinking: A survey on efficient reasoning for large language models,”

arXiv preprint arXiv:2503.16419

[229]    X. Chen, A. Zhao, H. Xia, X. Lu, H. Wang, Y. Chen, W. Zhang, J. Wang, W. Li, and X. Shen, “Reasoning beyond language: A comprehensive survey on latent chain-of-thought reasoning,”

arXiv preprint arXiv:2505.16782

[230]    S. Feng, S. Wang, S. Ouyang, L. Kong, Z. Song, J. Zhu, H. Wang, and X. Wang, “Can mllms guide me home? a benchmark study on fine-grained visual reasoning from transit maps,”

arXiv preprint arXiv:2505.18675

[231]    X. Ma, G. Wan, R. Yu, G. Fang, and X. Wang, “Cot-valve: Length-compressible chain-of-thought tuning,” in

Proc. Annu. Meet. Assoc. Comput. Linguist.

[232]    H. Xia, Y. Li, C. T. Leong, W. Wang, and W. Li, “Tokenskip: Controllable chain-of-thought compression in llms,”

arXiv preprint arXiv:2502.12067

[233]    T. Liu, Q. Guo, X. Hu, C. Jiayang, Y. Zhang, X. Qiu, and Z. Zhang, “Can language models learn to skip steps?”

arXiv preprint arXiv:2411.01855

[234]    G. Fang, X. Ma, and X. Wang, “Thinkless: Llm learns when to think,”

arXiv preprint arXiv:2505.13379

[235]    L. C. Magister, J. Mallinson, J. Adamek, E. Malmi, and A. Severyn, “Teaching small language models to reason,” in

Proc. Annu. Meet. Assoc. Comput. Linguist.

[236]    C. Li, Q. Chen, L. Li, C. Wang, Y. Li, Z. Chen, and Y. Zhang, “Mixed distillation helps smaller language model better reasoning,”

arXiv preprint arXiv:2312.10730

[237]    T. Feng, Y. Li, L. Chenglin, H. Chen, F. Yu, and Y. Zhang, “Teaching small language models reasoning through counterfactual distillation,” in

Proc. Conf. Empir. Methods Nat. Lang. Process.

[238]    N. Zhang, Y. Zhang, P. Mitra, and R. Zhang, “When reasoning meets compression: Benchmarking compressed large reasoning models on complex reasoning tasks,”

arXiv preprint arXiv:2504.02010

[239]    H. Sun, M. Haider, R. Zhang, H. Yang, J. Qiu, M. Yin, M. Wang, P. Bartlett, and A. Zanette, “Fast best-of-n decoding via speculative rejection,” in

Proc. Adv. Neural Inform. Process. Syst.

[240]    C. Ma, H. Zhao, J. Zhang, J. He, and L. Kong, “Non-myopic generation of language models for reasoning and planning,” in

Proc. Int. Conf. Learn. Represent.

[241]    F. Luo, Y.-N. Chuang, G. Wang, H. A. D. Le, S. Zhong, H. Liu, J. Yuan, Y. Sui, V. Braverman, V. Chaudhary

, “Autol2s: Auto long-short reasoning for efficient large language models,”

arXiv preprint arXiv:2505.22662

[242]    F. Xu, H. Yan, C. Ma, H. Zhao, J. Liu, Q. Lin, and Z. Wu, “ ϕ \phi italic_ϕ -decoding: Adaptive foresight sampling for balanced inference-time exploration and exploitation,”

arXiv preprint arXiv:2503.13288

[243]    Y. Ding, W. Jiang, S. Liu, Y. Jing, J. Guo, Y. Wang, J. Zhang, Z. Wang, Z. Liu, B. Du

, “Dynamic parallel tree search for efficient llm reasoning,”

arXiv preprint arXiv:2502.16235

