---
sourceFile: "Vision transformer - Wikipedia"
exportedBy: "Kortex"
exportDate: "2025-10-28T18:42:20.243Z"
---

# Vision transformer - Wikipedia

f57be8a8-b06c-4c0a-be96-c1472536029d

Vision transformer - Wikipedia

765fe2eb-57d8-4423-92ba-a1692bf29509

https://en.wikipedia.org/wiki/Vision_transformer

## Jump to content

## Main menu    Navigation

## Current events

## Random article

## About Wikipedia

## Contribute

## Learn to edit

## Community portal

## Recent changes

## Upload file

## Special pages

## Create account

## Create account

## Pages for logged out editors

https://en.wikipedia.org/wiki/Help:Introduction

## Contributions

https://en.wikipedia.org/wiki/Help:Introduction

https://en.wikipedia.org/wiki/Help:Introduction

https://en.wikipedia.org/wiki/Help:Introduction

1   History

https://en.wikipedia.org/wiki/Help:Introduction

2   Overview

https://en.wikipedia.org/wiki/Help:Introduction

3   Variants

https://en.wikipedia.org/wiki/Help:Introduction

3.1   Original ViT

https://en.wikipedia.org/wiki/Help:Introduction

3.2   Architectural improvements

https://en.wikipedia.org/wiki/Help:Introduction

3.2.1   Pooling

https://en.wikipedia.org/wiki/Help:Introduction

3.3   Masked Autoencoder

https://en.wikipedia.org/wiki/Help:Introduction

3.4   DINO

https://en.wikipedia.org/wiki/Help:Introduction

3.5   Swin Transformer

https://en.wikipedia.org/wiki/Help:Introduction

3.6   TimeSformer

https://en.wikipedia.org/wiki/Help:Introduction

3.7   ViT-VQGAN

https://en.wikipedia.org/wiki/Help:Introduction

3.8   Others

https://en.wikipedia.org/wiki/Help:Introduction

4   Comparison with CNNs

https://en.wikipedia.org/wiki/Help:Introduction

5   Applications

https://en.wikipedia.org/wiki/Help:Introduction

6   See also

https://en.wikipedia.org/wiki/Help:Introduction

7   References

https://en.wikipedia.org/wiki/Help:Introduction

8   Further reading

https://en.wikipedia.org/wiki/Help:Introduction

## Vision transformer

https://en.wikipedia.org/wiki/Help:Introduction

https://en.wikipedia.org/wiki/Help:Introduction

https://en.wikipedia.org/wiki/Help:Introduction

https://en.wikipedia.org/wiki/Help:Introduction

https://en.wikipedia.org/wiki/Help:Introduction

https://en.wikipedia.org/wiki/Help:Introduction

https://en.wikipedia.org/wiki/Help:Introduction

https://en.wikipedia.org/wiki/Help:Introduction

https://en.wikipedia.org/wiki/Help:Introduction

https://en.wikipedia.org/wiki/Help:Introduction

## Tools    Actions

https://en.wikipedia.org/wiki/Help:Introduction

https://en.wikipedia.org/wiki/Help:Introduction

## View history

https://en.wikipedia.org/wiki/Help:Introduction

## What links here

https://en.wikipedia.org/wiki/Help:Introduction

## Related changes

https://en.wikipedia.org/wiki/Help:Introduction

## Upload file

https://en.wikipedia.org/wiki/Help:Introduction

## Permanent link

https://en.wikipedia.org/wiki/Help:Introduction

## Page information

https://en.wikipedia.org/wiki/Help:Introduction

## Cite this page

https://en.wikipedia.org/wiki/Help:Introduction

## Get shortened URL

https://en.wikipedia.org/wiki/Help:Introduction

## Download QR code

https://en.wikipedia.org/wiki/Help:Introduction

Print/export

## Download as PDF

https://en.wikipedia.org/wiki/Help:Introduction

## Printable version

https://en.wikipedia.org/wiki/Help:Introduction

## In other projects

## Wikidata item

https://en.wikipedia.org/wiki/Help:Introduction

From Wikipedia, the free encyclopedia   Machine learning model for vision processing

The architecture of vision transformer. An input image is divided into patches, each of which is linearly mapped through a patch embedding layer, before entering a standard Transformer encoder.

vision transformer

transformer

https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)

designed for

computer vision

https://en.wikipedia.org/wiki/Computer_vision

A ViT decomposes an input image into a series of patches (rather than text into

https://en.wikipedia.org/wiki/Byte_pair_encoding

), serializes each patch into a vector, and maps it to a smaller dimension with a single

matrix multiplication

https://en.wikipedia.org/wiki/Matrix_multiplication

. These vector

https://en.wikipedia.org/wiki/Latent_space

are then processed by a

transformer encoder

https://en.wikipedia.org/wiki/BERT_(language_model)

as if they were token embeddings.

## ViTs were designed as alternatives to

convolutional neural networks

https://en.wikipedia.org/wiki/Convolutional_neural_network

(CNNs) in computer vision applications. They have different inductive biases, training stability, and data efficiency.

Compared to CNNs, ViTs are less data efficient, but have higher capacity. Some of the largest modern computer vision models are ViTs, such as one with 22B parameters.

Subsequent to its publication, many variants were proposed, with hybrid architectures with both features of ViTs and CNNs

. ViTs have found application in

image recognition

https://en.wikipedia.org/wiki/Image_recognition

image segmentation

https://en.wikipedia.org/wiki/Image_segmentation

weather prediction

https://en.wikipedia.org/wiki/Weather_forecasting

autonomous driving

https://en.wikipedia.org/wiki/Autonomous_driving

https://en.wikipedia.org/w/index.php?title=Vision_transformer&action=edit&section=1

## Transformers were introduced in

## Attention Is All You Need

and have found widespread use in

natural language processing

https://en.wikipedia.org/wiki/Natural_language_processing

. A 2019 paper

applied ideas from the Transformer to computer vision. Specifically, they started with a

https://en.wikipedia.org/wiki/Residual_neural_network

, a standard

convolutional neural network

https://en.wikipedia.org/wiki/Convolutional_neural_network

used for computer vision, and replaced all convolutional kernels by the self-attention mechanism found in a Transformer. It resulted in superior performance. However, it is not a Vision Transformer.

In 2020, an

encoder-only Transformer

https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)#encoder-only

was adapted for computer vision, yielding the ViT, which reached state of the art in image classification, overcoming the previous dominance of CNN.

The masked autoencoder (2022) extended ViT to work with unsupervised training. The vision transformer and the masked autoencoder, in turn, stimulated new developments in convolutional neural networks.

Subsequently, there was cross-fertilization between the previous CNN approach and the ViT approach.

In 2021, some important variants of the Vision Transformers were proposed. These variants are mainly intended to be more efficient, more accurate or better suited to a specific domain. Two studies

improved efficiency and robustness of ViT by adding a CNN as a preprocessor. The Swin Transformer

achieved state-of-the-art results on some object detection datasets such as

https://en.wikipedia.org/wiki/COCO_(dataset)

, by using convolution-like sliding windows of attention mechanism, and the

https://en.wikipedia.org/wiki/Pyramid_(image_processing)

process in classical computer vision.

https://en.wikipedia.org/w/index.php?title=Vision_transformer&action=edit&section=2

Vision Transformer architecture, showing the encoder-only Transformer blocks inside

The basic architecture, used by the original 2020 paper,

is as follows. In summary, it is a

https://en.wikipedia.org/wiki/BERT_(language_model)

-like encoder-only Transformer.

The input image is of type  R   H   ×   W   ×   C   {\displaystyle \mathbb {R} ^{H\times W\times C}}   , where  H   ,   W   ,   C   {\displaystyle H,W,C}    are height, width, channel (

https://en.wikipedia.org/wiki/RGB_color_model

). It is then split into square-shaped patches of type  R   P   ×   P   ×   C   {\displaystyle \mathbb {R} ^{P\times P\times C}}   .

For each patch, the patch is pushed through a linear operator, to obtain a vector ("patch embedding"). The position of the patch is also transformed into a vector by "position encoding". The two vectors are added, then pushed through several Transformer encoders.

The attention mechanism in a ViT repeatedly transforms representation vectors of image patches, incorporating more and more semantic relations between image patches in an image. This is analogous to how in natural language processing, as representation vectors flow through a transformer, they incorporate more and more semantic relations between words, from syntax to semantics.

The above architecture turns an image into a sequence of vector representations. To use these for downstream applications, an additional head needs to be trained to interpret them.

For example, to use it for classification, one can add a shallow MLP on top of it that outputs a probability distribution over classes. The original paper uses a linear-

https://en.wikipedia.org/wiki/Activation_function

-linear-softmax network.

https://en.wikipedia.org/w/index.php?title=Vision_transformer&action=edit&section=3

## Original ViT

https://en.wikipedia.org/w/index.php?title=Vision_transformer&action=edit&section=4

The original ViT was an encoder-only Transformer supervise-trained to predict the image label from the patches of the image. As in the case of

https://en.wikipedia.org/wiki/BERT_(language_model)

, it uses a special token  <CLS>  in the input side, and the corresponding output vector is used as the only input of the final output MLP head. The special token is an architectural hack to allow the model to compress all information relevant for predicting the image label into one vector.

Animation of ViT. The 0th token is the special  <CLS> . The other 9 patches are projected by a linear layer before being fed into the Transformer encoder as input tokens 1 to 9.

## Transformers found their initial applications in

natural language processing

https://en.wikipedia.org/wiki/Natural_language_processing

tasks, as demonstrated by

language models

https://en.wikipedia.org/wiki/Language_models

https://en.wikipedia.org/wiki/BERT_(language_model)

https://en.wikipedia.org/wiki/GPT-3

. By contrast the typical image processing system uses a

convolutional neural network

https://en.wikipedia.org/wiki/Convolutional_neural_network

(CNN). Well-known projects include Xception,

https://en.wikipedia.org/wiki/Residual_neural_network

## EfficientNet

https://en.wikipedia.org/wiki/EfficientNet

https://en.wikipedia.org/wiki/Inceptionv3

Transformers measure the relationships between pairs of input tokens (words in the case of text strings), termed

https://en.wikipedia.org/wiki/Attention_(machine_learning)

. The cost is quadratic in the number of tokens. For images, the basic unit of analysis is the

https://en.wikipedia.org/wiki/Pixel

. However, computing relationships for every pixel pair in a typical image is prohibitive in terms of memory and computation. Instead, ViT computes relationships among pixels in various small sections of the image (e.g., 16x16 pixels), at a drastically reduced cost. The sections (with positional embeddings) are placed in a sequence. The embeddings are learnable vectors. Each section is arranged into a linear sequence and multiplied by the embedding matrix. The result, with the position embedding is fed to the transformer.

## Architectural improvements

https://en.wikipedia.org/w/index.php?title=Vision_transformer&action=edit&section=5

https://en.wikipedia.org/w/index.php?title=Vision_transformer&action=edit&section=6

]   Main article:

## Pooling layer

https://en.wikipedia.org/wiki/Pooling_layer

After the ViT processes an image, it produces some embedding vectors. These must be converted to a single class probability prediction by some kind of network. In the original ViT and Masked Autoencoder, they used a dummy  [CLS]  token , in emulation of the

https://en.wikipedia.org/wiki/BERT_(language_model)

language model. The output at  [CLS]  is the classification token, which is then processed by a

https://en.wikipedia.org/wiki/LayerNorm

-feedforward-softmax module into a probability distribution.

Global average pooling (GAP)

does not use the dummy token, but simply takes the average of all output tokens as the classification token. It was mentioned in the original ViT as being equally good.

Multihead attention pooling (MAP)

applies a

multiheaded attention block

https://en.wikipedia.org/wiki/Attention_(machine_learning)

to pooling. Specifically, it takes as input a list of vectors  x   1   ,   x   2   ,   …   ,   x   n   {\displaystyle x_{1},x_{2},\dots ,x_{n}}   , which might be thought of as the output vectors of a layer of a ViT. The output from MAP is  M   u   l   t   i   h   e   a   d   e   d   A   t   t   e   n   t   i   o   n   (   Q   ,   V   ,   V   )   {\displaystyle \mathrm {MultiheadedAttention} (Q,V,V)}   , where  q   {\displaystyle q}    is a trainable query vector, and  V   {\displaystyle V}    is the matrix with rows being  x   1   ,   x   2   ,   …   ,   x   n   {\displaystyle x_{1},x_{2},\dots ,x_{n}}   .

## This was first proposed in the

## Set Transformer

architecture.

Later papers demonstrated that GAP and MAP both perform better than BERT-like pooling.

## A variant of MAP was proposed as

class attention

, which applies MAP, then feedforward, then MAP again.

Re-attention

was proposed to allow training deep ViT. It changes the multiheaded attention module.

## Masked Autoencoder

https://en.wikipedia.org/w/index.php?title=Vision_transformer&action=edit&section=7

## Masked Autoencoder architecture

## Masked Autoencoder

took inspiration from

denoising autoencoders

https://en.wikipedia.org/wiki/Denoising_autoencoder

and context encoders.

It has two ViTs put end-to-end. The first one ("encoder") takes in image patches with positional encoding, and outputs vectors representing each patch. The second one (called "decoder", even though it is still an encoder-only Transformer) takes in vectors with positional encoding and outputs image patches again.

During training, input images (224px x 224 px in the original implementation) are split along a designated number of lines on each axis, producing image patches

. A certain percentage of patches are selected to be masked out by mask tokens, while all others are retained in the image. The network is tasked with reconstructing the image from the remaining unmasked patches. Mask tokens in the original implementation are learnable

vector quantities

https://en.wikipedia.org/wiki/Vector_space

https://en.wikipedia.org/wiki/Vector_space

linear projection

https://en.wikipedia.org/wiki/Projection_(linear_algebra)

with positional embeddings is then applied to the vector of unmasked patches. Experiments varying mask ratio on networks trained on the

ImageNet-1K

https://en.wikipedia.org/wiki/ImageNet

dataset found 75% mask ratios

achieved high performance on both finetuning and linear-probing of the encoder's [Latent space|latent space]. The MAE processes only unmasked patches during training, increasing the efficiency of data processing in the encoder and lowering the memory usage of the [[Transformer (deep learning architecture) |transformer]]

A less computationally-intensive ViT is used for the decoder in the original implementation of the MAE. Masked patches are added back to the output of the encoder block as mask tokens and both are fed into the decoder. A reconstruction loss is computed for the masked patches to assess network performance.

In prediction, the decoder architecture is discarded entirely. The input image is split into patches by the same algorithm as in training, but no patches are masked out. A linear projection with a positional embedding is applied to each patch, and the resulting embedding vector representations of each patch are fed to the encoder.

## Uses and Derivatives

Several derivatives of the original MAE have been explored. The MAE has been applied for self-supervised pretraining in medical contexts, including for chest X-ray interpretation

. Derivatives of the MAE have been applied in this context to better serve as pretraining in medical contexts.

## Medically Supervised MAE

Medically Supervised MAE seeks to address the application of MAE's high mask ratios when applied to medical lesion datasets and uses a [[Supervised learning|supervised training] set to create local attention maps for medical images in order to constrain which patches are masked out. Medically Supervised MAE achieved state-of-the-art performance as of Jan. 2025 on the classification of medical lesions on the Messidor-2, BTMD, HAM10000, DeepLesion, and ChestXRay2017 datasets

Gray Level Co-occurrence Matrix MAE

(GLCM-MAE)

: GCLM-MAE uses GCLM to extract texture information from images in order to preserve texture information. It addresses an issue in which a classic MAE oversmooths images, causing a loss of granular detail that may be important in medical contexts. GLCM-MAE achieves state-of-the-art performance on the identification of gallbladder cancer, breast cancer imaged from ultrasound, pneumonia imaged from X-rays, and COVID-19 imaged from computed tomography as of Jul. 2025.

Region-aware MAE

R-MAE: R-MAE replaces patch-generating step in the original MAE with an algorithm for assigning individual pixels to regions of interest in an image, which are masked out together. The region encoding architecture is standalone, but can be combined with the MAE for region reconstruction.

Siamese MAEs (SiamMAE)

SiamMAE is a network designed to apply MAEs to video data. Samples two frames from a video (compared to one in the original MAE), and labels them as "past" and "future." The network masks out a majority of the patches (~95%) in the future frame, leaves the past frame untouched, and passes both through the MAE encoder block. The decoder architecture is replaced with attention blocks that map patches from the past frame to the future frame for reconstruction. SiamMAE achieves competitive performance against larger models on segmentation and propagation in videos.

A similar architecture was BERT ViT (BEiT), published concurrently.

https://en.wikipedia.org/w/index.php?title=Vision_transformer&action=edit&section=8

Like the Masked Autoencoder, the

stillation with

labels) method is a way to train a ViT by

self-supervision

https://en.wikipedia.org/wiki/Self-supervised_learning

DINO is a form of teacher-student

self-distillation

https://en.wikipedia.org/wiki/Knowledge_distillation

. In DINO, the student is the model itself, and the teacher is an exponential average of the student's past states. The method is similar to previous works like momentum contrast

and bootstrap your own latent (BYOL).

## The loss function used in DINO is the

cross-entropy loss

https://en.wikipedia.org/wiki/Cross-entropy

between the output of the teacher network (   f   θ   t   ′   {\displaystyle f_{\theta '_{t}}}   ) and the output of the student network (   f   θ   t   {\displaystyle f_{\theta _{t}}}   ). The teacher network is an exponentially decaying average of the student network's past parameters:  θ   t   ′   =   α   θ   t   +   α   (   1   −   α   )   θ   t   −   1   +   ⋯   {\displaystyle \theta '_{t}=\alpha \theta _{t}+\alpha (1-\alpha )\theta _{t-1}+\cdots }   . The inputs to the networks are two different crops of the same image, represented as  T   (   x   )   {\displaystyle T(x)}    and  T   ′   (   x   )   {\displaystyle T'(x)}   , where  x   {\displaystyle x}    is the original image. The loss function is written as   L   (   f   θ   t   ′   (   T   (   x   )   )   ,   f   θ   t   (   T   ′   (   x   )   )   )   {\displaystyle L(f_{\theta '_{t}}(T(x)),f_{\theta _{t}}(T'(x)))}   One issue is that the network can "collapse" by always outputting the same value (   y   {\displaystyle y}   ), regardless of the input. To prevent this collapse, DINO employs two strategies:

: The teacher network's output is sharpened using a softmax function with a lower temperature. This makes the teacher more "confident" in its predictions, forcing the student to learn more meaningful representations to match the teacher's sharpened output.

: The teacher network's output is centered by averaging it with its previous outputs. This prevents the teacher from becoming biased towards any particular output value, encouraging the student to learn a more diverse set of features.

In January 2024,

https://en.wikipedia.org/wiki/Meta_AI

Research released an updated version called DINOv2

with improvements in architecture, loss function, and optimization technique. It was trained on a larger and more diverse dataset. The features learned by DINOv2 were more

transferable

https://en.wikipedia.org/wiki/Transfer_learning

, meaning it had better performance in downstream tasks.

In August 2025, Meta AI Research released DINOv3, an update to DINOv2. It introduced image-text alignment like

https://en.wikipedia.org/wiki/Contrastive_Language-Image_Pre-training

. It scaled up the model to 7B parameters and the training dataset to 1.7B images (obtained by diversity-sampling an initial dataset with 17B images). Architecturally, it introduced two improvements: Gram anchoring and axial RoPE (

## Rotary Positional Embeddings

https://en.wikipedia.org/wiki/Rotary_positional_embedding

) with jittering. Gram anchoring applies teacher-student self-distillation for the

## Gram matrix

https://en.wikipedia.org/wiki/Gram_matrix

between the feature vectors of the patches of an image. It avoids the previously observed problem of degradation of dense feature maps: While performance on global tasks (like classification) continued to improve, performance on dense tasks (like segmentation) would peak early and then decline, with feature maps becoming noisy. Axial RoPE makes the model more robust to varying image resolutions, scales, and aspect ratios.

## Swin Transformer

https://en.wikipedia.org/w/index.php?title=Vision_transformer&action=edit&section=9

## Swin Transformer

took inspiration from standard CNNs:

Instead of performing self-attention over the entire sequence of tokens, one for each patch, it performs "shifted window based" self-attention, which means only performing attention over square-shaped blocks of patches. One block of patches is analogous to the receptive field of one convolution.

After every few attention blocks, there is a "merge layer", which merges neighboring 2x2 tokens into a single token. This is analogous to

https://en.wikipedia.org/wiki/Pooling_layer

(by 2x2 convolution kernels, with stride 2). Merging means concatenation followed by multiplication with a matrix.

It is improved by Swin Transformer V2,

which modifies upon the ViT by a different attention mechanism

: Figure 1

immediately after each attention and feedforward layer ("res-post-norm");

scaled cosine attention to replace the original dot product attention;

log-spaced continuous

relative position bias

https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)#Alternative_positional_encodings

, which allows

transfer learning

https://en.wikipedia.org/wiki/Transfer_learning

across different window resolutions.

## TimeSformer

https://en.wikipedia.org/w/index.php?title=Vision_transformer&action=edit&section=10

## TimeSformer

was designed for video understanding tasks, and it applied a factorized self-attention, similar to the factorized convolution kernels found in the

https://en.wikipedia.org/wiki/Inceptionv3

CNN architecture.

Schematically, it divides a video into frames, and each frame into a square grid of patches (same as ViT). Let each patch coordinate be denoted by  x   ,   y   ,   t   {\displaystyle x,y,t}   , denoting horizontal, vertical, and time.

A space attention layer is a self-attention layer where each query patch  q   x   ,   y   ,   t   {\displaystyle q_{x,y,t}}    attends to only the key and value patches  k   x   ′   ,   y   ′   ,   t   ′   ,   v   x   ′   ,   y   ′   ,   t   ′   {\displaystyle k_{x',y',t'},v_{x',y',t'}}    such that  t   =   t   ′   {\displaystyle t=t'}   .

A time attention layer is where the requirement is  x   ′   =   x   ,   y   ′   =   y   {\displaystyle x'=x,y'=y}    instead.

The TimeSformer also considered other attention layer designs, such as the "height attention layer" where the requirement is  x   ′   =   x   ,   t   ′   =   t   {\displaystyle x'=x,t'=t}   . However, they found empirically that the best design interleaves one space attention layer and one time attention layer.

https://en.wikipedia.org/w/index.php?title=Vision_transformer&action=edit&section=11

there are two ViT encoders and a discriminator. One encodes 8x8 patches of an image into a list of vectors, one for each patch. The vectors can only come from a discrete set of "codebook", as in

vector quantization

https://en.wikipedia.org/wiki/Vector_quantization

. Another encodes the quantized vectors back to image patches. The training objective attempts to make the reconstruction image (the output image) faithful to the input image. The discriminator (usually a convolutional network, but other networks are allowed) attempts to decide if an image is an original real image, or a reconstructed image by the ViT.

The idea is essentially the same as vector quantized variational autoencoder (VQVAE) plus

generative adversarial network

https://en.wikipedia.org/wiki/Generative_adversarial_network

After such a ViT-VQGAN is trained, it can be used to code an arbitrary image into a list of symbols, and code an arbitrary list of symbols into an image. The list of symbols can be used to train into a standard autoregressive transformer (like GPT), for autoregressively generating an image. Further, one can take a list of caption-image pairs, convert the images into strings of symbols, and train a standard GPT-style transformer. Then at test time, one can just give an image caption, and have it autoregressively generate the image. This is the structure of Google Parti.

https://en.wikipedia.org/w/index.php?title=Vision_transformer&action=edit&section=12

Other examples include the visual transformer,

the data-efficient ViT (DeiT),

In the Transformer in Transformer architecture, each layer applies a vision Transformer layer on each image patch embedding, add back the resulting tokens to the embedding, then applies another vision Transformer layer.

## Comparison with CNNs

https://en.wikipedia.org/w/index.php?title=Vision_transformer&action=edit&section=13

Typically, ViT uses patch sizes larger than standard CNN kernels (3x3 to 7x7). ViT is more sensitive to the choice of the optimizer,

hyperparameters

https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)

, and network depth. Preprocessing with a layer of smaller-size, overlapping (stride < size) convolutional filters helps with performance and stability.

## This different behavior seems to derive from the different

inductive biases

https://en.wikipedia.org/wiki/Inductive_bias

they possess.

CNN applies the same set of filters for processing the entire image. This allows them to be more data efficient and less sensitive to local perturbations.

ViT applies self-attention, allowing them to easily capture long-range relationships between patches

. They also require more data to train, but they can ingest more training data compared to CNN, which might not improve after training on a large enough training dataset. ViT also appears more robust to input image distortions such as adversarial patches or permutations.

## Applications

https://en.wikipedia.org/w/index.php?title=Vision_transformer&action=edit&section=14

ViT have been used in many computer vision tasks with excellent results and in some cases even state-of-the-art, such as in

image classification

https://en.wikipedia.org/wiki/Image_classification

object detection

https://en.wikipedia.org/wiki/Object_detection

https://en.wikipedia.org/wiki/Deepfake

detection,

image segmentation

anomaly detection

image synthesis

https://en.wikipedia.org/wiki/Image_synthesis

cluster analysis

https://en.wikipedia.org/wiki/Cluster_analysis

autonomous driving

https://en.wikipedia.org/wiki/Autonomous_driving

## ViT had been used for image generation as backbones for

https://en.wikipedia.org/wiki/Generative_adversarial_network

https://en.wikipedia.org/wiki/Generative_adversarial_network

diffusion models

https://en.wikipedia.org/wiki/Diffusion_model

(diffusion transformer, or DiT).

has been demonstrated to learn useful representations for clustering images and exploring morphological profiles on biological datasets, such as images generated with the

## Cell Painting

https://en.wikipedia.org/wiki/Cell_Painting

In 2024, a 113 billion-parameter ViT model was proposed (the largest ViT to date) for

weather and climate prediction

https://en.wikipedia.org/wiki/Weather_forecasting

, and trained on the

## Frontier supercomputer

https://en.wikipedia.org/wiki/Frontier_(supercomputer)

with a throughput of 1.6

https://en.wikipedia.org/wiki/Floating_point_operations_per_second

https://en.wikipedia.org/w/index.php?title=Vision_transformer&action=edit&section=15

Transformer (machine learning model)

https://en.wikipedia.org/w/index.php?title=Vision_transformer&action=edit&section=15

## Convolutional neural network

https://en.wikipedia.org/w/index.php?title=Vision_transformer&action=edit&section=15

Attention (machine learning)

https://en.wikipedia.org/w/index.php?title=Vision_transformer&action=edit&section=15

https://en.wikipedia.org/w/index.php?title=Vision_transformer&action=edit&section=15

## Deep learning

https://en.wikipedia.org/w/index.php?title=Vision_transformer&action=edit&section=15

https://en.wikipedia.org/w/index.php?title=Vision_transformer&action=edit&section=15

https://en.wikipedia.org/w/index.php?title=Vision_transformer&action=edit&section=15

https://en.wikipedia.org/w/index.php?title=Vision_transformer&action=edit&section=16

Dosovitskiy, Alexey; Beyer, Lucas; Kolesnikov, Alexander; Weissenborn, Dirk; Zhai, Xiaohua; Unterthiner, Thomas; Dehghani, Mostafa; Minderer, Matthias; Heigold, Georg; Gelly, Sylvain; Uszkoreit, Jakob (2021-06-03). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale".

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2010.11929

https://arxiv.org/archive/cs.CV

Raghu, Maithra; Unterthiner, Thomas; Kornblith, Simon; Zhang, Chiyuan; Dosovitskiy, Alexey (2021-08-19). "Do Vision Transformers See Like Convolutional Neural Networks?".

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2108.08810

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

Dehghani, Mostafa; Djolonga, Josip; Mustafa, Basil; Padlewski, Piotr; Heek, Jonathan; Gilmer, Justin; Steiner, Andreas; Caron, Mathilde; Geirhos, Robert (2023-02-10),

Scaling Vision Transformers to 22 Billion Parameters

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2302.05442

https://arxiv.org/abs/2302.05442

"Scaling vision transformers to 22 billion parameters"

https://arxiv.org/abs/2302.05442

research.google

. Retrieved  2024-08-07 .

Koresh, Ella; Gross, Ronit D.; Meir, Yuval; Tzach, Yarden; Halevi, Tal; Kanter, Ido (2025).

"Unified CNNs and transformers underlying learning mechanism reveals multi-head attention modus vivendi"

https://www.sciencedirect.com/science/article/pii/S0378437125001815

Physica A: Statistical Mechanics and Its Applications

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2501.12900

https://en.wikipedia.org/wiki/Bibcode_(identifier)

2025PhyA..66630529K

https://ui.adsabs.harvard.edu/abs/2025PhyA..66630529K

https://en.wikipedia.org/wiki/Doi_(identifier)

10.1016/j.physa.2025.130529

https://doi.org/10.1016%2Fj.physa.2025.130529

https://en.wikipedia.org/wiki/ISSN_(identifier)

https://en.wikipedia.org/wiki/ISSN_(identifier)

Han, Kai; Wang, Yunhe; Chen, Hanting; Chen, Xinghao; Guo, Jianyuan; Liu, Zhenhua; Tang, Yehui; Xiao, An; Xu, Chunjing; Xu, Yixing; Yang, Zhaohui; Zhang, Yiman; Tao, Dacheng (2023-01-01). "A Survey on Vision Transformer".

## IEEE Transactions on Pattern Analysis and Machine Intelligence

(1):  87– 110.

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2012.12556

https://en.wikipedia.org/wiki/Bibcode_(identifier)

2023ITPAM..45...87H

https://ui.adsabs.harvard.edu/abs/2023ITPAM..45...87H

https://en.wikipedia.org/wiki/Doi_(identifier)

10.1109/TPAMI.2022.3152247

https://doi.org/10.1109%2FTPAMI.2022.3152247

https://en.wikipedia.org/wiki/ISSN_(identifier)

https://en.wikipedia.org/wiki/ISSN_(identifier)

https://en.wikipedia.org/wiki/PMID_(identifier)

https://en.wikipedia.org/wiki/PMID_(identifier)

Khan, Salman; Naseer, Muzammal; Hayat, Munawar; Zamir, Syed Waqas; Khan, Fahad Shahbaz; Shah, Mubarak (2022-09-13). "Transformers in Vision: A Survey".

ACM Comput. Surv

(10s): 200:1–200:41.

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2101.01169

https://en.wikipedia.org/wiki/Doi_(identifier)

10.1145/3505244

https://doi.org/10.1145%2F3505244

https://en.wikipedia.org/wiki/ISSN_(identifier)

https://en.wikipedia.org/wiki/ISSN_(identifier)

https://en.wikipedia.org/wiki/ISSN_(identifier)

Vaswani, Ashish

https://en.wikipedia.org/wiki/ISSN_(identifier)

; Shazeer, Noam; Parmar, Niki; Uszkoreit, Jakob; Jones, Llion;

Gomez, Aidan N

https://en.wikipedia.org/wiki/Aidan_Gomez

; Kaiser, Łukasz; Polosukhin, Illia (2017).

"Attention is All you Need"

https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

## Advances in Neural Information Processing Systems

. Curran Associates, Inc.

Ramachandran, Prajit; Parmar, Niki; Vaswani, Ashish; Bello, Irwan; Levskaya, Anselm; Shlens, Jon (2019).

"Stand-Alone Self-Attention in Vision Models"

https://proceedings.neurips.cc/paper/2019/hash/3416a75f4cea9109507cacd8e2f2aefc-Abstract.html

## Advances in Neural Information Processing Systems

. Curran Associates, Inc.

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/1906.05909

https://arxiv.org/abs/1906.05909

Liu, Zhuang; Mao, Hanzi; Wu, Chao-Yuan; Feichtenhofer, Christoph; Darrell, Trevor; Xie, Saining (2022).

"A ConvNet for the 2020s"

https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html

:  11976– 11986.

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2201.03545

cite journal

https://en.wikipedia.org/wiki/Template:Cite_journal

}} :  Cite journal requires  |journal=  (

https://en.wikipedia.org/wiki/Help:CS1_errors#missing_periodical

https://en.wikipedia.org/wiki/Help:CS1_errors#missing_periodical

Woo, Sanghyun; Debnath, Shoubhik; Hu, Ronghang; Chen, Xinlei; Liu, Zhuang; Kweon, In So; Xie, Saining (2023).

"ConvNeXt V2: Co-Designing and Scaling ConvNets With Masked Autoencoders"

https://openaccess.thecvf.com/content/CVPR2023/html/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.html

:  16133– 16142.

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2301.00808

cite journal

https://en.wikipedia.org/wiki/Template:Cite_journal

}} :  Cite journal requires  |journal=  (

https://en.wikipedia.org/wiki/Help:CS1_errors#missing_periodical

https://en.wikipedia.org/wiki/Help:CS1_errors#missing_periodical

Wu, Bichen; Xu, Chenfeng; Dai, Xiaoliang; Wan, Alvin; Zhang, Peizhao; Yan, Zhicheng; Masayoshi, Tomizuka; Gonzalez, Joseph; Keutzer, Kurt; Vajda, Peter (2020). "Visual Transformers: Token-based Image Representation and Processing for Computer Vision".

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2006.03677

https://arxiv.org/archive/cs.CV

Xiao, Tete; Singh, Mannat; Mintun, Eric; Darrell, Trevor; Dollár, Piotr; Girshick, Ross (2021-06-28). "Early Convolutions Help Transformers See Better".

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2106.14881

https://arxiv.org/archive/cs.CV

Liu, Ze; Lin, Yutong; Cao, Yue; Hu, Han; Wei, Yixuan; Zhang, Zheng; Lin, Stephen; Guo, Baining (2021-03-25). "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows".

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2103.14030

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

Tan, Mingxing; Le, Quoc (23 June 2021).

"EfficientNetV2: Smaller Models and Faster Training"

https://proceedings.mlr.press/v139/tan21a/tan21a.pdf

Proceedings of the 38th International Conference on Machine Learning (PMLR)

:  10096– 10106.

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2104.00298

. Retrieved  31 October  2023 .

https://arxiv.org/abs/2104.00298

Huang, Gao; Liu, Zhuang; van der Maaten, Laurens; Q. Weinberger, Kilian (28 Jan 2018). "Densely Connected Convolutional Networks".

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/1608.06993

https://arxiv.org/archive/cs.CV

Sarkar, Arjun (2021-05-20).

"Are Transformers better than CNN's at Image Recognition?"

https://web.archive.org/web/20220511082728/https://towardsdatascience.com/are-transformers-better-than-cnns-at-image-recognition-ced60ccc7c8

. Archived from

the original

https://towardsdatascience.com/are-transformers-better-than-cnns-at-image-recognition-ced60ccc7c8

on 2022-05-11 . Retrieved  2021-07-11 .

Zhai, Xiaohua; Kolesnikov, Alexander; Houlsby, Neil; Beyer, Lucas (June 2022).

"Scaling Vision Transformers"

https://dx.doi.org/10.1109/cvpr52688.2022.01179

2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)

. IEEE. pp.  1204– 1213.

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2106.04560

https://en.wikipedia.org/wiki/Doi_(identifier)

10.1109/cvpr52688.2022.01179

https://doi.org/10.1109%2Fcvpr52688.2022.01179

https://en.wikipedia.org/wiki/ISBN_(identifier)

978-1-6654-6946-3

https://en.wikipedia.org/wiki/ISBN_(identifier)

https://en.wikipedia.org/wiki/ISBN_(identifier)

Lee, Juho; Lee, Yoonho; Kim, Jungtaek; Kosiorek, Adam; Choi, Seungjin; Teh, Yee Whye (2019-05-24).

"Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks"

https://proceedings.mlr.press/v97/lee19d.html

Proceedings of the 36th International Conference on Machine Learning

. PMLR:  3744– 3753.

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/1810.00825

https://arxiv.org/abs/1810.00825

Karamcheti, Siddharth; Nair, Suraj; Chen, Annie S.; Kollar, Thomas; Finn, Chelsea; Sadigh, Dorsa; Liang, Percy (2023-02-24),

Language-Driven Representation Learning for Robotics

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2302.12766

https://arxiv.org/abs/2302.12766

Touvron, Hugo; Cord, Matthieu; Sablayrolles, Alexandre; Synnaeve, Gabriel; Jégou, Hervé (2021).

"Going Deeper With Image Transformers"

https://openaccess.thecvf.com/content/ICCV2021/html/Touvron_Going_Deeper_With_Image_Transformers_ICCV_2021_paper.html

:  32– 42.

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2103.17239

cite journal

https://en.wikipedia.org/wiki/Template:Cite_journal

}} :  Cite journal requires  |journal=  (

https://en.wikipedia.org/wiki/Help:CS1_errors#missing_periodical

https://en.wikipedia.org/wiki/Help:CS1_errors#missing_periodical

Zhou, Daquan; Kang, Bingyi; Jin, Xiaojie; Yang, Linjie; Lian, Xiaochen; Jiang, Zihang; Hou, Qibin; Feng, Jiashi (2021-04-19),

DeepViT: Towards Deeper Vision Transformer

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2103.11886

https://arxiv.org/abs/2103.11886

He, Kaiming; Chen, Xinlei; Xie, Saining; Li, Yanghao; Dollár, Piotr; Girshick, Ross (2021). "Masked Autoencoders Are Scalable Vision Learners".

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2111.06377

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

Pathak, Deepak; Krahenbuhl, Philipp; Donahue, Jeff; Darrell, Trevor; Efros, Alexei A. (June 2016). "Context Encoders: Feature Learning by Inpainting".

2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)

. IEEE. pp.  2536– 2544.

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/1604.07379

https://en.wikipedia.org/wiki/Doi_(identifier)

10.1109/CVPR.2016.278

https://doi.org/10.1109%2FCVPR.2016.278

https://en.wikipedia.org/wiki/ISBN_(identifier)

978-1-4673-8851-1

https://en.wikipedia.org/wiki/ISBN_(identifier)

https://en.wikipedia.org/wiki/ISBN_(identifier)

Liu, Ze; Lin, Yutong; Cao, Yue; Hu, Han; Wei, Yixuan; Zhang, Zheng; Lin, Stephen; Guo, Baining (2021). "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows".

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2111.06377

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

Liu, Ze; Lin, Yutong; Cao, Yue; Hu, Han; Wei, Yixuan; Zhang, Zheng; Lin, Stephen; Guo, Baining (2021). "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows".

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2111.06377

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

Liu, Ze; Lin, Yutong; Cao, Yue; Hu, Han; Wei, Yixuan; Zhang, Zheng; Lin, Stephen; Guo, Baining (2021). "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows".

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2111.06377

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

Liu, Ze; Lin, Yutong; Cao, Yue; Hu, Han; Wei, Yixuan; Zhang, Zheng; Lin, Stephen; Guo, Baining (2021). "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows".

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2111.06377

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

Zhou, Lei; Liu, Huidong; Bae, Joseph; He, Junjun; Samaras, Dimitris; Prasanna, Prateek (2022). "Self Pre-Training with Masked Autoencoders for Medical Image Classification and Segmentation".

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2203.05573

https://arxiv.org/archive/eess.IV

https://arxiv.org/archive/eess.IV

Chen, Rui; Yang, Xiaotong; Li, Yue; Peng, Guocheng; Zhu, Qiuyue; Zhang, Zhenyu; Jiang, Hong (2024). "Deep learning-based model for automatic detection and grading of meibomian gland dysfunction from infrared images".

## Applied Soft Computing

https://en.wikipedia.org/wiki/Doi_(identifier)

10.1016/j.asoc.2024.112905

https://doi.org/10.1016%2Fj.asoc.2024.112905

(inactive 17 October 2025). {{

cite journal

https://en.wikipedia.org/wiki/Template:Cite_journal

}} : CS1 maint: DOI inactive as of October 2025 (

https://en.wikipedia.org/wiki/Category:CS1_maint:_DOI_inactive_as_of_October_2025

https://en.wikipedia.org/wiki/Category:CS1_maint:_DOI_inactive_as_of_October_2025

Chen, Rui; Yang, Xiaotong; Li, Yue; Peng, Guocheng; Zhu, Qiuyue; Zhang, Zhenyu; Jiang, Hong (2024). "Deep learning-based model for automatic detection and grading of meibomian gland dysfunction from infrared images".

## Applied Soft Computing

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2507.10869

https://arxiv.org/abs/2507.10869

Madan, Chetan; Satia, Aarjav; Basu, Soumen; Gupta, Pankaj; Dutta, Usha; Arora, Chetan (2025). "Focus on Texture: Rethinking Pre-training in Masked Autoencoders for Medical Image Classification".

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2507.10869

https://arxiv.org/archive/eess.IV

https://arxiv.org/archive/eess.IV

Nguyen, Duy Kien; Li, Yanghao; Aggarwal, Vaibhav; Oswald, Martin R.; Kirillov, Alexander; Snoek, Cees G. M.; Chen, Xinlei (2024).

"R-MAE: Regions Meet Masked Autoencoders"

https://openreview.net/pdf?id=ba84RDHFnz

OpenReview.net

. International Conference on Learning Representations.

Gupta, Agrim; Wu, Jiajun; Deng, Jia; Fei-Fei, Li (2023). "Siamese Masked Autoencoders".

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2305.14344

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

Bao, Hangbo; Dong, Li; Piao, Songhao; Wei, Furu (2021-10-06).

"BEiT: BERT Pre-Training of Image Transformers"

https://openreview.net/forum?id=p-BhZSz59o4

## International Conference on Learning Representations

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2106.08254

Caron, Mathilde; Touvron, Hugo; Misra, Ishan; Jegou, Herve; Mairal, Julien; Bojanowski, Piotr; Joulin, Armand (October 2021).

"Emerging Properties in Self-Supervised Vision Transformers"

https://dx.doi.org/10.1109/iccv48922.2021.00951

2021 IEEE/CVF International Conference on Computer Vision (ICCV)

. IEEE. pp.  9630– 9640.

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2104.14294

https://en.wikipedia.org/wiki/Doi_(identifier)

10.1109/iccv48922.2021.00951

https://doi.org/10.1109%2Ficcv48922.2021.00951

https://en.wikipedia.org/wiki/ISBN_(identifier)

978-1-6654-2812-5

https://en.wikipedia.org/wiki/ISBN_(identifier)

https://en.wikipedia.org/wiki/ISBN_(identifier)

He, Kaiming; Fan, Haoqi; Wu, Yuxin; Xie, Saining; Girshick, Ross (2020).

"Momentum Contrast for Unsupervised Visual Representation Learning"

https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html

:  9729– 9738.

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/1911.05722

cite journal

https://en.wikipedia.org/wiki/Template:Cite_journal

}} :  Cite journal requires  |journal=  (

https://en.wikipedia.org/wiki/Help:CS1_errors#missing_periodical

https://en.wikipedia.org/wiki/Help:CS1_errors#missing_periodical

Grill, Jean-Bastien; Strub, Florian; Altché, Florent; Tallec, Corentin; Richemond, Pierre; Buchatskaya, Elena; Doersch, Carl; Avila Pires, Bernardo; Guo, Zhaohan; Gheshlaghi Azar, Mohammad; Piot, Bilal; kavukcuoglu, koray; Munos, Remi; Valko, Michal (2020).

"Bootstrap Your Own Latent - A New Approach to Self-Supervised Learning"

https://proceedings.neurips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html

## Advances in Neural Information Processing Systems

. Curran Associates, Inc.:  21271– 21284.

Oquab, Maxime; Darcet, Timothée; Moutakanni, Théo; Vo, Huy; Szafraniec, Marc; Khalidov, Vasil; Fernandez, Pierre; Haziza, Daniel; Massa, Francisco (2023-04-14). "DINOv2: Learning Robust Visual Features without Supervision".

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2304.07193

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

"DINOv3: Self-supervised learning for vision at unprecedented scale"

https://arxiv.org/archive/cs.CV

ai.meta.com

https://web.archive.org/web/20250814224844/https://ai.meta.com/blog/dinov3-self-supervised-vision-model/

from the original on 2025-08-14 . Retrieved  2025-08-16 .

https://web.archive.org/web/20250814224844/https://ai.meta.com/blog/dinov3-self-supervised-vision-model/

Siméoni, Oriane; Vo, Huy V.; Seitzer, Maximilian; Baldassarre, Federico; Oquab, Maxime; Jose, Cijo; Khalidov, Vasil; Szafraniec, Marc; Yi, Seungeun; Ramamonjisoa, Michaël; Massa, Francisco; Haziza, Daniel; Wehrstedt, Luca; Wang, Jianyuan; Darcet, Timothée; Moutakanni, Théo; Sentana, Leonel; Roberts, Claire; Vedaldi, Andrea; Tolan, Jamie; Brandt, John; Couprie, Camille; Mairal, Julien; Jégou, Hervé; Labatut, Patrick; Bojanowski, Piotr (2025). "DINOv3".

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2508.10104

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

Liu, Ze; Hu, Han; Lin, Yutong; Yao, Zhuliang; Xie, Zhenda; Wei, Yixuan; Ning, Jia; Cao, Yue; Zhang, Zheng; Dong, Li; Wei, Furu; Guo, Baining (2022).

"Swin Transformer V2: Scaling Up Capacity and Resolution"

https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Swin_Transformer_V2_Scaling_Up_Capacity_and_Resolution_CVPR_2022_paper.html

. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp.  12009– 12019.

https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Swin_Transformer_V2_Scaling_Up_Capacity_and_Resolution_CVPR_2022_paper.html

Bertasius, Gedas; Wang, Heng; Torresani, Lorenzo (2021-02-09). "Is Space-Time Attention All You Need for Video Understanding?".

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2102.05095

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

Szegedy, Christian; Vanhoucke, Vincent; Ioffe, Sergey; Shlens, Jon; Wojna, Zbigniew (2016).

"Rethinking the Inception Architecture for Computer Vision"

https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.html

:  2818– 2826.

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/1512.00567

cite journal

https://en.wikipedia.org/wiki/Template:Cite_journal

}} :  Cite journal requires  |journal=  (

https://en.wikipedia.org/wiki/Help:CS1_errors#missing_periodical

https://en.wikipedia.org/wiki/Help:CS1_errors#missing_periodical

Yu, Jiahui; Li, Xin; Koh, Jing Yu; Zhang, Han; Pang, Ruoming; Qin, James; Ku, Alexander; Xu, Yuanzhong; Baldridge, Jason; Wu, Yonghui (2021). "Vector-quantized Image Modeling with Improved VQGAN".

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2110.04627

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

"Parti: Pathways Autoregressive Text-to-Image Model"

https://arxiv.org/archive/cs.CV

sites.research.google

. Retrieved  2023-11-03 .

Wu, Bichen; Xu, Chenfeng; Dai, Xiaoliang; Wan, Alvin; Zhang, Peizhao; Yan, Zhicheng; Tomizuka, Masayoshi; Gonzalez, Joseph; Keutzer, Kurt (2020-11-19),

Visual Transformers: Token-based Image Representation and Processing for Computer Vision

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2006.03677

https://arxiv.org/abs/2006.03677

Dai, Zihang; Liu, Hanxiao; Le, Quoc V.; Tan, Mingxing (2021-06-09). "CoAtNet: Marrying Convolution and Attention for All Data Sizes".

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2106.04803

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

Wu, Haiping; Xiao, Bin; Codella, Noel; Liu, Mengchen; Dai, Xiyang; Yuan, Lu; Zhang, Lei (2021-03-29). "CvT: Introducing Convolutions to Vision Transformers".

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2103.15808

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

Touvron, Hugo; Cord, Matthieu; Jégou, Hervé (2022). "DeiT III: Revenge of the ViT". In Avidan, Shai; Brostow, Gabriel; Cissé, Moustapha; Farinella, Giovanni Maria; Hassner, Tal (eds.).

Computer Vision – ECCV 2022

. Lecture Notes in Computer Science. Vol. 13684. Cham: Springer Nature Switzerland. pp.  516– 533.

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2204.07118

https://en.wikipedia.org/wiki/Doi_(identifier)

10.1007/978-3-031-20053-3_30

https://doi.org/10.1007%2F978-3-031-20053-3_30

https://en.wikipedia.org/wiki/ISBN_(identifier)

978-3-031-20053-3

https://en.wikipedia.org/wiki/ISBN_(identifier)

https://en.wikipedia.org/wiki/ISBN_(identifier)

Han, Kai; Xiao, An; Wu, Enhua; Guo, Jianyuan; XU, Chunjing; Wang, Yunhe (2021).

"Transformer in Transformer"

https://proceedings.neurips.cc/paper/2021/hash/854d9fca60b4bd07f9bb215d59ef5561-Abstract.html

## Advances in Neural Information Processing Systems

. Curran Associates, Inc.:  15908– 15919.

Gross, Ronit D.; Halevi, Tal; Koresh, Ella; Tzach, Yarden; Kanter, Ido (2025).

"Low-latency vision transformers via large-scale multi-head attention"

https://www.sciencedirect.com/science/article/pii/S037843712500487X

Physica A: Statistical Mechanics and Its Applications

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2506.23832

https://en.wikipedia.org/wiki/Bibcode_(identifier)

2025PhyA..67530835G

https://ui.adsabs.harvard.edu/abs/2025PhyA..67530835G

https://en.wikipedia.org/wiki/Doi_(identifier)

10.1016/j.physa.2025.130835

https://doi.org/10.1016%2Fj.physa.2025.130835

https://en.wikipedia.org/wiki/ISSN_(identifier)

https://en.wikipedia.org/wiki/ISSN_(identifier)

https://en.wikipedia.org/wiki/ISSN_(identifier)

Naseer, Muzammal; Ranasinghe, Kanchana; Khan, Salman; Hayat, Munawar; Khan, Fahad Shahbaz; Yang, Ming-Hsuan (2021-05-21). "Intriguing Properties of Vision Transformers".

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2105.10497

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

Coccomini, Davide; Messina, Nicola; Gennaro, Claudio; Falchi, Fabrizio (2022). "Combining Efficient

and Vision Transformers for Video Deepfake Detection".

Image Analysis and Processing – ICIAP 2022

. Lecture Notes in Computer Science. Vol. 13233. pp.  219– 229.

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2107.02612

https://en.wikipedia.org/wiki/Doi_(identifier)

10.1007/978-3-031-06433-3_19

https://doi.org/10.1007%2F978-3-031-06433-3_19

https://en.wikipedia.org/wiki/ISBN_(identifier)

978-3-031-06432-6

https://en.wikipedia.org/wiki/ISBN_(identifier)

https://en.wikipedia.org/wiki/S2CID_(identifier)

https://en.wikipedia.org/wiki/S2CID_(identifier)

https://en.wikipedia.org/wiki/S2CID_(identifier)

Kirillov, Alexander; Mintun, Eric; Ravi, Nikhila; Mao, Hanzi; Rolland, Chloe; Gustafson, Laura; Xiao, Tete; Whitehead, Spencer; Berg, Alexander C.; Lo, Wan-Yen; Dollar, Piotr; Girshick, Ross (2023).

"Segment Anything"

https://openaccess.thecvf.com/content/ICCV2023/html/Kirillov_Segment_Anything_ICCV_2023_paper.html

:  4015– 4026.   {{

cite journal

https://en.wikipedia.org/wiki/Template:Cite_journal

}} :  Cite journal requires  |journal=  (

https://en.wikipedia.org/wiki/Help:CS1_errors#missing_periodical

https://en.wikipedia.org/wiki/Help:CS1_errors#missing_periodical

Jiang, Yifan; Chang, Shiyu; Wang, Zhangyang (2021).

"TransGAN: Two Pure Transformers Can Make One Strong GAN, and That Can Scale Up"

https://proceedings.neurips.cc/paper_files/paper/2021/hash/7c220a2091c26a7f5e9f1cfb099511e3-Abstract.html

## Advances in Neural Information Processing Systems

. Curran Associates, Inc.:  14745– 14758.

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2102.07074

https://arxiv.org/abs/2102.07074

Peebles, William; Xie, Saining (March 2023). "Scalable Diffusion Models with Transformers".

https://en.wikipedia.org/wiki/ArXiv_(identifier)

2212.09748v2

https://arxiv.org/abs/2212.09748v2

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

Doron, Michael; Moutakanni, Théo; Chen, Zitong S.; Moshkov, Nikita; Caron, Mathilde; Touvron, Hugo; Bojanowski, Piotr; Pernice, Wolfgang M.; Caicedo, Juan C. (2023-06-18).

"Unbiased single-cell morphology with self-supervised vision transformers"

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10312751

BioRxiv: The Preprint Server for Biology

2023.06.16.545359.

https://en.wikipedia.org/wiki/Doi_(identifier)

10.1101/2023.06.16.545359

https://doi.org/10.1101%2F2023.06.16.545359

https://en.wikipedia.org/wiki/PMC_(identifier)

https://en.wikipedia.org/wiki/PMC_(identifier)

https://en.wikipedia.org/wiki/PMID_(identifier)

https://en.wikipedia.org/wiki/PMID_(identifier)

https://en.wikipedia.org/wiki/PMID_(identifier)

Wang, Xiao; Liu, Siyan; Tsaris, Aristeidis; Choi, Jong-Youl; Aji, Ashwin; Fan, Ming; Zhang, Wei; Yin, Junqi; Ashfaq, Moetasim; Lu, Dan; Balaprakash, Prasanna (2024). "ORBIT: Oak Ridge Base Foundation Model for Earth System Predictability".

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2404.14712

physics.ao-ph

https://arxiv.org/archive/physics.ao-ph

## Further reading

https://en.wikipedia.org/w/index.php?title=Vision_transformer&action=edit&section=17

Zhang, Aston; Lipton, Zachary; Li, Mu; Smola, Alexander J. (2024).

"11.8. Transformers for Vision"

https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html

## Dive into deep learning

. Cambridge New York Port Melbourne New Delhi Singapore: Cambridge University Press.

https://en.wikipedia.org/wiki/ISBN_(identifier)

978-1-009-38943-3

https://en.wikipedia.org/wiki/ISBN_(identifier)

Steiner, Andreas; Kolesnikov, Alexander; Zhai, Xiaohua; Wightman, Ross; Uszkoreit, Jakob; Beyer, Lucas (June 18, 2021). "How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers".

https://en.wikipedia.org/wiki/ArXiv_(identifier)

https://arxiv.org/abs/2106.10270

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

## Artificial intelligence

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

## Hyperparameter

https://arxiv.org/archive/cs.CV

## Loss functions

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

Bias–variance tradeoff

https://arxiv.org/archive/cs.CV

## Double descent

https://arxiv.org/archive/cs.CV

## Overfitting

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

## Gradient descent

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

Quasi-Newton method

https://arxiv.org/archive/cs.CV

## Conjugate gradient method

https://arxiv.org/archive/cs.CV

## Backpropagation

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

## Convolution

https://arxiv.org/archive/cs.CV

## Normalization

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

## Weight initialization

https://arxiv.org/archive/cs.CV

## Regularization

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

## Augmentation

https://arxiv.org/archive/cs.CV

## Prompt engineering

https://arxiv.org/archive/cs.CV

## Reinforcement learning

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

## Policy gradient

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

## Latent diffusion model

https://arxiv.org/archive/cs.CV

## Autoregression

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

## Uncanny valley

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

Self-supervised learning

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

Recursive self-improvement

https://arxiv.org/archive/cs.CV

## Hallucination

https://arxiv.org/archive/cs.CV

## Word embedding

https://arxiv.org/archive/cs.CV

## Vibe coding

https://arxiv.org/archive/cs.CV

https://arxiv.org/archive/cs.CV

https://en.wikipedia.org/wiki/AI_alignment

## Applications

## Machine learning

https://en.wikipedia.org/wiki/AI_alignment

In-context learning

https://en.wikipedia.org/wiki/AI_alignment

## Artificial neural network

https://en.wikipedia.org/wiki/AI_alignment

## Deep learning

https://en.wikipedia.org/wiki/AI_alignment

## Language model

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

## Model Context Protocol

https://en.wikipedia.org/wiki/AI_alignment

## Intelligent agent

https://en.wikipedia.org/wiki/AI_alignment

## Artificial human companion

https://en.wikipedia.org/wiki/AI_alignment

Humanity's Last Exam

https://en.wikipedia.org/wiki/AI_alignment

Artificial general intelligence (AGI)

https://en.wikipedia.org/wiki/AI_alignment

Implementations Audio–visual

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

## Human image synthesis

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

## Computer vision

https://en.wikipedia.org/wiki/AI_alignment

## Speech synthesis

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

## Speech recognition

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

## Facial recognition

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

Text-to-image models

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

## Stable Diffusion

https://en.wikipedia.org/wiki/AI_alignment

Text-to-video models

https://en.wikipedia.org/wiki/AI_alignment

## Dream Machine

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

## Music generation

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

## Chinchilla AI

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

Gemini (language model)

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

## Project Debater

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

## IBM Watsonx

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

## Decisional

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

## OpenAI Five

https://en.wikipedia.org/wiki/AI_alignment

Self-driving car

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

## Action selection

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

## Robot control

https://en.wikipedia.org/wiki/AI_alignment

## Alan Turing

https://en.wikipedia.org/wiki/AI_alignment

## Warren Sturgis McCulloch

https://en.wikipedia.org/wiki/AI_alignment

## Walter Pitts

https://en.wikipedia.org/wiki/AI_alignment

## John von Neumann

https://en.wikipedia.org/wiki/AI_alignment

Christopher D. Manning

https://en.wikipedia.org/wiki/AI_alignment

## Claude Shannon

https://en.wikipedia.org/wiki/AI_alignment

Shun'ichi Amari

https://en.wikipedia.org/wiki/AI_alignment

## Kunihiko Fukushima

https://en.wikipedia.org/wiki/AI_alignment

## Takeo Kanade

https://en.wikipedia.org/wiki/AI_alignment

## Marvin Minsky

https://en.wikipedia.org/wiki/AI_alignment

## John McCarthy

https://en.wikipedia.org/wiki/AI_alignment

## Nathaniel Rochester

https://en.wikipedia.org/wiki/AI_alignment

## Allen Newell

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

Herbert A. Simon

https://en.wikipedia.org/wiki/AI_alignment

## Oliver Selfridge

https://en.wikipedia.org/wiki/AI_alignment

## Frank Rosenblatt

https://en.wikipedia.org/wiki/AI_alignment

## Bernard Widrow

https://en.wikipedia.org/wiki/AI_alignment

## Joseph Weizenbaum

https://en.wikipedia.org/wiki/AI_alignment

## Seymour Papert

https://en.wikipedia.org/wiki/AI_alignment

## Seppo Linnainmaa

https://en.wikipedia.org/wiki/AI_alignment

## Paul Werbos

https://en.wikipedia.org/wiki/AI_alignment

## Geoffrey Hinton

https://en.wikipedia.org/wiki/AI_alignment

## John Hopfield

https://en.wikipedia.org/wiki/AI_alignment

Jürgen Schmidhuber

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

## Yoshua Bengio

https://en.wikipedia.org/wiki/AI_alignment

Lotfi A. Zadeh

https://en.wikipedia.org/wiki/AI_alignment

## Stephen Grossberg

https://en.wikipedia.org/wiki/AI_alignment

## Alex Graves

https://en.wikipedia.org/wiki/AI_alignment

## James Goodnight

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

## Alex Krizhevsky

https://en.wikipedia.org/wiki/AI_alignment

## Ilya Sutskever

https://en.wikipedia.org/wiki/AI_alignment

## Oriol Vinyals

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

## Ian Goodfellow

https://en.wikipedia.org/wiki/AI_alignment

## Demis Hassabis

https://en.wikipedia.org/wiki/AI_alignment

## David Silver

https://en.wikipedia.org/wiki/AI_alignment

## Andrej Karpathy

https://en.wikipedia.org/wiki/AI_alignment

## Ashish Vaswani

https://en.wikipedia.org/wiki/AI_alignment

## Noam Shazeer

https://en.wikipedia.org/wiki/AI_alignment

## Aidan Gomez

https://en.wikipedia.org/wiki/AI_alignment

## John Schulman

https://en.wikipedia.org/wiki/AI_alignment

## Mustafa Suleyman

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

## Daniel Kokotajlo

https://en.wikipedia.org/wiki/AI_alignment

François Chollet

https://en.wikipedia.org/wiki/AI_alignment

## Architectures

## Neural Turing machine

https://en.wikipedia.org/wiki/AI_alignment

## Differentiable neural computer

https://en.wikipedia.org/wiki/AI_alignment

## Transformer

https://en.wikipedia.org/wiki/AI_alignment

Vision transformer (ViT)

https://en.wikipedia.org/wiki/AI_alignment

Recurrent neural network (RNN)

https://en.wikipedia.org/wiki/AI_alignment

Long short-term memory (LSTM)

https://en.wikipedia.org/wiki/AI_alignment

Gated recurrent unit (GRU)

https://en.wikipedia.org/wiki/AI_alignment

## Echo state network

https://en.wikipedia.org/wiki/AI_alignment

Multilayer perceptron (MLP)

https://en.wikipedia.org/wiki/AI_alignment

Convolutional neural network (CNN)

https://en.wikipedia.org/wiki/AI_alignment

Residual neural network (RNN)

https://en.wikipedia.org/wiki/AI_alignment

## Highway network

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

## Autoencoder

https://en.wikipedia.org/wiki/AI_alignment

Variational autoencoder (VAE)

https://en.wikipedia.org/wiki/AI_alignment

Generative adversarial network (GAN)

https://en.wikipedia.org/wiki/AI_alignment

Graph neural network (GNN)

https://en.wikipedia.org/wiki/AI_alignment

https://en.wikipedia.org/wiki/AI_alignment

Retrieved from "

https://en.wikipedia.org/w/index.php?title=Vision_transformer&oldid=1318235876

https://en.wikipedia.org/w/index.php?title=Vision_transformer&oldid=1318235876

https://en.wikipedia.org/wiki/Help:Category

## Neural network architectures

https://en.wikipedia.org/wiki/Help:Category

## Computer vision

https://en.wikipedia.org/wiki/Help:Category

## Artificial neural networks

https://en.wikipedia.org/wiki/Help:Category

## Image processing

https://en.wikipedia.org/wiki/Help:Category

2020 in artificial intelligence

https://en.wikipedia.org/wiki/Help:Category

Hidden categories:

CS1 errors: missing periodical

https://en.wikipedia.org/wiki/Help:Category

CS1 maint: DOI inactive as of October 2025

https://en.wikipedia.org/wiki/Help:Category

## Articles with short description

https://en.wikipedia.org/wiki/Help:Category

## Short description is different from Wikidata

https://en.wikipedia.org/wiki/Help:Category

