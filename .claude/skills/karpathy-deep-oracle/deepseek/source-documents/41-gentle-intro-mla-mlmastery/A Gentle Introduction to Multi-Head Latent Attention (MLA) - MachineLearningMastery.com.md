---
sourceFile: "A Gentle Introduction to Multi-Head Latent Attention (MLA) - MachineLearningMastery.com"
exportedBy: "Kortex"
exportDate: "2025-10-29T02:36:03.146Z"
---

# A Gentle Introduction to Multi-Head Latent Attention (MLA) - MachineLearningMastery.com

32931290-c507-45b4-949b-f4b7621ac146

A Gentle Introduction to Multi-Head Latent Attention (MLA) - MachineLearningMastery.com

d078827b-d68a-47b8-b1d1-be2c7ae85ea9

https://machinelearningmastery.com/a-gentle-introduction-to-multi-head-latent-attention-mla/

## Making developers awesome at machine learning

## Making Developers Awesome at Machine Learning

Click to Take the FREE Crash-Course

## Get Started

## Better Deep Learning

Code Algorithms Implementing machine learning algorithms from scratch.

## Computer Vision

## Data Preparation

Deep Learning (keras) Deep Learning

## Deep Learning with PyTorch

## Ensemble Learning

## Foundations of Data Science

## Hugging Face Transformers

## Neural Net Time Series Deep Learning for Time Series Forecasting

## Imbalanced Learning

## Intermediate Data Science

## Intro to Time Series

## Intro to Algorithms

## Linear Algebra

LSTMs Long Short-Term Memory Networks

## Optimization

## Probability

Python (scikit-learn)

## Python for Machine Learning

## Stable Diffusion

Weka (no code)

## Making developers awesome at machine learning

Click to Take the FREE Crash-Course

## Making Developers Awesome at Machine Learning

Click to Take the FREE Crash-Course

## Get Started

## Better Deep Learning

Code Algorithms Implementing machine learning algorithms from scratch.

## Computer Vision

## Data Preparation

Deep Learning (keras) Deep Learning

## Deep Learning with PyTorch

## Ensemble Learning

## Foundations of Data Science

## Hugging Face Transformers

## Neural Net Time Series Deep Learning for Time Series Forecasting

## Imbalanced Learning

## Intermediate Data Science

## Intro to Time Series

## Intro to Algorithms

## Linear Algebra

LSTMs Long Short-Term Memory Networks

## Optimization

## Probability

Python (scikit-learn)

## Python for Machine Learning

## Stable Diffusion

Weka (no code)

A Gentle Introduction to Multi-Head Latent Attention (MLA)

https://machinelearningmastery.com/author/adriantam/

on   September 12, 2025   in

## Building Transformer Models

https://machinelearningmastery.com/category/building-transformer-models/

https://machinelearningmastery.com/category/building-transformer-models/

Not all Transformer models are called “large language models” because you can build a very small model using the Transformer architecture. The truly large Transformer models are often impractical to use at home because they’re too large to fit on a single computer and too slow to run without a cluster of GPUs.

The recent introduction of Multi-Head Latent Attention (MLA) proposed a new approach to running attention operations with a lower memory footprint. First proposed in DeepSeek-V2, it changes how you perform matrix multiplication in the attention operation. In this post, you will learn how MLA works and how to implement it in PyTorch.

Kick-start your project

with my book

## Building Transformer Models From Scratch with PyTorch

https://machinelearningmastery.com//building-transformer-models-from-scratch/

. It provides

self-study tutorials

working code

Let’s get started.

A Gentle Introduction to Multi-Head Latent Attention (MLA) 
 Photo by

## Victoriano Izquierdo

https://unsplash.com/photos/cars-on-gray-asphalt-road-in-the-city-during-nighttime-29Rh5DOS5Qs

. Some rights reserved.

This post is divided into three parts; they are:

Low-Rank Approximation of Matrices

Multi-head Latent Attention (MLA)

## PyTorch Implementation

Low-Rank Approximation of Matrices

Multi-Head Attention (MHA) and Grouped-Query Attention (GQA) are the attention mechanisms used in almost all transformer models. Recently, a new attention mechanism called

Multi-head Latent Attention

(MLA) was proposed in DeepSeek-V2 to further reduce computational cost and speed up inference.

The core idea is to use low-rank approximation to convert a large matrix into two smaller matrices, $M\approx UV$. If matrix $M$ is an $n\times m$ matrix, $U$ would be an $n\times r$ matrix and $V$ would be an $r\times m$ matrix, where $r$ is smaller than both $n$ and $m$. The product $UV$ won’t be identical to $M$, but it can be close enough for practical purposes. One method to decompose $M$ into $U$ and $V$ is to use singular value decomposition (SVD) and select the top $r$ orthonormal bases. Specifically, the SVD of matrix $M$ produces:

$$ 
  M = U \Sigma V^T 
  $$

where $U$ and $V$ are square matrices (the orthonormal bases) and $\Sigma$ is a diagonal matrix containing the singular values of $M$. If you zero out the lower singular values from the diagonal of $\Sigma$, you effectively remove the lower rows of $U$ and $V$. The result of this multiplication is an approximation of $M$. If the elements zeroed out from $\Sigma$ are numerically close to zero, this approximation will be quite accurate.

This concept isn’t new. Low-rank adaptation is a common technique for fine-tuning large transformer models, and it also uses such approximations of projection matrices to augment the model for new functionality.

Multi-head Latent Attention (MLA)

Similar to GQA, which only manipulates the key and value projections, Multi-head Latent Attention (MLA) also factorizes only the key and value projections. However, unlike GQA, MLA doesn’t share the key and value projections across multiple queries, but operates in the same way as multi-head attention. The original paper describes MLA as operating on the

compressed latent representation

of the key/value space during inference.

For input sequence $X$, self-attention using MLA computes:

$$ 
  \begin{aligned} 
  Q &= XW_Q^DW_Q^U = (XW_Q^D)W_Q^U = C_QW_Q^U \\ 
  K &= XW_{KV}^DW_K^U = (XW_{KV}^D)W_K^U = C_{KV}W_K^U \\ 
  V &= XW_{KV}^DW_V^U = (XW_{KV}^D)W_V^U = C_{KV}W_V^U 
  \end{aligned} 
  $$

$W_Q^D,W_{KV}^D \in \mathbb{R}^{d\times r}$ are low-rank compression matrices, with a small $r$, to reduce the dimension

$W_Q^U,W_K^U,W_V^U \in \mathbb{R}^{r\times(n_h d_h)}$ are decompression matrices, to recover the dimension

$r$ is the latent dimension, typically $r \ll n_h\cdot d_h$

You might notice that $K$, for example, is computed as a projection from $X$, but through two matrix multiplications instead of one. This might seem like a waste of computation, but you’ll see why this is actually efficient in the following explanation.

Now consider the standard attention operation:

$$ 
  \begin{aligned} 
  O_i &= \text{softmax}\big(\frac{QK^\top}{\sqrt{d_k}}\big)V \\ 
  &= \text{softmax}\big(\frac{(XW_Q^D W_{Q,i}^U)(XW_{KV}^D W_{K,i}^U)^\top}{\sqrt{d_k}}\big)XW_{KV}^D W_V^U \\ 
  &= \text{softmax}\big(\frac{XW_Q^D W_{Q,i}^U {W_{K,i}^U}^\top {W_{KV}^D}^\top X^\top}{\sqrt{d_k}}\big)XW_{KV}^D W_{V,i}^U \\ 
  &= \text{softmax}\big(\frac{C_Q W_{Q,i}^U {W_{K,i}^U}^\top C_{KV}^\top}{\sqrt{d_k}}\big)C_{KV} W_{V,i}^U 
  \end{aligned} 
  $$

This is where MLA’s computational savings come from: Instead of factoring the key and value projection matrices $W^K$ and $W^V$ independently, the compression matrices are shared. Recall that even in cross-attention, the key and value input sequences are the same, so you have a shared factor $C_{KV}$ for both the $K$ and $V$ projections.

Another key technique is that the multiple heads of attention are implemented only in the decompression matrices $W_Q^U, W_K^U, W_V^U$. Hence, for a single head, the equations above use the notations $W_{Q,i}^U, W_{K,i}^U, W_{V,i}^U$. In this way, both $C_Q$ and $C_{KV}$ are computed once and shared across all heads.

Furthermore, note the matrix multiplication $W_{Q,i}^U{W_{K,i}^U}^\top$ in the last line of the softmax above. This is a multiplication of two decompression matrices, independent of the input $X$. Therefore, this matrix multiplication can be pre-computed and cached as $W_{QK,i} = W_{Q,i}^U{W_{K,i}^U}^\top$, saving time during inference.

By breaking down the projection matrices and using a lower dimension for the latent representation, MLA saves computation and memory usage even though there are more matrices involved in the equation.

## PyTorch Implementation

Once you understand MLA’s design, implementing it in PyTorch is straightforward. Here’s an example:

import math import torch import torch.nn as nn class MultiHeadLatentAttention(nn.Module): def __init__(self, d_model=128*128, num_heads=128, q_latent_dim=12, kv_latent_dim=4): super().__init__() self.d_model = d_model self.num_heads = num_heads self.q_latent_dim = q_latent_dim self.kv_latent_dim = kv_latent_dim head_dim = d_model // num_heads # Query projections self.Wq_d = nn.Linear(d_model, q_latent_dim) # Precomputed matrix multiplications of W_q^U and W_k^U, for multiple heads self.W_qk = nn.Linear(q_latent_dim, num_heads * kv_latent_dim) # Key/Value latent projections self.Wkv_d = nn.Linear(d_model, kv_latent_dim) self.Wv_u = nn.Linear(kv_latent_dim, num_heads * head_dim) # Output projection self.Wo = nn.Linear(num_heads * head_dim, d_model) def forward(self, x): batch_size, seq_len, d_model = x.shape # Projections of input into latent spaces C_q = self.Wq_d(x) # shape: (batch_size, seq_len, q_latent_dim) C_kv = self.Wkv_d(x) # shape: (batch_size, seq_len, kv_latent_dim) # Attention score, shape: (batch_size, num_heads, seq_len, seq_len) C_qW_qk = self.W_qk(C_q).view(batch_size, seq_len, self.num_heads, self.kv_latent_dim) scores = torch.matmul(C_qW_qk.transpose(1, 2), C_kv.transpose(-2, -1)[:, None, ...]) / math.sqrt(self.kv_latent_dim) # Attention computation attn_weight = torch.softmax(scores, dim=-1) # Restore V from latent space V = self.Wv_u(C_kv).view(batch_size, seq_len, self.num_heads, -1) # Compute attention output, shape: (batch_size, seq_len, num_heads, head_dim) output = torch.matmul(attn_weight, V.transpose(1,2)).transpose(1,2).contiguous() # Concatentate the heads, then apply output projection output = self.Wo(output.view(batch_size, seq_len, -1)) return output   1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46   import  math import  torch import  torch . nn  as   nn   class   MultiHeadLatentAttention ( nn . Module ) :   def  __init__ ( self ,   d_model = 128 * 128 ,   num_heads = 128 ,   q_latent_dim = 12 ,   kv_latent_dim = 4 ) :   super ( ) . __init__ ( )   self . d_model   =   d_model   self . num_heads   =   num_heads   self . q_latent_dim   =   q_latent_dim   self . kv_latent_dim   =   kv_latent_dim   head_dim   =   d_model   // num_heads   # Query projections   self . Wq_d   =   nn . Linear ( d_model ,   q_latent_dim )   # Precomputed matrix multiplications of W_q^U and W_k^U, for multiple heads   self . W_qk   =   nn . Linear ( q_latent_dim ,   num_heads *   kv_latent_dim )   # Key/Value latent projections   self . Wkv_d   =   nn . Linear ( d_model ,   kv_latent_dim )   self . Wv_u   =   nn . Linear ( kv_latent_dim ,   num_heads *   head_dim )   # Output projection   self . Wo   =   nn . Linear ( num_heads *   head_dim ,   d_model )   def  forward ( self ,   x ) :   batch_size ,   seq_len ,   d_model   =   x . shape   # Projections of input into latent spaces   C_q   =   self . Wq_d ( x )   # shape: (batch_size, seq_len, q_latent_dim)   C_kv   =   self . Wkv_d ( x )   # shape: (batch_size, seq_len, kv_latent_dim)   # Attention score, shape: (batch_size, num_heads, seq_len, seq_len)   C_qW_qk   =   self . W_qk ( C_q ) . view ( batch_size ,   seq_len ,   self . num_heads ,   self . kv_latent_dim )   scores   =   torch . matmul ( C_qW_qk . transpose ( 1 ,   2 ) ,   C_kv . transpose ( - 2 ,   - 1 ) [ : ,   None ,   . . . ] )   /   math . sqrt ( self . kv_latent_dim )   # Attention computation   attn_weight   =   torch . softmax ( scores ,   dim = - 1 )   # Restore V from latent space   V   =   self . Wv_u ( C_kv ) . view ( batch_size ,   seq_len ,   self . num_heads ,   - 1 )   # Compute attention output, shape: (batch_size, seq_len, num_heads, head_dim)   output   =   torch . matmul ( attn_weight ,   V . transpose ( 1 , 2 ) ) . transpose ( 1 , 2 ) . contiguous ( )   # Concatentate the heads, then apply output projection   output   =   self . Wo ( output . view ( batch_size ,   seq_len ,   - 1 ) )   return   output

Comparing this code with the equations from the previous section, you can see that $W_{QK,i}$ is defined directly as a component in this module.

The input sequence  x  to the  forward()  method has a shape of  (batch_size, seq_len, d_model) , as does the final output. First, the input  x  is projected into  C_q  and  C_kv , which are shared by all attention heads. Next, the attention score is computed for each head using two matrix multiplications. First, you use  self.W_qk  to multiply  C_q , then reshape the result into  (batch_size, seq_len, num_heads, kv_latent_dim) . Then you multiply it with  C_kv , after appropriate axis transpositions, to get the attention score. Since  C_qW_qk  is a 4-dimensional tensor and  C_kv  is a 3-dimensional one, you add a dummy dimension to  C_kv  in place of the  num_heads  dimension.

Next, you obtain the attention weight by applying softmax to the attention score. To get the attention output, you multiply the attention weight with  V , which is computed from  C_kv  projected using  self.Wv_u . Finally, you concatenate the outputs of all heads and apply the output projection to get the final output.

The original MLA paper suggests that it outperforms GQA in both model quality and inference speed. Since the matrices are smaller in this case, it’s also more memory efficient. However, you don’t need to train a model specifically for MLA. You can also convert a model trained with traditional multi-head attention to MLA by factoring the projection matrices after training.

## Further Readings

Below are some resources you may find useful:

DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model

DeepSeek-V3 Technical Report

LORD: Low Rank Decomposition of Monolingual Code LLMs for One-Shot Compression

Low-Rank Adaptation of Large Language Models

Generalization Guarantees for Neural Networks via Harnessing the Low-rank Structure of the Jacobian

DeepSeek Multi-Head Latent Attention implementation

TransMLA: Multi-Head Latent Attention Is All You Need

In this post, you learned how MLA works and how to implement it in PyTorch. MLA is a new attention mechanism proposed in DeepSeek-V2 that uses low-rank approximation of projection matrices in multi-head attention. This approach can significantly reduce computational cost and memory usage while maintaining model performance.

## Building Transformer Models From Scratch with PyTorch

Build, train, and understand Transformers in pure PyTorch

...step by step

Learn how in my new Ebook:

## Building Transformer Models From Scratch with PyTorch

https://machinelearningmastery.com/building-transformer-models-from-scratch/

self-study tutorials

end-to-end projects

attention mechanisms

normalization layers

, and much more...

## Finally Bring Machine Learning To 
  Your Own Projects

Skip the Academics. Just Results.

See What's Inside

## More On This Topic

## How to Explore the GAN Latent Space When Generating Faces

A Gentle Introduction to Multi-Head Attention and…

Gentle Introduction to Global Attention for…

A Gentle Introduction to Attention Masking in…

## A Gentle Introduction to Attention and Transformer Models

Attention in Long Short-Term Memory Recurrent Neural…

## About Adrian Tam

Adrian Tam, PhD is a data scientist and software engineer.

View all posts by Adrian Tam  →

https://machinelearningmastery.com/author/adriantam/

Converting Pandas DataFrames to PyTorch DataLoaders for Custom Deep Learning Model Training

https://machinelearningmastery.com/author/adriantam/

Combining XGBoost and Embeddings: Hybrid Semantic Boosted Trees?

No comments yet.

## Leave a Reply

Click here to cancel reply.

## Jason Brownlee

## PhD   and I

help developers

get results with

machine learning

https://machinelearningmastery.com/about

Never miss a tutorial:

Picked for you:

## A Gentle Introduction to Attention and Transformer Models

https://machinelearningmastery.com/about

## Mixture of Experts Architecture in Transformer Models

https://machinelearningmastery.com/about

## Tokenizers in Language Models

https://machinelearningmastery.com/about

Building a Plain Seq2Seq Model for Language Translation

https://machinelearningmastery.com/about

## Word Embeddings in Language Models

https://machinelearningmastery.com/about

Loving the Tutorials?

## Buiding Transformer Models From Scratch wtih PyTorch

https://machinelearningmastery.com/building-transformer-models-from-scratch/

EBook is 
  where you'll find the

## Really Good

>> See What's Inside

Machine Learning Mastery is part of Guiding Tech Media, a leading digital media publisher focused on helping people figure out technology.

## Visit our corporate website

https://www.guidingtechmedia.com

to learn more about our mission and team.

https://www.guidingtechmedia.com

https://www.guidingtechmedia.com

https://www.guidingtechmedia.com

https://www.guidingtechmedia.com

https://www.guidingtechmedia.com

© 2025 Guiding Tech Media All Rights Reserved

https://www.guidingtechmedia.com

https://www.guidingtechmedia.com

https://www.guidingtechmedia.com

https://www.guidingtechmedia.com

https://www.guidingtechmedia.com

