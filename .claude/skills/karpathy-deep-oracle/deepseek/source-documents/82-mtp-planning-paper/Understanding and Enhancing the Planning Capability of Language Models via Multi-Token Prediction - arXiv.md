---
sourceFile: "Understanding and Enhancing the Planning Capability of Language Models via Multi-Token Prediction - arXiv"
exportedBy: "Kortex"
exportDate: "2025-10-29T02:36:40.779Z"
---

# Understanding and Enhancing the Planning Capability of Language Models via Multi-Token Prediction - arXiv

13760bd3-22fd-41a6-beb6-dbd1ba5a9f6c

Understanding and Enhancing the Planning Capability of Language Models via Multi-Token Prediction - arXiv

293a89da-f8e4-467d-9f50-4789e71629ec

https://www.arxiv.org/pdf/2509.23186

UNDERSTANDING AND ENHANCING THE PLANNING CAPABILITY OF LANGUAGE MODELS VIA MULTI-TOKEN PREDICTION

Qimin Zhong1, Hao Liao1, Siwei Wang2, Mingyang Zhou1

Xiaoqun Wu1, Rui Mao1, Wei Chen2

1College of Computer Science and Software Engineering, Shenzhen University, China 2Microsoft Research Asia

Large Language Models (LLMs) have achieved impressive performance across diverse tasks but continue to struggle with learning transitive relations, a corner-stone for complex planning (Balesni et al., 2024; Wang et al., 2024b). To address this issue, we investigate the Multi-Token Prediction (MTP) paradigm and its im-pact to transitive relation learning. We theoretically analyze the MTP paradigm using a Transformer architecture composed of a shared output head and a transfer layer. Our analysis reveals that the transfer layer gradually learns the multi-step adjacency information, which in turn enables the backbone model to capture unob-served transitive reachability relations beyond those directly present in the training data, albeit with some inevitable noise in adjacency estimation. Building on this foundation, we propose two strategies to enhance the transfer layer and overall learning quality: Next-Token Injection (NTI) and a Transformer-based transfer layer. Our experiments on both synthetic graphs and the Blocksworld planning benchmark validate our theoretical findings and demonstrate that the improve-ments significantly enhance the model’s path-planning capability. These findings deepen our understanding of how Transformers with MTP learn in complex plan-ning tasks, and provide practical strategies to overcome the transitivity bottleneck, paving the way toward structurally aware and general-purpose planning models.

1 INTRODUCTION

Transformer models have achieved remarkable success across natural language processing (Vaswani et al., 2017; Devlin et al., 2019; Brown et al., 2020), computer vision (Dosovitskiy et al., 2021; Car-ion et al., 2020; Liu et al., 2021), reinforcement learning (Parisotto et al., 2020; Chen et al., 2021a; Janner et al., 2021), program synthesis (Chen et al., 2021b; Nijkamp et al., 2023), and complex plan-ning (Chen et al., 2021a; Lehnert et al., 2024). However, a fundamental question remains: do these models truly possess planning capabilities, or do they merely rely on reconstructing patterns from training data? This question is particularly critical in complex planning tasks, which often require compositional planning to generate coherent sequences of actions toward a goal. In such tasks, it is natural and effective to abstract the problem as path finding on a graph, where nodes represent states and edges represent executable actions. path finding not only lies at the core of many classi-cal planning problems but also closely relates to sequential decision-making in real-world complex tasks, such as robotic motion planning, automated scheduling, and step-wise reasoning in mathemat-ical proofs. Under this abstraction, standard autoregressive Transformers typically perform reliable planning on paths observed during training (Wang et al., 2024b).

However, the performance of these models degrades substantially when the task requires transitive planning, which demands combining information from multiple path segments to infer new reach-ability relations: as demonstrated by Wang et al. (2024b), standard autoregressive Transformers would fail to infer that node A could reach node C when the training data contain both paths from A to B and from B to C but no paths from A to C. This limitation not only prevents the model from generalizing to unseen paths in complex planning tasks but also highlights a fundamental bottle-neck of current Transformers in structured planning (Zhang et al., 2024). Therefore, understanding

and improving the model’s compositional learning ability is crucial for enhancing Transformers in structured planning and sequential prediction tasks.

To address this issue, we explore the Multi-Token Prediction (MTP) paradigm, in which the model predicts multiple future nodes in a single training step, providing richer supervision sig-nals and showing potential for modeling long-range dependencies and structural relationships. MTP has been adopted in a number of leading AI companies and their models such as Meta and DeepSeek (Gloeckle et al., 2024; Liu et al., 2024), but their underlying mechanism, especially for planning, remain largely unexplored. In this work, we build on the analytical framework of ALPINE (Wang et al., 2024b) to systematically investigate the effect of MTP on path planning, and propose architectural enhancements to strengthen the ability of Transformers to learn transi-tive reachability relations. Our study provides both theoretical insights and practical guidance for developing future Transformers with stronger reasoning and planning capabilities.

To summarize, our contributions include: First, through a theoretical analysis on a simplified Trans-former, we show how the multi-token loss simultaneously shapes the transfer matrices and backbone network weights, revealing the coupled learning dynamics among the transfer layer and the adja-cency and reachability within the backbone model (Section 3). Second, based on these insights, we propose two enhancements to the architecture: (1) Next-Token Injection (NTI), to explicitly inject intermediate nodes as multi-hop supervision; and (2) a Transformer-based transfer layer to maintain structural consistency across prediction steps (Section 4). Third, we conduct extensive experiments on synthetic graphs as well as the Blocksworld planning benchmark and show that these methods significantly improve prediction accuracy, and the learned transfer matrices progressively approx-imate the ground-truth adjacency matrices (Section 5). These findings indicate that MTP together with our enhancements provides better support for transitive relations, advancing models toward stronger structural planning capabilities.

Additional Related Work Our work relates to recent studies on the structural planning ability of large language models (LLMs). Prior work has examined LLMs in path prediction and task planning over symbolic or graph-structured inputs. Other approaches use prompt engineering or structural injection to guide graph reasoning. Multi-Token Prediction (MTP) has been explored to improve training efficiency. Unlike these, we focus on the interpretability of structural learning under MTP. See Appendix A for more details.

2 SETTING AND PRELIMINARIES

We use a to denote a column vector, A denotes a matrix; the ith component of a is written as a(i); the (i, j) entry of A is written A(i,j); the i-th row (column) of A is denoted A(i,:) (A(:,i)).

2.1 PATH PLANNING OVER SIMPLE DIRECTED GRAPHS WITH LANGUAGE MODEL

To evaluate the planning capability of an autoregressive language model, we construct path-planning tasks on directed graphs. Let G = (V, E) be a directed acyclic graph with node set V and edge set E . For any u, v ∈ V , the presence of (u, v) ∈ E indicates a directed edge from u to v.

During training, each reachable source–target pair (s, t) (i.e., t can be reached from s via one or more edges) is encoded as a token sequence "s t s a b c t \n", where s and t denote the source and target, a, b, c represent intermediate nodes, and \n marks sequence termination. The model learns in an autoregressive fashion by predicting every next token in turn. At test time, only the prefix "s t" is provided; the model must autoregressively complete a valid path from s to t, respecting the graph’s adjacency and reachability constraints. This procedure measures the model’s ability to capture both one-step adjacency and long-range reachability information.

We denote the ground-truth adjacency matrix and reachability matrix of the graph by Atrue and Rtrue, respectively:

Atrue (i,k) =

{ 1, if (i, k) ∈ E , 0, otherwise.

Rtrue (t,k) =

{ 1, if there exists a path k → t in G, 0, otherwise.

2.2 HIERARCHICAL EVALUATION OF GENERALIZATION ABILITY

Given a training set D where u = (u1, . . . , un) ∈ D is a valid path in the graph, we define the observed adjacency and reachability matrices in the observation graph Gobs:

Aobs (i,k) =

{ 1, if ∃u ∈ D, n ∈ [3, N − 1] s.t. un = i, un+1 = k,

0, otherwise

Robs (t,k) =

{ 1, if ∃u ∈ D, n ∈ [4, N ] s.t. u2 = t, un = k,

0, otherwise.

Here, Robs is a subset of Rtrue, contains only the reachability relations directly observed in D.

We then partition test pairs (s, t) into four degrees based on their observed reachability in Gobs: a) degree-0 if Robs

(t,s) = 1; b) degree-1 if Robs (t,s) = 0 but there exists u such that Aobs

(s,u) = 1 and Robs

(t,u) = 1; c) degree-2 if it is neither degree-0 nor degree-1 but there exists u such that Aobs (s,u) = 1

and (u, t) is degree-1; d) degree-3 otherwise.

Following the standard architecture of Generative Pretrained Transformer (GPT) (Radford et al., 2018), each Transformer layer comprises multi-head attention (MHA), residual connections, layer normalization (LN), and a feed-forward network (FFN), as

Transformer(X) = FFN ( LN2(MHA(LN1(X)) +X)

) +MHA(LN1(X)) +X. (1)

A training sequence u = (u1, . . . , un) is first mapped to a sequence of corresponding embedding vectors X1:n = (x1, . . . ,xn) using an embedding matrix Wt. This sequence is then passed through the Transformer layers, which produce a sequence of contextualized hidden states

H1:n = (h1, . . . ,hn) = Transformer(X1:n). (2)

For next-token prediction, the model uses the final hidden state hn, which corresponds to the last token in the input sequence. The predictive distribution is then given by:

p(un+1 | u1:n) = softmax(Wohn).

The standard next-token training objective is the cross-entropy loss,

L = − N−1∑ n=1

log p(un+1 | u1:n). (3)

Wang et al. (2024b) point out that GPT trained solely with next-token loss achieves over 90% accu-racy on degree-0 and degree-1 tasks, but drops to about 60% on degree-2 tasks. The model can only learn the observed reachability matrix Robs, and fails to learn the complete true reachability matrix Rtrue, highlighting its inability to generalize to transitive paths unseen during training.

3 MECHANISTIC UNDERSTANDING OF MULTI-TOKEN PREDICTION

In this paper, we investigate the mechanism of MTP in which multi-step tokens are used during learning to enhance the learning effectiveness, while inference is still done by next-token prediction. Some prior work also uses MTP at inference to accelerate generation, but it is not our focus (See Appendix A for more discussions). We use a shared output head architecture for MTP (Figure 1(b)). Separate learnable transfer layers map the backbone output to different target positions in parallel before decoding, enabling the model to share parameters across steps. This design enhances the in-terpretability of structural learning. The shared output head is denoted as Wo, and the transfer layer as W T . In contrast, Meta’s MTP architecture assigns an independent output head to each prediction step (Gloeckle et al., 2024) (Figure 1(a)). While flexible, this approach lacks parameter sharing, making it harder to model unified patterns across steps. The two architectures are mathematically equivalent and thus we choose the first one for its better interpretability.

3.1 TRAINING DYNAMICS OF 2-TOKEN PREDICTION

To enable theoretical analysis on how 2-token supervision shapes structural learning, we simplify the Transformer model with several assumptions, similar to the ones in (Wang et al., 2024b). We

## Output HeadsOutput Heads

## Output Heads

## Output Head

## Transfer Layers

Multi-Head Attention

## Feed Forward

## Input Embedding

## Positional  Encoding

## Linear Output Layer

## Output Head

## Input sequence

s t s a b c d e t

## Training sequence

## Input sequence

c Step1s→a→b→c→d→e→t

## Training path

⇒ Linear Transfer layer

## Transfer layer

Output3 Output2 Output3

Figure 1: Multi-Token Prediction (MTP) architectures for 3-step prediction. (a) Meta’s MTP architecture with independent output heads for each step. (b) Ours with a shared output head: i) next-step predictions are generated directly from the backbone, ii) other steps require transforma-tion through a separate transfer layer.

consider a single-layer, single-head Transformer where positional embeddings and layer normaliza-tions are omitted; the feed-forward network is a single linear map FFN(X) = XWM ; and the token embedding matrix Wt and output embedding matrix Wo are identity matrices. The attention mechanism is also simplified, using a standard value projection matrix W V but with a manually set attention matrix α (replacing the standard softmax

( QK⊤ √ dk

) ) that restricts attention to the target

token (i.e., each row is a one-hot vector with a 1 in the second column). Finally, a transfer matrix W T ∈ RM×M maps next-step logits to subsequent-step logits. Complete derivations under these assumptions are provided in Appendix B.

Under this setup, the hidden state at position n, (H1)(n,:), is the sum of feed-forward and attention outputs derived from the one-hot input matrix U . Projecting this state through the output matrix Wo

yields the logits:

(H1)(n,:) Wo = ( UWtW

M + αUWtW V ) (n,:)

Wo = ( UWM + αUW V

= WM (un,:)

+W V (u2,:)

where un is the current token and u2 is the attended target token. Therefore, the logits for the next and the subsequent step are given by

logitn+1(k) = WM (un,k)

+W V (u2,k)

, logitn+2(k) = (WMW T )(un,k) + (W V W T )(u2,k).

For 2-Token Prediction, the objective is ℓ(D) = ℓ(1)(D)+ ℓ(2)(D), and we focus on the second loss ℓ(2)(D), which is formulated as:

ℓ(2)(D) = − ∑ u∈D

U(n+2,k) log exp

( (WMW T )(un,k) + (W V W T )(u2,k)

( (WMW T )(un,ℓ) + (W V W T )(u2,ℓ)

Let P̂i,j(k ′) be the softmax probability of predicting node k′ two steps ahead given current node i

and target j. Define Ni,j,k′ as the number of such occurrences in D and Ni,j = ∑

k′ Ni,j,k′ . This leads to the following theorem.

Theorem 1. For any pair (i, j) in dataset D with Ni,j > 0, let P data i,j (k′) =

Ni,j be the

empirical probability of the second-next node k′. The contribution of this pair to the gradient ∂ℓ(2)(D)

∂WT (d,k′)

is determined by the prediction error, for any d where (WM (i,d) + W V

(j,d)) > 0: (i) If

P̂i,j(k ′) < P data

i,j (k′), the contribution is negative, promoting an increase in the weight W T (d,k′).

(ii) Conversely, if P̂i,j(k ′) > P data

i,j (k′), the contribution is positive, promoting a decrease in the weight. The total gradient is the sum of these contributions over all pairs (i, j) in D.

The derivations and proofs of the theorems are provided in Appendix C.

Transfer Matrix Learned as an Adjacency Matrix. Theorem 1 shows that the transfer matrix W T is updated by 2nd-step prediction errors. When the model underestimates the probability of reaching node k′ from node i in two steps, the gradient increases the weight W T

(d,k′) from all the positive-correlated intermediate node d (e.g., all the feasible d’s for this i, j pair) to k′; otherwise, it decreases. Thus, if the backbone model correctly predicts the next-step node d, then by increasing the weight W T

(d,k′), it will enable W T to correctly learn the adjacency between d and k′.

We next analyze how the 2nd-step prediction affects the backbone parameters WM and W V

through gradients propagated from the transfer matrix. Theorem 2. For any pair (i, j) in dataset D with Ni,j > 0, the contribution of each

(current node i, second-step node k′) pair to the gradient ∂ℓ(2)(D)

∂W V (j,k)

is determined by the prediction

error, for any k where W T (k,k′) > 0: (i) If P̂i,j(k

′) < P data i,j (k′), the contribution is negative, pro-

moting an increase in the weight W V (j,k); (ii) Conversely, if P̂i,j(k

′) > P data i,j (k′), the contribution

is positive, promoting a decrease in the weight. The total gradient is the sum of contributions from all (i, k′) pairs. Analogous results hold for gradients w.r.t. WM .

Learning the Transitive Reachability. The next-token loss ℓ(1)(D) encourages the backbone ma-trix W V to capture the observed reachability from the training data (Wang et al., 2024b). For a given pair (i, k′), when the transfer matrix entry W T

(k,k′) is large (indicating a confident k → k′ transition), and the model predicts a lower probability for k′ than the ground truth along a path i → k → k′, the 2nd-step prediction loss ℓ(2)(D) applies a negative gradient to W V

(j,k) (i.e., increasing its weight), thereby strengthening the k ⇝ j relation. Therefore, when W T

(k,k′) captures the true adjacency relationship between k and k′, the 2nd-step prediction enables the backbone to learn the transitive reachability from k to j, based on the observed reachability from k′ to j is learned by W V

(j,k′) by the 1st-step token prediction, and the adjacency (k, k′) is learned by the transfer layer W T

(k,k′). This shows that 2-token prediction could achieve higher-order reachability beyond the observed reacha-bility of the next-token prediction.

Learning the Adjacency. While the next-token loss ℓ(1)(D) directly encourages WM to capture the adjacency relationship in the dataset (Wang et al., 2024b), the 2nd-step prediction loss ℓ(2)(D) operates indirectly. For a given pair (j, k′), when W T

(k,k′) is large and the model underestimates the probability of the second-step node k′, the loss applies a negative gradient to WM

(i,k), thereby strengthening the i → k connection. This suggests that spurious adjacency (i, k) may be introduced into WM when learning transitive reachability by the 2nd-step prediction, which is mechanically difficult to avoid due to the tight coupling between adjacency and reachability learning in the back-bone. Our empirical validation later (Section 5) demonstrates that this risk is limited and the overall benefit of learning transitive reachability outweighs the risk of spurious adjacency.

Next-Node Prediction. During the next-token prediction inference, the model samples the next node k with high WM

(un,k) + W V

(u2,k) , which favors nodes that are both neighbors of the current

node (high WM ) and reachable from the target (high W V ), and correctly reflects the essence of path planning. Moreover, the transitive reachability learned from ℓ(2) helps the model generate more accurate paths, leading to an improvement on its performance, especially for high-order test cases.

3.2 LEARNING MECHANISM OF MULTI-TOKEN PREDICTION

The total loss of MTP is defined as the sum of cross-entropy losses at each step: ℓ(D) =∑n k=1 ℓ

(k)(D), where each ℓ(k) corresponds to an independent transfer layer. The transfer layer used for generating the outputs of the n-th step token is denoted as WT (n−1).

Theorems 1 and 2 generalize naturally: the n-th transfer layer W T (n−1) takes the next-step logits from the backbone model as input and outputs the n-th step logits, such that W T (n−1) approximates the (n − 1)-th power of the adjacency matrix. Under the influence of W T (n−1), the model can capture the transitive reachability composed of the observed reachability from the n-th step node

s t s a b c d e t

## Current Node

Next 2 Node

## Target Node

2nd-step loss

s t s a b ... ... m ...

## Current Node

## Next n Node

## Target Node

backward adjacency

backward (n-1)-step adjacency

reachability

## Transfer layer

For any node k, k learned adjacency to d:

n-th step loss

reachability

## Transfer layer

For any node k, k learned (n-1)-step adjacency to m:

Figure 2: Illustration of the learning mechanism under Multi-Token Prediction. Left: 2nd-step loss; Right: n-th step loss. MTP learns transitive reachability and spurious adjacency.

m to target t and the (n − 1)-th power adjacency from some node k to m learned under W T (n−1)

(Figure 2). Meanwhile, it may learn some spurious adjacency from the current node b to k.

4 ENHANCING TRANSFER LAYERS FOR MULTI-TOKEN PREDICTION

4.1 NEXT-TOKEN INJECTION

## Output Head

## Transfer Layer

Ouput1 Output2

## Input sequence

Multi-Head Attention

## Feed Forward

## Transfer Layer

Figure 3: Enhanced transfer layer ar-chitecture with NTI and Transformer-based transfer layer.

The transfer layer projects the backbone representation at the next-step position to predict future tokens. Its perfor-mance is constrained by the backbone output hn’s ability to predict the next node, where the deviation between the predicted and true next step introduces noise.

We propose Next-Token Injection (NTI), which aug-ments the transfer input with information from the true next node to provide direct supervision. This is achieved by injecting the embedding vector of the true next token un+1 into the backbone’s hidden state hn, which is then mapped to different positions by separate transfer layers, as follows:

h̃n = hn + k(Wt)(:,un+1), logits2 = Woh̃nW T , (5)

where k is a learnable scalar balancing the internal representation and external supervision.

NTI’s residual shortcut reframes absolute prediction into a simple transformation, analogous to the shortcuts in ResNet (He et al., 2016), thus enabling gradients to bypass unstable backbone states and directly optimize the transfer layer. To analyze this from a gradient perspective, let p̂n+2 = softmax(logits2) denote the predicted probability distribution, and let en+2 be the one-hot vector for the true token un+2. The gradient of the loss with respect to the transfer matrix is then:

∂W T = ( Wo

( hn + k(Wt)(:,un+1)

))⊤ (p̂n+2 − en+2) , (6)

thus preserving the informativeness of supervision even when the predicted next-step hidden state is corrupted by noise, thereby enhancing stability and accuracy in structural modeling.

4.2 TRANSFORMER-BASED TRANSFER LAYER

To overcome the limitations of linear mappings, we replace the linear transfer layer with a Transformer-based transfer layer. The input to this layer is the hidden representation hn ∈ Rd

produced by the backbone at the next-step position. Unlike linear layers that treat each dimension independently, the Transformer leverages self-attention to model dependencies across dimensions, allowing each component of hn to interact and integrate information from all others.

This dynamic interaction significantly enhances the expressiveness of the transfer layer, enabling more precise modeling of multi-hop relations in the underlying graph structure. Consequently, it

Table 1: Path prediction accuracy (%) on degree-0/1/2/3 paths in 100-node DAGs. Metrics include graph-level accuracy (with ± standard error) and path-level accuracy. Results for degree-0/1 are averaged over 50 graphs; degree-2/3 over 200 graphs.

MODEL DEGREE-0 DEGREE-1 DEGREE-2 DEGREE-3 OVERALL

Graph ± / Path Graph ± / Path Graph ± / Path Graph ± / Path Path

1-Token (baseline) (Wang et al., 2024b) 92.58±0.18 / 92.56 86.57±0.24 / 86.60 63.80±0.65 / 64.34 30.76±1.30 / 33.25 89.31

2-Token (Meta’s) (Gloeckle et al., 2024) 92.03±0.19 / 92.00 85.54±0.27 / 85.60 66.37±0.56 / 66.78 36.09±1.33 / 35.55 88.65 2-Token (DeepSeek’s) (Liu et al., 2024) 93.71±0.18 / 93.62 88.82±0.22 / 88.79 67.78±0.58 / 68.48 31.85±1.34 / 34.36 90.90 2-Token + NTI (linear transfer layer) 94.09±0.24 / 94.14 90.00±0.29 / 90.02 69.51±0.60 / 69.56 39.25±1.28 / 42.08 91.74 2-Token + 1-layer Transformer 93.87±0.15 / 93.83 87.84±0.25 / 87.86 68.78±0.55 / 69.51 37.11±1.25 / 39.26 90.68 2-Token + NTI + 1-layer 96.43±0.12 / 96.43 88.92±0.21 / 88.94 71.32±0.51 / 71.52 43.49±1.17 / 44.37 92.65 2-Token + NTI + 3-layer 93.76±0.16 / 93.70 89.47±0.24 / 89.39 71.41±0.55 / 71.60 43.43±1.30 / 44.26 91.29 2-Token + NTI + 6-layer 94.56±0.17 / 94.50 90.09±0.23 / 90.10 72.56±0.47 / 72.97 43.81±1.23 / 45.70 92.07

3-Token (Meta’s) (Gloeckle et al., 2024) 90.76±0.22 / 90.72 83.37±0.29 / 83.34 62.17±0.58 / 62.24 34.63±1.22 / 37.23 86.90 3-Token (DeepSeek’s) (Liu et al., 2024) 94.45±0.16 / 94.37 89.42±0.25 / 89.43 66.21±0.60 / 66.42 30.22±1.29 / 32.27 91.54 3-Token + NTI (linear transfer layer) 92.19±0.15 / 92.17 87.37±0.20 / 87.39 63.38±0.56 / 64.06 42.96±1.36 / 46.28 89.44 3-Token + 1-layer Transformer 92.22±0.15 / 92.16 84.11±0.22 / 84.15 66.79±0.47 / 67.22 40.67±1.24 / 40.43 88.17 3-Token + NTI + 1-layer 92.35±0.16 / 92.31 85.34±0.24 / 85.37 69.61±0.49 / 70.10 44.82±1.31 / 46.10 88.84 3-Token + NTI + 3-layer 93.29±0.13 / 93.25 89.54±0.16 / 89.52 71.97±0.47 / 72.39 45.38±1.18 / 47.25 91.12 3-Token + NTI + 6-layer 93.55±0.14 / 93.52 89.67±0.18 / 89.66 72.82±0.49 / 73.09 45.18±1.24 / 46.99 91.34

achieves more accurate modeling of the transition from one-hop to multi-hop representations in Multi-Token Prediction tasks.

The overall architecture of the enhanced transfer layer, combining Next-Token Injection and Transformer-based transfer layer, is illustrated in Figure 3.

5 EMPIRICAL EVALUATION ON GRAPH PLANNING

5.1 OVERALL ACCURACY OF DIFFERENT MODELS ON PATH PLANNING

We evaluate model performance on randomly generated directed acyclic graphs (DAGs) by measur-ing prediction accuracy on test paths. To analyze performance under varying planning difficulties, test paths are categorized into degree-0/1/2/3 classes according to their reachability in the observa-tion graph Gobs, as defined in Section 2.

Experimental Setup. For each trial, we generate a random DAG with n = 100 nodes, where each potential edge (i, j) for i < j is included independently with probability p = 0.1. For every reachable source–target pair (s, t), we randomly sample m = 20 valid paths. To increase the number of test paths, 10% of (s, t) pairs are used for training and the remaining 90% for testing, while all one-hop edges (s, t) ∈ E are always included in the training set as direct paths “s t s t\n”. All models use 120-dimensional embeddings and adopt a 1-layer, 1-head Transformer as the backbone. “NTI” denotes models with Next-Token Injection (Section 4). The Transformer-based transfer layer uses the same hidden size as the backbone and varies in depth (1, 3, or 6 layers).

Metrics. We evaluate our models using three metrics. Graph-level accuracy is computed by first averaging the path-level accuracy within each graph, and then averaging across all graphs. Standard error is calculated by dividing the standard deviation of graph-level accuracies by the square root of the total number of graphs. Path-level accuracy is the average accuracy over all test paths.

Results. As shown in Table 1, the MTP models achieve notable improvements over the 1-Token Prediction (i.e., Next-Token Prediction) baseline on degree-2/3 test paths. Compared to the MTP baseline, incorporating NTI and Transformer-based transfer layers consistently boosts performance across all degrees. The significant gains in accuracy for degree-2/3 tests in particular demonstrate the effectiveness of MTP in handling transitive reachability.

Effect of Backbone Architecture. We investigate the impact of backbone model complexity, in-cluding Transformer depth and number of attention heads. As shown in Figure 4, increasing back-bone model complexity yields only marginal improvements. In contrast, the configuration combin-ing 2-Token, NTI, and a Transformer-based transfer layer consistently yields stable accuracy gains.

https://lh3.googleusercontent.com/notebooklm/AG60hOqpw1gVQyjXjIpQ92cWTVnkI5wmN0lBsjfZurV5i9PbN4-xqg4rVqFcj4oDQYQqEUqx-j2H6P6L-cPT_ZRPkkPFHYfv-Xgr-L9KbRzdjrq01Dh-j4OYIQ4IFUAf_jNLLHsw3O4k=w48-h768-v0

ea17bacf-5532-43a3-a305-f1ea3883832c

https://lh3.googleusercontent.com/notebooklm/AG60hOokGin3ijumMB-UVMe6TLd11wqN2oJcP2SfbuqzdxDctygnkzkCofrFRqXjWkGKlT6mRHinccoMdE4v0s6Wuv6zlOjH8_B9qREdVQ9ueNSDwwOvDExHAgmNlYUV_WWQKXNKumSidA=w908-h500-v0

341e8615-542d-4844-bbfb-c29164014842

https://lh3.googleusercontent.com/notebooklm/AG60hOqiWZIelmkSVq_63BZ7TjdFg7Y4vvixxkB_Uek0NfXpUVORf_Og0ivXnDkZXwYFnhkDrmouTWxfQ4G-oDZuebOKnfEeUY-_QAcK_f7Cfx9hqU81S7dazwQ9ryqvK_syEQmI-FAHZA=w908-h260-v0

5eb39c51-d356-4359-a20a-a2bdc5e6f147

https://lh3.googleusercontent.com/notebooklm/AG60hOqAkg-dbJSqZQ8Q9jCuD3vOMeKPHd0AkcBa1wgxRibL3xg2fE85_gYldSJRFFzi3_QtDGJwxGm31ZjtD1mZkQVXxBGvS-spU_zDPbSueVP44yXPYtaHy5l44GtJgx8Gd7kpWipaMA=w886-h500-v0

3a216a47-6b91-4b75-9479-0a88d52394b4

https://lh3.googleusercontent.com/notebooklm/AG60hOrWkVKpQEbTrxdIqEDQiyxAkon1656uvqT--KSq57jFHcQ-ecAJAFqN_74WB32RS37inFIqRRUImzXepSr8fB8O-NFg_vIVMnZkDzCJV5A13Ki5q5wEc7G-nHKFjnnUjjoAmNxE=w886-h260-v0

f20afa6b-e22d-470b-a533-0413df959a9c

https://lh3.googleusercontent.com/notebooklm/AG60hOqLNBmfRrMrGcuhlqVIQNHJgX4PdqRK9_Mtf9pb3F8qMz-QSj7n4mG0e_lrIWqK7JptNoIE7h8R7RPNkAMlhw8SKFD3LxT-LsGpaai0HHOgjYIYuz9BUJ72XfRx3zWCN9LBvRL8=w609-h609-v0

008c55b3-1707-4af1-893f-6b44f4983aab

https://lh3.googleusercontent.com/notebooklm/AG60hOoZXuITQWWg6KNKI-clI4Wa-ZSzvcsp47UrXnXwuisTgx77KObU0JptSCPYT_-YpViZnd-mfA9cHd0tMOJUXLgc345ZouBX7hmy-mBzJFZIBOlNvF8ePgpIBqDLBwLHOOBbC9W9Hg=w31-h612-v0

da54a5b0-365b-43f1-91f8-f2968ad6da01

https://lh3.googleusercontent.com/notebooklm/AG60hOpZYGOBTmBM1iWutSr-C5pSyH5mnrvnDcZHYcrZjzzwaTQamrEyKFtCu8kxBVlcfPD7NsEWcpUfrpTOMuZBzIU4xQuPeEU-1fTTF5P7mzbBMEva-rH_IS7E31746Crh8N-vNQ6bsg=w597-h597-v0

45a76212-3f24-4035-be9e-31305859ddb4

https://lh3.googleusercontent.com/notebooklm/AG60hOomURM6YL-iksSyCbUsL_99BVHkoaORz6SdAUqu_87cYfq0f42cW31ULg9BdeOrzsIQd7EH_6jHsZh1EdWffvtplybHUy8NYhwckUSTwVdO7y1ohuLfXIN_1t5045EuZ7pfG2Sybg=w30-h600-v0

db5bd178-f9a8-4cff-8daa-fe0518385dfd

1 head 3 head 5 head 1 layer - 61.86 60.17 62.15

62.69 65.55 64.91 70.69 68.93 68.63 66.57 69.62 68.74 69.82 70.44 70.68

3 layer - 64.79 65.7 65.65 70.88 70.11 69.59 70.85 70.98 71.78 68.72 68.62 68.46 71.83 71.1 71.19

5 layer - 62.33 60.75 60.75 68.53 68.19 67.06 69.83 68.11 68.25 71.14 67.86 69.66 72.0 71.08 71.27 0.60

Models 1-Token 2-Token

2-Token + NTI 2-Token + 1-layer

2-Token + NTI + 1-layer

Figure 4: Degree-2 path graph-level accu-racy (%) for Transformer depth and number of heads, averaged over 10 fixed graphs.

Table 2: Degree-2 path graph-level accuracy (with ± standard error) for node counts and embedding sizes, averaged over 100 graphs.

MODEL 100-NODE 200-NODE 300-NODE 300-NODE 120-DIM 120-DIM 120-DIM 320-DIM

1-Token 63.00±0.89 60.86±0.82 56.16±0.64 55.92±0.64

2-Token 65.99±0.88 62.43±0.60 50.13±0.69 57.47±0.80

2-Token + NTI 69.71±0.85 63.84±0.81 53.79±0.60 65.32±0.71

2-Token + 1-layer 69.02±0.69 61.30±0.66 46.62±0.52 60.51±0.68

2-Token + NTI + 1-layer 71.55±0.73 65.10±0.62 49.79±0.54 65.14±0.76

Effect of Graph Size. We further evaluate scalability on graphs with 200 and 300 nodes, using a fixed embedding size of 120 (shown in Table 2). Results show that 2-Token performance degrades with increasing graph size. On 300-node graphs, increasing the embedding dimension to 320 enables the 2-Token model to outperform the 1-Token baseline. When further increasing the graph size and allocating a matching embedding dimension, the experimental results exhibit the same behavior. It is likely that when the number of nodes exceeds the embedding capacity, supervision signals from different tokens may conflict, limiting the model’s ability to encode structural patterns.

5.2 WEIGHT ANALYSIS OF THE TRAINED MODEL

(a) Without NTI (b) With NTI

Figure 5: Visualization of the projected matrix on the 100-node graph without and with NTI. Red boxes indicate ground-truth adjacency, show-ing only the central submatrix.

In a 100-node path-planning task, we project the transfer layer W T onto the node represen-tations via WtW

TWo to evaluate its ability to learn transition relations, where Wt and Wo

denote the input embedding and output projec-tion matrices, respectively. Figure 5 illustrates the visual projection and Table 3 reports the average weights of the projection under true adjacency and non-adjacency entries. The re-sults clearly show that the transfer layer W T

is learning the true adjacency and with NTI the learning effect is much better.

Table 3: Average weights of adjacency and non-adjacency entries in the pro-jected matrix and their gap.

SETTING ADJ. AVG NON-ADJ. AVG GAP

Without NTI 0.82 -0.01 0.83 With NTI 4.01 -0.05 4.06

0 5 10 15 Target Node

-0.19 0.11 0.34 -0.10 -0.55 -0.26 6.29 -0.68 0.29 -0.09 -1.19 -1.25 5.80 -0.62 -0.27 -0.85 -0.57 0.01 -0.15 0.96

-0.63 -0.31 0.45 -0.69 1.28 0.02 -1.10 -0.33 0.07 4.76 4.02 -0.78 -0.48 -0.36 -0.51 0.36 1.12 -0.99 0.56 -0.25

-1.40 -2.04 1.06 4.40 -0.86 0.15 0.50 -1.96 0.57 0.04 0.84 -1.01 -2.12 0.44 -0.44 -0.13 -0.08 -0.37 1.75 -1.13

-0.28 0.02 -0.20 -0.90 -1.43 4.63 -0.15 0.20 -0.57 -0.31 -0.23 -0.12 -0.77 -0.55 0.30 3.67 -0.66 -0.23 -0.95 -0.20

-0.29 -0.02 0.19 -0.14 -0.87 1.21 -0.27 -0.28 -0.19 -0.20 0.30 -0.04 -0.13 -0.79 4.41 0.07 -0.57 -0.39 -0.68 -0.22

-0.24 0.23 0.25 -0.82 -0.14 -0.77 -0.11 0.62 0.46 0.05 -0.67 3.64 0.51 0.23 -0.19 0.60 0.36 -0.91 -0.70 -1.00

-0.10 0.19 -0.36 -0.50 -0.79 -0.30 -0.40 0.01 0.29 -0.21 0.06 -1.14 0.03 0.42 -0.46 0.09 -0.43 4.65 -1.06 -0.83

0.42 -0.21 0.47 -0.46 -0.11 -0.20 0.57 -0.25 0.20 0.49 -0.01 -0.06 -0.43 -0.17 0.01 0.17 -0.40 -0.30 0.09 0.50

-0.78 0.52 0.95 0.22 -0.47 -0.78 -0.37 -0.35 0.05 0.75 0.02 -0.70 -0.44 4.70 -0.16 0.11 -0.48 0.04 0.63 0.22

0.13 -0.12 -0.49 -0.10 -0.70 0.42 -0.78 0.18 -0.15 0.48 0.32 -0.74 0.53 -0.26 3.96 0.12 3.18 -1.29 -0.39 0.56

-0.36 0.84 0.04 0.03 0.27 0.44 0.07 -0.52 0.37 0.34 -0.34 -0.41 -0.02 -0.58 -0.17 -0.21 -0.22 -0.78 0.38 0.32

0.19 -0.31 0.84 -0.44 0.89 0.69 0.18 -0.15 -0.40 0.08 -0.28 -0.88 -0.11 -0.30 -0.07 0.41 -0.04 -0.09 -1.13 -0.35

-0.11 0.24 0.05 0.41 -0.14 -0.15 -0.19 1.25 0.25 -0.15 -0.54 -0.53 0.39 -0.00 -0.20 -0.66 -0.37 -0.94 0.22 -0.35

-0.18 -0.16 0.29 -0.42 0.52 0.23 -0.12 0.17 -0.15 0.69 -0.47 -0.38 0.58 -0.20 0.10 0.56 3.08 -1.27 0.08 0.55

-0.27 -0.25 -0.07 -0.24 1.71 -0.10 -0.19 -0.19 -0.30 -0.25 -1.10 -0.44 -0.16 0.12 -0.22 -0.32 0.31 -0.70 -0.24 0.26

-0.22 0.04 0.04 -0.39 0.92 -0.22 -0.12 -0.56 0.08 -0.05 -0.47 -0.36 -0.31 0.30 -0.38 -0.06 0.43 -0.56 -0.53 -0.03

-0.21 -0.54 -0.39 0.32 -0.42 -0.32 -0.28 0.61 0.42 -0.49 -0.00 0.05 0.19 -0.77 0.03 1.37 -0.08 0.77 0.17 0.87

0.10 0.11 -0.14 -0.33 -0.00 0.22 0.11 -0.23 -0.22 0.71 0.39 0.01 -0.34 -0.46 -0.10 0.18 -0.34 -0.02 -0.02 0.00

0.05 0.63 0.61 -0.18 -0.13 0.49 0.81 -0.72 0.44 0.61 -0.48 0.01 0.11 -0.42 -0.11 0.57 0.28 -0.75 -0.07 0.17

0.16 0.05 0.36 -0.20 -0.45 0.13 -0.35 -0.31 -0.17 -0.41 0.15 -0.56 -0.51 0.14 0.90 0.64 0.38 -0.20 -0.17 -0.70 2

(a) Learned WM

0 5 10 15 Target Node

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

0.20 0.20 0.19 0.82 0.20 0.19 0.19 0.20 0.19 0.19 0.19 0.20 0.20 0.19 0.19 0.19 0.19 0.20 0.18 0.20

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

0.13 -0.10 0.05 1.15 0.15 1.25 0.12 -0.02 0.11 0.11 0.00 0.14 -0.08 -0.12 0.16 -0.04 0.10 0.12 -0.42 -0.08

0.20 0.20 0.20 0.20 0.20 0.20 0.68 0.20 0.20 0.20 0.20 0.20 -0.02 0.20 0.20 0.20 0.20 0.20 0.20 0.19

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

0.20 0.20 0.19 0.20 0.18 0.19 0.20 0.20 0.19 0.79 0.09 0.20 0.20 0.19 0.20 0.19 0.19 0.20 0.19 0.20

0.20 0.20 0.19 0.20 0.18 0.19 0.20 0.20 0.19 -0.08 1.00 0.20 0.20 0.19 0.20 0.19 0.19 0.20 0.19 0.20

0.06 -0.13 -0.12 1.03 -0.05 1.07 -0.06 -0.05 0.04 -0.25 -0.06 1.38 -0.11 -0.31 -0.11 -0.17 -0.16 0.06 -0.53 -0.21

0.20 0.20 0.20 0.20 0.20 0.20 -0.16 0.20 0.20 0.20 0.20 0.20 0.84 0.20 0.20 0.20 0.20 0.20 0.20 0.19

0.20 0.19 0.19 0.19 0.20 0.20 0.20 0.19 0.19 0.19 0.19 0.20 0.19 0.83 0.19 0.19 0.19 0.19 0.19 0.19

0.14 0.05 0.01 0.13 1.05 -0.01 0.10 0.15 0.13 1.06 -0.07 0.10 0.14 -0.21 1.88 0.06 -0.17 0.15 0.03 0.12

0.12 -0.11 0.07 1.18 0.14 -0.24 0.13 -0.02 0.07 0.12 -0.05 0.14 -0.09 -0.12 0.15 1.56 0.11 0.11 -0.43 -0.07

0.16 0.85 0.04 0.15 0.06 0.04 0.15 0.16 0.15 0.45 -0.01 0.15 0.15 0.53 -0.01 0.13 2.30 0.16 0.05 0.15

0.12 0.04 0.01 0.01 0.07 -0.36 1.16 0.16 0.04 -0.45 0.13 0.15 -0.40 0.17 0.18 -0.06 0.01 1.53 -0.02 -0.46

0.20 0.20 0.18 -0.07 0.20 0.19 0.19 0.20 0.19 0.19 0.19 0.20 0.20 0.19 0.19 0.19 0.19 0.19 1.12 0.20

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.5

(b) Learned W V

Figure 6: Weight analysis on the first 20 nodes of the 100-node graph using a simplified model with fixed transfer layer W T . Red boxes indicate true adjacency and reachability, light dashed boxes show observed reach-ability, and white triangles mark theoretically learnable entries under ℓ(2)(D).

In the simplified 100-node model, with the transfer layer W T fixed to the ground-truth adjacency matrix, we vi-sualize the first 20 nodes of the learned weight matrices WM and W V , as shown in Figures 6a and 6b. In WM , this region fully corresponds to relations that are theoretically learn-able under ℓ(2)(D), with some weights slightly amplified, though still far less pronounced than those reinforced by ℓ(1)(D). In W V , a subset of relations is captured by ℓ(2)(D), with their weights clearly standing out above the background (Table 4).

The 2nd-step prediction encourages WM to assign small positive weights to certain spurious adja-cency relations, introducing mild noise. In contrast, W V significantly strengthens correct but unob-

served reachability relations, with weights clearly standing out from the background nonreachable entities. These enhanced reachability signals are concentrated near the diagonal in key positions, making them substantially more influential during the next-step prediction process. This indicates that the 2nd-step prediction indeed enables the backbone model to learn transitive reachability be-yond those learned by the next-token prediction. Further visualizations and analyses of weights for the general 2-Token Prediction can be found in Appendix D.

5.3 EVALUATION ON THE BLOCKSWORLD PLANNING TASK

Table 4: Average weights of different entry types in WM and W V in the 100-node graph with a fixed transfer layer W T in the simplified model.

## MATRIX TYPE VALUE

True adjacency 1.90 WM Theoretically learnable 0.16

Other entries -0.01

Observed reachability 0.63 W V Theoretically learnable 0.34

Other entries -0.01

To further assess the practicality and general-ization of our methods, we evaluate their per-formance on the classical Blocksworld plan-ning benchmark. This task provides a struc-tured, fixed-graph environment to test the mod-els’ ability in finding valid action sequences. Full details of the experimental setup, including the graph representation and dataset construc-tion methodology, are provided in Appendix E.

Results. As shown in Table 5, our proposed enhancements to multi-token prediction consis-tently outperform the 1-token baseline and standard MTP architectures across various training data sizes. Notably, the combination of NTI and a multi-layer Transformer transfer layer (e.g., 2-Token + NTI + 6-layer) achieves the highest accuracy, demonstrating the effectiveness of our approach in a complex, classical planning environment.

Table 5: Path prediction accuracy (%) on Blocksworld under varying training set sizes.

TRAIN SIZE (NUMBER OF PATHS PER LENGTH) 100 200 300 400 500

1-Token (baseline) 45.62 62.66 70.21 74.86 77.94

2-Token (Meta’s) (Gloeckle et al., 2024) 42.42 60.40 68.19 72.31 76.27 2-Token (DeepSeek’s) (Liu et al., 2024) 51.32 66.60 74.01 78.41 80.23 2-Token + NTI (linear transfer layer) 44.91 62.46 71.36 74.85 80.03 2-Token + 1-layer Transformer 51.25 66.51 75.84 77.11 79.32 2-Token + NTI + 1-layer 52.51 67.37 75.92 79.92 81.55 2-Token + NTI + 3-layer 52.01 66.73 74.86 79.56 83.30 2-Token + NTI + 6-layer 52.84 68.57 73.74 78.97 85.70

3-Token (Meta’s) (Gloeckle et al., 2024) 40.99 56.41 64.67 69.38 76.43 3-Token (DeepSeek’s) (Liu et al., 2024) 50.13 65.30 71.52 76.87 80.87 3-Token + NTI (linear transfer layer) 42.91 62.46 68.47 73.82 78.29 3-Token + 1-layer Transformer 49.98 64.95 72.63 76.83 78.37 3-Token + NTI + 1-layer 50.47 67.13 72.88 76.45 81.48 3-Token + NTI + 3-layer 49.84 66.57 71.70 77.47 80.40 3-Token + NTI + 6-layer 50.10 67.11 72.69 77.27 82.18

6 CONCLUSION AND FUTURE WORK

The paper provides an in-depth analysis of how Multi-Token Prediction enables the autoregressive Transformer to learn transitive relation in graph path planning. Based on this, we propose two enhancement strategies: Next-Token Injection (NTI) and a Transformer-based Transfer Layer. Ex-periments on synthetic graphs and the Blocksworld planning task demonstrate that these methods significantly improve the model’s accuracy and stability in transitive planning tasks.

Our work opens several promising avenues for future research. One is to bridge the gap to real-world applications by effectively abstracting continuous and ambiguous tasks into discrete state represen-tations for multi-step prediction. Methodologically, our framework can be enhanced by extending the NTI mechanism to generate guiding signals in unsupervised settings, or by combining the trans-fer layer with explicit planning modules like chain-of-thought and backtracking search. Integrating these architectural improvements within a reinforcement learning paradigm presents another promis-ing path toward creating more general and capable planning agents.

## REFERENCES

Zeyuan Allen-Zhu and Yuanzhi Li. Physics of Language Models: Part 1, Learning Hierarchical Language Structures. SSRN Electronic Journal, May 2023.

Zachary Ankner, Rishab Parthasarathy, Aniruddha Nrusimha, Christopher Rinard, Jonathan Ragan-Kelley, and William Brandon. Hydra: Sequentially-dependent draft heads for medusa decoding. In First Conference on Language Modeling, 2024.

Gregor Bachmann and Vaishnavh Nagarajan. The pitfalls of next-token prediction. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org, 2024.

Mikita Balesni, Tomasz Korbak, and Owain Evans. The two-hop curse: LLMs trained on A->B, B->C fail to learn A->C. arXiv preprint arXiv:2411.16353, 2024.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in Neural Information Processing Systems, 33:1877–1901, 2020.

Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, and Tri Dao. Medusa: Simple llm inference acceleration framework with multiple decoding heads. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org, 2024.

Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In European Conference on Computer Vision, pp. 213–229. Springer, 2020.

Ziwei Chai, Tianjie Zhang, Liang Wu, Kaiqiao Han, Xiaohai Hu, Xuanwen Huang, and Yang Yang. Graphllm: Boosting graph reasoning ability of large language model. arXiv preprint arXiv:2310.05845, 2023.

Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Misha Laskin, Pieter Abbeel, Aravind Srinivas, and Igor Mordatch. Decision transformer: Reinforcement learning via sequence modeling. Advances in Neural Information Processing Systems, 34:15084–15097, 2021a.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021b.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pp. 4171–4186, 2019.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszko-reit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.

Guhao Feng, Bohang Zhang, Yuntian Gu, Haotian Ye, Di He, and Liwei Wang. Towards revealing the mystery behind chain of thought: a theoretical perspective. Advances in Neural Information Processing Systems, 36:70757–70798, 2023.

Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Roziere, David Lopez-Paz, and Gabriel Synnaeve. Better & faster large language models via multi-token prediction. In Forty-first International Conference on Machine Learning, 2024.

Xavier Glorot, Antoine Bordes, and Yoshua Bengio. Deep sparse rectifier neural networks. In Proceedings of the fourteenth international conference on artificial intelligence and statistics, pp. 315–323. JMLR Workshop and Conference Proceedings, 2011.

Jiayan Guo, Lun Du, and Hengyu Liu. Gpt4graph: Can large language models understand graph structured data ? an empirical evaluation and benchmarking. CoRR, abs/2305.15066, 2023.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recog-nition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.

Michael Janner, Qiyang Li, and Sergey Levine. Offline reinforcement learning as one big sequence modeling problem. Advances in Neural Information Processing Systems, 34:1273–1286, 2021.

Lucas Lehnert, Sainbayar Sukhbaatar, Paul McVay, Michael Rabbat, and Yuandong Tian. Beyond a*: Better LLM planning via search dynamics bootstrapping. In ICLR 2024 Workshop on Large Language Model (LLM) Agents, 2024.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. CoRR, abs/2412.19437, 2024.

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 10012–10022, 2021.

Zihan Luo, Xiran Song, Hong Huang, Jianxun Lian, Chenhao Zhang, Jinqi Jiang, Xing Xie, and Hai Jin. Graphinstruct: Empowering large language models with graph understanding and reasoning capability. CoRR, abs/2403.04483, 2024.

William Merrill and Ashish Sabharwal. The parallelism tradeoff: Limitations of log-precision trans-formers. Transactions of the Association for Computational Linguistics, 11:531–545, 2023.

Ida Momennejad, Hosein Hasanbeig, Felipe Vieira Frujeri, Hiteshi Sharma, Nebojsa Jojic, Hamid Palangi, Robert Ness, and Jonathan Larson. Evaluating cognitive maps and planning in large language models with cogeval. Advances in Neural Information Processing Systems, 36:69736– 69751, 2023.

Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. Codegen: An open large language model for code with multi-turn program synthesis. In The Eleventh International Conference on Learning Representations, 2023.

Emilio Parisotto, Francis Song, Jack Rae, Razvan Pascanu, Caglar Gulcehre, Siddhant Jayakumar, Max Jaderberg, Raphael Lopez Kaufman, Aidan Clark, Seb Noury, et al. Stabilizing transformers for reinforcement learning. In International Conference on Machine Learning, pp. 7487–7498. PMLR, 2020.

Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language under-standing by generative pre-training. 2018.

Mohammad Samragh, Arnav Kundu, David Harrison, Kumari Nishu, Devang Naik, Minsik Cho, and Mehrdad Farajtabar. Your llm knows the future: Uncovering its multi-token prediction potential. arXiv preprint arXiv:2507.11851, 2025.

Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. Hugging-gpt: Solving ai tasks with chatgpt and its friends in hugging face. Advances in Neural Information Processing Systems, 36:38154–38180, 2023.

Jiabin Tang, Yuhao Yang, Wei Wei, Lei Shi, Lixin Su, Suqi Cheng, Dawei Yin, and Chao Huang. Graphgpt: Graph instruction tuning for large language models. In Proceedings of the 47th In-ternational ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 491–500, 2024.

Karthik Valmeekam, Matthew Marquez, Sarath Sreedharan, and Subbarao Kambhampati. On the planning abilities of large language models-a critical investigation. Advances in Neural Informa-tion Processing Systems, 36:75993–76005, 2023.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in Neural Informa-tion Processing Systems, 30, 2017.

Heng Wang, Shangbin Feng, Tianxing He, Zhaoxuan Tan, Xiaochuang Han, and Yulia Tsvetkov. Can language models solve graph problems in natural language? Advances in Neural Information Processing Systems, 36:30840–30861, 2023.

Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6):186345, 2024a.

Siwei Wang, Yifei Shen, Shi Feng, Haoran Sun, Shang-Hua Teng, and Wei Chen. Alpine: Unveil-ing the planning capability of autoregressive learning in language models. Advances in Neural Information Processing Systems, 37:119662–119688, 2024b.

Yuhao Wang, Heyang Liu, Ziyang Cheng, Ronghua Wu, Qunshan Gu, Yanfeng Wang, and Yu Wang. Vocalnet: Speech llm with multi-token prediction for faster and high-quality generation. CoRR, abs/2504.04060, April 2025.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik R Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.

Siyu Yuan, Jiangjie Chen, Ziquan Fu, Xuyang Ge, Soham Shah, Charles Jankowski, Yanghua Xiao, and Deqing Yang. Distilling script knowledge from large language models for constrained lan-guage planning. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 4303–4325, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.236.

Eric Zelikman, Qian Huang, Gabriel Poesia, Noah D. Goodman, and Nick Haber. Parsel: algorith-mic reasoning with language models by composing decompositions. In Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA, 2023. Curran Associates Inc.

Ce Zhang, Yale Song, Ruta Desai, Michael Louis Iuzzolino, Joseph Tighe, Gedas Bertasius, and Satwik Kottur. Enhancing visual planning with auxiliary tasks and multi-token prediction. arXiv preprint arXiv:2507.15130, 2025.

Dylan Zhang, Curt Tigges, Zory Zhang, Stella Biderman, Maxim Raginsky, and Talia Ringer. Transformer-based models are not yet perfect at learning to emulate structural recursion. Trans-actions on Machine Learning Research, 2024. ISSN 2835-8856.

## A RELATED WORK

A.1 LLMS FOR STRUCTURED PLANNING AND REASONING

Recent studies have examined the capacity of large language models (LLMs) to perform structured planning and reasoning. In planning domains, benchmarks such as CogEval (Momennejad et al., 2023) and Blocksworld (Valmeekam et al., 2023) highlight significant challenges: while humans achieve over 70% success in Blocksworld, GPT-3 attains only 5%, suggesting that LLMs struggle to capture underlying task structures. Nevertheless, LLMs exhibit promising behaviors in multi-step decision-making for autonomous agents (Wang et al., 2024a), which can often be abstracted as path-finding over graphs. For example, HuggingGPT (Shen et al., 2023) coordinates external APIs through dependency relations, naturally forming a graph-based planning problem. Our work adopts this abstraction but focuses on consistency and interpretability in multi-step prediction.

Beyond planning, researchers have explored the graph reasoning abilities of LLMs. Frameworks such as GPT4Graph (Guo et al., 2023) and NLGraph (Wang et al., 2023) show that while LLMs can process graph-structured inputs, performance remains fragile and sensitive to spurious correlations, with GPT-4 reaching only around 50% accuracy on shortest-path tasks. To improve reasoning, recent efforts augment LLMs with external modules such as GNN encoders (Chai et al., 2023; Tang et al., 2024) or explicitly train them to imitate classical algorithms like BFS and DFS (Luo et al., 2024). Other work focuses on extracting structured task knowledge, e.g., distilling temporal and causal relations into compact representations (Yuan et al., 2023). While these approaches improve empirical performance, they provide limited insights into why LLMs fail on more complex planning scenarios.

A complementary line of research examines the algorithmic foundations of LLM reasoning. Trans-formers have been shown to belong to the TC0 complexity class (Merrill & Sabharwal, 2023), but techniques like chain-of-thought prompting (Feng et al., 2023) and Tree of Thoughts search (Yao et al., 2023) allow them to simulate more complex procedures sequentially. Parsel (Zelikman et al., 2023), for instance, decomposes reasoning into structured subroutines, closely related to the multi-step prediction framework considered here. However, these studies largely overlook how the au-toregressive training paradigm itself may impose fundamental limitations on consistent multi-step planning—a gap that our theoretical analysis seeks to address.

A.2 MULTI-TOKEN PREDICTION (MTP)

Traditional language models are trained with autoregressive next-token prediction (NTP), which inherently suffers from slow inference and discrepancies between training and inference (Allen-Zhu & Li, 2023). Moreover, NTP tends to overfit local transitions and struggles to capture long-range dependencies (Bachmann & Nagarajan, 2024). Multi-Token Prediction (MTP) has recently emerged as an alternative paradigm. Instead of predicting a single token at each step, MTP predicts multiple future tokens in parallel, often through several independent output heads attached to a shared Transformer backbone (Gloeckle et al., 2024; Cai et al., 2024).

MTP offers two primary advantages. First, it enables self-speculative decoding, substantially ac-celerating inference. Approaches like Medusa (Cai et al., 2024) and Hydra (Ankner et al., 2024) verify multiple candidate tokens simultaneously, achieving up to 3× speedup in low-batch scenar-ios. Second, by jointly predicting multiple steps, MTP encourages the model to capture longer-term dependencies and more global structures (Gloeckle et al., 2024; Samragh et al., 2025), in contrast to NTP’s purely local supervision.

These benefits translate to strong empirical performance. In code generation, a 13B MTP-trained model outperforms its autoregressive counterpart by 12–17% on HumanEval and MBPP (Gloeckle et al., 2024). Beyond code, MTP has been applied to speech modeling and visual planning, demon-strating both substantial speedups and improved multi-step reasoning (Wang et al., 2025; Zhang et al., 2025).

From a theoretical perspective, ALPINE shows that autoregressive models face fundamental barriers in transitive planning problems (Wang et al., 2024b). By directly supervising multiple future tokens, MTP can mitigate exposure bias and better capture reachability structures essential for planning. In this work, we adopt MTP for training in graph-based path-finding tasks to explore its potential in

overcoming the core limitations of NTP, while inference is still performed via NTP, as speedup is not our focus and we aim to study how multi-token supervision enhances learning.

## B SIMPLIFIED TRANSFORMER SETUP

To theoretically analyze how the 2-Token Prediction objective helps the model learn structural infor-mation, we follow the analysis framework of (Wang et al., 2024b), starting from a standard Trans-former model and gradually simplifying it into an analytically tractable form. We begin by reviewing the standard Transformer computations and then describe the simplifications introduced.

In the original model, the input tokens x1, x2, . . . , xN are mapped into embedding vectors X ∈ RN×d, which are then processed through a stack of Transformer blocks. Each block consists of a multi-head self-attention (MHA) module and a feed-forward network (FFN). The attention mecha-nism is defined as:

Attention(Q,K,V ) = softmax

( QK⊤ √ dk

where Q,K,V ∈ RN×dk are the query, key, and value matrices respectively, dk = d/H is the per-head embedding dimension, and H is the number of attention heads. For input X ∈ RN×d, the i-th head computes: Qi = XWQ

i , Ki = XWK i , Vi = XW V

i , with learnable parame-ters WQ

i ,WK i ,W V

i ∈ Rd×dk . The multi-head attention output is the concatenation of all heads: MHA(X) = ConcatHi=1 (Attention(Qi,Ki,Vi)) .

The feed-forward network is a two-layer MLP:

FFN(X) = max(0,XW1 + 1b⊤1 )W2 + 1b⊤2 , (8)

where W1 ∈ Rd×4d, W2 ∈ R4d×d, b1 ∈ R4d, b2 ∈ Rd, 1 ∈ RN×1 is the all-one vector for broadcasting biases, and 0 ∈ RN×4d is the zero matrix used in the ReLU activation (Glorot et al., 2011).

Each Transformer block applies residual connections and layer normalizations:

Transformer(X) = FFN(LN2(MHA(LN1(X)) +X)) +MHA(LN1(X)) +X. (9)

Backbone (Transformer)

## Output Head

⇒ Backbone

(a) Simplified 1-Token Prediction model

Backbone (Transformer)

## Output Head

## Transfer Layer

Ouput1 Output2

⇒ Backbone

Transfer Layer (Wᵀ)

Output1 Output2

(b) Simplified 2-Token Prediction model

Figure 7: Illustration for the simplified model architectures.

To enable tractable visualization and analysis, we structurally simplify the architecture. First, we retain only a single-layer, single-head self-attention module. The attention matrix softmax

( QK⊤ √ dk

) is manually set to a one-hot matrix in which the second column is filled with ones and all other entries are zero, i.e.,

( QK⊤ √ dk

 0 1 0 · · · 0 0 1 0 · · · 0 ...

0 1 0 · · · 0

 ∈ Rn×n. (10)

This setting simulates the attention matrix learned by the model after training on the task, where each token attends exclusively to the target node (i.e., the second token in the sequence). In the

https://lh3.googleusercontent.com/notebooklm/AG60hOp1pFHHiT0ygJACEUqC9meHrMbJ0DV8vGVTd-OF3_NLxi1UuVJKOdd_Vi6789lWZYage-EXQ7BtlObfxJ5Y-Ol83QdVKyhYoahaI0fs3ueAIA95cIw0kP7JpdsgLi3D1DWyaL7_Sg=w1764-h500-v0

285851dd-93dd-4484-a16b-b06002b8c5d1

https://lh3.googleusercontent.com/notebooklm/AG60hOpzQkV_Zm8g2PrffPuz1jIacdkRL9-bjzC9rD8DRHfqfHD9_LUJp_lmzx294JU37Rq8ROcG5PkykynPd8ZcDWpgayRujaJPCJexkh-MR3vWZvish5MRnkZ1VoUPiQvKhajf9esX=w1764-h500-v0

e5de0444-1bed-4f28-80be-dba3bcf449c7

https://lh3.googleusercontent.com/notebooklm/AG60hOoM1WOW8XvgSXlDG2kyCDo75Lts1dJ96jBK4SlPCr1jza8j9tZWbnP33R6UvMSbV33LnH5nCKJblBCMMzmiK49wTZEqW5WE_22F_3Q1e6IiNzf2hIqfi3RpaHkmJH2PafsUPNQ3=w1764-h500-v0

51489ad8-18d8-4253-acf5-94f33349cd64

https://lh3.googleusercontent.com/notebooklm/AG60hOr1sHpbZWwcvp_Bv7ZTWImZkCO1n-W_p6Y8VevKfhBWSHrvqwBpSXMa45_xaLSyhuFb-3Lk4V9cuonlhf4I3cNOYcxVNzMDKjdvgy9W_UvlAbwlQFCuWgSliuC1N59fK_KT2jE9Rg=w1764-h39-v0

5527ba9c-6b29-47c0-b33e-f0950b3dc9c5

https://lh3.googleusercontent.com/notebooklm/AG60hOpGP3i6vz2bpYK5evZNm1ZxRSjxakSryVjxJUW5HJ3UoM4LXGHxO1KxZ80tU4vfK3wP1JRuAdHFrNH-UvpenvMGJayt42TI8hl0FYd-BAO05DoKdZtXehW9f6zh6YugVyTeyP3RPw=w1770-h500-v0

5b31e8e6-b4d0-4b7e-9c29-7d5e6a027ffd

https://lh3.googleusercontent.com/notebooklm/AG60hOpz9CwrFWhkWnZBDXrFUWmwMUOIfIzzxwMzeq4P6pRBeJL-sLoia_MTM_XMtroa6_v7nj8ucqm0sP6oQtzssFL5nf75g5-Ua7pejZiD3nMHrkvmfrXzIyWxVcSQMyg-My0B00G_=w1770-h500-v0

bdab2bc6-bd5b-4c3f-9be1-18f610d1fb89

https://lh3.googleusercontent.com/notebooklm/AG60hOoyGp4juX6ORGS1HW3urz-7MiNHCGi1ppnLiJilqhW4Ba-zxbEHiG1Tx2YfEfXO8KPz3aPkO-srsUisU9rpWalw9w9wCABL6q3YFjHx7ZX58Ix8C5-vhSjKM-oh6-DPEhWxLtP_iw=w1770-h500-v0

687a3ba8-6c87-4446-ac94-3aef5071b869

https://lh3.googleusercontent.com/notebooklm/AG60hOoC-YksG_5wDXbAzpf5EndTILF_vDowvK7oMs9AdW8svgt-Qb_7RjqR28P7zNaOo3Pa_AGXgR6rI-bsN-trdL5Z0zFGFV71AtN16ngKOHlFklKy0RMoxCG44gCwlhqKweQgeB9vZA=w1770-h48-v0

e014a238-6b07-4347-ac3c-e854a9c6924b

(a) 1-Token Prediction model (b) 2-Token Prediction model

Figure 8: Visualization of attention matrices for (a) 1-Token Prediction and (b) 2-Token Prediction Transformer models.

actual path-planning task with 100 nodes, the relevant results are shown in Figure 8. These results are obtained by analyzing the attention mechanism of a single-layer single-head Transformer model, presenting the averaged attention matrix computed over the test dataset. Each row n of the matrix corresponds to the attention distribution vector when predicting next token.

We remove all positional encodings by setting Wp = 0, eliminate all layer normalization operations, and replace the two-layer FFN with a single linear transformation:

FFN(X) = XWM , (11)

the forward propagation becomes an additive form. The resulting Transformer block is:

Transformer(X) = FFN(X) +MHA(X). (12)

To further reduce complexity, we set both the token embedding matrix Wt and the output projection matrix Wo to identity matrices, and assume that the embedding dimension equals the vocabulary size, i.e., d = M . This allows direct interpretation of logits in the vocabulary space.

Finally, in the 2-Token Prediction setting, we introduce a transfer matrix W T ∈ RM×M after the logits layer. This transfer layer is used to map the predicted next-step logits to the logits for the following step. The two resulting model variants are shown in Figure 7.

## C DERIVATIONS AND PROOFS

Let Ni,j,k′ denote the number of times in D that the following conditions are satisfied: (a) the current node is i; (b) the attention target is j; and (c) the token two steps ahead is k′. Let Ni,j =

∑ k′ Ni,j,k′

denote the total count of such (i, j) pairs.

This leads to the following theorem:

Theorem 1. For any pair (i, j) in dataset D with Ni,j > 0, let P data i,j (k′) =

Ni,j be the

empirical probability of the second-next node k′. The contribution of this pair to the gradient ∂ℓ(2)(D)

∂WT (d,k′)

is determined by the prediction error, for any d where (WM (i,d) + W V

(j,d)) > 0: (i) If

P̂i,j(k ′) < P data

i,j (k′), the contribution is negative, promoting an increase in the weight W T (d,k′).

(ii) Conversely, if P̂i,j(k ′) > P data

i,j (k′), the contribution is positive, promoting a decrease in the weight. The total gradient is the sum of these contributions over all pairs (i, j) in D.

Theorem 2.

For any pair (i, j) in dataset D with Ni,j > 0, the contribution of each

(current node i, second-step node k′) pair to the gradient ∂ℓ(2)(D)

is determined by the pre-

diction error, for any k where W T (k,k′) > 0: (i) If P̂i,j(k

′) < P data i,j (k′), the contribution is

negative, promoting an increase in the weight W V (j,k); (ii) Conversely, if P̂i,j(k

′) > P data i,j (k′),

the contribution is positive, promoting a decrease in the weight. The total gradient is the sum of contributions from all (i, k′) pairs.

For any pair (i, j) in dataset D with Ni,j > 0, the contribution of each

(target node j, second-step node k′) pair to the gradient ∂ℓ(2)(D)

is determined by the predic-

tion error, for any k where W T (k,k′) > 0: (i) If P̂i,j(k

′) < P data i,j (k′), the contribution is negative,

promoting an increase in the weight WM (i,k); (ii) Conversely, if P̂i,j(k

′) > P data i,j (k′), the contribu-

tion is positive, promoting a decrease in the weight. The total gradient is the sum of contributions from all (j, k′) pairs.

According to the definition of cross-entropy loss and the predicted weight vectors in our simplified model, the total cross-entropy loss (involving matrices WM , W V , and W T ) is given by

ℓ(2)(D) = − ∑ u∈D

U(n+2,k) log exp

+ ( W V W T

ℓ exp ( (WMW T )(un,ℓ)

+ (W V W T )(u2,ℓ)

) Let A(i,k) =

, B(j,k) = ( W V W T

ℓ(2)(D) = − ∑ u∈D

I[un = i, u2 = j] log exp(A(i,k) +B(j,k))∑ ℓ exp(A(i,ℓ) +B(j,ℓ))

= − ∑ i,j,k′

Ni,j,k′ log exp(A(i,k′) +B(j,k′))∑ ℓ exp(A(i,ℓ) +B(j,ℓ))

= − ∑ i,j,k′

Ni,j,k′(A(i,k′) +B(j,k′)) + ∑ i,j

exp(A(i,ℓ) +B(j,ℓ))

= − ∑ i,j,k′

+ ( W V W T

WMW T ) (i,ℓ)

+ ( W V W T

We define P̂i,j(k ′) as the softmax probability—under current model parameters—that, given current

node i and target node j, the model predicts node k′ as the token at step n+ 2:

P̂i,j(k ′) =

exp ( A(i,k′) +B(j,k′)

( A(i,ℓ) +B(j,ℓ)

) . Then we have that the total gradient is the sum of contributions from all pairs (i, j):

∂W T (d,k′)

[( P̂i,j(k

′)− P data i,j (k′)

(i,d) +W V (j,d)

Contribution from pair (i, j)

The proof of Theorem 1 follows by analyzing the sign of each term—or contribution—in this sum-mation. For any specific pair (i, j) with Ni,j > 0, the sign of its contribution is determined by the prediction error.

If P̂i,j(k ′) < P data

i,j (k′), the first factor in the term, ( P̂i,j(k

′)− P data i,j (k′)

) , is negative. Given the

theorem’s conditions that Ni,j > 0 and (WM (i,d) + W V

(j,d)) > 0, the entire term corresponding to

this (i, j) pair is therefore negative, which proves part (i) of the theorem. Conversely, if P̂i,j(k ′) >

P data i,j (k′), the first factor is positive, making the entire term for this (i, j) pair positive. This proves

part (ii) of the theorem.

The total gradient is the aggregation of all such positive and negative contributions from every pair (i, j) in the dataset D, as stated in the final sentence of the theorem. This concludes the proof.

https://lh3.googleusercontent.com/notebooklm/AG60hOrKX-me_poFXxBMhUtKHG9J82hNR1TKRuvV1VySyuAC6sUqSdQGyAGhOpJSZL0dBsQ8yVho0Ni1HOUtmVmfhQcWli2g4XiUz6m1RDNuaE6CMQAlUN5uo554nfeK-EZoNgNrhmvS=w597-h597-v0

cb5eddc1-aabc-4bce-bbc4-dd70bda2cbb1

https://lh3.googleusercontent.com/notebooklm/AG60hOqUEDJ2lVdezVQv724wAAgX7s_UCDIXnzDWmLESR7nMO-onN5qnK7ZPiE_Ra4yHatT-QSgqATfzqLyUNRpGACzQ7De69t0SVRJQxwlmSPGx2IZvhRTQRdUkbtUbEmNAC2LJ-jrjjw=w30-h600-v0

dda94ba7-8919-4ea0-8417-01539cbafe10

https://lh3.googleusercontent.com/notebooklm/AG60hOo5gsJQUcMDTlRIAzypWCD-JS8cWk8EVWTFmTO9bRLOdmubC0mshsxV7uyl7vJSrpywAbsvQFk9e579bRUM4upL7S4aFDJDz22WZUEOW15AEfLkyN9yTF-qili9WjC3gpKpIO0Ixg=w609-h609-v0

4602ba25-1cc8-4c70-a2d2-b3aabf5c8694

https://lh3.googleusercontent.com/notebooklm/AG60hOo9V4b9gQBIIZ9yx5hGfyz057szcnTOOAxk4L7rPyfAat2pwi3e8SKfA8bflwhiFQYmfVq1CtPjHxH01wItqsQmTspFHlqojCjQCNKvo-KPSwn9a-KBnG5c1xOeMTPBqkDaIwSEew=w31-h613-v0

de71d3df-8fd7-436a-92e2-464b3aabe979

Similarly, the gradient with respect to WM is derived as the sum of contributions from all applicable pairs (j, k′):

[( P̂i,j(k

′)− P data i,j (k′)

) ·Ni,j ·W T

Contribution from pair (j, k′)

The proof of Theorem 2 for the backbone parameter WM follows from analyzing the sign of each term in this summation. For any term in this sum that corresponds to a context (i, j) with Ni,j > 0, its sign is determined by the prediction error for that context.

If P̂i,j(k ′) < P data

i,j (k′), the initial factor ( P̂i,j(k

′)− P data i,j (k′)

) is negative. Given the theorem’s

conditions that Ni,j > 0 and W T (k,k′) > 0, the entire term corresponding to this (j, k′) pair is

therefore negative, which proves part (i) of the theorem regarding the contribution to the gradient of WM

(i,k). Conversely, if P̂i,j(k ′) > P data

i,j (k′), the initial factor is positive, making the entire term for this (j, k′) pair positive. This proves part (ii) of the theorem.

The total gradient for WM (i,k) is the sum of all such positive and negative contributions over all

applicable pairs (j, k′). As stated in the theorem, analogous reasoning holds for gradients w.r.t. W V . This concludes the proof.

## D FURTHER WEIGHT ANALYSIS OF SIMPLIFIED MODEL

Fixing the transfer layer W T to the ground-truth adjacency, we train the simplified model on a 20-node graph.

Figure 9 shows the learned weight matrices after training with 1-Token Prediction. As can be seen, the model learns the adjacency relations in WM and the observed reachability relations in W V . Figure 10 shows the learning results of 2-Token Prediction, where the matrices contain entries that are theoretically learned under the ℓ(2)(D) constraint. These entries have weights slightly higher than the background, but remain significantly weaker than those reinforced by ℓ(1)(D). Ultimately, spurious adjacency relations are observed in WM , while W V partially captures unobserved reach-ability relations.

0 5 10 15 Target Node

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

0.05 0.05 0.05 0.05 0.05 3.42 0.07 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05

-0.56 -0.56 -0.56 -0.56 -0.56 3.72 9.87 -1.40 -0.56 -0.56 -0.56 3.14 -0.56 -1.30 -0.56 -5.31 -0.56 -0.56 4.13 -0.56

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

-0.18 -0.18 -0.18 -0.18 -0.18 -0.18 -0.18 -0.18 -0.18 -0.18 -0.18 -0.18 -0.18 -1.76 -0.12 -0.17 9.21 -0.18 -0.15 0.27

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

-0.17 -0.17 -0.17 -0.17 -0.17 -0.16 -0.48 13.36 -0.17 -0.17 -0.17 -2.10 -0.17 -0.17 -0.17 -3.37 -0.17 -0.17 -0.16 -0.17

-0.22 -0.22 -0.22 -0.22 -0.22 -0.21 -0.85 -0.78 -0.22 -0.22 -0.22 5.04 -0.22 -0.22 -0.22 4.98 -0.22 -0.22 -0.21 -0.22

-0.06 -0.06 -0.06 -0.06 -0.06 -0.06 -0.05 -0.06 -0.06 -0.06 -0.06 -0.06 -0.06 9.94 -0.06 -0.06 -0.06 -0.06 -4.38 -0.06

0.10 0.10 0.10 0.10 0.10 0.10 -0.79 -0.64 0.10 0.10 0.10 0.11 0.10 0.10 0.10 3.89 0.10 0.10 0.11 0.10

-0.12 -0.12 -0.12 -0.12 -0.12 -0.12 -0.12 -0.12 -0.12 -0.12 -0.12 -0.12 9.78 -0.11 -2.91 -0.12 -0.12 -0.12 -0.12 -0.12

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

-0.38 -0.38 -0.38 -0.38 -0.38 -0.38 -0.38 -0.38 -0.38 -0.38 -0.38 -0.38 -1.72 7.28 5.85 -0.37 -1.23 -0.38 -0.35 0.63

-0.24 -0.24 -0.24 -0.24 -0.24 -0.23 -1.08 -0.96 -0.24 -0.24 -0.24 -0.23 -0.24 -1.54 -0.23 3.67 -0.52 -0.24 4.53 3.85

0.11 0.11 0.11 0.11 0.11 0.11 -0.83 -0.66 0.11 0.11 0.11 0.11 0.11 0.11 0.11 3.83 0.11 0.11 0.11 0.11

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 -0.49 0.04 0.03 -0.20 0.03 0.04 4.47

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 5.0

(a) Learned WM

0 5 10 15 Target Node

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

-0.07 -0.07 -0.07 -0.07 -0.07 10.99 -4.46 -0.07 -0.07 -0.07 -0.07 -0.28 -0.07 -0.07 -0.07 -0.06 -0.07 -0.07 -0.56 -0.07

0.08 0.08 0.08 0.08 0.08 -0.58 4.51 0.08 0.08 0.08 0.08 -0.26 0.08 0.08 0.08 0.10 0.08 0.08 -0.67 0.08

-0.06 -0.06 -0.06 -0.06 -0.06 -0.72 4.31 3.04 -0.06 -0.06 -0.06 -0.39 -0.06 -0.06 -0.06 -0.03 -0.06 -0.06 -0.82 -0.06

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

-0.19 -0.19 -0.19 -0.19 -0.19 -1.57 1.70 2.20 -0.19 -0.19 -0.19 8.65 -0.19 -0.17 -0.19 -1.83 -0.19 -0.19 -1.75 -0.19

0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 3.83 0.03 0.05 0.03 0.03 0.03 0.03 0.03

-0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.05 -0.04 7.00 -1.57 -0.05 -0.04 -0.05 -0.05 -0.11

-0.21 -0.21 -0.21 -0.21 -0.21 -0.21 -0.21 -0.21 -0.21 -0.21 -0.21 -0.21 4.51 -3.81 7.72 -0.21 -0.19 -0.21 -0.21 -0.33

-0.63 -0.63 -0.63 -0.63 -0.63 -0.88 4.42 3.75 -0.63 -0.63 -0.63 -1.32 -0.63 -0.62 -0.63 10.30 -0.62 -0.63 -1.48 -1.00

0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.02 -0.00 0.00 4.41 0.00 0.00 -0.04

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

-0.25 -0.25 -0.25 -0.25 -0.25 -0.69 -4.76 -0.24 -0.25 -0.25 -0.25 -0.47 -0.25 5.14 -0.25 -0.59 -0.25 -0.25 10.26 -0.76

-0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.49 -0.49 -0.50 -0.50 -0.50 -0.50 -0.46 4.14 -3.40 -0.80 3.10 -0.50 -1.36 10.62 4

(b) Learned W V

Figure 9: Weight visualizations on a 20-node graph trained with 1-Token Prediction. Red boxes highlight true adjacency in WM (left) and true reachability in W V (right). In the right plot, light dashed boxes indicate observed reachability.

To rigorously validate our theoretical analysis, Figure 6 in the main text shows the training results of 2-Token Prediction on a 100-node graph when the transfer layer W T is fixed to the ground-truth adjacency matrix. However, in the general 2-Token Prediction setting, the transfer layer W T is learned, with its input being the backbone-predicted next-step logits rather than the true next-step one-hot vectors. As a result, it exhibits certain deviations from the ground-truth adjacency matrix.

https://lh3.googleusercontent.com/notebooklm/AG60hOpV0tPUndPLyBbKKX0VW6ybfLh9uBM3Wwfx7ptME29ZdxdvlWfAGMpkqgWKUv621mX9rUsh5zuLKb6V51xVkq8E_30FiM7LO8y7mtl7X4PO5pbdqBnBZoJBiEz7YH2SUSjytYgDCg=w609-h609-v0

168c8013-64b2-47aa-b497-0085227a1dc7

https://lh3.googleusercontent.com/notebooklm/AG60hOqjb9BiEI25qIjCie2SV37VBUI8G-88M2RU_UZ6vjxR_8-F1nelBf4hpkAnAITG6XQ-8-lOBTP1nz5CIz6QUh_kIkTk6ogHPXiQP9cLvDl8Zfp1-U6TOxZnkQ_ieJo3RVitIo0Ogw=w31-h613-v0

8ad6317b-a19a-4e1e-af8d-b2d7c6c85d7c

https://lh3.googleusercontent.com/notebooklm/AG60hOo6uyT__8EZfkPMwasoTUorBFfzfiW0hPyUB94fJxiNK1T3VIq8ykzMPipyT4hY_BGd_l9AMK4FouaXni4oevTmigreTsSV7HVLGb8knFalj8iMrWl0h8CGa0DH5a8cIOkUHCZ1eA=w609-h609-v0

67b9fa36-eabc-41d5-953b-cbe9f22985dc

https://lh3.googleusercontent.com/notebooklm/AG60hOrv4_AIf7IuimAC9IYYBIZUrZM3ZiaaKyx40qwKP6HExbLPrZGLovUR2S387T8bFd3cnvspxhCQou64VUJMpFAmydfcmXOZxs0HZgPRmqI1d9hfefW40j5eKm7JVm6GNPj1Cxdd2w=w31-h613-v0

0fa3d6c8-6d37-4dfa-90dd-a4e5f0d3cc3b

0 5 10 15 Target Node

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

0.09 0.09 0.09 0.09 0.09 2.42 0.14 0.09 0.09 0.09 0.09 0.10 0.09 0.09 0.09 0.09 0.09 0.09 0.10 0.09

-0.51 -0.69 -1.52 -0.51 -0.84 2.62 6.63 -1.41 -0.66 -0.98 -0.76 2.31 -0.97 -1.46 -0.98 -1.82 -0.61 -0.51 2.68 -0.50

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

-0.03 -0.05 -0.11 -0.03 0.31 -0.03 -0.05 -0.09 -0.07 -0.07 -0.05 -0.03 0.28 -0.03 -0.07 -0.03 5.48 -0.03 -0.03 0.30

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

-0.06 -0.15 0.39 -0.06 -0.19 -0.06 -0.61 8.18 -0.10 0.63 -0.10 -0.68 -0.23 0.47 0.63 -1.76 -0.15 -0.06 -0.05 -0.06

-0.02 -0.01 -0.13 -0.02 0.00 -0.02 -0.52 -0.41 -0.01 -0.09 -0.01 2.90 0.01 -0.06 -0.09 3.01 -0.01 -0.02 -0.01 -0.02

0.02 -0.21 1.74 0.02 -0.69 0.02 0.00 -0.81 -0.02 -0.65 -0.02 0.02 -0.74 5.86 -0.64 0.02 -0.66 0.02 -1.44 0.03

0.14 0.14 0.17 0.14 0.15 0.14 -0.24 -0.10 0.14 0.05 0.15 0.15 0.15 0.07 0.05 2.24 0.15 0.14 0.15 0.14

-0.05 -0.16 -0.41 -0.05 -1.06 -0.05 -0.19 -0.33 -3.08 -0.24 -0.19 -0.05 6.38 -1.24 -0.80 -0.05 -0.98 -0.05 -0.05 -0.05

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

-0.18 -0.18 -0.48 -0.18 0.90 -0.18 -0.18 -0.77 -0.12 -0.77 -0.18 -0.18 -0.24 4.92 3.73 -0.17 0.42 -0.18 -0.17 0.52

-0.07 -0.05 -0.23 -0.07 -0.14 -0.06 -0.38 -0.25 -0.06 -0.11 -0.06 -0.06 -0.13 -0.90 -0.10 2.10 -0.35 -0.07 3.01 2.53

0.14 0.14 0.17 0.14 0.15 0.14 -0.24 -0.10 0.14 0.05 0.14 0.15 0.15 0.07 0.05 2.24 0.15 0.14 0.15 0.14

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

0.11 0.11 0.12 0.11 -0.00 0.11 0.11 0.13 0.11 0.12 0.11 0.11 -0.00 -0.17 0.13 0.11 -0.20 0.11 0.11 2.87

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

(a) Learned WM

0 5 10 15 Target Node

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

0.03 0.03 0.05 0.03 0.04 6.44 -2.32 0.05 0.03 0.04 0.03 -0.14 0.04 0.05 0.04 0.04 0.03 0.03 -0.21 0.03

0.14 0.14 0.16 0.14 0.15 -0.16 2.16 0.16 0.14 0.15 0.14 -0.07 0.15 0.16 0.15 0.15 0.14 0.14 -0.14 0.14

-0.03 -0.08 -0.31 -0.03 -0.16 -0.13 3.05 1.87 -0.09 -0.14 -0.12 -0.09 -0.22 -0.22 -0.14 -0.01 -0.08 -0.03 -0.12 -0.03

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

-0.15 -0.32 1.41 -0.15 -0.30 -0.39 1.01 1.38 -0.22 -1.84 -0.27 5.90 -0.36 -1.98 -1.84 -0.63 -0.19 -0.15 -0.38 -0.15

0.10 0.10 0.12 0.10 0.14 0.10 0.10 0.11 0.16 0.11 0.10 0.10 2.10 0.14 0.12 0.10 0.13 0.10 0.10 0.10

0.06 0.06 -0.04 0.06 0.01 0.06 0.06 0.09 0.06 0.08 0.06 0.06 0.06 3.98 -0.55 0.06 0.02 0.06 0.06 0.03

-0.12 -0.22 -0.45 -0.12 -1.16 -0.12 -0.25 -0.37 -3.11 -0.28 -0.25 -0.12 2.97 -2.21 4.83 -0.12 -1.08 -0.12 -0.12 -0.18

-0.39 -0.45 -2.28 -0.39 -0.57 -0.47 3.02 2.32 -0.46 1.23 -0.49 -0.72 -0.64 1.16 1.23 6.72 -0.49 -0.39 -0.72 -0.58

0.08 0.08 0.09 0.08 0.02 0.08 0.08 0.09 0.08 0.08 0.08 0.08 0.03 0.04 0.08 0.08 2.87 0.08 0.08 0.06

0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20

-0.10 -0.31 1.48 -0.10 -0.77 -0.33 -2.39 -0.88 -0.13 -0.73 -0.13 -0.26 -0.81 3.64 -0.73 -0.27 -0.74 -0.10 6.25 -0.42

-0.22 -0.25 -0.63 -0.22 1.10 -0.22 -0.24 -0.90 -0.28 -0.87 -0.24 -0.22 1.22 2.66 -1.40 -0.34 2.52 -0.22 -0.56 6.81

(b) Learned W V

Figure 10: Weight visualizations on a 20-node graph with fixed transfer layer W T trained with 2-Token Prediction. Red boxes highlight true adjacency in WM (left) and true reachability in W V

(right). In the right plot, light dashed boxes indicate observed reachability. White triangles indicate theoretically learnable entries under ℓ(2)(D).

Figure 11 shows the training results of general 2-Token Prediction. The behavior of WM is sim-ilar to the previous analysis: the true adjacency relations learned via ℓ(1)(D) remain prominent. Since W T is not a perfect ground-truth adjacency matrix, W V not only captures the theoretically learnable reachability relations under ℓ(2)(D), but also learns additional high-weight reachability relations, which also belong to the true reachability in the graph. Table 6 shows the average weight statistics.

To explain this phenomenon in a more general way, training paths often contain sequences that differ only by additional intermediate nodes. As a result, the model tends to assign relatively high logits to tokens corresponding to these intermediate nodes, which can cause the transfer layer to learn transitions that do not exist in the true adjacency. This represents common noise in the transfer layer. At the same time, due to the structure of the paths, the transfer layer can propagate reachability through multiple steps, enabling it to capture additional correct reachability relations. Consequently, the resulting reachability relations learned by W V include both theoretically learnable entries and other valid relations present in the graph.

This process allows the model to capture a broader set of correct reachability relations. As a result, the 2-Token Prediction models continue to achieve strong performance on the degree-3 paths in the test set, as reported in Table 1.

Table 6: Average weights of different entry types in WM and W V in the 100-node graph under general 2-Token Prediction in the simplified model.

## MATRIX TYPE VALUE

True adjacency 2.57 WM Theoretically learnable 0.13

Other entries -0.11

Observed reachability 0.89 True but not observed reachability 0.31

W V Theoretically learnable 0.52 True but not observed & theoretically learnable reachability 0.61 Other entries (not true reachability) -1.21

## E BLOCKSWORLD EXPERIMENTAL SETUP

The Blocksworld benchmark (Valmeekam et al., 2023) is a well-known task in classical planning. It involves a set of colored blocks, where each block can be either placed on the table or stacked on

https://lh3.googleusercontent.com/notebooklm/AG60hOqt-Om5Oh-VaPga70kUqcSvlJbWtU1Pf7Ufm1NKoca0zmKrQkns2Tydxyca-pHb4O-PsBALf9Di9mVvw5rnd_IkOPgzSa4pk9nc0YR2DmSmapb2_vYAxrINccN2-hPYxoGxeLf5=w597-h597-v0

4c1d79b3-4032-421f-9306-9ac168a6f78d

https://lh3.googleusercontent.com/notebooklm/AG60hOpiHcUCKGnTSHGiSvocT-dqOeSGA6XaqU03s7jTQA89j68JiqGWf4JBFoboII-4bHle_a7SF5r9EPiIKLYiNgwsmciKlZPE3XELYr7P6B8ZYbVytZa2lCNl0oAukj-HVSD7UEUVlQ=w30-h600-v0

ee82f3ea-6491-42a4-a617-2246ade07781

https://lh3.googleusercontent.com/notebooklm/AG60hOqvKbF4uATcpdm4D0WzSFCmkID2ilqDXFbGx2AgZj75Zn6UAEjnq23c__Zj6nb5aMzEFvFvZp0oRn3F7EU2WzeHDj_6PZysBMmEOjXn9ab56YwvZyi5PUOJIxoAsUgi0JgNoaiNEA=w609-h609-v0

ead55f4b-9096-4ce6-91cd-94ad106531cb

https://lh3.googleusercontent.com/notebooklm/AG60hOr-LZyE0FmUj4W6FcqCROhM4qu1V3Kh_INgq3EocIl2wIH8617VmaG1kZTLHl44W5cuTpAiZ7icvgwS_wCz5PKTlgwmPpNVmbb7Fid9K_CnVkpCzVcEoISgRTvq_1X9urn-HAnJ=w31-h612-v0

62e29ac0-b07e-4ae9-b6fc-e9a41699f319

https://lh3.googleusercontent.com/notebooklm/AG60hOq15GpG2Lt9VM_P5KF3wiILY57SH0t9PsKVJ_yJifPo7iE0WWDKhwONtLpj8aegHiGOyta4pCeq2oNfjDj0tZ06Hn5i8rhDBuUvSD2SdHrZQI9tpFGLFvDq8CR-aDNJYquVtYHMMw=w601-h601-v0

dc5a58a5-df3d-4318-8ec4-70c1bc34ed02

https://lh3.googleusercontent.com/notebooklm/AG60hOooM9WZJE8ZNyoxEIYNT0VkHq-o-kQ2yWLT5shxg1IDbfe8GIlgTtu_olATllNpN9XTq-afEAz-Am0DUkqTVPAMa0s9buQbTN0afv4Fw2GwkOmHekdJDeD8LpxLmwEjCPKxPWOdOA=w30-h604-v0

e151f209-12db-40c0-911d-eed43d8e8331

0 5 10 15 Target Node

-2.24 -2.24 -2.24 0.05 -2.24 0.24 1.25 -2.24 -2.24 -0.13 -0.07 -0.14 1.15 0.48 0.06 -0.00 0.24 -0.22 0.03 -2.24

-1.12 -1.12 -1.12 0.06 -1.12 0.01 -0.00 -1.12 -1.12 1.29 1.19 0.19 0.19 0.08 0.27 0.13 0.36 0.04 0.23 -1.12

-1.33 -1.33 -1.33 1.32 -1.33 0.16 0.08 -1.33 -1.33 0.30 0.27 0.33 0.07 0.14 0.34 0.10 0.11 -0.39 1.02 -1.33

-0.80 -0.80 -0.80 0.32 -0.80 0.96 0.11 -0.80 -0.80 -0.01 -0.07 -0.07 0.24 0.18 0.16 0.93 -0.00 -0.12 -0.13 -0.80

-0.57 -0.57 -0.57 0.07 -0.57 0.14 -0.20 -0.57 -0.57 0.24 0.10 0.20 0.03 -0.03 0.75 0.12 0.12 0.12 0.14 -0.57

-1.28 -1.28 -1.28 0.24 -1.28 0.05 0.00 -1.28 -1.28 0.14 -0.37 1.02 0.03 0.17 0.02 -0.63 -0.02 -0.10 -0.11 -1.28

-0.78 -0.78 -0.78 -0.08 -0.78 -0.17 0.30 -0.78 -0.78 -0.08 -0.12 -0.39 -0.16 0.02 0.25 0.01 -0.08 0.76 -0.19 -0.78

0.02 0.02 0.02 -0.05 0.02 0.09 -0.05 0.02 0.02 -0.06 -0.08 0.05 0.00 0.08 -0.04 0.02 0.01 0.06 0.12 0.02

-0.92 -0.92 -0.92 0.13 -0.92 -0.02 0.05 -0.92 -0.92 0.17 0.15 0.02 0.25 0.90 0.08 -0.06 -0.09 -0.07 0.11 -0.92

-1.13 -1.13 -1.13 0.27 -1.13 0.06 0.12 -1.13 -1.13 0.43 -0.29 0.01 0.02 0.34 0.68 -0.19 0.37 0.05 -0.19 -1.13

-0.97 -0.97 -0.97 -0.04 -0.97 -0.10 -0.01 -0.97 -0.97 -0.03 0.34 0.17 0.09 0.01 0.34 -0.01 -0.04 0.16 0.09 -0.97

-0.27 -0.27 -0.27 0.08 -0.27 0.08 -0.08 -0.27 -0.27 -0.09 0.03 0.20 0.10 -0.21 0.21 -0.08 -0.14 0.35 -0.03 -0.27

-0.35 -0.35 -0.35 -0.10 -0.35 -0.14 -0.27 -0.35 -0.35 -0.27 0.00 -0.06 0.35 -0.22 -0.07 -0.15 0.02 0.32 -0.11 -0.35

-0.48 -0.48 -0.48 0.35 -0.48 0.01 0.08 -0.48 -0.48 0.34 0.36 0.24 0.06 -0.02 -0.28 -0.00 0.66 -0.08 0.30 -0.48

-1.29 -1.29 -1.29 -0.10 -1.29 0.01 -0.04 -1.29 -1.29 0.12 -0.17 -0.11 -0.02 0.02 0.40 0.02 0.00 -0.04 -0.20 -1.29

-0.31 -0.31 -0.31 0.06 -0.31 -0.05 0.12 -0.31 -0.31 0.18 0.16 0.12 0.05 0.05 0.13 0.38 0.08 -0.07 -0.05 -0.31

-0.28 -0.28 -0.28 -0.10 -0.28 0.11 0.11 -0.28 -0.28 -0.19 -0.13 -0.41 0.37 -0.03 0.02 -0.03 0.61 -0.03 -0.04 -0.28

-0.09 -0.09 -0.09 -0.21 -0.09 -0.08 -0.06 -0.09 -0.09 -0.12 0.02 0.15 0.10 -0.06 -0.01 0.14 -0.24 0.35 -0.21 -0.09

-0.46 -0.46 -0.46 -0.22 -0.46 0.24 -0.06 -0.46 -0.46 -0.29 0.29 -0.13 0.14 0.03 -0.07 0.11 -0.16 -0.11 0.36 -0.46

0.36 0.36 0.36 0.15 0.36 0.03 0.07 0.36 0.36 -0.09 0.00 -0.08 0.05 -0.05 0.04 0.16 0.04 -0.04 0.02 0.36

(a) Learned W T with true adjacency

0 5 10 15 Target Node

-0.53 0.10 0.07 -0.10 0.04 0.01 5.72 -0.02 -0.09 -0.02 -0.09 -0.09 5.44 -0.09 -0.05 0.06 -0.21 -0.34 -0.18 0.01

-0.22 -1.69 -0.02 -0.22 -0.22 -0.19 0.37 -0.14 -0.31 4.92 4.24 -0.04 0.00 -0.10 -0.51 -0.31 3.30 -0.17 -0.10 0.09

-0.21 -0.34 -1.33 4.78 -0.18 -0.42 -0.06 -0.23 -0.43 -0.43 -0.08 -0.49 -0.12 -0.22 -0.16 -0.29 -0.07 0.16 4.17 -0.29

-0.36 -0.30 -0.86 -1.09 -0.22 4.79 0.14 -0.18 -0.29 -0.16 -0.17 -0.32 -0.13 -0.56 0.04 3.97 -0.15 0.02 -0.28 -0.23

-0.43 -0.31 -0.26 -0.42 -0.69 0.20 -0.67 -0.44 -0.24 -0.51 -0.35 -0.30 -0.38 -0.13 4.30 -0.20 -0.31 -0.28 -0.37 -0.06

-0.86 -0.02 0.07 0.13 -0.24 -0.91 -0.19 -0.04 -0.24 -0.40 -0.22 3.67 -0.24 -0.22 -0.47 -0.30 -0.50 -0.30 -0.28 -0.41

-0.67 0.11 -0.01 -0.14 -0.09 -0.19 -0.88 -0.11 -0.28 -0.02 -0.08 -0.19 -0.29 -0.10 -0.26 -0.03 -0.27 4.80 -0.12 -0.09

-0.49 0.00 0.22 -0.03 -0.16 0.16 -0.24 -0.36 -0.06 -0.05 -0.11 0.03 -0.54 -0.07 0.19 0.00 0.09 -0.27 0.05 -0.18

-0.58 -0.11 -0.07 0.09 -0.17 0.01 -0.48 -0.21 -1.11 0.06 -0.01 0.06 -0.35 5.25 -0.21 -0.09 -0.17 -0.15 0.09 -0.09

-0.73 -0.65 -0.18 -0.50 -0.39 -0.11 -0.57 0.02 -0.47 -1.14 -0.23 -0.32 -0.06 -0.57 4.03 -0.38 3.31 -0.23 -0.13 -0.32

-0.28 -0.11 -0.47 -0.27 -0.07 -0.13 0.53 -0.09 -0.25 -0.28 -1.02 -0.38 0.02 -0.57 -0.43 -0.18 -0.06 -0.05 -0.32 -0.00

-0.39 -0.32 0.66 -0.17 -0.17 0.95 0.10 -0.17 -0.16 -0.23 -0.21 -0.90 0.09 -0.18 -0.34 0.07 -0.07 -0.65 0.13 -0.12

-0.32 -0.08 -0.07 0.02 -0.24 -0.31 -0.25 -0.30 -0.18 -0.11 -0.25 -0.29 -0.56 -0.16 -0.11 -0.16 -0.46 -0.22 -0.18 -0.08

-0.86 -0.23 -0.51 -0.50 -0.19 -0.45 -0.09 -0.13 -0.41 -0.57 -0.67 -0.23 0.48 -1.04 -0.39 -0.35 3.42 -0.36 -0.33 -0.25

-0.26 0.47 -0.27 -0.30 0.47 -0.16 0.09 -0.07 -0.07 0.42 -0.14 -0.14 0.14 -0.27 -1.23 -0.33 -0.08 0.14 -0.15 -0.23

-0.78 -0.21 0.14 0.36 -0.15 0.13 -0.34 -0.17 -0.10 -0.19 -0.37 -0.22 -0.41 -0.37 -0.55 -1.28 -0.71 -0.21 -0.22 -0.08

-1.29 0.35 0.18 -0.32 -0.25 -0.29 -0.01 -0.24 -0.25 0.78 0.01 -0.23 -0.04 0.32 -0.07 0.02 -1.93 -0.42 -0.13 -0.09

0.51 0.02 0.28 -0.26 -0.32 -0.67 0.32 -0.33 0.02 -0.10 -0.22 -0.47 -0.96 -0.04 -0.23 -0.24 -0.30 -1.10 -0.33 0.14

0.02 -0.06 -0.04 0.11 0.03 -0.05 0.39 -0.10 -0.16 -0.04 -0.23 -0.30 0.13 -0.37 -0.16 -0.13 -0.12 -0.14 -0.65 -0.03

-0.69 0.54 -0.49 -0.38 -0.00 -0.15 -1.09 0.03 -0.46 -0.71 -0.44 0.04 -0.55 -0.56 0.63 0.26 0.05 -0.21 -0.38 -1.24

(b) Learned WM with true adjacency

0 5 10 15 Target Node

0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00

0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00

0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00

-8.50 -8.21 3.08 0.13 -8.45 -11.32 -8.74 -8.65 -8.83 -8.00 -7.58 -9.09 -8.92 -8.65 -7.80 -11.56 -8.30 -8.90 -8.89 -8.46

0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00

0.63 0.10 5.28 5.47 0.10 3.97 0.23 -0.12 -0.26 0.50 0.33 -1.35 -0.26 -0.57 0.58 -0.51 0.42 -0.55 -0.15 -0.13

3.15 -9.11 -8.72 -8.28 -8.83 -8.84 0.12 -8.05 -8.20 -8.96 -8.41 -8.10 -10.21 -8.46 -7.17 -8.17 -8.55 -11.49 -8.98 -8.71

0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00

0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00

-7.61 3.62 -8.67 -7.69 -8.23 -8.29 -8.62 -6.76 -8.53 0.55 -8.71 -8.25 -8.23 -8.73 -9.66 -8.28 -9.96 -8.01 -8.21 -8.61

-8.67 2.69 -8.27 -8.34 -8.71 -7.85 -8.94 -7.79 -8.22 -9.17 0.19 -8.05 -8.61 -8.15 -8.37 -8.07 -9.60 -8.49 -8.25 -7.55

0.36 -0.44 4.76 2.90 0.03 4.77 -0.24 -0.08 0.06 -0.03 -0.33 4.50 -0.16 -0.31 -0.17 -0.69 0.15 -0.04 -0.69 -0.10

2.44 -9.61 -7.85 -9.14 -8.81 -9.33 -10.66 -9.40 -9.40 -9.33 -8.92 -9.48 -0.43 -9.00 -8.98 -9.53 -9.20 -9.54 -9.11 -9.66

-7.40 -8.51 -7.84 -8.49 -8.00 -8.29 -7.41 -6.99 3.04 -7.89 -7.30 -8.26 -7.98 0.17 -8.65 -8.01 -11.02 -8.03 -6.70 -7.90

-0.16 4.57 -0.07 -0.30 4.65 -0.49 -0.56 -0.23 0.11 4.80 -0.98 -0.52 -0.24 0.47 4.50 0.21 -1.35 -0.11 -0.39 -0.39

0.26 -0.08 5.55 5.66 -0.39 -1.12 0.36 -0.19 0.20 0.28 0.09 -0.30 -0.42 -0.27 -0.18 4.85 -0.01 0.37 -1.66 0.05

-0.71 4.71 0.36 -0.16 -0.02 0.11 -0.58 -0.12 0.28 3.91 -2.04 -0.27 -0.43 4.86 -0.49 0.10 5.24 -0.09 -0.93 -0.58

5.09 0.46 -0.52 0.19 -0.35 -0.15 5.43 -0.24 -0.80 -0.65 0.04 0.41 -1.83 0.01 0.56 -0.12 -0.20 4.25 -0.52 0.01

-9.15 -8.93 2.30 -10.28 -8.66 -8.48 -9.26 -7.60 -8.70 -8.76 -7.61 -8.63 -7.35 -8.74 -8.53 -8.97 -9.05 -8.14 -0.14 -8.93

0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00

(c) Learned W V with true reachability

Figure 11: Weight visualizations on a 100-node graph under general 2-Token Prediction for the first 20 nodes. (a) W T with true adjacency highlighted in red boxes. (b) WM with true adja-cency in red boxes and theoretically learnable adjacency (under ℓ(2)(D)) marked by white triangles: extra entries are incorrect. (c) W V with true reachability in red boxes, observed reachability in light dashed boxes, and theoretically learnable reachability in white: extra entries are correct and unobserved.

top of another block. The objective is to plan a sequence of valid actions that transform an initial configuration into a goal configuration.

We consider a version with 4 blocks. This setup results in a complete state transition graph con-taining 73 nodes, where each node represents a unique, valid block configuration, and each edge (u, v) denotes a legal atomic action that transitions the system from state u to state v. This task can therefore be seen as an instance of the path-planning problem described in the main text, but on a fixed, predefined graph. For example, node 49 represents the state where block A is on the table, B is on C, C is on D, and D is on the table.

To simulate a more general planning scenario where the model does not see the full state space during training, we use only a small subset of state transition processes to construct the training set. Each training example corresponds to a valid path sampled from the graph. The test set is composed of randomly selected source–target state pairs, and the model is required to generate the sequence of intermediate states connecting them, i.e., a valid path.

Dataset Construction. The training set includes all one-hop edges (s, t) ∈ E added as direct paths to ensure learning of adjacency relations. For each path length from 2 to 6, we sample n acyclic paths, where n is varied as 100, 200, 300, 400, or 500 to create training sets of different sizes. Each path is formatted as “s t s a ... t \n”.

The test set is fixed and consists of 5,000 randomly sampled paths with lengths greater than 1. For each training size, all models are trained on the same set of paths and evaluated on this same test set.

## F USE OF LARGE LANGUAGE MODELS

We acknowledge the use of Large Language Models (LLMs) in the preparation of this manuscript. Their role was limited to aiding in and polishing the writing.

In adherence to the principles of transparency and academic integrity, we herein detail the auxiliary role that Large Language Models (LLMs) played in the preparation of this manuscript. We followed a strict workflow to ensure that all intellectual contributions are the original work of the human authors. The entire substantive content of this paper—including the formulation of the research problem, the design of the theoretical framework, the execution of experiments, the collection and analysis of data, and the derivation of our conclusions—was conceived and executed exclusively by the human authors. Based on this work, we authored a complete and comprehensive first draft that fully encapsulated our research, argumentation, and findings.

Only upon the completion of this draft did we employ an LLM as a tool for linguistic refinement. Its application was strictly confined to enhancing the quality of the prose, not to generating content. Specifically, the model was utilized to check for and correct potential grammatical errors, optimize sentence structures for improved fluency and readability, and suggest more precise or varied aca-demic terminology to enhance the overall clarity and professionalism of the text. It also assisted in ensuring the consistent use of key terms throughout the manuscript. Critically, this refinement process remained under our rigorous supervision. Every suggestion generated by the LLM was treated as a candidate for review, not an automatic edit. Our team conducted a scrupulous evaluation and critical analysis of each proposed change, carefully judging whether it improved the language without altering our original academic intent, the nuances of our arguments, or the precision of our scientific claims. Any suggestion that could potentially introduce ambiguity or weaken our line of reasoning was not adopted.

Therefore, while we leveraged an LLM to polish the language of this manuscript, the intellectual ownership, the academic core, and the final phrasing of every sentence are the result of our indepen-dent work. The full and final responsibility for the scientific accuracy, the validity of the arguments, and the originality of this paper rests entirely with us, the human authors.

