---
sourceFile: "How the NanoGPT Speedrun WR dropped by 20% in 3 months - LessWrong"
exportedBy: "Kortex"
exportDate: "2025-10-28T19:01:55.219Z"
---

# How the NanoGPT Speedrun WR dropped by 20% in 3 months - LessWrong

2362c81e-8418-4cdb-98ef-fbd86f4be732

How the NanoGPT Speedrun WR dropped by 20% in 3 months - LessWrong

4be96d14-6a79-4033-8be2-b8d6b6e134b0

https://www.lesswrong.com/posts/j3gp8tebQiFJqzBgg/how-the-nanogpt-speedrun-wr-dropped-by-20-in-3-months

https://www.lesswrong.com/w/ai

## Personal Blog

https://www.lesswrong.com/w/ai

How the NanoGPT Speedrun WR dropped by 20% in 3   months

https://www.lesswrong.com/w/ai

https://www.lesswrong.com/users/larry-dial?from=post_header

5th Oct 2025 10  min read

https://www.lesswrong.com#comments

https://www.lesswrong.com#comments

## Personal Blog

https://www.lesswrong.com#comments

How the NanoGPT Speedrun WR dropped by 20% in 3 months

https://www.lesswrong.com#comments

3 Vladimir_Nesov

https://www.lesswrong.com#comments

https://www.lesswrong.com#comments

7 Vladimir_Nesov

https://www.lesswrong.com#comments

16 the gears to ascension

https://www.lesswrong.com#comments

16 larry-dial

https://www.lesswrong.com#comments

7 the gears to ascension

https://www.lesswrong.com#comments

5 larry-dial

https://www.lesswrong.com#comments

3 faul_sname

https://www.lesswrong.com#comments

2 Vladimir_Nesov

https://www.lesswrong.com#comments

New Comment   9 comments , sorted by

top scoring

Click to highlight new comments since:

Today at 6:55 PM

Vladimir_Nesov

Many of the improvements in the last 20% have not yet been published outside of the modded-nanogpt repo. This post summarizes those improvements.

Thus this is ~original capability research/exposition, not framed/filtered for potential relevance to either interpretability/alignment/control or forecasting.

This is a rather famous competition project, competent people and orgs are supposed to be aware of it.

An interesting question is, how far the algorithmic improvements are likely to go in general and how fast. This competition is mostly a laboratory of that (but there are plenty of comparable open efforts on larger scales, cf. DeepSeek V3).

We have been under impression that after 15x improvement in 8 months or so, this one had saturated. It turns out that this is not the case. This informs us about the likely future trends (tells us not to update too hard on the slowdown of this one). We should be aware of that.

Vladimir_Nesov

It's possible to frame this content in a relevant way, to study forecasting of algorithmic improvements (where most of the technical details of the improvements themselves aren't relevant). Similarly, if it was already published elsewhere (at a given level of accessibility) and well-known to ~saturation, it would've been neutral to discuss it.

I think LW shouldn't be taking steps to advance (open) capability research for its own sake, however trivially. A post being actually good then makes it proportionally worse.

the gears to ascension

24d 16 14

It's possible to frame this content in a relevant way

I appreciate that this post is clearly (rather than covertly) capabilities. too many posts pretend to be alignment which aren't.  I wouldn't want OP to dress it up in lies in order to fit it in.

OP, I'm curious about your views on alignment.

My current view is that alignment of advanced future AI systems will  need to be approached from a large number of angles simultaneously :  how public perception of AI is managed ,  how regulatory body's set incentives for research ,  how investors direct funds , and how researchers build thoughtful systems and anticipate change to model behavior. I believe I can best contribute by  gaining a deep technical understanding of AI systems , such that I can better anticipate how changes to data/architecture/compute will impact behavior. Right now I find that  exploring capabilities  gives the strongest feedback signal to build this intuition, because the system immediately tells you when your next idea sucks, or when your intuition is off.

the gears to ascension

I appreciate your willingness to explain your view. Replying with how mine responds to that viewpoint, as a person who was doing something quite similar about 10 years ago and came to my capabilities knowledge that way:

The fact that capabilities gives good feedback signal and alignment does not, seems to me to be much of why we're finding it difficult to solve alignment. If we knew a thing to line-go-up about was a good thing to line-go-up about to solve alignment, then just line-go-up about it, and you've solved alignment! compare to a hypothetical research field starting three thousand years ago, "machine motion". machine motion studies making machines cause motion. machine motion papers can push forward "make things go fast", and eventually, someone figures out how to make machines cause a lot of motion all at once and tries it out in the new mexico desert a few decades ago. but, the sister field, machine aim, has less progress. aiming a machine requires making it go a

direction, and it turns out that, at least for simple machines, making it go at all is easy to measure, but ... the metaphor breaks down because the space one aims through for literal throwing is so low dimensional (and relatively low effective lyapunov exponent) compared to the one we need to aim through (which includes all forms of throwing, as well as every other physical interaction downstream of the starkly superintelligent model we eventually build and align.)

I agree that understanding capabilities is very important for having plausible alignment ideas. I don't agree that trying to push the frontier of a problem, especially when focused on others' understanding, is a necessary way to do that. I did a lot of keeping up to date of the kind you're doing here over the years. but even though doing that is normal and important in order to contribute to alignment seriously, I've been very careful not to narrate my thoughts on it online, so as to not be a marginal contribution to the frontier unless I can tell whether I'm improving the ratio of alignment outcomes. if everyone did this, there would be little progress on AI except when it was a good idea to do so. the flip side of this is, I don't think you're doing a very good job of understanding capabilities, in a similar way to how most people in the field aren't; but see above for why that's all I'll say on that.

It seems to me that in order to matter, alignment work has to be able to work at the frontier. so working with the frontier is important and not a mistake. but I'm not a fan of anything that pushes that frontier. I want to know how to push it in some directions, but those directions involve figuring out how to make loss functions and learning algorithms that quickly and asymptotically organize an AI into a thing that actually works towards indefinite-term good.

I'm optimistic we can define that, likely many of the tools of capabilities will matter, but I think we'll want to be on the pretty math-heavy end of capabilities to do it right, where you derive your training setup from a theoretical insight. and I'm optimistic that scaling has already put us close to being able to figure out inhumanly hard math questions with the help of an AI that is only locally aligned to solving problems, and have it help us figure out the math of training a successor that is asymptotically aligned to control the world into states where no other AI breaks humans' autonomy as we enter this new era.

I think we have different viewpoints of what the frontier is. The majority of the 20% improvements mentioned in this post are things I came up with and are pretty surface level. I have only been looking at LLMs for 6 months when I have free time outside work as something to tinker with, and I don't consider myself an expert, obviously. I would anticipate that the actual research frontier at labs is substantially ahead, such that any moral discussions around this post are akin to debating if a 11th grade Chemistry lab will encourage the creation of nuclear weapons.

I don't think you're doing a very good job of understanding capabilities

Part of my hope in posting was to get technical feedback from a crowd that is knowledgeable on AI systems. Curious if you can be more specific on why you believe this.

AI development feels more similar to biology than to chemistry. Bright 11th graders shouldn't be doing experiments on culturing some previously unculturabke pathogen which would be a good bioweapon target and discussing their results, since the field is wide and shallow and it's not entirely impossible that their experiments are novel. On the other hand, if they're running basic experiments on culturing some specific common bacterium (e.g. e coli) better, they probably don't need to worry about accelerating bioweapon development even if there is a chance of them making a slight advancement to the field of biology as a whole.

The nanogpt speedrun feels more like developing better methods to culture e coli at a hobbyist level , and quite unlikely to lead to any substantial advancement applicable to the operational efficiency of well-funded companies at the frontier. Still, it probably is worth keeping track of when the work you're doing approaches the "this is actually something novel the frontier labs might use" mark, particularly if it's something more substantial than "here's how to use the hardware more efficiently to train this particular model".

Vladimir_Nesov

Framing isn't about being covert, it's about a particular emphasis on what kinds of considerations are in scope, naturally resulting in almost complete omission of obviously irrelevant (technical) details (and occasionally in comical misunderstanding of content produced from a mutually unintelligible framing).

## Moderation Log

https://www.lesswrong.com/moderation

## More from

https://www.lesswrong.com/users/larry-dial

https://www.lesswrong.com/users/larry-dial

## Curated and popular this week

https://www.lesswrong.com#comments

In early 2024 Andrej Karpathy stood up an llm.c repo to train GPT-2 (124M), which took an equivalent of 45 minutes on 8xH100 GPUs to reach 3.28 cross entropy loss. By Jan 2025, collaborators of

modded-nanogpt

https://github.com/KellerJordan/modded-nanogpt

brought that time down to 3 minutes. It sat near 3 minutes until July 2025, having a large swath of optimization already applied: RoPE, value embeddings, reduce scatter grad updates, Muon, QK Norm, Relu^2, a custom FP8 head, skip connections, flex attention, short-long windows, attention window warmup, linear lr cooldown, and more. Yet, in the last 3 months the record has fallen by another 20% to 2 minutes and 20 seconds.

Many of the improvements in the last 20% have not yet been published outside of the modded-nanogpt repo. This post summarizes those improvements.  Not everything will generalize to larger scales, but there are some core concepts that I believe are promising. Improvements are sorted into ML and Engineering, grouped by concept, and subjectively ranked by their general applicability. Each change includes an estimated runtime impact and links to the associated PRs. The post concludes with general thoughts on the process of finding improvements in transformer architectures and training recipes.

Cat Rank Description Rough Est Impact  PR ML 1 Document Alignment 3s

ML 2 Dynamic Attention Window 4s

https://github.com/KellerJordan/modded-nanogpt/pull/118

https://github.com/KellerJordan/modded-nanogpt/pull/122

https://github.com/KellerJordan/modded-nanogpt/pull/127

https://github.com/KellerJordan/modded-nanogpt/pull/131

ML 3 Heterogenous Batch Sizes 4s

https://github.com/KellerJordan/modded-nanogpt/pull/136

ML 4 Backout 2s

https://github.com/KellerJordan/modded-nanogpt/pull/140

ML 5 Polar Express 1s

https://github.com/KellerJordan/modded-nanogpt/pull/134

ML 6 Smear Module 1.5s

https://github.com/KellerJordan/modded-nanogpt/pull/130

ML 7 Sparse Attention Gate 0.5s

https://github.com/KellerJordan/modded-nanogpt/pull/117

ML 8 More Bfloat16 0.5s

https://github.com/KellerJordan/modded-nanogpt/pull/125

https://github.com/KellerJordan/modded-nanogpt/pull/133

ML 9 Softmax Skip Gate 1s

https://github.com/KellerJordan/modded-nanogpt/pull/125

ML 10 Drop initial MLP Layer 1.5s

https://github.com/KellerJordan/modded-nanogpt/pull/120

ENG 1 Flash Attention 3 3s

ENG 2 Parameter reshaping for shared reduce scatter 1.5s

https://github.com/KellerJordan/modded-nanogpt/pull/109

https://github.com/KellerJordan/modded-nanogpt/pull/132

ENG 3 Async Data Fetch and Index 1.5s

https://github.com/KellerJordan/modded-nanogpt/pull/127

ENG 4 Vectorized Optimizer Step 0.5s

https://github.com/KellerJordan/modded-nanogpt/pull/125

ENG 5 Triton Kernel for Symmetric Matmul 1s

https://github.com/KellerJordan/modded-nanogpt/pull/109

ENG 6 Resize Lambda Params 1.5s

https://github.com/KellerJordan/modded-nanogpt/pull/140

Latest version with all implemented code:

https://github.com/KellerJordan/modded-nanogpt/blob/ba3e54f378b11af1ee33c2d518820e4532020190/train_gpt.py

https://github.com/KellerJordan/modded-nanogpt/blob/ba3e54f378b11af1ee33c2d518820e4532020190/train_gpt.py

(Updates must be found through open PR list due to inactive repo owner)

## ML Improvements

#1: Document Alignment

Intra-document masking

is a common technique used in models such as Llama 3 to prevent attention queries from attending to positions in other documents. However, masking is only half the picture. NanoGPT applies a data processing step during training such that each GPU receives the first 2048 tokens of at least 16 unique documents per step. The 2048 limit was optimized by Varun Srivastava. This approach has several benefits:

Lower variance gradients. FineWeb documents can have up to 70,000 tokens. Since each training step contains 262,144 tokens, a naïve data sampling strategy may have 1/4 of its gradient estimates for a step coming from a single highly correlated document. This sampling approach ensures that each gradient is informed by at least 128 documents.

Beginning of Sentence token is kept in context window. Prior research

demonstrated that having the bos token in the context window can improve performance.

No mid-context learning. The model does not need to waste effort trying to learn from samples that start in the middle of a document.

#2: Dynamic Attention Window Management by Layer

NanoGPT applies a window sizing scheme across its 10 attention layers of [short, short, short, long, short, short, short, short, short, long]. The short window is initialized to 128 tokens and the long window to 384 tokens. 3 transformations occur during training:

At 1/3 of training: Increase from 128/384 to 384/896. Apply YaRN

At 2/3 of training: Increase from 384/896 to 640/1408. Apply YaRN.

At 3/3 of training: Increase from 640/1408 to 768/2560. Apply YaRN.

Partial RoPE is applied to 50% of the head dimensions. It was observed that the long windows primarily attend to the stationary dimensions, and are responsible for model tasks such 'find activations that look very similar to me, regardless of their position'. These long windows showed much more flexibility with window extensions, in particular the jump from 1408 to 2560 after training is complete.

#3 Heterogenous Batch Sizes

Critical batch size theory focuses on finding the single optimal batch size for a given model and dataset. However, parameters within a model have distinct training characteristics that lead to different optimal batch sizes. NanoGPT uses gradient accumulation to only update the embedding and lm_head weights every other step, creating heterogenous batch sizes within the same model. This means the gradients for these parameters across all 50,000 tokens in the vocabulary only need to be synced across GPUs half as often, leading to faster time per step.

# on even steps, only step Muon params # on odd steps, step all params if step%2==0: optimizer2.step() optimizer2.zero_grad(set_to_none=True) else: for opt in optimizers: opt.step() model.zero_grad(set_to_none=True)

#4 Backout: Enabling a model to back out context for predictions

In the standard transformer architecture contributions to the residual stream have to serve two purposes at once: provide context to downstream layers, and add to the final prediction. However, information may be valuable for downstream context but not directly map to the lm_head vector of the token needed to be predicted. To enable the model to modulate these two functions independently, its given the ability to back out prior contributions just before making a prediction. The model learns to back out 50% of the contributions from the first 2/3 layers. The core of this idea is from Sebastian Müller.

x -= backout_lambda*residual_stream_after_layer8 x = norm(x) logits = self.lm_head(x)

#5 Polar Express

This is a more accurate method for computing the orthogonalization step in Muon compared to Newton Schulz. See the original paper

for details. Incorporation into the repo was performed by Varun Srivastava.

#6 Smear Module

It is common for several heads in a transformer to devolve into a "Previous token head" that always attends to the previous position. However, attention is a computationally inefficient way to attend to the previous position. NanoGPT introduces the Smear Module, which enables tokens to directly peer backwards 1 position and smear the prior token forward. The contribution is gated on a sigmoid gate that is fed by the first 12 dimensions of the token embedding space. On average, the model learns that (token + 0.07prior_token) is a more useful representation than (token).

x = self.embed(input_seq) smear_lambda = self.scalars[5 * len(self.blocks)] smear_gate_out = smear_lambda * torch.sigmoid(self.smear_gate(x[1:, :self.smear_gate.weight.size(-1)])) x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]]) x = x0 = norm(x[None])

#7 Sparse Attention Gate

Attention does not have a built in way to perform a no-op. Many mechanisms to alleviate this have been proposed, but they are often either not directly compatible with Flash Attention 3, or incur high runtime overhead (in the context of this speedrun). NanoGPT uses a sigmoid gate for each attention head to modulate the attention output. The gate is fed by only the first 12 dimensions of the residual stream, enabling fast updates while significantly reducing the bos token attention sink behavior.

# init self.attn_gate = CastedLinear(12, num_heads) # perform attn out projection y = y * torch.sigmoid(self.attn_gate(x[..., :self.attn_gate.weight.size(-1)])).view(B, T, self.num_heads, 1)

#8 More Bfloat16

The cross entropy loss calculation is left in bfloat16 instead of casting up to float32. The language model head and rotary cos and sin terms are stored in bfloat16 instead of bfloat32. This gives a faster runtime on the forward pass with minimal increase in loss. Adam gradient calculations are left in bfloat16. The parameter storage for MLP and attention matrices are left in float32 due to higher sensitivity to loss. This change for rotary was discovered by the

https://www.hiverge.ai/blog/introducing-hiverge

, while the improvement for cross entropy loss was discovered by Daniil Sedov.

#9 Softmax Skip Gate

Skip connections had previously been setup and initialized with weights of 1:1 with the main pathway. This is replaced with a sigmoid gate that is initialized to produce 0.18. The smaller initialization for skip connections gives the model worse initial training, but better final performance due to encouraging the formation of deeper pathways. This change was implemented by the

https://www.hiverge.ai/blog/introducing-hiverge

#10 Drop MLP Layer

EmelyanenkoK dropped the initial MLP layer and increased the step count to partially compensate, after running an ablation that showed it had the least impact of all MLP layers in the model.

## Engineering Improvements

#1 Flash Attention 3

In order to make Flash Attention 3 compatible with torch.compile, an unmerged version is used. Varun Srivastava streamlined this process with the huggingface kernels library, and implemented flash_attn_varlen_func() to maintain the intra-document masking that was previously applied via flex attention.

y = flash_attn_interface.flash_attn_varlen_func( q[0], k[0], v[0], cu_seqlens_q=seqlens, cu_seqlens_k=seqlens, max_seqlen_q=max_len, max_seqlen_k=max_len, causal=True, softmax_scale=attn_scale, window_size=(bm_size, 0) )

#2 Parameter reshaping for shared reduce scatter

The optimizer implementation for MLP and Attention parameters uses Muon, which requires that the entire gradient for a matrix be collected onto a single GPU to perform an accurate orthogonalization update. After each training step each GPU has its own gradients, which need to get collected in one place. Torch has a distributed API call to take 8 parameters, pick a GPU to own each, and have the other 7 GPUs send their copy to the designated owner. However, the API call only works if each GPU has a parameter of equal size. This means that if there aren't exactly 8 parameters, extra padding variables get created.

To minimize padding variables, all attention and MLP weights are reshaped to the same dimensions of [d_model, 4*d_model]. Bryon Xu implemented this for MLP. This means that on the forward pass the MLP out projection gets transposed back to shape (4*d_model, d_model) and the attention matrix gets reshaped to (4, d_model, d_model), for the 4 projections for Q,K,V, Out. The attention parameters also get reshaped prior to the orthogonalization update, shown below.

# Compute zeropower for the entire chunk in a single, batched call. original_shape = batched_update_grads.shape # Reshape attn params from [hdim, dim*4] to [4,hdim,dim] to apply NS indepedently to Q,K,V,O module_idx = start_idx if start_idx<len(params) else 0 if getattr(params[module_idx],'module','none')=='attn': batch = 4 * original_shape[0] d1 = original_shape[1] d2 = original_shape[2] // 4 batched = batched_update_grads.view(batch, d1, d2) v_chunk = polar_express(batched) v_chunk = v_chunk.view(original_shape) else: v_chunk = polar_express(batched_update_grads)

#3 Async Data Fetch and Index

Start prefetching and indexing the next shard immediately. Since this occurs on the CPU, there is ample time to perform this during the GPU heavy workload, and we shouldn't be bottlenecking GPU activities on CPU data indexing. Only partially index the first shard before starting to train on it. Kickoff a parallel thread to finish indexing it, which gets picked up on the 5th step.

#4 Vectorized Optimizer Step

Torch reduce scatter, Muon orthogonalization, and torch all gather can be executed across multiple parameters at once, as long as the total parameter count is divisible by 8. This change was implemented by the

https://www.hiverge.ai/blog/introducing-hiverge

#5 Triton Kernel for Symmetric Matmul

Multiplying two matrices with shape (m, k) and (k, n) requires 2*m*k*n FLOPS. However, multiplying a matrix (m, k) with its own transpose (k, m) can be done with only m*k*m FLOPS. The result is symmetric, so we only need to compute half of it and copy the result across the diagonal. This is used in the first step of Newton Schulz for the Muon update. This update is from Bryon Xu.

#6 Resize Lambda Parameters

The model has a host of scalars to manage the weighting of various connections. Originally it was assumed that the exact update process of these scalars was less relevant, since the count (<100) is dwarfed by the core model parameters. However, it was later observed when the count was set to 56 or 72 scalars the runtime increased meaningfully compared to 64 scalars. While the exact cause is not fully understood, it is weakly hypothesized that coalesced memory access patterns are playing a role here, where each GPU can access 4 data values simultaneously. After the Adam optimizer splits the parameters 8 ways across the GPUs, 56 scalars was leading to 7 parameters per GPU, and 72 scalars was leading to 9 parameters per GPU.

## Takeaways from the Journey

All changes above without a listed author were from myself. I have learned a lot in the last 3 months about the process of discovering model improvements (and my bank account has also lost some weight). I hope that I can keep learning, to the point where I'll look back and consider current me a bit clueless. Here are my takeaways from where I stand today.

Optimize for many ideas over good ideas.

The best way to have a good idea is to first have 20 bad ideas and learn from them. When I was first starting on the speedrun, I spent a lot of effort trying to mentally anticipate how an idea might pan out. I have found it more advantageous to limit my initial mental effort to 'Is this idea plausible'. If so, immediately shift into 'How can I test it'. The most fruitful spot for learning is after I have test results and have to think through why an idea failed or worked. I want to go from ideation to that spot as quickly as possible.

Work backwards from the data.

Moving the needle on a pre-optimized task means you have to find ideas that no one has thought of yet. The approach of 'read a paper', 'apply the paper', 'repeat', is a good way to keep your inspiration in the same spot as the people who have already been testing ideas. If you instead work backwards from the data, it will give you a completely unique perspective- just from the fact that there are millions of different ways to look at the data.

https://medium.com/@larry36d/formation-of-induction-heads-in-modded-nanogpt-5eb899de89e4

is an example where I explore how the phrase

http://stickygooeycreamychewy.com

gets perfectly predicted by the model on its second time ever seeing it. This gives me a unique perspective on how the last layer is functioning, leading to the post training attention window improvements.

Let gradient magic work for you.

I'm using the phrase 'gradient magic' to refer to how backpropagation can almost instantly find the local gradient across the entire parameter space. This is something I've heard for years but didn't understand until recently, because it is so remarkably different from how humans approach problems. If a human was in a footrace and they had 100 million doors in front of them and needed to pick a route, it would be tremendously helpful if someone could remove the worse half of the doors. Choice  parallelizes  humans. Backpropagation cuts through it. Instead of trying to help the model by eliminating choices, give it more context and more choices.

Find environments with feedback.

I don't work in AI, I don't have professors or peers in AI, and none of my friends work on anything related to AI. As a result, I am rarely ever getting feedback. Most of my knowledge consumption in this space is unidirectional, which I'm realizing is horribly inefficient. I got lunch one time with a super cool guy at AI2, and had a video call a year ago with a very friendly research engineer. Those two experiences were very helpful for me, though sparse. The consistent feedback from a speedrun timing and some of the discussions coming off of it has been incredibly productive for my rate of learning. In a sense, it helps level the playing field between myself and those already in feedback rich environments. If I was better about networking I probably could have leveled that playing field a long time ago. Now that I've had this experience, "What is the level of feedback" is a question I'm asking about every new learning environment.

http://stickygooeycreamychewy.com/

Analysing The Impact of Sequence Composition on Language Model Pre-Training https://arxiv.org/pdf/2402.13991

http://stickygooeycreamychewy.com/

Efficient Streaming Language Models with Attention Sinks https://arxiv.org/abs/2309.17453

http://stickygooeycreamychewy.com/

YaRN: Efficient Context Window Extension of Large Language Models. https://arxiv.org/pdf/2309.00071

http://stickygooeycreamychewy.com/

The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm. https://arxiv.org/pdf/2505.16932

