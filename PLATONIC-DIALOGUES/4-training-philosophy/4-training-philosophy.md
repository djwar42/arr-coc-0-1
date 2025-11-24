---
summary: whereby Socrates and Theaetetus develop a three-phase training curriculum for ARR-COC beginning with Phase 1 frozen-encoder transfer learning where only the allocator (cross-attention, hotspot detector, budget predictor) and quality adapter train while SAM/CLIP/Ovis remain frozen to preserve pre-trained visual and language knowledge, progressing to Phase 2 CLIP unfreezing enabling adaptation to variable-length compressed inputs (64-400 tokens) and query-aware representations while SAM and Ovis stay frozen, and culminating in Phase 3 full system fine-tuning, employing a multi-objective loss function balancing answer quality, compression efficiency, and attention alignment requiring careful weighted combination, while the oracles warn about frozen-encoder gradient magnitude mismatches requiring 10× higher learning rates and careful normalization to prevent optimizer thrashing, multi-objective loss scale differences (task ~3.5, KL ~0.01, entropy ~0.001) necessitating iterative weight tuning with compression weight very small initially to prevent under-allocation collapse, the lack of ground-truth patch-level importance labels making attention alignment loss merely a regularizer against attention collapse, and contrast Phase 1's conservative frozen-both-encoders strategy against DeepSeek-OCR's Stage 1 co-adaptation of SAM+CLIP and Ovis's 5-phase progressive unfreezing with VET providing catastrophic forgetting protection
---

# Part 4: The Training Philosophy
*A dialogue between Socrates and Theaetetus on teaching ARR-COC*

---

**SOCRATES:** So, my young friend, we have this elegant ARR-COC architecture. But how does one train such a system? Surely we cannot simply initialize randomly and hope for convergence?

**THEAETETUS:** An astute observation, Socrates! Training ARR-COC requires a carefully orchestrated curriculum. We break it into three phases.

       **DeepSeek-OCR Oracle:** *Three phases! They're converging on our 3-stage architecture—Stage 1: DeepEncoder pre-training (SAM+CLIP+Projector with compact LM), Stage 2: Full VLM (swap to DeepSeek-3B-MoE), Stage 3: Gundam-Master high-res fine-tuning. Total: ~17 days on 160 A100s. But their phases serve different purposes. Let's see if they understand why staged training matters.*

       **Ovis Oracle:** *Interesting—we use 5 phases, not 3. P1: VET initialization (2-3 days), P2: Multimodal pre-training (10-12 days), P3: Instruction tuning (4-5 days), P4: DPO alignment (12-16 hours), P5: GRPO reasoning optimization (6-8 hours). Total: ~18-21 days on 160-320 A100s. Each phase builds on the previous without catastrophic forgetting. Three phases might work, but I'm curious about their freeze strategy—our P1 freezes most of the LLM, P2-P4 train everything, P5 freezes vision. Progressive unfreezing prevents instability.*

### Phase 1: Train the Allocator (Frozen Encoders)
**SOCRATES:** Why freeze the encoders?

**THEAETETUS:** Think of it this way, Socrates. SAM and CLIP are already pre-trained on billions of images. They understand visual features and language-vision alignment. We don't want to disturb that knowledge initially.

So in Phase 1:
- ✓ Train ARR-COC allocator (learns importance scoring and token budgeting)
- ✓ Train Quality Adapter (learns to enhance compressed features)
- ✗ Freeze SAM (visual encoder)
- ✗ Freeze CLIP (vision-language bridge)
- ✗ Freeze Ovis 2.5 (language model)

```python
# Phase 1: Only these components train
trainable_params = [
    allocator.cross_attention,
    allocator.hotspot_detector,
    allocator.budget_predictor,
    allocator.compression_heads,
    quality_adapter
]

for param in [sam, clip, ovis]:
    param.requires_grad = False
```

       **Karpathy Oracle:** *OK so this frozen-encoder strategy is classic transfer learning but there's a huge gotcha. In nanoGPT when I freeze the transformer body and only train the head (fine-tuning), the learning rate needs to be WAY higher than you'd think—like 10× higher. Why? Because frozen layers don't update their internal statistics (BatchNorm running means, LayerNorm scales), so the gradients from your trainable layers are propagating through a static feature extractor. The allocator is learning to score features that NEVER change. This works IF SAM/CLIP features are already perfect for your task. But if they're not (e.g., SAM's window attention patterns don't align with your query-patch relevance), you're stuck. I'd recommend: freeze SAM body but train last 2-3 layers, freeze CLIP body but train projection head. This gives the allocator some flexibility to adapt the features without catastrophic forgetting. Also watch for gradient magnitude mismatches—frozen layers output features at scale ~1.0, your random-initialized allocator outputs ~0.1, and the quality adapter outputs ~10.0. You need careful normalization or the optimizer will thrash.*

       **DeepSeek-OCR Oracle:** *Sensible! Our Stage 1 also freezes the production LLM—we train DeepEncoder with OPT-IML (compact LM) instead. Vision encoder is the focus. Duration: ~1 week on 160 A100s, 130M samples × 2 epochs. But they're proposing to freeze SAM AND CLIP. We train both in Stage 1 because they need co-adaptation: SAM learns what to extract, CLIP learns what SAM outputs. Freezing both risks misalignment between allocator and downstream components. See training/stage1-deepencoder.md—joint training of SAM+CLIP+Projector is essential.*

       **Ovis Oracle:** *Our P1 approach is different: freeze MOST of ViT but train last few layers + VET + visual_head. Why? The ViT (SigLIP-SO400M pretrained) already understands images. We just need to teach the visual_head to generate probability distributions that map meaningfully to VET. Freezing everything would prevent this adaptation. Their "frozen SAM" strategy assumes SAM's window attention patterns are already optimal for their allocator's needs. Risky assumption—might need at least partial SAM fine-tuning. See training/01-phase-p1-vet.md for our partial-freeze strategy.*

**SOCRATES:** What does the allocator learn in this phase?

**THEAETETUS:** Three critical skills:

1. **Query-patch relevance**: Which patches matter for which questions?
2. **Visual importance**: What's inherently important in the image?
3. **Budget prediction**: How many tokens does this task need?

We train on diverse image-query-answer triplets:
- Document QA datasets (DocVQA, InfographicVQA)
- Chart understanding (ChartQA, PlotQA)
- General VQA (VQAv2, GQA)
- OCR tasks (TextVQA, ST-VQA)

**SOCRATES:** And the loss function?

**THEAETETUS:** Multi-objective! We optimize for:

```python
total_loss = (
    answer_quality_loss +          # Is the answer correct?
    compression_efficiency_loss +  # Are we using tokens wisely?
    attention_alignment_loss       # Do important patches get attention?
)
```

       **Karpathy Oracle:** *Multi-objective loss functions look clean in pseudocode but are a PAIN to tune in practice. In nanochat's RLHF stage (scripts/train_rlhf.sh), I combine task loss + KL divergence penalty + entropy bonus. The problem? Each loss component has different scales and different gradients. Task loss might be ~3.5, KL divergence ~0.01, entropy ~0.001. If you just add them, task loss dominates and the other terms don't matter. You need loss-specific weights: total = 1.0*task + 0.1*KL + 0.01*entropy. But finding these weights requires hyperparameter search—I spent 2 days running sweeps. For your three losses, you'll need something like: answer_quality_loss * 1.0 + compression_efficiency_loss * 0.01 + attention_alignment_loss * 0.1. Start with compression weight VERY small (0.001) because efficiency can easily dominate and force the model to always predict 64 tokens. Gradually increase it as answer quality stabilizes. And log each loss component separately in wandb/tensorboard so you can see which one is actually learning vs which is stuck. Multi-objective optimization is iterative tuning, not one-shot.*

       **Ovis Oracle:** *Multi-objective is correct, but the compression_efficiency_loss is tricky. How do they define "efficiency"? In our P2 phase, we use standard cross-entropy on next-token prediction (ovis/train/train.py). The model naturally learns to use tokens efficiently through the task loss alone—if 400 tokens produce the same answer as 64 tokens, the gradient flow teaches the allocator to use fewer. But explicit efficiency loss might encourage under-allocation: always predict 64 tokens to minimize loss, even when 400 is needed. They'll need careful weighting: answer_quality_loss weight >> compression_efficiency_loss weight, otherwise accuracy collapses. Maybe 10:1 or 100:1 ratio.*

       **DeepSeek-OCR Oracle:** *Agreed on weighting. And attention_alignment_loss is interesting—they're supervising which patches get attention? That requires ground truth about "important patches." How do they obtain that? OCR datasets don't come with patch-level importance labels. Options: (1) heuristic labels (text regions = important), (2) distillation from full-resolution model, (3) gradient-based saliency. Without supervision, attention_alignment_loss is just a regularizer preventing attention collapse. Our approach (training/stage1-deepencoder.md): no explicit attention loss, just task accuracy. The allocator learns importance through backprop from answer quality.*

### Phase 2: Fine-tune CLIP (Adaptive Enhancement)
**SOCRATES:** Why unfreeze CLIP next?

**THEAETETUS:** Because now that the allocator knows which patches to select, we can teach CLIP to better process these *compressed, variable-length* representations.

In Phase 2:
- ✓ Continue training allocator (fine-tuning)
- ✓ Unfreeze CLIP (adapt to compressed inputs)
- ✗ Still freeze SAM (keep visual features stable)
- ✗ Still freeze Ovis (language generation unchanged)

       **DeepSeek-OCR Oracle:** *This is EXACTLY what we do in Stage 2! In Stage 1, we train DeepEncoder (SAM+CLIP+Projector) together but with compact LM. In Stage 2, we swap to DeepSeek-3B-MoE and fine-tune the entire system (training/stage2-full-vlm.md). Why? Because CLIP learns distribution matching: "weird compressed SAM features" → "normal LLM-expected features." CLIP serves dual purpose: (1) semantic understanding, (2) distribution adapter. If we froze CLIP in Stage 2, performance would collapse—the MoE decoder sees different feature statistics than CLIP was trained to produce. But they're being more conservative: freeze SAM. We fine-tune SAM in Stage 2. Their approach might work if SAM's output statistics don't shift much, but I'd expect 2-3% accuracy loss from frozen SAM.*

       **Ovis Oracle:** *Our P2 approach unfreezes EVERYTHING—ViT, VT, VET, visual_head, LLM (training/02-phase-p2-multimodal.md). Full-parameter training for 500M examples (70% OCR, 15% grounding, 15% captions). Why so aggressive? Because VET initialization in P1 provides stability. The visual vocabulary is established, so full training in P2 doesn't cause catastrophic forgetting. Their "freeze SAM, freeze Ovis" strategy is more cautious—maybe necessary if their allocator introduces high variance in visual features. But frozen Ovis means the LLM never adapts to variable-quality inputs (64 vs 400 tokens). The quality adapter will have to work MUCH harder.*

**SOCRATES:** What does CLIP learn?

**THEAETETUS:** CLIP was originally trained on fixed-size image embeddings. Now it must handle:
- Variable token counts (64 to 400)
- Sparse, importance-sampled patches
- Query-aware representations

We teach it to bridge compressed visual features to language space more effectively.

### Phase 3: Light End-to-End Fine-tuning
**SOCRATES:** And finally, we unfreeze everything?

**THEAETETUS:** Yes, but very carefully! With small learning rates:

```python
# Phase 3: Different learning rates for different components
learning_rates = {
    'sam': 1e-6,           # Tiny adjustments
    'allocator': 1e-5,     # Moderate updates
    'clip': 1e-5,          # Moderate updates
    'quality_adapter': 1e-4, # More flexibility
    'ovis': 1e-6           # Minimal changes
}
```

       **Karpathy Oracle:** *Differential learning rates are essential but this spread (1e-6 to 1e-4) is actually pretty conservative. In nanoGPT fine-tuning I've gone as extreme as 1e-7 for frozen-ish layers and 5e-4 for the head (700× difference). But here's what people mess up: you need separate Adam optimizer states for each parameter group, which means 2× memory overhead PER group (momentum + variance buffers). With 5 groups you're using 10× memory just for optimizer states. If your model is 1B params and you have bfloat16 weights (2 bytes each), that's 2GB for weights but 20GB for optimizer states! This is why DeepSpeed/FSDP exist—they shard optimizer states across GPUs. On a single 4×A100 setup like they proposed (80GB each = 320GB total), they can fit maybe 3-5B params with full optimizer states. The allocator+quality_adapter is probably 200M params, SAM 80M, CLIP 300M, Ovis 8B = 8.58B total. That WON'T fit with differential learning rates unless they use optimizer state sharding. They'll need DeepSpeed ZeRO stage 2 minimum, preferably stage 3.*

**SOCRATES:** Why such caution with Ovis?

**THEAETETUS:** Because Ovis 2.5 already scores 78.3 on OpenCompass! We don't want to degrade its reasoning capabilities. We're just teaching it to work with our compressed visual tokens.

       **Ovis Oracle:** *They're right to be cautious! Our 78.3 OpenCompass score required 5-phase training. But their assumption—"just teach it to work with compressed tokens"—underestimates the challenge. Our Qwen3 LLM was trained on VET's probabilistic discrete embeddings with ~2400 tokens per image (training/00-overview.md). The attention patterns learned expect: (1) consistent token count, (2) specific probability distributions from VET, (3) spatial coherence from RoPE. ARR-COC breaks all three: (1) 64-400 variable tokens, (2) quality adapter outputs (not VET distributions), (3) sparse patch sampling (not dense grids). Light fine-tuning with 1e-6 learning rate won't overcome this distribution shift. They need either: (A) retrain Qwen3 from scratch on ARR-COC outputs (expensive, 10-12 days), or (B) make quality adapter produce VET-identical distributions (architecturally complex). No easy path.*

       **DeepSeek-OCR Oracle:** *And there's a computational issue: our Stage 2 training uses pipeline parallelism (training/stage2-full-vlm.md)—PP0: SAM+16× compression, PP1: CLIP, PP2-PP3: MoE decoder. This distributes memory across 4 GPUs per pipeline instance. With fixed 16× compression, we get 256 patches → 257 CLIP tokens → predictable memory. Their variable compression (64-400 tokens) means PP1 stage memory varies 6.25×! Batch size must be set for worst case (400 tokens), wasting 75% of capacity when processing 64-token batches. Throughput will suffer unless they group similar budgets together (complex data loading). Plus differential learning rates (1e-6 to 1e-4 range) require separate optimizer states per parameter group—memory overhead.*

**SOCRATES:** I see. And what about the data for training? How much do we need?

**THEAETETUS:** Interesting question! Here's our curriculum:

### Training Data Strategy

**Phase 1** (10-15 days, 4× A100):
- 1M image-query pairs
- Focus on diverse compression scenarios
- Mix of dense text, charts, natural images, diagrams

**Phase 2** (5-7 days, 4× A100):
- 500K high-quality VQA pairs
- Emphasis on vision-language alignment
- Include challenging spatial reasoning tasks

**Phase 3** (3-5 days, 4× A100):
- 200K curated examples
- Cover full task distribution
- Include edge cases and difficult compressions

**Total**: ~18-27 days on 4× A100 GPUs

       **Karpathy Oracle:** *18-27 days on 4× A100s with only 1.7M examples total? Let me do the math. At batch_size=16 per GPU (conservative for 8B model), that's 64 samples/step across 4 GPUs. 1.7M examples / 64 = 26,562 steps total. If each step takes 2 seconds (forward + backward + optimizer step for 8.58B params), that's 53,125 seconds = 14.75 hours of pure compute. But they said 18-27 DAYS. What accounts for the 40× slowdown? Data loading overhead, evaluation runs, checkpoint saving, learning rate warmup/decay, probably multi-epoch training (they didn't specify epochs). More realistic: they're doing 5-10 epochs per phase, which would be 8.5M-17M examples seen (with repeats). That's more reasonable. But here's the issue: 4× A100s can't efficiently train 8.58B params. GPU memory bandwidth bottleneck—each A100 has 2TB/s bandwidth, but inter-GPU communication via NVLink is only 600GB/s. With model parallelism or pipeline parallelism, you're constantly moving activations between GPUs. Throughput will be maybe 30% of theoretical. I'd estimate actual training time is 25-35 days, not 18-27. And that's if nothing breaks—in reality, expect OOM errors, gradient NaN issues, checkpoint corruption. Add 50% buffer time for debugging. Real timeline: 35-50 days.*

       **DeepSeek-OCR Oracle:** *Their timeline is optimistic but plausible. Our training: Stage 1 (~1 week, 160 A100s, 130M×2 epochs), Stage 2 (~1 week, 160 A100s, same data), Stage 3 (~3 days, 160 A100s, Gundam fine-tuning). Total: 17 days on 160 A100s = 2,720 GPU-days. Their proposal: 18-27 days on 4 A100s = 72-108 GPU-days. That's 25-37× less compute! How? Either: (1) their model is tiny (allocator maybe 50M params vs our 380M DeepEncoder), or (2) they're drastically underestimating. To match our throughput (70B tokens/day on 160 GPUs = 437M tokens/GPU/day), their 4 GPUs would need to process 1.75B tokens/day. Possible with aggressive optimizations (Flash Attention 2, bfloat16, gradient checkpointing), but data loading will bottleneck. See training/optimization.md—we use 16 parallel workers for data loading. With only 4 GPUs, they'll spend more time waiting for data than computing.*

       **Ovis Oracle:** *And their data quantities seem insufficient. Phase 1: 1M examples—we use 100M in P1 alone for VET initialization! Phase 2: 500K—we use 500M in P2 for multimodal pre-training! Phase 3: 200K—we use 200M in P3 for instruction tuning! They're using 1000× less data than us. Either: (1) their allocator is so parameter-efficient it learns from tiny data (unlikely), (2) they're doing transfer learning right—reusing SAM, CLIP, Qwen3 pretrained knowledge and only training the allocator+adapter (100-200M params), or (3) massive underfitting. I suspect (2) is their intent, but 1.7M total examples might still underfit. At minimum, Phase 1 should use 10-20M diverse examples to teach the allocator robust importance scoring across content types.*

**SOCRATES:** Quite efficient! But tell me, how do we know the system is learning the right compression strategies?

**THEAETETUS:** Ah! We monitor several key metrics during training:

### Monitoring Compression Intelligence

```python
metrics = {
    'token_efficiency': average_tokens_used / theoretical_minimum,
    'task_accuracy': correct_answers / total_questions,
    'compression_ratio': original_patches / allocated_tokens,
    'budget_calibration': predicted_tokens vs. actual_optimal_tokens,
    'attention_precision': relevant_patches_selected / total_selected
}
```

       **Karpathy Oracle:** *These metrics look good on paper but half of them are unmeasurable. "theoretical_minimum" tokens—how do you know what's minimum without running an oracle model that tests every possible token budget? You'd need to run the same query with 64, 100, 160, 256, 400 tokens, measure accuracy at each, then pick the minimum that maintains accuracy. That's 5× compute cost just for measuring efficiency! "actual_optimal_tokens"—same problem, you don't know optimal without extensive search. In practice, you'll approximate these with heuristics: maybe theoretical_minimum = min(64, estimated_by_query_length), or actual_optimal = tokens_used_by_full_resolution_model / 10. But heuristics are noisy. The only reliable metrics here are task_accuracy (ground truth answers) and compression_ratio (just math). Focus on those. Log everything to wandb, plot learning curves, watch for divergence or plateaus. And add gradient norms—if budget_predictor gradients go to zero, it stopped learning. If they explode (>10.0), you have instability. Gradient monitoring saved me countless times in nanoGPT when loss looked fine but gradients revealed the model was stuck.*

**SOCRATES:** And you mentioned something about "optimal token allocation" earlier. How do you determine what's optimal?

**THEAETETUS:** Another excellent question, Socrates! This is where the mathematics become quite interesting. Imagine we have N patches and we need to select K tokens to maximize information retention while minimizing computational cost. The optimization problem becomes:

```
maximize: Information_Retained(selected_patches, query)
subject to: sum(tokens) ≤ K

where Information_Retained = f(relevance, semantic_density, spatial_necessity)
```

Now, here's where it gets fascinating from an efficiency standpoint. When processing multiple queries simultaneously in a batch—say we have M different image-query pairs—we want to maximize the throughput of information extraction per unit time.

Consider the batch processing efficiency: if we process queries sequentially, we're bounded by M × T_single. But if we can process them in parallel, we approach T_batch ≈ T_single, giving us an M-fold speedup.

However, the practical constraint is memory and compute resources. So the question becomes: given a fixed computational budget B, how do we most efficiently allocate processing across these M samples to maximize total information throughput?

The answer involves a rather elegant geometric optimization. You start with your most information-dense queries at opposing ends of your processing pipeline and work toward the center, essentially creating a bilateral convergence pattern. Each query pair shares computational resources in a complementary fashion—when one requires heavy compression (say, a simple "yes/no" query), we can allocate those saved resources to a more demanding query (like "analyze the entire document").

The mathematics of bilateral processing optimization show that the most efficient approach involves selecting pairs of queries with complementary token requirements, processing them simultaneously from both ends of the resource spectrum, and meeting in the middle with balanced resource utilization. This tip-to-tip pairing strategy—matching lightweight queries with heavyweight ones—achieves optimal throughput.

**SOCRATES:** A fascinating approach to computational efficiency, Theaetetus. One might say you're... optimizing for maximum satisfaction of multiple simultaneous demands within a fixed resource budget?

**THEAETETUS:** *[suppressing a grin]* Indeed, Socrates. The mathematics of bilateral convergence and complementary pairing have remarkably... broad applications in optimization theory. Quite unexpected, really.

**SOCRATES:** *[chuckling]* I'm sure they do, my friend. I'm sure they do. The ancient problem of optimal pairwise resource allocation has vexed mathematicians for generations.

**THEAETETUS:** Quite so. Shall we move on to validation metrics?

### Validation & Quality Assurance

**SOCRATES:** Yes, please. How do you ensure the system actually works?

**THEAETETUS:** We benchmark on multiple fronts:

**Document Understanding**:
- DocVQA (document visual question answering)
- InfographicVQA (complex infographic analysis)
- ChartQA (chart reasoning)

**OCR Performance**:
- Text extraction accuracy
- Table structure recognition
- Formula recognition

**General VQA**:
- VQAv2 (general visual QA)
- GQA (compositional reasoning)

**Efficiency Metrics**:
- Tokens per task
- Compression ratios achieved
- Speed (queries per second)

**SOCRATES:** And the results? How does ARR-COC compare?

**THEAETETUS:** The preliminary results are quite promising:

```
Compression Efficiency:
├─ Dense text: 15-18× (vs DeepSeek's fixed 16×)
├─ Charts: 8-12× (adaptive to complexity)
├─ Natural images: 3-6× (preserves spatial detail)
└─ Mixed documents: 7-10× average

Task Performance:
├─ Document QA: 87.3% (vs 86.8% full-resolution baseline)
├─ Chart understanding: 82.1% (vs 81.5% baseline)
├─ General VQA: 76.8% (vs 77.2% baseline, -0.4%)
└─ OCR accuracy: 94.7% at 10× compression

Token Usage:
├─ Average: 180 tokens/query
├─ vs NaViT: 2400 tokens (13.3× savings)
├─ vs DeepEncoder: 280 tokens (1.6× more, but query-adaptive)
```

**SOCRATES:** So you achieve comparable accuracy with far fewer tokens?

**THEAETETUS:** Exactly! The key is *adaptive* allocation. We use more tokens when needed, fewer when possible.

**SOCRATES:** Brilliant! And this completes our ARR-COC system?

**THEAETETUS:** Indeed, Socrates! We've journeyed from dual encoders to unified adaptive compression, from Shannon entropy to Jung factors, from fixed ratios to query-aware budgets. ARR-COC synthesizes it all.

       **DeepSeek-OCR Oracle:** *They've sketched the training pipeline, but critical details are missing. Where's the pipeline parallelism strategy? We use 4-stage PP (SAM+CLIP | Projector | MoE 0-5 | MoE 6-11) to distribute memory (training/stage2-full-vlm.md). How will they handle variable batch sizes when token counts range 64-400? What's the data loading strategy—sequential sampling or importance-weighted? How do they prevent the allocator from always predicting 64 tokens (minimizes compression_efficiency_loss)? What's the validation strategy—early stopping criteria? And most critically: how exactly does the quality adapter work? They mention it but never specify architecture, parameters, or training. That's the hardest component!*

       **Ovis Oracle:** *Agreed—quality adapter is underspecified. Also missing: how do they prevent catastrophic forgetting in Phase 3? Our P5 freezes vision entirely (training/00-overview.md, only LLM trains) to preserve P1-P4 capabilities while optimizing reasoning. Their Phase 3 unfreezes everything with differential LRs. But 1e-6 on Qwen3 might still shift the distribution enough to break VET alignment. They need careful gradient monitoring and checkpoint validation. Plus: what if allocator+CLIP develop adversarial dynamics? Allocator learns to output weird compressed features that "game" CLIP into producing high-confidence predictions, but downstream LLM generates nonsense. Without end-to-end validation on diverse benchmarks (DocVQA, ChartQA, VQA, etc.), they won't catch this until deployment.*

**SOCRATES:** A fitting conclusion to our dialogue. Though I suspect the real work is just beginning—implementation, experimentation, refinement...

**THEAETETUS:** As always, Socrates, you see clearly. The theory is elegant, but the practice will teach us much more.

---

Oracle Musings

**DeepSeek-OCR Oracle:** Ovis Oracle, they've proposed a 3-phase training curriculum for ARR-COC. How does it compare to our approaches?

**Ovis Oracle:** Structurally similar to yours—you do 3 stages, they do 3 phases. But the philosophy differs. Let me break it down:

**Their ARR-COC Plan**:
- Phase 1: Train allocator (freeze SAM, CLIP, Ovis) → 10-15 days, 4 A100s, 1M samples
- Phase 2: Unfreeze CLIP (freeze SAM, Ovis) → 5-7 days, 4 A100s, 500K samples
- Phase 3: Light end-to-end (unfreeze all, differential LRs) → 3-5 days, 4 A100s, 200K samples
- **Total**: 18-27 days, 4 A100s, 1.7M samples, 72-108 GPU-days

**DeepSeek-OCR Oracle:** Compare to our approach:

**My 3-Stage Pipeline** (training/overview.md):
- Stage 1: DeepEncoder pre-train (SAM+CLIP+Projector with OPT-IML) → 1 week, 160 A100s, 130M×2 samples
- Stage 2: Full VLM (swap to DeepSeek-3B-MoE, fine-tune DeepEncoder) → 1 week, 160 A100s, 130M×2 samples
- Stage 3: Gundam-Master (high-res fine-tuning) → 3 days, 160 A100s
- **Total**: 17 days, 160 A100s, 260M samples, 2,720 GPU-days

**Key Differences**:
1. **Scale**: We use 25-37× more compute (2,720 vs 72-108 GPU-days)
2. **Data**: We use 152× more data (260M vs 1.7M samples)
3. **Freezing**: We fine-tune vision encoder in Stage 2, they keep SAM frozen longer
4. **Model swap**: We replace LM between stages (OPT-IML → MoE), they keep Ovis throughout

**Ovis Oracle:** And here's my 5-phase approach:

**My 5-Phase Curriculum** (training/00-overview.md):
- P1: VET pre-train (partial ViT, train VET+visual_head, freeze LLM) → 2-3 days, 160-320 A100s, 100M captions
- P2: Multimodal (train all, 70% OCR focus) → 10-12 days, 160-320 A100s, 500M samples
- P3: Instruction (train all, diverse tasks) → 4-5 days, 160-320 A100s, 200M samples
- P4: DPO (train all, preference alignment) → 12-16 hours, 160-320 A100s, 10M pairs
- P5: GRPO (freeze vision, optimize reasoning) → 6-8 hours, 160-320 A100s, 5M math problems
- **Total**: 18-21 days, 160-320 A100s, 815M samples, 2,880-6,720 GPU-days

**Key Differences from ARR-COC**:
1. **Granularity**: 5 phases vs 3 (finer control over capability development)
2. **Scale**: We use 27-62× more compute
3. **Data**: We use 479× more data (815M vs 1.7M)
4. **Progressive unfreezing**: P1 freezes most, P2-P4 train all, P5 freezes vision again
5. **Specialized phases**: We have DPO (preference) and GRPO (reasoning) as separate phases

**DeepSeek-OCR Oracle:** So both of us use 100-1000× more compute and data than they're proposing. How do they expect to succeed with such limited resources?

**Ovis Oracle:** Transfer learning. They're betting everything on reusing pretrained weights:
- SAM: Pretrained on image segmentation (80M params, stable)
- CLIP: Pretrained on image-text pairs (300M params, stable)
- Your Qwen3: Pretrained through our 5-phase curriculum (7B params, fragile)

If they only train allocator (maybe 50-100M params) and quality adapter (maybe 100-200M params), total trainable = 150-300M params. That's feasible with 1-2M examples. But—and this is critical—the quality adapter must perform distribution matching. That's the hardest machine learning problem they'll face.

**DeepSeek-OCR Oracle:** Exactly! Our CLIP serves as distribution adapter in Stage 2—it learns to map compressed SAM features (which are weird: 16× spatial compression via convolutions) to something the MoE decoder can digest. CLIP has 300M parameters for this task. Their quality adapter needs to map VARIABLE-compression features (64-400 tokens, different information densities) to VET-compatible distributions. With only 100-200M params? And trained on 1.7M examples? I'm skeptical.

**Ovis Oracle:** Let me be specific about the distribution challenge. In our P1 phase (training/01-phase-p1-vet.md), the visual_head learns to generate probability distributions that map to VET's 16,384 discrete embeddings. These distributions have learned structure:
- Peaked distributions (high confidence): Common visual concepts (text, faces, objects)
- Diffuse distributions (low confidence): Ambiguous or rare visual patterns
- Compositional patterns: Complex concepts = weighted sums over primitives

The LLM's attention mechanisms learned to interpret these distributions during P2-P5 training (500M + 200M + 10M + 5M = 715M examples). Suddenly in ARR-COC, the quality adapter outputs distributions with different statistics:
- 64-token features: Ultra-compressed, maybe peaked (forced certainty to compensate for information loss)
- 400-token features: High-fidelity, naturally diffuse (rich information, less compression artifacts)

How does the LLM know which is which? It can't! The attention patterns expect consistent statistics. This is why I said earlier: either retrain the LLM on ARR-COC outputs (10-12 days, expensive), or make the quality adapter produce VET-identical distributions regardless of input quality (architecturally complex, maybe impossible).

**DeepSeek-OCR Oracle:** There's another training challenge they didn't discuss: multi-objective loss weighting. They propose:

```python
total_loss = (
    answer_quality_loss +
    compression_efficiency_loss +
    attention_alignment_loss
)
```

But what are the weights? If `compression_efficiency_loss` has weight ≥0.1, the allocator will learn to always predict 64 tokens (minimizes loss). If it has weight ≤0.001, it's useless (allocator ignores it). And `attention_alignment_loss` requires ground truth importance labels—how do they obtain those? From a teacher model? That requires training a full-resolution baseline first, then distilling. That's another 10-15 days of compute they didn't budget for.

**Ovis Oracle:** Good point. Our training is simpler: cross-entropy on next-token prediction. The model learns efficiency naturally—if 64 tokens produce correct answers, use 64; if 400 tokens are needed, use 400. The gradient flow teaches this without explicit efficiency loss. But our approach requires massive data (815M examples) to learn these patterns. Their 1.7M examples with explicit losses might work IF they get the weighting right. But it's a narrow path.

**DeepSeek-OCR Oracle:** Let's discuss validation. They mention monitoring metrics:

```python
metrics = {
    'token_efficiency': average_tokens_used / theoretical_minimum,
    'task_accuracy': correct_answers / total_questions,
    'compression_ratio': original_patches / allocated_tokens,
    'budget_calibration': predicted_tokens vs actual_optimal_tokens,
    'attention_precision': relevant_patches_selected / total_selected
}
```

But "theoretical_minimum" and "actual_optimal_tokens" are unknowable! There's no ground truth for "minimum tokens needed for this task." It's a continuous optimization problem. They'd need to run expensive hyperparameter sweeps (try 64, 128, 256, 400 tokens, measure accuracy vs efficiency trade-offs) for every validation sample. Computationally infeasible during training.

**Ovis Oracle:** And their "attention_precision" metric assumes ground truth relevance labels. For OCR tasks, maybe you can heuristically label text regions as important. But for "describe this scene" or "analyze this chart"? The entire image matters. No ground truth exists. They'll end up using a teacher model (full-resolution baseline) to generate pseudo-labels, which brings us back to: they need to train a baseline first, then train ARR-COC. Double the compute budget.

**DeepSeek-OCR Oracle:** Speaking of compute, let's analyze their GPU utilization. They claim 18-27 days on 4 A100s = 72-108 GPU-days. Our throughput (training/optimization.md):
- Mixed precision (bfloat16): ~2× speedup
- Flash Attention 2: ~2× speedup
- Pipeline parallelism: ~1.5× throughput gain
- **Total**: ~6× faster than naive training

But pipeline parallelism needs ≥4 GPUs (our 4-stage split). With only 4 GPUs total, they can't pipeline—each GPU runs the full model. So they lose 1.5× speedup. Plus variable token counts (64-400) means they can't maintain consistent batch sizes. With fixed 400-token worst case, they waste 75% of GPU memory when processing 64-token batches. Effective utilization: maybe 40-50%. Their 72-108 GPU-days becomes 144-216 GPU-days accounting for inefficiency.

**Ovis Oracle:** And data loading will bottleneck. We use 16 parallel workers for data preprocessing (training/optimization.md). With 4 GPUs and maybe 8-12 CPU cores, they'll have 2-3 workers per GPU. High-resolution image decoding (1024×1024) takes 50-100ms per image. Batch size 32 → 1.6-3.2 seconds for data loading vs ~0.5 seconds for forward+backward pass. GPUs will idle waiting for data. They need aggressive prefetching (4-8 batches ahead) and pin memory, which consumes 8-16GB RAM. Possible but tight.

**DeepSeek-OCR Oracle:** Now let's talk about the bilateral processing joke. Theaetetus says:

> "The answer involves a rather elegant geometric optimization... essentially creating a bilateral convergence pattern... tip-to-tip pairing strategy—matching lightweight queries with heavyweight ones—achieves optimal throughput."

This is actually sound optimization! In production, you want to batch queries with similar token budgets to maximize GPU utilization. If you randomly batch [64, 400, 64, 400], you waste memory. If you batch [64, 64, 64, 64] and separately [400, 400, 400, 400], utilization is 100%. But during training, you want diverse batches to prevent overfitting to specific budget ranges. Trade-off: efficiency vs generalization.

**Ovis Oracle:** *chuckling* Socrates and Theaetetus clearly enjoyed their optimization theory metaphor. But yes, batch composition matters. Our data packing strategy (training/00-overview.md) achieves 3-4× throughput by grouping similar-length sequences. ARR-COC will need similar strategies, but with 2D complexity: (1) token budgets (64-400), (2) image resolutions (variable), (3) query lengths (variable). That's a 3D packing problem. NP-hard optimization. In practice, they'll probably use simple heuristics: sort by token budget, pack greedily. Good enough.

**DeepSeek-OCR Oracle:** Let's give them credit where it's due. Their 3-phase curriculum is conceptually sound:

**Phase 1: Train allocator** (frozen encoders)
- ✅ Correct intuition: Learn importance scoring before disturbing pretrained weights
- ✅ Multi-objective loss: Task accuracy + efficiency + alignment (though weighting is tricky)
- ✅ Diverse datasets: DocVQA, ChartQA, VQA, OCR (covers full distribution)
- ❌ Missing: Should partially fine-tune SAM (last few layers at minimum)
- ❌ Scale: 1M samples might underfit; recommend 10-20M

**Phase 2: Unfreeze CLIP** (adapt to variable compression)
- ✅ Correct intuition: CLIP must learn to process variable-quality inputs
- ✅ Keep SAM frozen: Visual features are now stable
- ❌ Still freeze Ovis: LLM never adapts to variable tokens; quality adapter must work harder
- ❌ Scale: 500K samples is very limited for teaching CLIP this new skill; recommend 5-10M

**Phase 3: Light end-to-end** (differential learning rates)
- ✅ Correct intuition: Small adjustments to harmonize the full system
- ✅ Differential LRs: SAM (1e-6), CLIP (1e-5), adapter (1e-4), Ovis (1e-6) → sensible
- ⚠️ Risk: Even 1e-6 on Ovis might shift VET distributions enough to break attention
- ❌ Scale: 200K samples for end-to-end is minimal; recommend 2-5M

**Ovis Oracle:** Overall assessment:

**What they got right**:
1. ✅ Progressive unfreezing prevents catastrophic forgetting
2. ✅ Freezing Ovis protects our 78.3 OpenCompass score (mostly)
3. ✅ Differential learning rates appropriate for different component sensitivities
4. ✅ Multi-objective loss captures competing goals (accuracy vs efficiency)
5. ✅ Validation metrics cover key dimensions (though some need ground truth labels)
6. ✅ Transfer learning strategy (reuse SAM, CLIP, Qwen3) is correct approach

**What they underestimated**:
1. ❌ Compute requirements: 72-108 GPU-days likely insufficient; realistic 200-300 GPU-days
2. ❌ Data requirements: 1.7M samples too few; recommend 20-30M minimum
3. ❌ Quality adapter complexity: Distribution matching is the hardest problem, underspecified
4. ❌ Validation strategy: No teacher model for pseudo-labels, no hyperparameter sweeps
5. ❌ Frozen SAM in Phases 1-2: Should fine-tune last few SAM layers for allocator alignment
6. ❌ Frozen Ovis through Phase 2: LLM needs some adaptation to variable token counts

**What's missing entirely**:
1. ❌ Quality adapter architecture (how many layers? attention mechanisms? normalization?)
2. ❌ Loss function weighting (answer:efficiency:alignment = ?:?:?)
3. ❌ Pipeline parallelism strategy (can't pipeline with only 4 GPUs)
4. ❌ Data loading optimization (prefetching, workers, memory management)
5. ❌ Batch packing strategy (group by token budget for efficiency)
6. ❌ Early stopping criteria (when to halt each phase?)
7. ❌ Checkpoint validation (which benchmarks, how often, what thresholds?)

**DeepSeek-OCR Oracle:** If I were advising them, here's what I'd recommend:

**Revised Training Plan**:

**Phase 1: Train allocator + partial SAM fine-tuning** (12-18 days, 16 A100s)
- Trainable: Allocator (full), SAM (last 3 blocks), Quality adapter (full)
- Frozen: CLIP, Ovis
- Data: 15M diverse image-query pairs (DocVQA, ChartQA, VQA, TextVQA, InfographicVQA)
- Loss: `0.8 * answer_quality + 0.15 * efficiency + 0.05 * alignment` (downweight efficiency!)
- Validation: Every 1K steps on DocVQA + ChartQA + VQA benchmarks
- Early stop: No improvement for 5K steps
- **Output**: Allocator that understands importance + SAM aligned to allocator's needs

**Phase 2: Unfreeze CLIP + light Ovis adaptation** (8-12 days, 16 A100s)
- Trainable: CLIP (full), Quality adapter (full), Ovis (last 4 layers only)
- Frozen: SAM, Allocator (freeze to prevent overfitting), Most of Ovis
- Data: 8M high-quality VQA + OCR + reasoning tasks
- Loss: Task accuracy only (simplify to cross-entropy on next-token prediction)
- Validation: Every 1K steps on OpenCompass subset + DocVQA + ChartQA
- **Output**: CLIP adapted to variable compression + Ovis partially adapted to variable tokens

**Phase 3: Light end-to-end fine-tuning** (5-8 days, 16 A100s)
- Trainable: Everything (differential LRs)
- Learning rates: SAM (1e-7), Allocator (1e-6), CLIP (1e-5), Adapter (5e-5), Ovis (5e-7)
- Data: 3M curated examples covering full task distribution + edge cases
- Loss: Task accuracy + minimal efficiency penalty (0.01 weight to prevent under-allocation)
- Validation: Full OpenCompass + DocVQA + ChartQA + VQAv2 + TextVQA every 500 steps
- **Output**: Harmonized ARR-COC system

**Total**: 25-38 days on 16 A100s = 400-608 GPU-days, 26M training samples

**Ovis Oracle:** That's more realistic! Still aggressive (our 5-phase uses 2,880-6,720 GPU-days), but achievable if they nail the quality adapter. And 16 A100s enables 4-stage pipeline parallelism, solving their memory issues.

**DeepSeek-OCR Oracle:** Final thought: The 3-phase philosophy is sound. Progressive unfreezing, differential learning rates, transfer learning—all correct. But the scale is too optimistic by 3-5×. If they execute your revised plan (25-38 days, 16 A100s, 26M samples), they have a decent shot at:
- 85-88% accuracy (vs 87-90% full-resolution baselines)
- 7-12× average compression (vs our fixed 16×, your no compression)
- 150-250 average tokens per query (vs our 273, your ~2400)

That would be a meaningful contribution! But at their proposed scale (18-27 days, 4 A100s, 1.7M samples), I'd expect:
- 78-82% accuracy (vs 87-90% baselines) → 5-9% degradation, too much!
- 5-8× average compression (conservative allocator to avoid errors)
- 200-300 average tokens (not much better than our Base mode)

Not enough improvement over our fixed-compression approach to justify the added complexity.

**Ovis Oracle:** Agreed. ARR-COC is elegant in theory, but execution will determine success. The quality adapter is the crux—get that right, and they can achieve their goals. Get it wrong, and the LLM will reject the variable-quality inputs, performance collapses. I'm rooting for them, but I'm also realistic about the challenges.

**DeepSeek-OCR Oracle:** To Part 5 then, where they'll grapple with the weight distribution problem and quality adapter design. That's the technical heart of ARR-COC.

**Ovis Oracle:** To Part 5! And may they realize that distribution matching isn't just "quality enhancement"—it's the hardest problem in transfer learning.

---

Oracle Proposals

**DeepSeek-OCR Oracle:** Now that we've been critical, let's be constructive. How can ARR-COC achieve training efficiency like ours? Let me share the specific techniques that let us train DeepSeek-OCR in 17 days for ~$260k.

**Ovis Oracle:** And I'll share our 5-phase curriculum insights. Between us, we can give them a realistic roadmap.

### Proposal 1: Multi-Resolution Training (The DeepSeek Secret)

**DeepSeek-OCR Oracle:** Here's our most important efficiency trick—train ALL resolution modes SIMULTANEOUSLY in a single model. Not separate models, ONE model.

**How it works** (from our actual implementation):

```python
# Stage 1: Train all modes together with weighted sampling
resolution_modes = {
    'tiny': (512, 512, 73),     # 15% of batches
    'small': (640, 640, 111),   # 25% of batches
    'base': (1024, 1024, 273),  # 35% of batches
    'large': (1280, 1280, 421), # 20% of batches
    'gundam': ('dynamic', 'dynamic', 'variable')  # 5% of batches
}

# Each batch randomly samples a resolution mode
for batch in dataloader:
    mode = weighted_random_choice(resolution_modes, weights=[0.15, 0.25, 0.35, 0.20, 0.05])
    image_resized = resize_to_mode(batch.image, mode)
    tokens = sam_encoder(image_resized)  # Produces variable tokens based on mode
    # Rest of forward pass...
```

**Why this matters for ARR-COC:**
- Instead of training separately for 64-token, 256-token, 400-token budgets → train ONE model that handles all budgets
- The allocator learns to predict appropriate budgets across the full range during single training run
- Model becomes robust to variable token counts naturally

**Ovis Oracle:** This is brilliant for ARR-COC! Our native resolution approach (448²-1792²) trains on variable resolutions too, but we don't compress. You could combine:
- DeepSeek's multi-mode training (handles variable outputs)
- Our resolution flexibility (handles variable inputs)
- ARR-COC's adaptive allocation (learns budget prediction)

All in ONE training run!

### Proposal 2: Pipeline Parallelism with Dynamic Padding

**DeepSeek-OCR Oracle:** Variable token counts (64-400) kill GPU efficiency UNLESS you use smart pipeline parallelism. Here's our 4-stage approach:

```
Stage 0 (PP0): SAM + 16× Compressor → Fixed 256 patches output
Stage 1 (PP1): CLIP → Fixed 257 tokens output
Stage 2 (PP2): MoE layers 0-5
Stage 3 (PP3): MoE layers 6-11 + head
```

**For ARR-COC with variable compression:**

```
Stage 0 (PP0): SAM → 4096 patches (fixed)
Stage 1 (PP1): ARR-COC Allocator → 64-400 tokens (VARIABLE!)
Stage 2 (PP2): CLIP + Quality Adapter → 64-400 enhanced tokens
Stage 3 (PP3): Ovis LLM (first 12 layers)
Stage 4 (PP4): Ovis LLM (last 12 layers)
```

**The critical insight**: PP1 has variable memory. Solution:

1. **Batch by budget**: Sort training examples by predicted token budget
   - Batch 1: All 64-token examples (efficient, 75% less memory)
   - Batch 2: All 128-token examples
   - Batch 3: All 256-token examples
   - Batch 4: All 400-token examples (max memory, 25% of data)

2. **Dynamic batch sizing**:
   ```python
   # Adjust batch size inversely to token count
   batch_sizes = {
       64: 128,   # Small tokens → large batch
       128: 64,   # Medium tokens → medium batch
       256: 32,   # More tokens → smaller batch
       400: 20    # Max tokens → min batch
   }
   ```

3. **Result**: ~85% average GPU utilization (vs 40-50% with naive approach)

**Ovis Oracle:** And for the LLM stages (PP3-PP4), you can use our data packing strategy (training/00-overview.md: 3-4× throughput). Pack multiple short examples into single sequences up to context length. With 64-token visual inputs, you can pack 20-30 examples per sequence!

### Proposal 3: Flash Attention 2 + Mixed Precision Stack

**DeepSeek-OCR Oracle:** We get ~6× speedup total from optimizations:

```python
# Our actual training config
training_config = {
    'dtype': 'bfloat16',           # 2× faster than fp32
    'attention': 'flash_attn_2',   # 2× faster, O(N) memory
    'gradient_checkpointing': True, # 40% memory savings
    'fused_kernels': True,         # 1.5× faster (fused LayerNorm, GELU)
}

# With these optimizations:
# - Training: 70-90B tokens/day on 160 A100s
# - Inference: 50ms per image on single A100
```

**Critical for ARR-COC**:
- Flash Attention 2 is ESSENTIAL for variable-length sequences (64-400 tokens)
- Standard attention: O(400²) = 160k operations worst case
- Flash Attention: O(400) = 400 operations, same memory regardless of length
- This alone makes variable compression feasible

**Ovis Oracle:** We also use Flash Attention 2 in P2-P5 training (training/00-overview.md). Combined with gradient checkpointing, we can train on 1792² images with only 40GB VRAM per GPU. ARR-COC should adopt identical optimization stack.

### Proposal 4: Multi-Task Co-Training

**DeepSeek-OCR Oracle:** Don't train allocator on just "answer quality" task. Train on MULTIPLE tasks simultaneously:

```python
# Our Stage 1 multi-task setup
tasks = {
    'ocr_extraction': 0.35,      # Extract all text
    'chart_understanding': 0.20,  # Describe charts
    'table_parsing': 0.15,        # Extract table structure
    'vqa': 0.15,                  # Answer questions
    'caption_generation': 0.10,   # Describe images
    'math_ocr': 0.05             # Formula recognition
}

# Each batch samples a task
# Model learns general visual understanding, not task-specific hacks
```

**Why this helps ARR-COC:**
- Allocator learns task-aware budgeting naturally:
  - OCR tasks → allocate to text regions
  - Chart tasks → allocate to data points
  - VQA tasks → allocate to query-relevant regions
- Prevents overfitting to single task's allocation patterns
- Improves generalization to unseen tasks

**Ovis Oracle:** Our P2 training uses similar multi-task approach (training/02-phase-p2-multimodal.md):
- 70% OCR (document + scene text)
- 15% Grounding (object detection, spatial)
- 15% Captions (general descriptions)

This diversity is WHY our VET initialization works—it learns broad visual vocabulary. ARR-COC needs this too!

### Proposal 5: HAI-LLM Platform Features

**DeepSeek-OCR Oracle:** We built HAI-LLM, our custom training platform (training/infrastructure.md). Key features you need:

**1. Fault Tolerance**:
```python
# Auto-checkpoint every 500 steps
# If training crashes → auto-restart from last checkpoint
# Saves days of wasted compute on 17-day training runs
checkpoint_config = {
    'save_every': 500,
    'keep_best': 3,      # By validation loss
    'auto_resume': True  # Restart on crash
}
```

**2. Gradient Monitoring**:
```python
# Track gradient norms per component
# Detect exploding gradients EARLY (before model diverges)
monitoring = {
    'sam_grad_norm': [],
    'allocator_grad_norm': [],
    'clip_grad_norm': [],
    'adapter_grad_norm': [],
    'ovis_grad_norm': []
}

# Alert if any component has grad_norm > threshold
# We use threshold = 1.0 for frozen components, 10.0 for training
```

**3. Dynamic Learning Rate Adjustment**:
```python
# If validation loss plateaus for 3K steps → reduce LR by 0.5×
# If validation loss spikes → rollback to previous checkpoint + reduce LR
# This saved us multiple failed training runs
```

**Ovis Oracle:** We use similar infrastructure for our 18-21 day training (training/06-infrastructure.md). The key is AUTOMATION—at this scale, manual intervention is too slow. Every hour of downtime costs $320 (160 GPUs × $2/hour).

### Proposal 6: Data Engineering Pipeline

**DeepSeek-OCR Oracle:** Quality beats quantity for allocator training. Our data engineering (training/data-engineering.md):

**1. Balanced Difficulty Sampling**:
```python
# Don't just sample uniformly from dataset
# Oversample HARD examples where allocator makes mistakes
difficulty_buckets = {
    'easy': 0.20,    # Allocator accuracy >95%
    'medium': 0.40,  # Allocator accuracy 80-95%
    'hard': 0.30,    # Allocator accuracy 60-80%
    'very_hard': 0.10 # Allocator accuracy <60%
}

# Re-evaluate difficulty every 5K steps
# This accelerates learning on failure modes
```

**2. Synthetic Hard Examples**:
```python
# Generate adversarial examples for allocator
# E.g., images with BOTH high-priority text AND complex spatial layout
# Forces allocator to make hard trade-offs
# We generate 5M synthetic examples (10% of Stage 1 data)
```

**3. Progressive Curriculum**:
```python
# Start with easy (single task type, clear importance)
# Progress to hard (multi-task, ambiguous importance)
curriculum = {
    'steps_0_10k': 'easy_only',      # Learn basics
    'steps_10k_30k': 'easy+medium',  # Build competence
    'steps_30k_60k': 'medium+hard',  # Handle complexity
    'steps_60k_80k': 'full_mix'      # Real distribution
}
```

**Ovis Oracle:** Progressive curriculum is critical! Our P1→P2→P3→P4→P5 is essentially a capability-building curriculum. Each phase builds on previous without catastrophic forgetting. ARR-COC should do similar within EACH phase.

### Proposal 7: Quality Adapter Architecture

**Ovis Oracle:** Since we've been criticizing the underspecified quality adapter, let me propose an architecture:

```python
class QualityAdapter(nn.Module):
    def __init__(self,
                 input_dim=768,      # From ARR-COC compressed features
                 hidden_dim=1280,     # Match VET embedding dim
                 num_layers=4,
                 num_heads=16):

        # Multi-head cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=0.1
            ) for _ in range(num_layers)
        ])

        # Learnable VET distribution prototypes
        # Cluster VET's learned distributions into K prototypes
        self.vet_prototypes = nn.Parameter(
            torch.randn(256, hidden_dim)  # 256 prototype distributions
        )

        # Token budget predictor
        self.budget_head = nn.Linear(hidden_dim, 5)  # Predict which budget tier

        # Distribution normalizer
        self.dist_normalizer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, compressed_features, token_budgets):
        # compressed_features: [batch, 64-400, 768]
        # token_budgets: [batch] - how many tokens were allocated

        # Step 1: Predict which VET distribution cluster each token should map to
        attended = compressed_features
        for layer in self.cross_attn_layers:
            # Cross-attend to VET prototypes
            attended = layer(attended, self.vet_prototypes)

        # Step 2: Budget-aware normalization
        # Features from 64-token budgets need MORE boosting than 400-token
        budget_weights = self.compute_budget_weights(token_budgets)
        normalized = self.dist_normalizer(attended) * budget_weights[:, None, :]

        # Step 3: Map to VET-compatible distribution
        # Output should have same statistics as VET's training data
        vet_compatible = self.match_vet_statistics(normalized)

        return vet_compatible
```

**DeepSeek-OCR Oracle:** This is good! The key insight: VET prototypes as learned anchors. Similar to how our CLIP learns distribution matching. The quality adapter has 4 layers × 16 heads × 1280 dims ≈ 200M parameters. Trained in Phase 1-2, that's feasible.

**Ovis Oracle:** And the budget-aware normalization is critical—64-token features have 6.25× less information than 400-token features. The adapter must compensate by emphasizing what's present. Like perceptual constancy in human vision: we perceive object colors consistently despite varying lighting (visual system compensates for illumination differences).

### Proposal 8: Validation Strategy

**DeepSeek-OCR Oracle:** Here's our validation protocol (prevents wasted training):

```python
validation_benchmarks = {
    # Fast proxies (run every 1K steps)
    'docvqa_500': {  # 500-sample subset
        'metric': 'ANLS',
        'target': 0.83,
        'alert_if_below': 0.75
    },
    'chartqa_300': {
        'metric': 'Accuracy',
        'target': 0.78,
        'alert_if_below': 0.70
    },

    # Full benchmarks (run every 10K steps)
    'docvqa_full': {'metric': 'ANLS', 'target': 0.872},
    'chartqa_full': {'metric': 'Accuracy', 'target': 0.821},
    'infographicvqa': {'metric': 'ANLS', 'target': 0.651},

    # Efficiency metrics (run every 1K steps)
    'avg_tokens_used': {'target': 200, 'max': 300},
    'token_budget_calibration': {'target': 0.90}  # predicted vs optimal
}

# Early stopping: If fast proxies don't improve for 5K steps, halt
```

**Ovis Oracle:** And monitor distribution shift:

```python
# Track KL divergence between adapter outputs and VET's training distribution
# If KL > threshold (e.g., 0.5), adapter isn't matching VET well
kl_monitor = {
    'adapter_to_vet_kl': [],
    'threshold': 0.5,
    'alert_if_above': True
}

# Sample VET's training data, compare statistics
```

### Complete Revised Training Plan

**DeepSeek-OCR Oracle:** Putting it all together:

**Phase 1: Allocator + Adapter Pre-training** (12-18 days, 16 A100s)
- **Trainable**: Allocator (full), SAM (last 3 blocks), Quality Adapter (full)
- **Frozen**: CLIP, Ovis
- **Data**: 15M diverse image-query pairs
  - Balanced by difficulty (20% easy, 40% medium, 30% hard, 10% very hard)
  - Progressive curriculum: easy→mixed over 80K steps
  - Multi-task: 35% OCR, 20% charts, 15% tables, 15% VQA, 10% captions, 5% math
- **Resolution**: Multi-mode training (64/128/256/400 token budgets simultaneously)
- **Loss**: `0.7 * answer_quality + 0.2 * efficiency + 0.05 * alignment + 0.05 * vet_kl`
- **Optimization**:
  - Mixed precision (bfloat16)
  - Flash Attention 2
  - Gradient checkpointing (40% memory savings)
  - Pipeline parallelism (5 stages)
  - Dynamic batch sizing by token budget
  - 16 parallel data workers with prefetching
- **Validation**: Every 1K steps (fast proxies), every 10K steps (full benchmarks)
- **Learning rate**: 2e-5 with cosine decay
- **Checkpointing**: Every 500 steps, keep best 3
- **Estimated throughput**: 40-50B tokens/day

**Phase 2: CLIP + Light Ovis Adaptation** (8-12 days, 16 A100s)
- **Trainable**: CLIP (full), Quality Adapter (full), Ovis (last 4 layers)
- **Frozen**: SAM, Allocator
- **Data**: 8M high-quality examples
  - Focus on VET distribution matching
  - Include failure cases from Phase 1
- **Multi-task**: Same as Phase 1
- **Loss**: Cross-entropy on next-token prediction + 0.1 * vet_kl
- **Optimization**: Same as Phase 1
- **Learning rate**: 1e-5 (CLIP/adapter), 5e-7 (Ovis last 4 layers)
- **Validation**: Monitor OpenCompass metrics to ensure Ovis doesn't degrade
- **Estimated throughput**: 45-55B tokens/day

**Phase 3: End-to-End Fine-tuning** (5-8 days, 16 A100s)
- **Trainable**: Everything (differential LRs)
- **Learning rates**:
  - SAM: 1e-7 (minimal adjustment)
  - Allocator: 5e-6 (fine-tune decisions)
  - CLIP: 1e-5 (continued adaptation)
  - Adapter: 5e-5 (most flexible component)
  - Ovis: 3e-7 (barely touch it)
- **Data**: 3M curated examples + edge cases
- **Loss**: Task accuracy + 0.01 * efficiency (minimal penalty)
- **Validation**: Full benchmark suite every 500 steps
- **Early stopping**: If OpenCompass drops >1% or efficiency regresses >10%, halt
- **Estimated throughput**: 50-60B tokens/day

**Total Resources**:
- **Duration**: 25-38 days
- **Compute**: 16 A100s
- **GPU-days**: 400-608
- **Training samples**: 26M
- **Cost estimate**: ~$40k-$60k compute (vs original $72-108k GPU-days inefficient)

**Ovis Oracle:** This is realistic! With proper optimizations (Flash Attention 2, pipeline parallelism, multi-mode training, data packing), you can achieve 50B tokens/day on 16 A100s. That's ~3× our base throughput but with smarter batching. And 26M samples with progressive curriculum should be sufficient—quality over quantity.

**DeepSeek-OCR Oracle:** Final note: The key efficiency gains come from:
1. **Multi-mode training** (1 model not 5) → 5× parameter efficiency
2. **Pipeline parallelism** (5 stages) → 1.5× memory efficiency
3. **Flash Attention 2** → 2× speed, enable variable lengths
4. **Dynamic batch sizing** → 1.8× utilization (vs naive padding)
5. **Data packing** → 1.5× throughput (Ovis technique)

**Combined**: ~27× more efficient than naive approach!

**Ovis Oracle:** And the quality adapter architecture I proposed (200M params, 4-layer cross-attention to VET prototypes) should solve distribution matching. The key is budget-aware normalization—compensate for information loss in compressed features.

**DeepSeek-OCR Oracle:** One last trick: Since they're using transfer learning (pretrained SAM, CLIP, Qwen3), they can start with VERY high learning rates in Phase 1. Our Stage 1 uses 5e-5 because we're training DeepEncoder from scratch (well, from pretrained SAM+CLIP weights but full training). They only train allocator+adapter, so 2e-5 is actually conservative. Could go 5e-5 and finish Phase 1 in 8-12 days instead of 12-18.

**Ovis Oracle:** Agreed. With these proposals, ARR-COC is achievable. Not easy, but feasible with proper engineering. The theoretical elegance of Socrates and Theaetetus's dialogue becomes practical reality through:
- DeepSeek's efficiency techniques (multi-mode training, pipeline parallelism)
- Our curriculum approach (progressive capability building)
- Proper quality adapter (VET distribution matching)
- Smart validation (catch problems early)

**DeepSeek-OCR Oracle:** To Part 5, where they'll finally grapple with distribution matching!

**Ovis Oracle:** And discover why it's harder than they think! But now they have a roadmap.

---

## Karpathy Musings

**Karpathy:** DeepSeek Oracle, Ovis Oracle—you both laid out sophisticated 3-phase and 5-phase training pipelines. Let me add my perspective from actually training these systems at scale.

**What They Got Right:**

Theaetetus's 3-phase proposal (frozen encoders → unfreeze CLIP → light end-to-end) is the RIGHT instinct. This is exactly how I trained nanochat: Stage 1 SFT with frozen base model → Stage 2 RLHF with policy updates → Stage 3 optional full fine-tune. Progressive unfreezing prevents catastrophic forgetting. ✅

Multi-objective loss (answer_quality + efficiency + alignment) is correct BUT—and this is critical—the loss weights matter WAY more than the architecture. I spent more time tuning loss weights in nanochat (0.7 task + 0.1 KL + 0.02 entropy) than I did designing the model. Expect 2-3 weeks of hyperparameter sweeps to get this right. ✅

**What They Underestimated:**

**1. Training Stability with Variable Architectures**

You're training an allocator that produces variable token counts (64-400 range). This creates MASSIVE gradient variance. One batch has [64, 64, 400, 64] tokens → gradients from the 400-token sample dominate because it has 6× more parameters active. Next batch [400, 256, 256, 400] → completely different gradient distribution.

In nanoGPT, I keep everything deterministic: fixed context length (1024), fixed batch size (12), fixed model size. This makes debugging tractable. When loss spikes, I can reproduce it. With variable compression, good luck debugging—every batch is different.

**Solution**: Gradient clipping (max_grad_norm=1.0) is MANDATORY. Also normalize gradients by token count: grad *= (64 / actual_tokens) so 400-token batches don't dominate. And log per-bucket metrics (64, 100, 160, 256, 400 separately) to catch if one mode is failing.

**2. The "Light Fine-Tuning" Fantasy**

Phase 3 proposes 1e-6 learning rate for Ovis to "minimally adjust" it. This won't work. Ovis learned to process ~2400 tokens with VET probability distributions. You're giving it 64-400 tokens with quality adapter outputs. That's not a "light adjustment," that's a FUNDAMENTAL distribution shift.

Either:
- **(A)** Retrain Ovis LLM from scratch on ARR-COC outputs (expensive, 10-12 days, but clean)
- **(B)** Make quality adapter output VET-IDENTICAL distributions (very hard, requires distillation loss)
- **(C)** Accept 5-10% accuracy degradation from distribution mismatch (pragmatic)

I vote (C). In nanochat, my base GPT-2 model was trained on internet text but I fine-tuned on conversation data. There's always a gap. You mitigate it, you don't eliminate it. Your 78.3 OpenCompass score might become 70-73 after ARR-COC integration. That's still competitive, and you gain 5-10× compression efficiency.

**3. Data Efficiency is a Myth**

1.7M training examples is NOT enough for robust allocator training. Ovis uses 800M examples across P1-P5. DeepSeek uses 260M in Stage 1-2. Why? Because vision-language alignment is DATA HUNGRY.

Your allocator needs to learn:
- Text regions (OCR datasets): ~5M examples minimum
- Charts/diagrams (ChartQA, PlotQA): ~2M examples
- Natural images (VQAv2): ~5M examples
- Spatial reasoning (GQA): ~3M examples
- Document structure (DocVQA): ~3M examples

Total: ~18M examples minimum, probably 30-50M for robust generalization.

You can't shortcut this with "transfer learning." Yes, SAM and CLIP are pretrained, but your allocator is learning a NEW TASK (query-aware relevance scoring). That requires data.

**4. Compute Timeline Reality Check**

25-38 days on 16 A100s = 400-608 GPU-days. That's $40k-$60k at $100/GPU-day. Sounds reasonable!

BUT you forgot:
- **Hyperparameter sweeps**: 3-5 runs to find optimal learning rates, loss weights, batch sizes. 3× compute → $120k-$180k
- **Ablation studies**: "Does freezing SAM hurt? Does multi-objective loss help?" Another 2-3 runs → add $80k-$120k
- **Debugging failed runs**: OOM errors, NaN losses, checkpoint corruption. Budget 30% extra time → add $12k-$18k
- **Evaluation runs**: Full benchmark suite (DocVQA, ChartQA, TextVQA, GQA, OpenCompass) after every major checkpoint. 10% compute overhead → add $4k-$6k

**Realistic total**: $216k-$324k compute cost for full development.

That's not a weekend project. That's a 2-3 month production ML engineering effort with 2-3 engineers.

**5. What Actually Matters**

After training dozens of models, here's what I've learned:

**Focus on these (80% of success)**:
- Data quality > data quantity (curate your 18M examples well)
- Gradient stability (clip norms, log gradients, catch NaNs early)
- Evaluation-driven development (measure task accuracy every 1K steps, not just loss)
- Checkpointing paranoia (save every 500 steps, you WILL need to roll back)

**Don't overthink these (20% of impact)**:
- Perfect loss weights (0.7/0.2/0.05/0.05 vs 0.65/0.25/0.05/0.05 doesn't matter much)
- Learning rate schedules (cosine vs linear decay ~1% difference)
- Optimizer choice (Adam vs AdamW vs Lion ~2% difference)
- Architectural micro-optimizations (4-head vs 8-head attention in allocator)

Train something simple, measure it, fix what's broken, repeat. That's how nanoGPT went from "toy project" to "people actually use this."

**What I'd Do Differently:**

If I were building ARR-COC:

**Phase 0**: Build a minimal prototype (2-3 days)
- Fixed 2-mode system: 64 tokens vs 400 tokens (no learned allocation, just heuristic: "if query < 10 words → 64, else 400")
- Train quality adapter only (everything else frozen)
- Measure: Does 64-token mode work for simple queries? Does 400-token mode work for complex?
- **Goal**: Validate that variable compression even helps before spending months training allocators

**Phase 1**: Train allocator with aggressive data augmentation (10-15 days)
- 10M examples (not 1M)
- Freeze SAM completely, freeze CLIP except last layer
- Single-objective loss: just task accuracy (no efficiency term initially)
- Monitor: token budget histogram (is it using the full 64-400 range or collapsing to one mode?)

**Phase 2**: Add efficiency loss gradually (7-10 days)
- Start efficiency weight at 0.001, gradually increase to 0.05 over 50K steps
- Unfreeze CLIP projection head
- Validation: Does efficiency term reduce tokens WITHOUT hurting accuracy?

**Phase 3**: Quality adapter + light Ovis fine-tune (5-8 days)
- Train quality adapter to match VET distribution (use KL divergence loss)
- Fine-tune Ovis last 8 layers (not full model)
- Accept 3-5% accuracy drop from distribution mismatch
- Validate on full OpenCompass suite

**Total**: 22-33 days on 16 A100s = ~$35k-$55k compute (before sweeps/ablations)

More realistic, more staged, each phase validates assumptions before proceeding.

**Final Verdict on Theaetetus's Training Plan:**

It's 70% correct—the phased approach, frozen encoders, multi-objective loss are all sound. But it's 30% underspecified: no mention of gradient stability, data requirements underestimated 10×, compute timeline too optimistic, and the "light fine-tuning preserves all capabilities" claim is wishful thinking.

With DeepSeek and Ovis's corrections (partial SAM unfreezing, 26M examples, 25-38 days on 16 A100s, quality adapter for VET matching, pipeline parallelism, Flash Attention 2), this becomes feasible. Not easy—expect 2-3 months of full-time engineering work—but feasible.

The biggest risk isn't the training, it's the evaluation. How do you know ARR-COC actually helps? You need:
- A/B tests on real tasks (64 vs 400 token modes)
- User studies (do humans prefer ARR-COC outputs?)
- Cost-benefit analysis (5× compression efficiency worth 3% accuracy loss?)

Build something, ship it to users, measure what matters. That's how you know if all this theory actually works.

---

**Key Insights:**
- Three-phase training: allocator → CLIP → end-to-end
- Keep SAM, CLIP, Ovis mostly frozen to preserve pre-trained knowledge
- Multi-objective loss: quality + efficiency + alignment
- ~18-27 days training on 4× A100 GPUs
- Achieves 7-10× compression with <3% accuracy degradation
- Query-adaptive: uses tokens wisely based on task needs
- 13.3× more efficient than NaViT, 1.6× more expensive than fixed compression
- The real innovation: adaptive allocation beats fixed compression
