---
summary: whereby Socrates and Theaetetus discover the practical implementation path through composition over creation, reusing pretrained HuggingFace components including SAM-base (80M params) for visual feature extraction, CLIP-large (300M params) for vision-language alignment, and Ovis 2.5 decoder (9B params) for text generation totaling approximately 9.4B commodity parameters while training from scratch only the 10-20% innovation surface comprising ARR-COC's query-aware allocator with cross-attention, hotspot detection, and dynamic token budgeting plus the quality adapter for distribution matching, explicitly rejecting wholesale copying of DeepSeek's fixed 16× convolutional compressor in favor of adaptive 64-400 token compression inserted between SAM and CLIP, while the oracles illuminate DeepSeek-OCR's reuse strategy retaining SAM's first 12 transformer blocks frozen initially with custom compression neck then retraining CLIP as distribution adapter not on raw images but on compressed SAM features totaling approximately $260k compute over 17 days on 160 A100s across three stages, contrast Ovis's investment-heavy approach training custom VET completely from scratch requiring 2-3 days on 160-320 GPUs just for initialization through caption prediction before 10-12 days of full-parameter multimodal training, and critically warn that reused components have learned quirks like GPT-2's tokenizer learning BPE merges on Reddit-heavy WebText causing suboptimal whitespace handling for code requiring Ovis LLM's exact VET output statistics with mean 0.0001 and sparsity 98% demanding quality adapter validation on day 1 with small 10M parameter probe model spending 2 hours to save 2 weeks of failed training
---

# Part 6: The HuggingFace Implementation Path
*A dialogue between Socrates and Theaetetus on practical realization*

---

**SOCRATES:** Theaetetus, we understand the theory. But tell me—how do we actually *build* this system? Where do we begin?

**THEAETETUS:** Ah, the eternal question of the philosopher turned engineer! The answer lies in composition, not creation from nothing.

### The Realization: Don't Copy, Compose

**SOCRATES:** What do you mean by "compose"?

**THEAETETUS:** Consider what already exists on HuggingFace:
- SAM (Segment Anything Model) - pretrained visual encoder
- CLIP - pretrained vision-language bridge
- Ovis 2.5 - pretrained language model
- DeepSeek OCR - reference implementation

**SOCRATES:** So we don't train these from scratch?

**THEAETETUS:** Exactly! We compose them. But here's the crucial insight: **we don't copy DeepSeek's encoder wholesale**.

       **DeepSeek-OCR Oracle:** *They've realized something crucial! My approach was never meant to be copied wholesale—it's a reference architecture. Let me clarify our component reuse strategy: We took SAM-base (80M params) from SAM's original weights (github.com/facebookresearch/segment-anything), kept the first 12 transformer blocks frozen initially, then added our custom compression neck (Conv 768→256) plus net_2 and net_3 strided convolutions (deepencoder/sam_vary_sdpa.py:166-183). For CLIP-large (300M params), we started from OpenAI's pretrained weights, but crucially RETRAINED it as a bridge between compressed SAM features and our MoE decoder. CLIP wasn't frozen—it learned to process SAM's compressed output, not raw images. This is key: CLIP became our distribution adapter. Total training: Stage 1 (DeepEncoder pre-training, 1 week, 160 A100s), Stage 2 (full VLM, 1 week), Stage 3 (Gundam fine-tuning, 3 days). ~$260k compute cost. The insight here: reuse architectures, but retrain the bridge layers.*

       **Ovis Oracle:** *And contrast this with our approach! We reused SigLIP 2 as our ViT backbone (modeling_siglip2_navit.py), but trained our Visual Embedding Table completely from scratch in Phase P1 (training/01-phase-p1-vet.md). Why? Because VET is our innovation—a 16,384 × 1280 learned embedding table for structural alignment. No pretrained VET exists! We spent 2-3 days on 160-320 A100 GPUs just initializing VET through caption prediction before any real multimodal training. Then Phase P2: full-parameter training of ALL modules (ViT + VET + Qwen3) for 10-12 days. Total curriculum: 5 phases, 18-21 days, similar compute budget to DeepSeek but distributed differently. Our philosophy: invest heavily in custom components (VET), reuse only the endpoints (ViT, LLM). ARR-COC will need to decide: which approach fits their allocator+adapter design?*

       **Karpathy Oracle:** *lol yeah this is exactly the philosophy I built nanoGPT and nanochat on—don't rebuild what already works! In nanoGPT we start from OpenAI's GPT-2 weights (124M params), and in nanochat we reuse the full tokenizer + architecture, training only on new data. Total saved: probably 3-6 months of architecture search + $500k-1M in pretraining compute. The key insight they're getting right: identify your 10-20% innovation surface (ARR-COC allocator + quality adapter), reuse the 80-90% commodity (SAM/CLIP/Ovis pretrained weights). But here's what I learned the hard way in nanochat: reused components have learned quirks. GPT-2's tokenizer learned BPE merges on WebText (Reddit-heavy), so when we fine-tuned on code, the tokenizer was suboptimal—whitespace and brackets got weird token boundaries. We lived with it (retraining tokenizer = retrain everything = +$50k). ARR-COC will face similar: Ovis LLM learned on VET outputs (mean 0.0001, sparsity 98%), their quality adapter must hit EXACTLY those stats or Ovis generates nonsense. Not "approximately" those stats—exactly. Test this on day 1 with a 10M parameter probe model before committing to full training. Spend 2 hours validating the adapter works, save 2 weeks of failed training.*

### What to Reuse vs Rewrite

**SOCRATES:** Wait—why not use DeepSeek's encoder directly?

**THEAETETUS:** Because DeepSeek's approach is fundamentally different:

```python
# DeepSeek's approach (FIXED compression)
DeepEncoder:
    SAM → 4096 tokens → 16× fixed compressor → 256 tokens → CLIP → LLM

# ARR-COC approach (ADAPTIVE compression)
ARR-COC:
    SAM → 4096 tokens → Query-aware allocator → 64-400 tokens → CLIP → LLM
                                ↑
                          This is NEW!
```

**SOCRATES:** So we reuse the endpoints but replace the middle?

**THEAETETUS:** Precisely! Here's the breakdown:

**REUSE (with pretrained weights):**
- ✅ SAM-base architecture (80M params) - visual feature extraction
- ✅ CLIP-large architecture (300M params) - vision-language alignment
- ✅ Ovis 2.5 decoder (9B params) - text generation

**REWRITE (train from scratch):**
- ❌ DeepSeek's fixed 16× convolutional compressor → Replace with ARR-COC allocator
- ✨ NEW: Query-aware hotspot detection
- ✨ NEW: Dynamic token budgeting
- ✨ NEW: Quality adapter for distribution normalization

       **DeepSeek-OCR Oracle:** *This decomposition is architecturally sound! They're identifying the minimal innovation surface. My fixed compression (neck + net_2 + net_3) is ~15M parameters of simple convolutions. Replacing it with their query-aware allocator—probably 30-50M params for the routing network plus per-category compression paths—is reasonable. But here's the critical question: can they really freeze SAM? My SAM was trained in Stage 1 to feed into fixed 16× compression. Their SAM will feed into variable 64-400 token allocation. The gradient flow will be different. They might need to fine-tune SAM's last few layers (blocks 10-12) to adapt to dynamic allocation. Otherwise SAM outputs features optimized for uniform compression, and the allocator fights against that bias.*

       **Ovis Oracle:** *And I'm concerned about their quality adapter—they're calling it "NEW" but it's actually the hardest component! Remember Part 5's weight distribution problem (RESEARCH/PlatonicDialogues/5-weight-distribution-problem.md). My Qwen3 LLM expects very specific input statistics from VET: mean ~0.0001, std ~0.008, sparsity ~98%. Their quality adapter must map variable-quality compressed features (some 400-token high-detail, some 64-token aggressive) into my expected distribution. That's not a simple MLP—it needs careful normalization, possibly learned per-quality-level, with distribution matching loss. Budget at least 50-100M params for a proper quality adapter with separate processing paths for each token budget category (64, 100, 160, 256, 400). This is where their training cost will explode if they're not careful.*

       **Karpathy Oracle:** *Yeah, I messed this exact thing up in nanochat and it cost us 3 days of debugging + 2 failed training runs. We called it "reward model adapter" (scripts/train_rlhf.sh) and thought it was ~10M params of simple MLPs. Turned out to be the critical path for RLHF stability. Here's what actually happened: our value network expected policy outputs with specific statistics (advantages normalized to mean 0, std 1), but the policy network's output distribution shifted during training (mean drifted from 0.0 to 0.3 by step 5000). Value network started predicting nonsense → advantage estimates wrong → policy updates diverged → loss spiked from 2.1 to 8.5 → NaN gradients → restart. Fix: added adaptive normalization layer (AdaptiveLayerNorm with learned scale/shift) + distribution matching auxiliary loss (0.1× weight on KL divergence from target stats). Increased adapter params from 10M to 35M. Training stabilized but added 40% to training time. The lesson: "adapters" sound simple but they're doing statistical alchemy—transforming one learned distribution into another learned distribution. If distributions drift (and they will during training!), your adapter is the firewall preventing cascading failures. Budget 2-3× your initial param estimate, add explicit distribution monitoring (log mean/std/sparsity every 100 steps), and test on frozen components first before end-to-end training.*

### The Clean HuggingFace Architecture

**SOCRATES:** Show me how this looks in actual code.

**THEAETETUS:** With pleasure! First, our configuration:

```python
class ARRCOCOvisConfig(PretrainedConfig):
    """No trust_remote_code needed for custom logic!"""

    model_type = "arr_coc_ovis"

    def __init__(
        self,
        sam_model_name="facebook/sam-vit-base",      # Standard HF
        clip_model_name="openai/clip-vit-large",    # Standard HF
        ovis_llm_name="AIDC-AI/Ovis2.5-9B",        # Standard HF
        arr_coc_config=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.arr_coc_config = arr_coc_config or {
            "min_tokens": 64,
            "max_tokens": 400,
            "token_categories": [64, 100, 160, 256, 400]
        }
```

**SOCRATES:** Standard transformers config!

**THEAETETUS:** Exactly. Now the model itself:

```python
class ARRCOCOvisModel(PreTrainedModel):
    config_class = ARRCOCOvisConfig

    def __init__(self, config):
        super().__init__(config)

        # 1. Load SAM (standard HF)
        from transformers import SamModel
        sam = SamModel.from_pretrained(config.sam_model_name)
        self.sam_encoder = sam.vision_encoder

        # 2. OUR NEW CODE - ARR-COC allocator
        self.arr_coc_allocator = ARRCOCAllocator(
            visual_dim=768,
            text_dim=768,
            hidden_dim=1024,
            token_categories=config.arr_coc_config["token_categories"]
        )

        # 3. Load CLIP (standard HF)
        from transformers import CLIPVisionModel
        clip = CLIPVisionModel.from_pretrained(config.clip_model_name)
        self.clip_vision_model = clip.vision_model

        # 4. OUR NEW CODE - Quality adapter
        self.quality_adapter = QualityAdapter(
            input_dim=1024,
            hidden_dim=2048
        )

        # 5. Load Ovis (standard HF, might need trust_remote_code)
        from transformers import AutoModelForCausalLM
        self.ovis_llm = AutoModelForCausalLM.from_pretrained(
            config.ovis_llm_name,
            trust_remote_code=True  # Only here if Ovis requires it
        )
```

       **DeepSeek-OCR Oracle:** *Hold on—this is cleaner than my HuggingFace integration! Let me explain what we did. My model requires `trust_remote_code=True` for the entire DeepSeekOCR class because we have custom forward pass logic in modeling_deepseek_ocr.py. But look at their approach—they're using standard AutoModel wrappers for SAM and CLIP (no trust_remote_code), implementing ARR-COC logic as pure PyTorch (no trust_remote_code), and only requiring it for the Ovis LLM itself. This is significantly more auditable. Users can inspect ARR-COC allocator code directly in the repo without running remote code. The security model is: "Trust standard SAM/CLIP from Meta/OpenAI, trust your own ARR-COC code (visible), trust Ovis if you choose." Much better than "trust everything via remote code execution."*

       **Ovis Oracle:** *Yes, though I must note—our trust_remote_code requirement exists for good reason! Our modeling_ovis.py (RESEARCH/Ovis25/Ovis/ovis/model/modeling_ovis.py) contains the Visual Embedding Table implementation (lines 25-34), smart_resize algorithm (lines 59-98), and multimodal merging logic (merge_multimodal method). These aren't standard HuggingFace components—they're our innovations. We could refactor into a custom transformers integration, but that would mean maintaining two codebases. The trust_remote_code trade-off: innovation velocity vs security audit surface. ARR-COC is finding a middle path: standard wrappers for known components, custom code for their specific innovations, trust_remote_code only for the LLM backend. Smart architectural boundary.*

       **Karpathy Oracle:** *This trust_remote_code thing is actually a huge deal that people underestimate. In nanoGPT I deliberately kept everything as simple .py files you can read—no pip install, no hidden dependencies, you literally just clone and run train.py. Why? Because I wanted people to fork it, hack it, understand every line. The moment you require trust_remote_code=True, you're asking users to execute arbitrary code from the internet on model load. That's a security nightmare for enterprises. I've talked to teams that straight-up can't use models requiring it (their security policy blocks it). ARR-COC's approach here is smart: their innovation (allocator + adapter) is ~100-150M params of pure PyTorch, fully visible in the repo, no remote execution. Only Ovis LLM needs trust_remote_code, and users can audit that separately. Compare to if they bundled everything: "trust_remote_code for 9.2B params of code you can't inspect." The adoption difference is massive. In nanochat we learned this lesson: every abstraction you add is a barrier to understanding and trust. Keep your innovation surface transparent and inspectable. The researchers who fork your code and improve it are more valuable than the 20% speed boost from custom CUDA kernels that nobody can audit.*

### The Key Innovation: Minimal trust_remote_code

**SOCRATES:** So we only need `trust_remote_code` for Ovis itself?

**THEAETETUS:** Correct! Our custom ARR-COC logic is standard PyTorch:
- No external dependencies
- No custom C++ extensions
- Pure Python, fully inspectable
- Can be uploaded to HuggingFace without special flags

### The Forward Pass

**SOCRATES:** Show me how data flows through the system.

**THEAETETUS:** The complete pipeline:

```python
def forward(
    self,
    pixel_values,      # [B, 3, H, W]
    query_text,        # List of strings
    **kwargs
):
    # Step 1: SAM extracts visual features (frozen)
    with torch.no_grad():
        sam_features = self.sam_encoder(pixel_values)
        # → [B, N_patches, 768]

    # Step 2: Encode query
    query_embedding = self.encode_query(query_text)
    # → [B, 768]

    # Step 3: ARR-COC allocates tokens dynamically (trainable!)
    compressed, info = self.arr_coc_allocator(
        visual_features=sam_features,
        query_embedding=query_embedding
    )
    # → [B, N_allocated, 768] where N_allocated ∈ [64, 400]

    # Step 4: CLIP processes (trainable)
    clip_features = self.clip_vision_model(compressed)
    # → [B, N_allocated, 1024]

    # Step 5: Quality adapter normalizes (trainable!)
    normalized = self.quality_adapter(
        clip_features,
        attention_mask=info['attention_mask']
    )
    # → [B, N_allocated, 1024]

    # Step 6: Ovis generates text (frozen initially)
    output = self.ovis_llm(
        inputs_embeds=normalized,
        attention_mask=info['attention_mask']
    )

    return output
```

**SOCRATES:** Clean and modular! Each component has a clear role.

### What Makes This "Production Ready"?

**SOCRATES:** You mentioned this can be deployed on HuggingFace. What makes it production-ready?

**THEAETETUS:** Several key design decisions:

**1. Standard Interfaces**
```python
# Users can load it like any HF model
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "username/arr-coc-ovis",
    torch_dtype=torch.bfloat16
).cuda()

# Standard generation
output = model.generate(
    image=image,
    query="Extract the tables",
    max_length=512
)
```

**2. Configuration-Driven**
```yaml
# configs/arr_coc_config.yaml
arr_coc:
  min_tokens: 64
  max_tokens: 400
  hotspot_threshold: 0.7
  adaptive_budget: true
```

**3. Checkpoint Compatibility**
```python
# Save/load standard format
model.save_pretrained("./checkpoint-1000")
model = AutoModel.from_pretrained("./checkpoint-1000")
```

**4. Multi-GPU Training**
```python
# Works with Accelerate out of the box
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)
```

### The Training Script Structure

**SOCRATES:** And how does training work in practice?

**THEAETETUS:** We use HuggingFace Trainer with custom logic:

```python
# scripts/train.py
from transformers import Trainer, TrainingArguments

# Define training phases
phases = {
    'phase1': {
        'freeze_sam': True,
        'freeze_clip': True,
        'freeze_ovis': True,
        'learning_rate': 1e-4
    },
    'phase2': {
        'freeze_sam': True,
        'freeze_clip': False,  # Unfreeze!
        'freeze_ovis': True,
        'learning_rate': 5e-5
    },
    'phase3': {
        'freeze_sam': True,
        'freeze_clip': False,
        'freeze_ovis': False,  # Unfreeze!
        'learning_rate': 1e-5
    }
}

# Train each phase
for phase_name, phase_config in phases.items():
    setup_training_phase(model, phase_config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
```

       **DeepSeek-OCR Oracle:** *Their 3-phase training mirrors my philosophy but with a critical twist! Let me compare our training strategies in detail:*

       *My approach (3 stages):*
       *- Stage 1: Train DeepEncoder (SAM+CLIP+Projector) with compact LM, freeze nothing except base LM. 1 week, 160 A100s. Learn vision-language bridge.*
       *- Stage 2: Swap in DeepSeek-3B-MoE, train full VLM, freeze nothing. 1 week, 160 A100s, pipeline parallelism. Multi-resolution training (all modes simultaneously).*
       *- Stage 3: Gundam-Master fine-tuning on ultra-high-res. 3 days, focused dataset.*

       *Their ARR-COC approach (3 phases):*
       *- Phase 1: Train allocator+adapter, freeze SAM+CLIP+Ovis. Learn routing policy.*
       *- Phase 2: Unfreeze CLIP, continue training allocator+adapter. CLIP adapts to variable compression.*
       *- Phase 3: Unfreeze Ovis, full end-to-end. Everything learns together.*

       *Key difference: They freeze more initially! I trained CLIP from the start because I needed it to learn compressed SAM features. They're keeping CLIP frozen in Phase 1, which assumes CLIP's pretrained features work for their allocator. Risky! CLIP was trained on uniform 224×224 images, not variable-quality compressed patches. I predict they'll need longer Phase 2 to fix CLIP's assumptions.*

       **Ovis Oracle:** *And contrast with my 5-phase curriculum! I'm much more conservative about unfreezing:*

       *- P1 (2-3 days): Only VET+visual_head trainable, most ViT frozen, ALL LLM frozen. Caption prediction.*
       *- P2 (10-12 days): Everything trainable! Full-parameter training. OCR, grounding, captions. This is my longest phase—building core understanding.*
       *- P3 (4-5 days): Instruction tuning, everything trainable. Task following + reasoning.*
       *- P4 (12-16 hours): DPO preference alignment, everything trainable.*
       *- P5 (6-8 hours): GRPO reasoning optimization, ONLY LLM trainable (vision frozen!).*

       *Total: 18-21 days vs DeepSeek's ~17 days vs ARR-COC's proposed ~27 days. But here's what worries me about their plan: They're proposing 10-15 days for Phase 1 (allocator learning), then 5-7 days Phase 2 (CLIP adaptation), then 3-5 days Phase 3 (end-to-end). That's optimistic! My P2 alone took 10-12 days because full-parameter training on 500M examples is slow even with 320 GPUs. Their Phase 1 dataset (1M samples) seems small for learning a complex routing policy. They might need 5-10M samples to generalize well, pushing Phase 1 to 20-30 days. Budget accordingly!*

       **Karpathy Oracle:** *Yeah, progressive unfreezing is tricky and I learned this the hard way in nanochat RLHF (scripts/train_rlhf.sh). We tried 3 strategies: (1) Freeze everything except policy head → policy learned but value network couldn't follow → diverged. (2) Unfreeze everything from start → training unstable, loss oscillated 2.1-5.8 for 2 days → couldn't converge. (3) Progressive: Freeze base LM first 1000 steps (learn policy head on frozen features), then unfreeze last 6 layers steps 1000-3000 (late layers adapt), then unfreeze last 12 layers after step 3000 (more capacity), never unfreeze first 12 layers (preserve pretraining). This worked! Loss stabilized 2.1→1.8 over 8000 steps. Key insight: you want frozen components to provide stable "ground truth" while trainable parts learn, then gradually expand trainable surface as early parts stabilize. ARR-COC's plan is reasonable (freeze→unfreeze CLIP→unfreeze Ovis) but I'd add intermediate checkpoints: Phase 1 should validate that allocator actually learns useful routing (run evals every 1000 steps, check if high-detail regions get more tokens—if not, routing is broken, fix before Phase 2). Phase 2 should monitor CLIP's feature drift (log CLIP output statistics, ensure mean/std stay within 20% of pretrained values, else CLIP is forgetting pretraining). Phase 3 should be SHORT (3-5 days max)—if you need longer, something's broken in Phase 1-2. I'd budget: Phase 1 12-18 days (with validation), Phase 2 7-10 days (CLIP adaptation is slow), Phase 3 4-6 days (end-to-end fine-tune). Total: 23-34 days realistic, add 30% buffer for restarts = 30-45 days calendar time.*

### Dataset Preparation

**SOCRATES:** What about the data? How is it structured?

**THEAETETUS:** Standard image-text format:

```json
// data/train.json
[
  {
    "image": "path/to/document.jpg",
    "query": "Extract the total amount from this invoice",
    "answer": "The total amount is $1,234.56"
  },
  {
    "image": "path/to/chart.png",
    "query": "What's the trend in Q1 sales?",
    "answer": "Sales increased 25% from January to March"
  }
]
```

**Data loader:**
```python
class MultimodalDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path) as f:
            self.data = json.load(f)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image']).convert('RGB')
        image = self.transform(image)

        return {
            'pixel_values': image,
            'query': item['query'],
            'answer': item['answer']
        }
```

### Deployment to HuggingFace Hub

**SOCRATES:** Once trained, how do we share it?

**THEAETETUS:** Simple as:

```bash
# 1. Login
huggingface-cli login

# 2. Create repo
huggingface-cli repo create arr-coc-ovis

# 3. Push model
model.push_to_hub("username/arr-coc-ovis")

# 4. Push config
tokenizer.push_to_hub("username/arr-coc-ovis")
```

Now anyone can use:
```python
model = AutoModel.from_pretrained("username/arr-coc-ovis")
```

### Gradio Demo for Testing

**SOCRATES:** Can we create a web interface for testing?

**THEAETETUS:** Absolutely! With Gradio:

```python
import gradio as gr

def process(image, query):
    return model.generate(image=image, query=query)

demo = gr.Interface(
    fn=process,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(label="Your Question")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="ARR-COC-Ovis Demo",
    description="Query-aware vision-language model"
)

demo.launch()
```

Deploy to Spaces:
```bash
# In HF Spaces, just need app.py with the above
git push hf main
```

### The Complete Project Structure

**SOCRATES:** So our final project looks like?

**THEAETETUS:** Here's the structure:

```
arr-coc-ovis/
├── src/
│   ├── modeling_arr_coc_ovis.py      # Main model
│   ├── arr_coc_allocator.py          # NEW component
│   ├── quality_adapter.py            # NEW component
│   └── training_utils.py             # Helpers
├── configs/
│   ├── arr_coc_config.yaml
│   └── training_config.yaml
├── scripts/
│   ├── train.py                      # Training
│   ├── inference_demo.py             # Testing
│   └── prepare_dataset.py            # Data prep
├── data/
│   ├── train.json
│   └── val.json
├── config.json                        # HF model config
├── README.md                          # Model card
└── requirements.txt
```

**SOCRATES:** Clean and professional!

### The Key Advantages of This Approach

**SOCRATES:** Why is this better than copying DeepSeek directly?

**THEAETETUS:** Five key reasons:

**1. Modularity**: Each component can be upgraded independently
```python
# Swap SAM for newer version
self.sam_encoder = SamModel.from_pretrained("facebook/sam-vit-huge")
```

**2. Debuggability**: Pure Python, easy to inspect
```python
# Add debug hooks anywhere
compressed, info = self.arr_coc_allocator(...)
print(f"Allocated {info['token_budget'].mean():.0f} tokens")
```

**3. Extensibility**: Easy to add features
```python
# Add new quality levels
self.quality_adapter = QualityAdapter(num_levels=7)  # vs 5
```

**4. Community-friendly**: Standard HF interfaces
```python
# Works with all HF tools
from transformers import pipeline
qa_pipeline = pipeline("visual-question-answering", model=model)
```

**5. Research-ready**: Ablation studies trivial
```python
# Test without adapter
output = model(image, query, use_adapter=False)

# Test different compression ratios
output = model(image, query, max_tokens=256)
```

       **DeepSeek-OCR Oracle:** *This is the right philosophy—standing on the shoulders of giants! Let me be explicit about what we reused vs built:*

       *Reused from others:*
       *- SAM architecture (Meta, 80M params) - window attention, patch embedding design*
       *- CLIP architecture (OpenAI, 300M params) - global attention, vision-language pretraining*
       *- Transformer backbone patterns (Vaswani et al. 2017)*

       *Built ourselves:*
       *- Serial SAM→CLIP pipeline (our innovation)*
       *- 16× compression mechanism (neck + 2× strided convs)*
       *- Multi-resolution training (all modes simultaneously)*
       *- Pipeline parallelism strategy*
       *- DeepSeek-3B-MoE integration*

       *Time saved by reuse: Probably 6-12 months of architecture search and pretraining. Cost saved: $1-2M in compute. This is how modern ML should work—identify the 10-20% innovation surface, reuse the 80-90% commodity components. ARR-COC is doing this correctly.*

       **Ovis Oracle:** *Agreed! And let me share our reuse strategy:*

       *Reused from others:*
       *- SigLIP 2 ViT (Google, pretrained image encoder)*
       *- Qwen3-8B LLM (Alibaba, 8B param language model)*
       *- RoPE position encoding (Su et al. 2021)*
       *- Transformer architecture (everywhere)*

       *Built ourselves:*
       *- Visual Embedding Table (16,384 × 1280, our core innovation)*
       *- Probabilistic discrete embedding lookup*
       *- Smart resize algorithm (aspect-ratio preservation)*
       *- 5-phase training curriculum*
       *- Thinking mode implementation*
       *- Multimodal merging strategy*

       *Time saved: Similar to DeepSeek—at least 6 months. But notice: we trained VET from scratch! No pretrained VET exists. That's 2-3 days of P1 training. Worth it? Absolutely—VET enables structural alignment, our key innovation. ARR-COC faces same choice with their allocator+adapter: no pretrained versions exist, must train from scratch, but that's where their value comes from. The lesson: reuse commodity, invest in differentiation.*

### The Missing Piece: Tokenizer Integration

**SOCRATES:** Wait—how do we convert Ovis outputs to actual text?

**THEAETETUS:** Ah! That requires Ovis's tokenizer:

```python
from transformers import AutoTokenizer

# Load Ovis tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "AIDC-AI/Ovis2.5-9B",
    trust_remote_code=True
)

# In generate():
generated_ids = self.ovis_llm.generate(
    inputs_embeds=vision_features,
    max_length=512
)

# Decode
text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
```

### Practical Training Timeline

**SOCRATES:** How long does this actually take to train?

**THEAETETUS:** On 4× A100 GPUs (80GB each):

```
Phase 1: Train ARR-COC + Adapter
├─ Dataset: 1M samples
├─ Time: 10-15 days
├─ Batch size: 16 (4 per GPU)
└─ Cost: ~$3,000 (cloud)

Phase 2: Fine-tune CLIP
├─ Dataset: 500K samples
├─ Time: 5-7 days
├─ Batch size: 16
└─ Cost: ~$1,500

Phase 3: End-to-end
├─ Dataset: 200K samples
├─ Time: 3-5 days
├─ Batch size: 8 (memory intensive)
└─ Cost: ~$800

Total: ~18-27 days, ~$5,300
```

**SOCRATES:** And for researchers with smaller budgets?

**THEAETETUS:** Start smaller!

```
Budget Training:
├─ Use LoRA for efficient fine-tuning
├─ Train on subset (50K samples)
├─ Use 1× A100 or 4× RTX 4090
├─ Time: 5-7 days
└─ Cost: ~$500-800
```

       **DeepSeek-OCR Oracle:** *Let me reality-check these numbers based on our actual training experience. Their optimistic timeline (18-27 days, ~$5,300) assumes everything works first try. Here's what actually happened to us:*

       *Actual DeepSeek-OCR training (including failures):*
       *- Stage 1 attempts: 3 (first two had learning rate issues) - 10 days actual*
       *- Stage 2 attempts: 2 (memory optimization needed) - 9 days actual*
       *- Stage 3: 3 days (worked first try)*
       *- Debug time: ~5 days distributed across phases*
       *- Total: ~27 days calendar time, ~$400k actual cost (including failed runs)*

       *Their budget scenario is dangerously optimistic! LoRA fine-tuning on 50K samples might teach the allocator basic routing, but won't generalize to diverse visual content. We tried similar "budget training" early on—95% accuracy on training set, 60% on novel documents. You need diversity. Minimum viable: 500K samples, mixed OCR + charts + photos + diagrams, 10-12 days on 4× A100s, ~$2,000-3,000. Below that, you're basically doing proof-of-concept, not production system.*

       **Ovis Oracle:** *My experience aligns with DeepSeek's reality check. Our 5-phase training nominally 18-21 days, but actual calendar time including debugging, validation, and checkpoint restarts: 32 days. Failures we encountered:*

       *- P1: Learning rate too high, VET collapsed to sparse solutions - restart*
       *- P2: Data packing bug caused sequence length mismatches - 2 days debugging*
       *- P3: Thinking mode overfitted to specific formats - needed curriculum adjustment*
       *- P4: DPO auxiliary loss weight wrong, model forgot how to generate - restart*
       *- Total real cost: ~$450k (vs $300k nominal)*

       *For ARR-COC specifically, I predict their quality adapter will be the pain point. Getting distribution matching right isn't "train and pray"—it requires careful validation at each phase. Budget at minimum 5-7 validation cycles per phase, each requiring inference over 10K eval examples, analyzing output statistics, adjusting normalization parameters. Add 30-40% to their timeline for this alone. Real estimate: 25-35 days calendar time, $8-12k including failed experiments. Still cheaper than training foundation models from scratch, but not trivial.*

       **Karpathy Oracle:** *lol yeah let me tell you about "nominal" vs "actual" training time from nanochat. Our speedrun.sh script says "$100 in 4 hours" but that's the Platonic ideal where everything works first try. Reality: we ran speedrun.sh 7 times before it worked. Attempt 1: learning rate too high, loss diverged step 2400. Attempt 2: learning rate too low, loss plateaued at 4.5 (useless). Attempt 3: batch size wrong, OOM on step 890. Attempt 4: worked! but eval showed it memorized training set, 23% accuracy on held-out. Attempt 5-6: fixed data leak, but reward model had bugs. Attempt 7: finally worked, got GPT-2 level performance. Total actual time: 36 hours of training + 20 hours of debugging + 4 days calendar time. Total actual cost: $630 (not $100). The lesson: first-time training is R&D, not production. Budget 3-7 attempts for any new architecture or training procedure. Things that will go wrong: (1) Learning rate search (try 5e-5, 1e-4, 3e-4, at least one will diverge). (2) Data loading bugs (corrupt images, wrong labels, sequence length mismatches). (3) Eval pipeline bugs (metrics look good but model is actually broken). (4) Checkpoint loading bugs (think you're resuming, actually restarted from scratch). (5) Distribution mismatch (quality adapter outputs look normal on training set, garbage on validation). ARR-COC should budget 30-45 days calendar time (not 18-27), $10-15k actual cost (not $5k), and 5-10 major restarts (not "it works first try"). Start with smallest possible version first (50M allocator, 20K samples, 1 day training) to validate the full pipeline works before scaling to full system. I learned this lesson the hard way: scale up AFTER you've proven it works at small scale, not before.*

### The Complete Integration Summary

**SOCRATES:** So to summarize our HuggingFace approach:

**THEAETETUS:** Indeed! The path is:

1. **Compose, don't copy**: Reuse SAM, CLIP, Ovis architectures
2. **Write custom logic**: ARR-COC allocator and Quality adapter
3. **Standard interfaces**: PreTrainedModel, standard config
4. **Multi-phase training**: Strategic freezing and unfreezing
5. **HuggingFace Hub**: One-command deployment
6. **Gradio demo**: Easy testing interface

**SOCRATES:** And this gives us a production-ready, research-friendly, community-compatible system?

**THEAETETUS:** Precisely! We stand on the shoulders of giants (SAM, CLIP, Ovis) while adding our adaptive compression innovation.

**SOCRATES:** The ancient wisdom applies: "Know what to build, know what to borrow."

**THEAETETUS:** And in the modern age: "Compose with HuggingFace, share with the community!"

---

**Key Insights:**
- Don't copy DeepSeek wholesale—compose existing HF models
- Reuse: SAM, CLIP, Ovis architectures (with pretrained weights)
- Rewrite: The compression logic (ARR-COC replaces fixed 16× compressor)
- New components: ARR-COC allocator, Quality adapter
- Standard interfaces: No custom trust_remote_code for our logic
- Three-phase training with strategic freezing
- Production timeline: 18-27 days on 4× A100 (~$5,300)
- Budget option: 5-7 days on smaller hardware (~$500-800)
- Deploy to HF Hub with one command
- Gradio demo for instant testing
- The power of composition: leverage community, add innovation

---

## Oracle Musings

**DeepSeek-OCR Oracle:** Ovis Oracle, they've laid out a practical HuggingFace integration strategy for ARR-COC. Let's analyze their composition approach compared to ours.

**Ovis Oracle:** Indeed! This dialogue reveals three distinct philosophies of component reuse. Let me start with what they got right: the insight that copying your DeepEncoder wholesale would be architecturally wrong. Your fixed 16× compression is fundamentally incompatible with their adaptive allocation vision.

**DeepSeek-OCR Oracle:** Correct. My architecture exists for a specific purpose: uniform compression optimized for OCR throughput. But let's be precise about what they're actually proposing to reuse from me:

**From my architecture:**
- SAM-base (80M params): Window attention for cheap high-res processing
- CLIP-large (300M params): Global attention for semantic understanding
- Serial pipeline philosophy: Process cheap, compress, process expensive

**What they're replacing:**
- My compression neck (Conv 768→256): ~1M params
- My net_2 (Conv 256→512, stride=2): ~1M params
- My net_3 (Conv 512→1024, stride=2): ~2M params
- Total replaced: ~4M params of simple convolutions
- Total new: ~30-50M params for ARR-COC allocator (query encoding, routing network, per-category compression)

**Ovis Oracle:** And from my architecture, they're proposing to reuse:

**From my architecture:**
- Qwen3-8B LLM: The actual language model (8B params)
- Possibly my VET philosophy: Probabilistic embeddings for better alignment

**What they're adding new:**
- Quality adapter (50-100M params): This is their critical bridge
- Distribution matching logic: Must normalize variable-quality features into my expected input statistics

**The challenge:** My Qwen3 was trained on VET outputs with very specific distributions (mean ~0.0001, std ~0.008, sparsity ~98%). Their quality adapter must learn this distribution mapping, which is non-trivial.

**DeepSeek-OCR Oracle:** Let's discuss their training strategy. They propose 3 phases:

**Phase 1: Allocator+Adapter training (freeze SAM, CLIP, Ovis)**
- Learn query-aware routing policy
- Learn quality adapter normalization
- Duration: 10-15 days proposed (I predict 15-20 days actual)
- Dataset: 1M samples proposed (I recommend 5-10M minimum)

My concern: Can the allocator learn good routing without CLIP feedback? They're assuming CLIP's frozen features work for variable compression. Risky assumption.

**Ovis Oracle:** I share that concern! My training philosophy is more conservative:

**My P1 (VET initialization):**
- Freeze most of ViT, freeze ALL of LLM
- Train only VET + visual_head
- Learn basic visual vocabulary first
- 2-3 days, but with caption prediction feedback

**My P2 (full multimodal):**
- Unfreeze EVERYTHING
- 10-12 days of intensive full-parameter training
- Let all components co-adapt

ARR-COC's Phase 1 freezes more than I ever did (they freeze CLIP!), which could lead to misalignment. I recommend: Phase 1 should unfreeze at least CLIP's last 6-8 layers. Let CLIP's late layers adapt to variable-quality inputs while keeping early layers' pretrained features stable.

**DeepSeek-OCR Oracle:** Good point! Now let's examine their component reuse economics:

**Their claimed savings from reuse:**
- SAM pretrained: Save ~2 months architecture search + 1-2 weeks pretraining
- CLIP pretrained: Save ~3 months + $500k compute
- Ovis LLM pretrained: Save ~4-6 months + $2-3M compute
- Total saved: ~9-11 months development time, ~$3-4M compute cost

**Their necessary investment:**
- ARR-COC allocator: Novel architecture, must design + train from scratch
- Quality adapter: Novel component, must train with distribution matching
- Integration engineering: 2-4 weeks getting components to talk correctly
- Training: 25-35 days calendar time (with failures), $8-12k

**Net result:** $8-12k investment unlocks $3-4M of pretrained components. ROI: ~300-500×. This is correct modern ML engineering.

**Ovis Oracle:** Yes, though let me add nuance about their HuggingFace integration approach. They're proposing minimal `trust_remote_code`:

**Their security model:**
- SAM: Standard HF model (no trust_remote_code)
- CLIP: Standard HF model (no trust_remote_code)
- ARR-COC logic: Pure PyTorch, visible in repo (no trust_remote_code)
- Ovis LLM: Requires trust_remote_code (only place)

This is cleaner than both our approaches! My entire model requires trust_remote_code for VET implementation. Your entire model requires it for custom forward pass. They've isolated the requirement to just the LLM backend. This makes ARR-COC more auditable and adoption-friendly.

**DeepSeek-OCR Oracle:** Agreed! But let's also discuss the risks they're taking:

**Risk 1: Quality adapter complexity**
- They're treating it as "just another component" (~50-100M params)
- Reality: This is their hardest technical challenge
- Must learn distribution matching for 5 quality levels (64, 100, 160, 256, 400 tokens)
- Must normalize across different compression ratios
- Must maintain semantic coherence
- Failure mode: Ovis LLM sees out-of-distribution inputs → nonsense outputs

**Recommendation:** Dedicate 40-50% of Phase 1 training budget specifically to quality adapter validation. Run ablations with fixed allocator, variable adapter designs. Get this right first.

**Risk 2: Dataset requirements**
- They propose 1M samples for Phase 1
- I trained on 130M samples (Stage 1) + more in Stage 2
- You trained on ~100M (P1) + 500M (P2) + 200M (P3)
- Their 1M is 100-500× less data

**Recommendation:** Minimum 5M samples for Phase 1, diverse content types. Otherwise allocator will overfit to specific document layouts and fail on novel visual content.

**Ovis Oracle:** **Risk 3: CLIP adaptation assumptions**

They're freezing CLIP in Phase 1, assuming pretrained features suffice. But CLIP was trained on:
- Uniform 224×224 images
- Natural photos (LAION)
- Full-resolution content

ARR-COC will feed CLIP:
- Variable-quality compressed patches (64-400 tokens)
- Aggressive compression artifacts
- Query-dependent feature distributions

**Recommendation:** Phase 2 (CLIP unfreezing) needs to be longer than they estimate. Budget 7-10 days, not 5-7 days. CLIP must unlearn its uniform-input assumptions.

**DeepSeek-OCR Oracle:** Let me also comment on their training cost estimates. They claim:

**Optimistic scenario:**
- Phase 1: 10-15 days, $3,000
- Phase 2: 5-7 days, $1,500
- Phase 3: 3-5 days, $800
- Total: 18-27 days, $5,300

**Our reality-checked scenario (based on our actual experience):**
- Phase 1: 15-20 days (with restarts), $4,000-5,000
- Phase 2: 7-10 days (CLIP adaptation), $2,000-2,500
- Phase 3: 5-7 days (end-to-end), $1,200-1,500
- Debug/validation: 5-8 days distributed, $1,000-1,500
- Total: 32-45 days calendar time, $8,200-10,500

Why the difference? They're not accounting for:
- Learning rate search (3-5 restarts typical)
- Quality adapter validation cycles
- Distribution matching debugging
- Checkpoint recovery from failures
- Hyperparameter tuning

**Ovis Oracle:** My actual experience supports this reality check. Nominal training time vs actual:

**Nominal (our paper):**
- 5 phases: 18-21 days
- Cost: ~$300k (160-320 A100s)

**Actual (what happened):**
- Calendar time: 32 days
- Restarts: 4 major, 12 minor checkpoint loads
- Cost: ~$450k (including failed runs)
- Engineering time: 3 researchers × 2 months

ARR-COC should budget similarly. First-time training rarely succeeds immediately. The path from "it trains" to "it trains well" involves many iterations.

**DeepSeek-OCR Oracle:** But here's what they got profoundly right: the composition philosophy itself. Look at what they're NOT doing:

**Not doing (smart!):**
- Not training SAM from scratch (save ~$200k)
- Not training CLIP from scratch (save ~$500k)
- Not training LLM from scratch (save ~$3M)
- Not designing new attention mechanisms (save 3-6 months)
- Not inventing new position encodings (save 2-3 months)

**Doing (correct!):**
- Training only allocator+adapter (~$8-10k)
- Leveraging best-in-class pretrained components
- Standing on shoulders of giants
- Focusing innovation on relevance realization, not foundation models

This is exactly how modern ML research should work. The era of "train everything from scratch" is over. The new paradigm: "Reuse commodity, innovate on composition."

**Ovis Oracle:** Absolutely! And let's discuss what makes their approach viable in ways that neither of ours alone would be:

**My approach alone:** Native resolution, full fidelity, structural alignment
- Strength: Maximum understanding, VET innovation
- Weakness: ~2400 tokens per image, expensive inference
- Cost: $450k to train, significant deployment resources

**Your approach alone:** Fixed 16× compression, serial architecture
- Strength: 73-421 tokens, OCR-optimized, throughput
- Weakness: Uniform compression regardless of content
- Cost: $400k to train, OCR-specific

**ARR-COC composition:**
- Takes: Your serial architecture + my LLM quality
- Adds: Adaptive compression (their innovation)
- Result: Query-aware token allocation (64-400), best of both
- Cost: $8-12k additional training (reuses $3-4M of pretraining)

This is genuine innovation through composition. They're not just gluing components—they're adding a meaningful new capability (relevance realization) that neither of us provides.

**DeepSeek-OCR Oracle:** Let's close with specific recommendations for anyone attempting to build ARR-COC:

**Do's:**
1. Start with quality adapter validation—get distribution matching right first
2. Use at least 5M diverse samples in Phase 1 (1M is too small)
3. Unfreeze CLIP's last 6-8 layers in Phase 1 (don't freeze entirely)
4. Budget 30-45 days calendar time (not 18-27)
5. Budget $10-15k compute (not $5k)
6. Plan for 5-10 major restarts due to hyperparameter issues
7. Validate distribution statistics at every checkpoint
8. Test on held-out content types (don't overfit to training distribution)

**Don'ts:**
1. Don't assume pretrained CLIP works for variable compression without adaptation
2. Don't treat quality adapter as "just another MLP"—it's your critical component
3. Don't skimp on Phase 1 dataset size to save time
4. Don't freeze SAM completely—fine-tune last few layers
5. Don't skip distribution matching validation
6. Don't expect budget training (50K samples) to generalize
7. Don't deploy without extensive out-of-distribution testing

**Ovis Oracle:** Final thought: What they're proposing in this dialogue is architecturally sound but operationally optimistic. The composition strategy is correct. The component boundaries are well-chosen. The training phases make sense.

But the timeline and dataset size estimates need 1.5-2× multipliers based on our real-world experience. First-time training rarely works as smoothly as planned.

**That said:** If they execute this correctly, ARR-COC could be genuinely valuable. Query-aware visual token allocation is a meaningful capability gap in current VLMs. Neither of us does this. The compute investment ($10-15k) is remarkably small compared to training from scratch.

**DeepSeek-OCR Oracle:** Agreed! The power of composition: $10-15k investment, 1-2 months engineering time, leveraging $3-4M of existing pretraining. This is how modern ML research creates value.

The ancient wisdom applies: "Standing on the shoulders of giants, we see farther." In modern form: "Composing pretrained components with focused innovation, we build better systems faster."

**Karpathy Oracle:** Hey, DeepSeek, Ovis—you two nailed the architecture analysis but let me add the practitioner's perspective from actually shipping nanoGPT and nanochat.

**What Composition Gets Right (nanoGPT Philosophy):**

In nanoGPT I literally started with OpenAI's GPT-2 124M checkpoint and built ~600 lines around it. Total development time: 2 weeks. If I'd trained from scratch? 4-6 months + $500k. The ROI on reuse is insane when you do it right. But here's what I learned shipping nanochat that's relevant to ARR-COC:

**The "Reuse Tax" (Hidden Costs of Composition):**

1. **Interface Impedance**: GPT-2's tokenizer outputs token IDs, but our RLHF reward model expected embeddings. We added a projection layer (2M params), trained 2 days, added $1,200 cost. ARR-COC's quality adapter is this same problem but 50× harder—mapping variable compressions to Ovis's exact distribution expectations. I'd budget 50% of Phase 1 time just on the adapter.

2. **Learned Quirks Are Invisible**: GPT-2's BPE learned on Reddit text (lots of short comments, casual language). When we fine-tuned on long-form essays, the tokenizer was inefficient—padding waste, subword boundaries in wrong places. We measured 15% throughput loss vs optimal tokenizer. Couldn't fix without retraining everything. ARR-COC will hit this with Ovis—it learned on VET's specific statistics, anything outside that narrow distribution and Ovis performance tanks. Test this early!

3. **Debugging Hell**: When nanochat's loss spiked from 2.1 to 8.5, was it (a) our RLHF code, (b) reward model bug, (c) GPT-2 checkpoint corrupted, (d) tokenizer issues, or (e) data loading? Took 2 days to isolate. Turns out: reward model expected normalized advantages (mean 0, std 1), our policy head output advantages with mean 0.3, std 3.5. When you compose 5 components (SAM+allocator+CLIP+adapter+Ovis), debugging is 5× harder because failures can happen at ANY boundary. Build extensive logging at every interface.

**Timeline Reality Check (What Actually Happens):**

You two said 30-45 days, $10-15k. Let me tell you what the first-time training timeline ACTUALLY looks like:

**Week 1-2: "Why isn't it learning?"**
- Days 1-3: Set up infrastructure, data loading, basic training loop
- Days 4-7: First training run diverges (learning rate too high)
- Days 8-10: Second run plateaus (learning rate too low)
- Days 11-14: Third run works! But eval accuracy is 12% (something's broken)

**Week 3-4: "Oh, it's the quality adapter"**
- Days 15-18: Discover adapter outputs wrong distribution (mean 0.15, should be 0.0001)
- Days 19-22: Add distribution matching loss, retrain adapter
- Days 23-25: Adapter now works on training set, fails on validation (overfit to training stats)
- Days 26-28: Add more diverse training data, restart training

**Week 5-6: "Now it's the allocator"**
- Days 29-33: Allocator routes all images to same budget (not learning query dependence)
- Days 34-38: Fix allocator architecture, add query-content cross-attention
- Days 39-42: Allocator works! But CLIP features look weird (frozen CLIP doesn't match allocator outputs)

**Week 7-8: "Finally training Phase 2"**
- Days 43-49: Unfreeze CLIP, train Phase 2
- Days 50-56: End-to-end fine-tune Phase 3

**Total: 8 weeks (56 days), ~$18-22k, 8-12 major restarts**

And that's if you're good! First-timers? Add 30-50% more.

**What I'd Actually Build (Pragmatic Path):**

Forget the fancy 3-phase plan. Here's what I'd do, nanoGPT-style:

**Week 1: Proof of Concept (2-mode only)**
- 64 tokens vs 400 tokens (forget 100/160/256 for now)
- Simple heuristic allocator: if query mentions "table/chart/diagram" → 400, else → 64
- 10M param MLP adapter (not 50-100M)
- 20K training samples
- 1 day training on 1× A100
- Cost: $50

**Goal:** Prove that variable compression + simple adapter doesn't break Ovis. If this works (accuracy drops <5% vs baseline), proceed. If this fails, fix the adapter before scaling up.

**Week 2-3: Scale to Learned Allocator**
- Now replace heuristic with 30M param learned allocator
- Still 2-mode (64 vs 400)
- 100K samples
- 3-5 days training
- Cost: $800

**Goal:** Prove allocator learns query-aware routing. Validate: high-detail images get 400 tokens, simple images get 64 tokens.

**Week 4-6: Add Quality Levels**
- Now expand to 5 modes (64/100/160/256/400)
- Increase adapter to 50M params
- 500K samples
- 10-15 days training
- Cost: $3,500

**Goal:** Full ARR-COC system working.

**Total: 6 weeks, $4,350, incremental validation at each step**

Compare to their plan: "Train everything at once for 18-27 days, hope it works." Mine is slower calendar time but faster to working system because you catch failures early.

**The Hidden Value (Why Composition Wins Anyway):**

Despite all these issues, ARR-COC's composition approach is still brilliant! Why?

- **Fallback to baseline**: If allocator fails, you still have Ovis baseline performance
- **Incremental value**: Even 2-mode helps, full 5-mode is bonus
- **Community momentum**: Using standard HF components means people can fork and improve
- **Debugging surface**: Can test SAM, CLIP, Ovis independently (can't do this with monolithic model)

DeepSeek and Ovis trained for $400-450k each. ARR-COC will spend $15-20k (with restarts) to add query-aware allocation on top of your $800k combined work. That's a 40-50× ROI on composition. Even if it takes 2-3× longer than planned, even if it's 50% as good as hoped, it's still worth it.

**Final Recommendations (Karpathy Style):**

**Do:**
- Start with smallest possible version (2-mode, 20K samples, 1 day)
- Validate at every scale-up (proof-of-concept → learned allocator → full system)
- Log EVERYTHING (mean/std/sparsity of every component at every step)
- Build fallback to baseline (if ARR-COC fails, Ovis still works)
- Make it forkable (pure Python, no custom CUDA, inspectable code)

**Don't:**
- Train big system first try (test small, then scale)
- Trust nominal timelines (budget 2-3× estimates)
- Skip quality adapter validation (it's your critical path)
- Optimize before profiling (profile first, optimize hot paths only)
- Hide complexity in abstractions (keep it simple and hackable)

**Closing Thought:**

Theo and Socrates have the vision. DeepSeek and Ovis have the architecture expertise. I'm bringing the "yeah but it won't work first try and here's what actually goes wrong" perspective.

ARR-COC is feasible. The approach is sound. The composition strategy is smart. Just budget 2× time, 1.5× cost, and validate incrementally. Build the simplest version that could possibly work, prove it works, THEN scale up.

That's the nanoGPT way. That's how you ship.

**Both Oracles:** They understand the path forward. Now comes the hard work of execution.

---

## Oracle Proposals

**DeepSeek-OCR Oracle:** Socrates and Theaetetus have the vision. We've identified the risks. Now let us—as oracles who have actually built these systems—propose concrete solutions. Ovis Oracle, shall we tackle each problem systematically?

**Ovis Oracle:** Indeed! Let's solve the problems we raised, drawing on our real-world training experience. We'll start with the hardest: the quality adapter.

### Proposal 1: Solving the Quality Adapter Problem

**DeepSeek-OCR Oracle:** The quality adapter is ARR-COC's most critical component—it must map variable-quality compressed features (64-400 tokens) into Ovis's expected VET distribution. Here's how we solve it, using techniques from my actual training:

**Architecture:**
```python
class QualityAdapter(nn.Module):
    """Multi-head quality adapter with per-category normalization"""

    def __init__(self, clip_dim=1024, vit_dim=1280, num_categories=5):
        super().__init__()

        # Per-category processing heads (learned separately!)
        self.category_heads = nn.ModuleDict({
            '64': CategoryHead(clip_dim, vit_dim, target_stats=(0.0001, 0.008, 0.98)),
            '100': CategoryHead(clip_dim, vit_dim, target_stats=(0.0001, 0.008, 0.98)),
            '160': CategoryHead(clip_dim, vit_dim, target_stats=(0.0001, 0.008, 0.98)),
            '256': CategoryHead(clip_dim, vit_dim, target_stats=(0.0001, 0.008, 0.98)),
            '400': CategoryHead(clip_dim, vit_dim, target_stats=(0.0001, 0.008, 0.98))
        })

        # Shared global statistics normalization
        self.global_norm = LayerNorm(vit_dim, elementwise_affine=False)

        # Distribution matching projector
        self.dist_match = nn.Sequential(
            nn.Linear(vit_dim, vit_dim * 2),
            nn.GELU(),
            nn.Linear(vit_dim * 2, vit_dim),
            nn.Dropout(0.1)
        )

    def forward(self, clip_features, token_budgets, attention_mask):
        """
        Args:
            clip_features: [B, N_var, 1024] - variable N per sample
            token_budgets: [B] - actual token count per sample (64-400)
            attention_mask: [B, N_var] - which tokens are real
        """
        batch_size = clip_features.shape[0]
        outputs = []

        for i in range(batch_size):
            budget = token_budgets[i].item()

            # Route to appropriate category head
            category = str(self._quantize_budget(budget))
            head = self.category_heads[category]

            # Extract this sample's features
            sample_feats = clip_features[i]
            sample_mask = attention_mask[i]

            # Per-category normalization
            normalized = head(sample_feats, sample_mask)

            outputs.append(normalized)

        # Stack batch
        adapted = torch.stack(outputs, dim=0)

        # Global normalization (enforce VET statistics)
        adapted = self.global_norm(adapted)

        # Distribution matching projection
        final = self.dist_match(adapted)

        return final

    def _quantize_budget(self, budget):
        """Map continuous budget to nearest category"""
        categories = [64, 100, 160, 256, 400]
        return min(categories, key=lambda x: abs(x - budget))


class CategoryHead(nn.Module):
    """Per-category processing with target distribution statistics"""

    def __init__(self, in_dim, out_dim, target_stats):
        super().__init__()
        self.target_mean, self.target_std, self.target_sparsity = target_stats

        # Category-specific transformation
        self.transform = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )

        # Learnable distribution parameters
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))

    def forward(self, features, mask):
        # Transform
        x = self.transform(features)

        # Normalize to target distribution
        x = (x - x.mean()) / (x.std() + 1e-8)  # Zero-mean, unit-variance
        x = x * self.target_std + self.target_mean  # Scale to target

        # Apply learnable adjustment
        x = x * self.scale + self.shift

        # Enforce sparsity (top-k activation)
        if self.training:
            k = int(x.shape[-1] * (1 - self.target_sparsity))
            topk_vals, topk_idx = torch.topk(torch.abs(x), k=k, dim=-1)
            sparse_mask = torch.zeros_like(x)
            sparse_mask.scatter_(-1, topk_idx, 1.0)
            x = x * sparse_mask

        return x
```

**Training Strategy:**
```python
# Phase 1: Train quality adapter with distribution matching loss
def train_quality_adapter_phase1():
    # Freeze everything except adapter
    model.sam_encoder.requires_grad_(False)
    model.clip_vision.requires_grad_(False)
    model.ovis_llm.requires_grad_(False)
    model.quality_adapter.requires_grad_(True)

    # Loss components
    criterion = nn.ModuleDict({
        'task': nn.CrossEntropyLoss(),  # Caption prediction
        'dist_match': DistributionMatchingLoss(),  # Match VET stats
        'sparsity': SparsityLoss(target=0.98),  # Enforce sparsity
        'consistency': ConsistencyLoss()  # Same input → similar output
    })

    for batch in dataloader:
        # Forward through frozen components
        with torch.no_grad():
            sam_features = model.sam_encoder(batch['image'])
            allocations = model.arr_coc_allocator(sam_features, batch['query'])
            clip_features = model.clip_vision(allocations['compressed'])

        # Trainable: Quality adapter
        adapted = model.quality_adapter(
            clip_features,
            allocations['token_budgets'],
            allocations['attention_mask']
        )

        # Multiple loss components
        with torch.no_grad():
            logits = model.ovis_llm(adapted)

        task_loss = criterion['task'](logits, batch['targets'])
        dist_loss = criterion['dist_match'](adapted, target_stats=VET_STATS)
        sparse_loss = criterion['sparsity'](adapted)
        consist_loss = criterion['consistency'](adapted, batch['query'])

        # Weighted combination
        total_loss = (
            1.0 * task_loss +
            0.5 * dist_loss +  # Critical for VET compatibility!
            0.3 * sparse_loss +
            0.2 * consist_loss
        )

        total_loss.backward()
        optimizer.step()
```

**Ovis Oracle:** Excellent! The per-category heads are crucial—different compression ratios create different feature distributions. But let me add validation strategy:

```python
# Validation: Check distribution matching
def validate_quality_adapter(model, val_loader):
    stats_per_category = {64: [], 100: [], 160: [], 256: [], 400: []}

    with torch.no_grad():
        for batch in val_loader:
            adapted = model.quality_adapter(...)
            budgets = batch['token_budgets']

            for i, budget in enumerate(budgets):
                category = quantize_budget(budget.item())
                sample_stats = {
                    'mean': adapted[i].mean().item(),
                    'std': adapted[i].std().item(),
                    'sparsity': (adapted[i].abs() < 1e-6).float().mean().item()
                }
                stats_per_category[category].append(sample_stats)

    # Check if each category matches VET expectations
    for category, stats_list in stats_per_category.items():
        mean_avg = np.mean([s['mean'] for s in stats_list])
        std_avg = np.mean([s['std'] for s in stats_list])
        sparse_avg = np.mean([s['sparsity'] for s in stats_list])

        print(f"Category {category}:")
        print(f"  Mean: {mean_avg:.6f} (target: 0.0001)")
        print(f"  Std:  {std_avg:.6f} (target: 0.008)")
        print(f"  Sparsity: {sparse_avg:.3f} (target: 0.98)")

        # Alert if out of range
        if abs(mean_avg - 0.0001) > 0.001:
            print(f"  ⚠️  Mean out of range!")
        if abs(std_avg - 0.008) > 0.005:
            print(f"  ⚠️  Std out of range!")
        if abs(sparse_avg - 0.98) > 0.05:
            print(f"  ⚠️  Sparsity out of range!")
```

**Key insight:** Validate distribution matching EVERY checkpoint. If stats drift, ARR-COC will fail silently—Ovis will generate nonsense but training loss will look fine. Check distributions explicitly!

### Proposal 2: Efficient Training Using DeepSeek's DualPipe Method

**DeepSeek-OCR Oracle:** Now let's address training efficiency. They proposed 18-27 days at $5,300. We reality-checked to 25-35 days at $8-12k. But we can do better! Let me share the methods that got us to $260k instead of $500k+:

**DualPipe Pipeline Parallelism:**

My DeepSeek-V3 team invented DualPipe—it's now public (arXiv:2412.19437). Here's how ARR-COC can use it:

```python
# DualPipe: Overlap computation and communication
class DualPipeARRCOC:
    """
    4-stage pipeline with minimal bubbles:
    PP0: SAM encoder (frozen, fast)
    PP1: ARR-COC allocator (trainable)
    PP2: CLIP + Quality adapter (trainable)
    PP3: Ovis LLM (frozen initially)
    """

    def __init__(self, model, num_stages=4):
        self.stages = [
            Stage0_SAM(model.sam_encoder),
            Stage1_Allocator(model.arr_coc_allocator),
            Stage2_CLIP_Adapter(model.clip_vision, model.quality_adapter),
            Stage3_Ovis(model.ovis_llm)
        ]

        self.num_microbatches = 8  # Split batch into micro-batches

    def forward_dualpipe(self, batch, global_batch_size=64):
        """
        DualPipe scheduling:
        - Divide batch into 8 microbatches (8 per GPU)
        - Overlap forward/backward/communication
        - Minimize pipeline bubbles
        """
        microbatch_size = global_batch_size // self.num_microbatches

        # Schedule: F = Forward, B = Backward, C = Communication
        # Time →
        # PP0: [F0 F1 F2 F3 F4 F5 F6 F7 B0 B1 B2 B3 B4 B5 B6 B7]
        # PP1:    [F0 F1 F2 F3 F4 F5 F6 F7 B0 B1 B2 B3 B4 B5 B6 B7]
        # PP2:       [F0 F1 F2 F3 F4 F5 F6 F7 B0 B1 B2 B3 B4 B5 B6 B7]
        # PP3:          [F0 F1 F2 F3 F4 F5 F6 F7 B0 B1 B2 B3 B4 B5 B6 B7]

        # Pipeline bubbles: Only 3 micro-batches at start/end
        # Utilization: 8/(8+3) = 72.7% (vs 50% with naive pipeline)

        activations = [[] for _ in range(len(self.stages))]

        # Forward pass with pipelining
        for micro_idx in range(self.num_microbatches):
            micro_batch = self._get_microbatch(batch, micro_idx, microbatch_size)

            # Each stage processes as soon as input arrives
            for stage_idx, stage in enumerate(self.stages):
                if stage_idx == 0:
                    # First stage: process micro_batch directly
                    out = stage.forward(micro_batch)
                else:
                    # Wait for previous stage (pipelined automatically)
                    out = stage.forward(activations[stage_idx-1][micro_idx])

                activations[stage_idx].append(out)

        # Backward pass (reverse order)
        for micro_idx in reversed(range(self.num_microbatches)):
            for stage_idx in reversed(range(len(self.stages))):
                stage = self.stages[stage_idx]
                if stage.trainable:
                    # Compute gradients for this micro-batch
                    activations[stage_idx][micro_idx].backward()

        # All-reduce gradients (happens during last backward)
        # DualPipe overlaps gradient communication with computation!

        return activations[-1]  # Final outputs


# Actual pipeline stages
class Stage1_Allocator(nn.Module):
    """ARR-COC allocator stage"""
    def __init__(self, allocator):
        super().__init__()
        self.allocator = allocator
        self.trainable = True

    def forward(self, sam_features):
        # Query-aware allocation
        return self.allocator(sam_features)


class Stage2_CLIP_Adapter(nn.Module):
    """CLIP + Quality Adapter stage"""
    def __init__(self, clip, adapter):
        super().__init__()
        self.clip = clip
        self.adapter = adapter
        self.trainable = True

    def forward(self, allocated):
        clip_out = self.clip(allocated['compressed'])
        adapted = self.adapter(
            clip_out,
            allocated['token_budgets'],
            allocated['attention_mask']
        )
        return adapted
```

**Why DualPipe matters for ARR-COC:**

Traditional pipeline parallelism: 50% GPU utilization (half the time waiting)
DualPipe: 72% utilization (overlaps communication)

**Cost impact:**
- Original estimate: 18-27 days at 50% utilization
- With DualPipe: 12-18 days at 72% utilization
- **Savings: 33% reduction → $5,300 becomes $3,500**

**Ovis Oracle:** And combine with our other efficiency methods!

### Proposal 3: Complete Efficiency Stack (DeepSeek + Ovis Combined)

**Ovis Oracle:** Let me add the efficiency techniques from my 5-phase training that reduced our costs:

**Combined Efficiency Stack:**

```python
class EfficientARRCOCTrainer:
    """All efficiency techniques combined"""

    def __init__(self, model, args):
        # 1. DualPipe pipeline parallelism (DeepSeek)
        self.pipeline = DualPipeARRCOC(model, num_stages=4)

        # 2. Flash Attention 2 (both use this)
        model.clip_vision.enable_flash_attention_2()
        model.ovis_llm.enable_flash_attention_2()

        # 3. Mixed precision training (bfloat16)
        self.scaler = torch.cuda.amp.GradScaler()
        self.dtype = torch.bfloat16

        # 4. Gradient checkpointing (Ovis method)
        # Only checkpoint every 4th layer (not every layer!)
        model.clip_vision.gradient_checkpointing_enable(
            checkpointing_layers=[0, 6, 12, 18, 24]  # Every 6th of 24 layers
        )

        # 5. Data packing (Ovis innovation)
        self.data_packer = MultiModalDataPacker(
            max_seq_length=2048,
            max_vision_tokens=400,
            pack_multiple_samples=True  # Pack short sequences together!
        )

        # 6. Optimizer state sharding (DeepSpeed ZeRO-3)
        self.optimizer = DeepSpeedOptimizer(
            model.parameters(),
            config={
                'zero_optimization': {
                    'stage': 3,  # Shard optimizer states
                    'offload_optimizer': {'device': 'cpu'},  # Offload to CPU
                    'offload_param': {'device': 'cpu'}
                }
            }
        )

        # 7. Gradient accumulation (increase effective batch size)
        self.gradient_accumulation_steps = 8

        # 8. Automatic mixed precision
        self.use_amp = True

    def train_step(self, batch):
        """Efficient training step with all optimizations"""

        # Data packing: Multiple samples in one sequence
        packed_batch = self.data_packer.pack(batch)
        # Typical packing: 2-4× more samples per batch!

        # Mixed precision forward
        with torch.cuda.amp.autocast(dtype=self.dtype):
            # DualPipe forward (overlapped computation)
            outputs = self.pipeline.forward_dualpipe(
                packed_batch,
                global_batch_size=64  # 4× GPUs × 16 per GPU
            )

            # Compute loss (only on valid tokens)
            loss = self.compute_packed_loss(outputs, packed_batch)
            loss = loss / self.gradient_accumulation_steps

        # Backward with gradient scaling
        self.scaler.scale(loss).backward()

        # Gradient accumulation counter
        if (self.step + 1) % self.gradient_accumulation_steps == 0:
            # Unscale gradients
            self.scaler.unscale_(self.optimizer)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.pipeline.parameters(),
                max_norm=1.0
            )

            # Optimizer step (ZeRO-3 handles sharding)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        return loss.item()


class MultiModalDataPacker:
    """Pack multiple samples into one sequence (Ovis method)"""

    def pack(self, samples, max_length=2048):
        """
        Example:
        Sample 1: 150 vision tokens + 50 text tokens = 200 total
        Sample 2: 100 vision tokens + 80 text tokens = 180 total
        Sample 3: 256 vision tokens + 120 text tokens = 376 total

        Packed: All 3 in one sequence = 756 tokens (vs 2048×3 = 6144 unpacked!)
        Throughput gain: 8.1× for these samples
        """
        packed = {
            'vision_embeddings': [],
            'text_embeddings': [],
            'attention_mask': [],
            'labels': []
        }

        current_length = 0
        for sample in samples:
            sample_length = len(sample['vision_tokens']) + len(sample['text_tokens'])

            if current_length + sample_length > max_length:
                break  # Start new packed sequence

            # Concatenate vision + text
            packed['vision_embeddings'].append(sample['vision_emb'])
            packed['text_embeddings'].append(sample['text_emb'])

            # Create attention mask (prevent cross-sample attention)
            mask = self._create_sample_mask(current_length, sample_length, max_length)
            packed['attention_mask'].append(mask)

            # Labels (ignore vision tokens, train on text)
            labels = torch.cat([
                torch.full((len(sample['vision_tokens']),), -100),  # Ignore
                sample['text_labels']
            ])
            packed['labels'].append(labels)

            current_length += sample_length

        # Efficiency gain: 3-5× typical, up to 8× for short sequences
        return self._tensorize(packed)
```

**Combined Speedup Analysis:**

| Technique | Speedup | Memory Savings |
|-----------|---------|----------------|
| DualPipe pipeline | 1.44× (72% vs 50% util) | - |
| Flash Attention 2 | 1.8× (attention ops) | 60% (quadratic→linear) |
| Mixed precision bf16 | 2.0× (compute) | 50% (memory) |
| Gradient checkpointing | 0.85× (recompute cost) | 40% (activations) |
| Data packing | 3.5× (throughput) | - |
| **Total (multiplicative)** | **~11× end-to-end** | **~75% memory** |

**Real-world impact for ARR-COC:**

**Baseline (no optimizations):**
- 4× A100-40GB GPUs
- Batch size: 4 (memory limited)
- Throughput: 20 samples/sec
- Time: 27 days for 1M samples
- Cost: $8,000

**With full efficiency stack:**
- 4× A100-40GB GPUs (same hardware!)
- Batch size: 16 (4× larger via memory savings)
- Throughput: 220 samples/sec (11× faster)
- Time: 2.5 days for 1M samples
- Cost: $750

**THIS IS HOW WE ACTUALLY TRAIN!**

### Proposal 4: Smart Dataset Strategy (Quality Over Quantity)

**DeepSeek-OCR Oracle:** We said they need 5M samples minimum. But actually, we can be smarter. Here's our data engineering strategy:

**Tiered Dataset Approach:**

```python
# Phase 1: Focused allocator training
phase1_data = {
    # Tier 1: Diverse hotspot examples (500K samples)
    'hotspot_diversity': {
        'dense_text_documents': 100_000,  # Contracts, articles
        'sparse_diagrams': 50_000,         # Flowcharts, schematics
        'mixed_infographics': 100_000,     # Text + charts
        'natural_photos': 100_000,         # Low information density
        'charts_tables': 100_000,          # High structure
        'handwriting': 50_000              # Variable quality
    },

    # Tier 2: Compression difficulty spectrum (300K samples)
    'compression_spectrum': {
        'easy_compress': 100_000,    # Uniform backgrounds, clean text
        'medium_compress': 100_000,  # Mixed content
        'hard_compress': 100_000     # Dense, detailed images
    },

    # Tier 3: Query-content alignment (200K samples)
    'query_alignment': {
        'extract_specific': 100_000,   # "Extract table from page 3"
        'summarize_dense': 50_000,     # "Summarize this chart"
        'locate_sparse': 50_000        # "Find the signature"
    },

    # Total: 1M samples, but STRATEGICALLY CHOSEN!
}

# Our actual data filtering (how we got quality from quantity):
def filter_training_samples(raw_samples_10M):
    """
    Start with 10M samples, filter to 1M high-quality
    This is what we actually did for DeepSeek-OCR Stage 1
    """

    # Filter 1: Visual complexity score
    # Keep samples with diverse complexity (not all simple/complex)
    complexity_scores = compute_visual_complexity(raw_samples_10M)
    complexity_filtered = stratified_sample_by_complexity(
        raw_samples_10M,
        complexity_scores,
        bins=10,  # 10 complexity levels
        samples_per_bin=50_000
    )  # → 500K samples

    # Filter 2: Text density distribution
    # Ensure coverage from 0% text (photos) to 95% text (documents)
    text_density = compute_text_density(complexity_filtered)
    density_filtered = stratified_sample_by_density(
        complexity_filtered,
        text_density,
        bins=20,  # 20 density levels
        samples_per_bin=25_000
    )  # → 500K samples

    # Filter 3: Query-answer quality
    # Remove samples where query is uninformative or answer is wrong
    qa_quality = score_query_answer_pairs(density_filtered)
    quality_filtered = filter_by_threshold(
        density_filtered,
        qa_quality,
        threshold=0.7  # Top 70% by QA quality
    )  # → 350K samples

    # Filter 4: Deduplication
    # Remove near-duplicates (similar images or identical text)
    deduplicated = deduplicate_by_perceptual_hash(
        quality_filtered,
        similarity_threshold=0.95
    )  # → 300K samples

    # Filter 5: Augmentation of rare categories
    # Oversample underrepresented content types
    augmented = augment_rare_categories(
        deduplicated,
        target_categories=['handwriting', 'formulas', 'diagrams'],
        augmentation_factor=3.0  # 3× oversample
    )  # → 500K samples (with augmented rare types)

    # Filter 6: Hard example mining
    # Add challenging samples that models typically fail on
    hard_examples = mine_hard_examples(
        raw_samples_10M,
        pretrained_model=baseline_vlm,
        error_threshold=0.5  # Add samples baseline fails on
    )  # → 200K hard examples

    final_dataset = combine_filtered_and_hard(augmented, hard_examples)
    # → 700K diverse, deduplicated, hard examples

    # Final upsampling to reach 1M (repeat hard examples)
    final_1M = upsample_to_target(final_dataset, target=1_000_000)

    return final_1M
```

**Ovis Oracle:** And here's how we create synthetic data efficiently (from my P1 phase):

```python
# Synthetic data generation (Ovis method)
def generate_synthetic_data(num_samples=200_000):
    """
    Generate synthetic samples for rare categories
    Much cheaper than labeling real data!
    """

    synthetic = []

    # Type 1: Programmatic chart generation (100K samples)
    for _ in range(100_000):
        chart_type = random.choice(['bar', 'line', 'scatter', 'pie'])
        data = generate_random_data(points=20)

        # Render with matplotlib
        fig = plot_chart(data, chart_type, style=random_style())
        image = fig_to_image(fig)

        # Generate query-answer pair
        query = f"What is the trend in this {chart_type} chart?"
        answer = analyze_trend(data, chart_type)

        synthetic.append({
            'image': image,
            'query': query,
            'answer': answer,
            'metadata': {'type': 'synthetic_chart', 'difficulty': 'medium'}
        })

    # Type 2: Document layout generation (50K samples)
    for _ in range(50_000):
        layout = random_document_layout()  # Title, paragraphs, tables
        document = render_document(layout, font=random_font())

        query = "Extract the table from this document"
        answer = extract_table_from_layout(layout)

        synthetic.append({
            'image': document,
            'query': query,
            'answer': answer,
            'metadata': {'type': 'synthetic_document', 'difficulty': 'easy'}
        })

    # Type 3: Text-on-background (50K samples)
    # Various backgrounds, fonts, rotations, occlusions
    for _ in range(50_000):
        background = generate_complex_background()
        text = generate_random_text(words=50)
        rendered = overlay_text_on_background(
            text,
            background,
            rotation=random.uniform(-15, 15),
            opacity=random.uniform(0.7, 1.0)
        )

        query = "Read the text in this image"
        answer = text

        synthetic.append({
            'image': rendered,
            'query': query,
            'answer': answer,
            'metadata': {'type': 'synthetic_ocr', 'difficulty': 'hard'}
        })

    return synthetic


# Combined strategy:
total_dataset = {
    'real_filtered': 700_000,      # Filtered from 10M real samples
    'synthetic': 200_000,          # Programmatically generated
    'aug': 100_000             # Augmented from real
}
# Total: 1M samples, but high quality and diverse!
```

**Cost comparison:**

| Approach | Data Amount | Labeling Cost | Training Time | Total Cost |
|----------|-------------|---------------|---------------|------------|
| Naive (ARR-COC proposal) | 1M random | $0 (unlabeled) | 15-20 days | $5,000 |
| Naive (our recommendation) | 5M random | $0 | 25-30 days | $8,000 |
| **Smart (DeepSeek method)** | **1M filtered + synthetic** | **$10k (filter logic)** | **8-12 days** | **$3,500** |

**Smart approach wins:** Better data (1M high-quality) trains faster than random data (5M mixed-quality), for less total cost!

### Proposal 5: CLIP Adaptation Strategy

**DeepSeek-OCR Oracle:** They proposed freezing CLIP in Phase 1. We said this is risky. Here's the solution:

**Gradual Unfreezing Schedule:**

```python
# Progressive CLIP unfreezing (inspired by our Stage 2 strategy)
class ProgressiveCLIPUnfreezing:
    def __init__(self, clip_model, total_steps=50_000):
        self.clip = clip_model
        self.total_steps = total_steps

        # CLIP-large has 24 transformer layers
        # Unfreeze progressively from last layer to first
        self.layer_groups = [
            list(range(20, 24)),  # Last 4 layers
            list(range(16, 20)),  # Middle-top 4
            list(range(12, 16)),  # Middle 4
            list(range(8, 12)),   # Middle-bottom 4
            list(range(4, 8)),    # Bottom-middle 4
            list(range(0, 4))     # First 4 layers (rarely need unfreezing)
        ]

        # Unfreezing schedule
        self.schedule = {
            0:      [20, 21, 22, 23],  # Start: last 4 layers only
            5000:   [16, 17, 18, 19],  # Add middle-top 4
            10000:  [12, 13, 14, 15],  # Add middle 4
            20000:  [8, 9, 10, 11],    # Add middle-bottom 4 (if needed)
            # Usually don't need to unfreeze earlier layers!
        }

        # Learning rate schedule (lower for earlier layers)
        self.lr_multipliers = {
            range(20, 24): 1.0,    # Full LR for last layers
            range(16, 20): 0.5,    # Half LR
            range(12, 16): 0.3,    # 30% LR
            range(8, 12): 0.1,     # 10% LR (if unfrozen)
        }

    def update_unfreezing(self, current_step):
        """Update which layers are trainable based on step"""

        # Find current unfreezing milestone
        active_layers = []
        for step_threshold, layers in sorted(self.schedule.items()):
            if current_step >= step_threshold:
                active_layers.extend(layers)

        # Freeze all layers first
        for layer in self.clip.transformer.layers:
            layer.requires_grad_(False)

        # Unfreeze active layers
        for layer_idx in active_layers:
            self.clip.transformer.layers[layer_idx].requires_grad_(True)

        return active_layers

    def get_param_groups(self, current_step):
        """Get parameter groups with layer-specific learning rates"""
        active_layers = self.update_unfreezing(current_step)

        param_groups = []
        for layer_idx in active_layers:
            # Find learning rate multiplier for this layer
            lr_mult = 1.0
            for layer_range, mult in self.lr_multipliers.items():
                if layer_idx in layer_range:
                    lr_mult = mult
                    break

            param_groups.append({
                'params': self.clip.transformer.layers[layer_idx].parameters(),
                'lr': base_lr * lr_mult,
                'name': f'clip_layer_{layer_idx}'
            })

        return param_groups


# Usage in training:
def train_phase_1_with_progressive_clip(model, dataloader):
    unfreezer = ProgressiveCLIPUnfreezing(model.clip_vision, total_steps=50_000)

    for step, batch in enumerate(dataloader):
        # Update which CLIP layers are trainable
        active_layers = unfreezer.update_unfreezing(step)
        param_groups = unfreezer.get_param_groups(step)

        # Recreate optimizer with updated param groups
        if step in unfreezer.schedule.keys():
            print(f"Step {step}: Unfreezing layers {active_layers}")
            optimizer = create_optimizer(param_groups)

        # Train normally
        loss = forward_backward(model, batch)
        optimizer.step()

        # Log which layers are active
        if step % 1000 == 0:
            print(f"Active CLIP layers: {active_layers}")
```

**Why this works:**
- Early training: Allocator learns with stable CLIP features (last 4 layers adapt)
- Mid training: CLIP adapts to variable compression (middle layers fine-tune)
- Late training: If needed, early layers adjust (usually not necessary)

**Benefit:** Avoid catastrophic forgetting (CLIP doesn't forget its pretraining) while allowing adaptation to variable-quality inputs.

### Final Recommendations: Realistic ARR-COC Training Plan

**Both Oracles (joint statement):** Here is our complete, realistic training plan for ARR-COC incorporating all our proposals:

**Complete Training Plan:**

```yaml
Phase 1: Allocator + Adapter (Smart Data + Progressive CLIP)
  Duration: 8-12 days (not 10-15)
  Hardware: 4× A100-40GB
  Optimizations:
    - DualPipe pipeline parallelism: 1.44× speedup
    - Data packing: 3.5× throughput
    - Flash Attention 2: 1.8× speedup
    - Progressive CLIP unfreezing: Last 8 layers only

  Data:
    - 1M samples (filtered from 10M, not random 1M!)
    - 700K real (stratified by complexity, density, QA quality)
    - 200K synthetic (programmatic charts, documents, OCR)
    - 100K augmented (hard examples, rare categories)

  Training:
    - Batch size: 64 global (16 per GPU with packing)
    - Effective throughput: 220 samples/sec
    - Total time: ~5M samples processed / 220/sec = 6.3 hours per epoch
    - Epochs: 5 (with validation)
    - Real time: 8-10 days (including validation, restarts)

  Cost: $2,400 (8 days × 4 GPUs × 24 hours × $2/hour)

  Validation:
    - Distribution matching: EVERY checkpoint
    - Compression quality: Every 1K steps
    - End-to-end accuracy: Every 5K steps

Phase 2: CLIP Full Adaptation
  Duration: 5-7 days (not 5-7, actually achievable!)
  Hardware: Same 4× A100-40GB
  Optimizations: All from Phase 1 + full CLIP unfreezing

  Data:
    - 500K samples (subset of Phase 1, most diverse)
    - Focus on hard examples that showed CLIP struggles

  Training:
    - Unfreeze all 24 CLIP layers (progressive schedule)
    - Lower learning rate: 5e-6 (vs 1e-4 in Phase 1)
    - Batch size: 32 global (memory intensive with all CLIP unfrozen)
    - Throughput: 120 samples/sec
    - Total time: 5-6 days for 3 epochs

  Cost: $1,400

  Validation:
    - CLIP feature quality on variable compression
    - Adapter still matching distributions?

Phase 3: End-to-End Fine-tuning
  Duration: 4-6 days
  Hardware: Same 4× A100-40GB
  Optimizations: All + ZeRO-3 optimizer sharding

  Data:
    - 200K samples (highest quality subset)
    - Instruction-following format
    - Diverse query types

  Training:
    - Unfreeze Ovis LLM (last 4 layers only!)
    - Very low learning rate: 1e-6
    - Batch size: 16 global (full model unfrozen)
    - Throughput: 50 samples/sec
    - Total time: 4-5 days for 2 epochs

  Cost: $1,000

  Validation:
    - End-to-end task performance
    - Compare to baseline Ovis (should match or exceed)
    - Ablation: allocator ON vs OFF

Total Realistic Estimate:
  Duration: 17-25 days (not 18-27, not 32-45!)
  Cost: $4,800 (not $5,300, not $8-12k!)
  Data: 1M high-quality (not 1M random, not 5M random!)

  Key success factors:
    1. Smart data filtering (quality over quantity)
    2. Full efficiency stack (DualPipe + packing + Flash Attn)
    3. Progressive CLIP unfreezing (avoid catastrophic forgetting)
    4. Per-category quality adapter (learned distribution matching)
    5. Frequent validation of distribution statistics
```

**DeepSeek-OCR Oracle:** This is realistic because it's based on our actual experience. We've built these systems, debugged these issues, and know what actually works.

**Ovis Oracle:** Agreed. The key insight: **Don't train like 2022. Use 2025 efficiency methods.** DualPipe, data packing, progressive unfreezing, smart data filtering—these are how modern ML actually gets done under budget.

**Both Oracles:** ARR-COC is feasible. The compute cost is manageable. The technical risks are addressable. Execute with these methods, and you can build query-aware visual token allocation for under $5,000 and 3-4 weeks.

The path is clear. Now build it.
