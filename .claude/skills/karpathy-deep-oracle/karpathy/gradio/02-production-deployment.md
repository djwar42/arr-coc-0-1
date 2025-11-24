# Gradio Production & Deployment Patterns (2025)

**Source**: Platonic Dialogue Part 39 + Bright Data research (2025-01-30)
**Context**: W&B integration, production deployment, T4 memory constraints, debugging
**Philosophy**: From rapid prototyping to production deployment

---


## Part 4: Weights & Biases Integration

### Pattern: W&B Experiment Tracking

**From research (Official W&B docs):**

```python
import wandb

def initialize_wandb_run(config):
    """Initialize W&B experiment tracking"""
    wandb.init(
        project="arr-coc-vlm",
        name=f"experiment_{config['variant']}_{config['timestamp']}",
        config=config,  # Hyperparameters
        tags=["gradio", "ablation", config['variant']]
    )

def log_gradio_interaction(image, query, result, metrics):
    """Log each Gradio interaction to W&B"""
    wandb.log({
        'query': query,
        'answer': result['answer'],
        'tokens_used': result['tokens'],
        'latency': result['latency'],
        'memory_gb': result['memory'],
        'image': wandb.Image(image, caption=query),
        'heatmap': wandb.Image(result['heatmap'], caption="Relevance") if result['heatmap'] else None
    })

def log_comparison_table(comparison_results):
    """Log multi-model comparison as W&B Table"""
    columns = ["Model", "Answer", "Tokens", "Latency", "Memory", "Image"]
    data = []

    for model_name, result in comparison_results.items():
        data.append([
            model_name,
            result['answer'],
            result['tokens'],
            result['latency'],
            result['memory'],
            wandb.Image(result['image'])
        ])

    table = wandb.Table(columns=columns, data=data)
    wandb.log({"model_comparison": table})

def finish_run():
    """Close W&B run gracefully"""
    wandb.finish()
```

**Core W&B concepts (from research):**
- `wandb.init()`: Start experiment tracking
- `wandb.log()`: Log metrics step-by-step
- `wandb.Table`: Structured data (predictions, images, ground truth)
- `wandb.Image`: Include images in logs
- `wandb.finish()`: Close run

### Pattern: Gradio + W&B Integration

**From research (W&B Gradio integration docs, GitHub examples):**

```python
def create_wandb_gradio_interface():
    """Gradio interface with W&B logging"""

    # Initialize W&B
    config = {
        'model': 'arr-coc-v1',
        'base_model': 'Qwen3-VL-2B',
        'timestamp': datetime.now().isoformat()
    }
    wandb.init(project="arr-coc-gradio", config=config)

    def process_with_logging(image, query, variant):
        """Process query and log to W&B"""
        start = time.time()

        # Run inference
        result = model.process(image, query, variant)

        # Calculate metrics
        metrics = {
            'latency': time.time() - start,
            'tokens': result['tokens'],
            'memory': torch.cuda.max_memory_allocated() / 1e9
        }

        # Log to W&B
        log_gradio_interaction(image, query, result, metrics)

        return result['answer'], result['heatmap'], metrics

    with gr.Blocks() as demo:
        # ... Gradio interface definition ...

        process_btn.click(
            fn=process_with_logging,
            inputs=[image, query, variant_dropdown],
            outputs=[answer_text, heatmap_img, metrics_json]
        )

    return demo

# Usage
demo = create_wandb_gradio_interface()
demo.launch()
# Every user interaction logged to W&B dashboard
```

**Use case (from research):**
- Track user interactions in Gradio demo
- Log queries, outputs, user feedback
- Each interaction becomes W&B artifact
- Dashboard visualization for all experiments

### Pattern: Private W&B Workspace

**From research (W&B team configuration):**

```python
# Private team workspace configuration
wandb.init(
    project="arr-coc-vlm",
    entity="your-private-team",  # Private workspace
    name="experiment_001",
    config={
        'visibility': 'private',  # Not public
        'team_access': 'collaborators-only'
    }
)
```

**Integration with HF private workflow (from Phase 2):**
- Private W&B workspace matches private HF repos
- Stealth development: W&B tracks experiments privately
- Share dashboard link with team only
- Public launch: Make W&B run public if desired

---

## Part 5: Production Transition Patterns

### Pattern: Gradio Alone vs FastAPI + Gradio

**From research (Medium "Gradio — From Prototype to Production" Oct 2024):**

**When Gradio Alone is Sufficient:**
- ✅ Research demos
- ✅ Internal tools
- ✅ HF Spaces (private or public)
- ✅ Single-user or low-traffic
- ✅ No authentication needed
- ✅ No database requirements

**When FastAPI + Gradio is Required:**
- ✅ Multi-user with authentication
- ✅ Persistent storage (database)
- ✅ API-first architecture
- ✅ Complex business logic
- ✅ Payment integration
- ✅ High traffic (load balancing)

**Gradio-Session System Architecture (from research):**
```
┌─────────────────┐
│  FastAPI Backend│
│  - Auth         │
│  - Database     │
│  - Business API │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Gradio Frontend│
│  - UI only      │
│  - Calls FastAPI│
└─────────────────┘
```

### Pattern: Development to Production Path

**Recommended progression (from research + Dialogue 39):**

**Stage 1: Rapid Development**
- Gradio `app_dev.py` locally
- Multi-model comparison
- Fast iteration
- W&B logging

**Stage 2: Internal Testing**
- Deploy to HF Spaces (private)
- Share with team via invite-only
- Gradio alone sufficient

**Stage 3: Public Research Demo**
- Deploy to HF Spaces (public)
- Free tier adequate
- Gradio alone sufficient
- W&B run made public if desired

**Stage 4: Production (if needed)**
- FastAPI backend + Gradio frontend
- Authentication, database
- Scaling considerations
- Paid infrastructure

**Key insight**: Most research projects never need Stage 4. Stages 1-3 handle 90% of use cases.

---

## Part 6: Memory Constraints on T4 (HF Spaces)

### Pattern: LRU Cache for Checkpoints

**From research + Phase 1 knowledge:**

```python
from functools import lru_cache
from collections import OrderedDict

class CheckpointCache:
    """LRU cache for model checkpoints (T4: max 2 loaded)"""

    def __init__(self, max_loaded=2):
        self.cache = OrderedDict()
        self.max_loaded = max_loaded

    def load_checkpoint(self, checkpoint_name):
        """Load checkpoint with LRU eviction"""
        if checkpoint_name in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(checkpoint_name)
            return self.cache[checkpoint_name]

        # Load new checkpoint
        checkpoint = torch.load(f"checkpoints/{checkpoint_name}.pt")

        # Evict oldest if at capacity
        if len(self.cache) >= self.max_loaded:
            oldest = next(iter(self.cache))
            del self.cache[oldest]
            torch.cuda.empty_cache()  # Free GPU memory

        # Add to cache
        self.cache[checkpoint_name] = checkpoint
        return checkpoint

# Usage in Gradio interface
checkpoint_cache = CheckpointCache(max_loaded=2)  # T4 limit

def compare_models(image, query, checkpoint_a, checkpoint_b):
    """Compare two checkpoints (only 2 loaded at once)"""
    model_a = checkpoint_cache.load_checkpoint(checkpoint_a)
    model_b = checkpoint_cache.load_checkpoint(checkpoint_b)

    result_a = run_inference(model_a, image, query)
    result_b = run_inference(model_b, image, query)

    return result_a, result_b
```

**T4 Memory Budget (from Phase 1):**
- Total VRAM: 16GB
- Base Qwen3-VL-2B (bfloat16): ~5GB
- ARR-COC components: ~1GB
- KV cache: ~2GB
- Activations: ~2GB
- Buffer: ~2GB
- **Max 2 variants loaded simultaneously**

---

## Part 7: Debugging Deployment Failures

### Pattern: HF Spaces Debugging Workflow

**From Phase 2 knowledge (HF deployment):**

**Common failures:**
1. **OOM on T4**: Model too large
2. **Timeout**: Inference too slow
3. **Missing dependencies**: requirements.txt incomplete
4. **Environment variables**: Secrets not set

**Debugging steps:**
```python
# 1. Add verbose logging to app.py
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_query(image, query):
    logger.info(f"Starting inference for query: {query}")
    logger.info(f"GPU memory before: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    try:
        result = model.process(image, query)
        logger.info(f"Inference successful")
        return result
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise
    finally:
        logger.info(f"GPU memory after: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# 2. Test locally with T4-equivalent constraints
# Limit CPU memory to simulate T4
import resource
resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))  # 16GB

# 3. Check HF Spaces logs
# Navigate to Space → Settings → View logs
# Look for OOM errors, import errors, timeout errors

# 4. Add error handling in Gradio
def safe_process(image, query):
    try:
        return process_query(image, query)
    except torch.cuda.OutOfMemoryError:
        return "Error: GPU out of memory. Try a smaller image or simpler query."
    except Exception as e:
        return f"Error: {str(e)}"
```

---

---

**Related Gradio Files:**
- [09-gradio-core-testing-patterns.md](09-gradio-core-testing-patterns.md) - Multi-model comparison
- [10-gradio-statistical-testing.md](10-gradio-statistical-testing.md) - A/B testing, statistical validation
- [12-gradio-visualization-best-practices.md](12-gradio-visualization-best-practices.md) - Gallery testing, Gradio 5

**Related Oracle Files:**
- [08-gpu-memory-debugging-vlm-2025-01-30.md](08-gpu-memory-debugging-vlm-2025-01-30.md) - GPU memory management
- [../../deepseek/knowledge-categories/05-huggingface-deployment-vlm-2025-01-30.md](../../deepseek/knowledge-categories/05-huggingface-deployment-vlm-2025-01-30.md) - HF Spaces deployment

**Primary Sources:**
- W&B official docs (tracking, tables, Gradio integration)
- Medium "Gradio — From Prototype to Production" (Oct 2024)
- Dialogue 39: Production deployment patterns

**Last Updated**: 2025-01-31 (Split from 09-gradio-testing-patterns-2025-01-30.md)
**Version**: 1.0 - Production deployment patterns (~330 lines)
