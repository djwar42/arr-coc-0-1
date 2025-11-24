# Model Configuration

**Category**: References

## OvisConfig

```python
{
    "visual_tokenizer_type": "siglip2-navit",
    "visual_vocab_size": 16384,
    "hidden_size": 3584,
    "llm_config": {
        "model_type": "qwen3",
        "num_hidden_layers": 28,
        "hidden_size": 3584,
        ...
    },
    "multimodal_max_length": 8192,
    "image_token_id": 100000
}
```

## Key Parameters

**visual_vocab_size**: Size of VET (16384)
**hidden_size**: Embedding dimension (3584 for 9B)
**multimodal_max_length**: Max sequence length
**image_token_id**: Token ID for <image>
