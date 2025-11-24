# <claudes_code_comments>
# ** Class List **
# DeepseekV2Config - Configuration class for DeepSeekV2/V3 models
#
# ** Technical Review **
# Configuration class for DeepSeekV2 and DeepSeekV3 models, supporting both dense and
# Mixture of Experts (MoE) architectures with Multi-head Latent Attention (MLA).
#
# ** Key Parameter Groups **:
#
# 1. Model Architecture (lines 119-125, 162-168):
#    - vocab_size: Vocabulary size (default 102400)
#    - hidden_size: Model dimension (default 4096)
#    - num_hidden_layers: Number of transformer layers (default 30)
#    - num_attention_heads: Attention heads (default 32)
#    - intermediate_size: Dense MLP hidden dim (default 11008, ~2.7x hidden_size)
#    - moe_intermediate_size: Expert MLP hidden dim (default 1407, smaller for efficiency)
#
# 2. Multi-head Latent Attention (MLA) (lines 130-134, 173-177):
#    - use_mla: Enable MLA (default True) vs standard multi-head attention
#    - kv_lora_rank: Compressed KV cache dimension (default 512)
#    - q_lora_rank: Query low-rank projection (default 1536, None=no compression)
#    - qk_rope_head_dim: Rotary positional head dim (default 64)
#    - qk_nope_head_dim: Non-positional head dim (default 128)
#    - v_head_dim: Value head dimension (default 128)
#    Total q_head_dim = qk_nope_head_dim + qk_rope_head_dim = 192
#
# 3. Mixture of Experts (MoE) (lines 126-144, 169-187):
#    - n_routed_experts: Total routed experts (None=dense model, e.g., 64 or 256)
#    - n_shared_experts: Always-activated experts (None=dense, e.g., 2)
#    - num_experts_per_tok: Top-K activated experts (None=dense, e.g., 6)
#    - routed_scaling_factor: Expert output scaling (default 1.0)
#    - ep_size: Expert parallelism world size (default 1=no parallelism)
#    - scoring_func: Expert scoring ("softmax" or "sigmoid", default "softmax")
#    - norm_topk_prob: Renormalize top-K expert weights (default False)
#    - topk_method: Top-K selection method (default "gready" [sic])
#    - n_group / topk_group: Grouped expert routing (None=no grouping)
#
# 4. MoE Layer Placement (lines 139-140, 182-183):
#    - moe_layer_freq: MoE layer frequency (default 1=every layer)
#      * freq=1: All layers use MoE
#      * freq=2: Alternating dense/MoE layers
#    - first_k_dense_replace: Number of initial dense layers (default 0)
#      * e.g., first_k_dense_replace=1: First layer dense, rest MoE
#
# 5. Auxiliary Loss (lines 141-144, 184-187):
#    - aux_loss_alpha: Load balancing loss weight (default 0.001)
#    - seq_aux: Per-sample auxiliary loss (default True) vs per-batch
#
# 6. Positional Embeddings (lines 155-156, 198-199):
#    - rope_theta: RoPE base frequency (default 10000.0)
#    - rope_scaling: Context extension config (None=standard RoPE)
#      * {"type": "linear", "factor": 2.0}: Linear scaling for 2x context
#      * {"type": "dynamic", "factor": 2.0}: Dynamic NTK scaling
#      * {"type": "yarn", "factor": 8.0, ...}: YaRN scaling with wavelength correction
#        - original_max_position_embeddings: Original context (e.g., 4096)
#        - beta_fast / beta_slow: Wavelength correction aggressiveness
#        - mscale / mscale_all_dim: Attention score scaling parameters
#
# 7. Training & Inference (lines 147-159, 194-202):
#    - max_position_embeddings: Maximum sequence length (default 2048)
#    - use_cache: Enable KV caching for inference (default True)
#    - attention_bias: Add bias to attention projections (default False)
#    - attention_dropout: Attention dropout rate (default 0.0)
#    - hidden_act: Activation function (default "silu" for SwiGLU)
#    - initializer_range: Weight init std dev (default 0.02)
#    - rms_norm_eps: RMSNorm epsilon (default 1e-6)
#
# 8. Tokenization (lines 150-152, 204-209):
#    - pad_token_id: Padding token (None=no padding)
#    - bos_token_id: Beginning of sequence (default 100000)
#    - eos_token_id: End of sequence (default 100001)
#    - tie_word_embeddings: Share input/output embeddings (default False)
#
# ** Usage Patterns **:
#
# Dense Model (No MoE):
#   config = DeepseekV2Config(
#       hidden_size=4096,
#       num_hidden_layers=30,
#       n_routed_experts=None,  # No MoE
#       n_shared_experts=None,
#   )
#
# MoE Model:
#   config = DeepseekV2Config(
#       hidden_size=4096,
#       num_hidden_layers=60,
#       n_routed_experts=160,
#       n_shared_experts=2,
#       num_experts_per_tok=6,
#       ep_size=8,  # Expert parallelism across 8 GPUs
#   )
#
# Extended Context (YaRN):
#   config = DeepseekV2Config(
#       max_position_embeddings=2048,  # Don't update when using rope_scaling
#       rope_scaling={
#           "type": "yarn",
#           "factor": 32.0,  # 4K → 128K context
#           "original_max_position_embeddings": 4096,
#           "beta_fast": 32,
#           "beta_slow": 1,
#           "mscale": 1.0,
#           "mscale_all_dim": 0.8,
#       },
#   )
#
# ** Architecture Variations **:
# - DeepSeekV2-Lite: hidden_size=2048, n_routed_experts=64, kv_lora_rank=512
# - DeepSeekV2-Base: hidden_size=4096, n_routed_experts=160, kv_lora_rank=512
# - DeepSeekV3-671B: hidden_size=7168, n_routed_experts=256, kv_lora_rank=512
#
# This configuration supports the full range of DeepSeek model architectures from
# small dense models to massive MoE models with expert parallelism and compressed KV cache.
# </claudes_code_comments>

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

DEEPSEEK_PRETRAINED_CONFIG_ARCHIVE_MAP = {}
class DeepseekV2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DeepseekV2Model`]. It is used to instantiate an DeepSeek
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the DeepSeek-V2 with multi-latent attention.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 102400):
            Vocabulary size of the Deep model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`DeepseekV2Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        moe_intermediate_size (`int`, *optional*, defaults to 1407):
            Dimension of the MoE representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        n_shared_experts (`int`, *optional*, defaults to None):
            Number of shared experts, None means dense model.
        n_routed_experts (`int`, *optional*, defaults to None):
            Number of routed experts, None means dense model.
        routed_scaling_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor or routed experts.
        topk_method (`str`, *optional*, defaults to `gready`):
            Topk method used in routed gate.
        n_group (`int`, *optional*, defaults to None):
            Number of groups for routed experts.
        topk_group (`int`, *optional*, defaults to None):
            Number of selected groups for each token(for each token, ensuring the selected experts is only within `topk_group` groups).
        num_experts_per_tok (`int`, *optional*, defaults to None):
            Number of selected experts, None means dense model.
        moe_layer_freq (`int`, *optional*, defaults to 1):
            The frequency of the MoE layer: one expert layer for every `moe_layer_freq - 1` dense layers.
        first_k_dense_replace (`int`, *optional*, defaults to 0):
            Number of dense layers in shallow layers(embed->dense->dense->...->dense->moe->moe...->lm_head).
                                                            \--k dense layers--/
        norm_topk_prob (`bool`, *optional*, defaults to False):
            Whether to normalize the weights of the routed experts.
        scoring_func (`str`, *optional*, defaults to 'softmax'):
            Method of computing expert weights.
        aux_loss_alpha (`float`, *optional*, defaults to 0.001):
            Auxiliary loss weight coefficient.
        seq_aux = (`bool`, *optional*, defaults to True):
            Whether to compute the auxiliary loss for each individual sample.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        use_mla (`bool`, *optional*, defaults to `True`): Use multi-latent attention or multi-head attention. If True,
            the model will use multi-latent attention, otherwise, it will use multi-head attention.

    ```python
    >>> from transformers import DeepseekV2Model, DeepseekV2Config

    >>> # Initializing a Deepseek-V2 style configuration
    >>> configuration = DeepseekV2Config()

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "deepseek_v2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=102400,
        hidden_size=4096,
        intermediate_size=11008,
        moe_intermediate_size = 1407,
        num_hidden_layers=30,
        num_attention_heads=32,
        num_key_value_heads=32,
        n_shared_experts = None,
        n_routed_experts = None,
        ep_size = 1,
        routed_scaling_factor = 1.0,
        kv_lora_rank = 512,
        q_lora_rank = 1536,
        qk_rope_head_dim = 64,
        v_head_dim = 128,
        qk_nope_head_dim = 128,
        topk_method = 'gready',
        n_group = None,
        topk_group = None,
        num_experts_per_tok = None,
        moe_layer_freq = 1,
        first_k_dense_replace = 0,
        norm_topk_prob = False,
        scoring_func = 'softmax',
        aux_loss_alpha = 0.001,
        seq_aux = True,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=100000,
        eos_token_id=100001,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        use_mla=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.ep_size = ep_size
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = float(rms_norm_eps)
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_mla = use_mla

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
