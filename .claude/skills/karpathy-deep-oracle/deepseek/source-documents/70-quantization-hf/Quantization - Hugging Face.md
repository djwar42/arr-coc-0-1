---
sourceFile: "Quantization - Hugging Face"
exportedBy: "Kortex"
exportDate: "2025-10-29T02:37:44.837Z"
---

# Quantization - Hugging Face

bfb921fe-c5d3-4f00-9c3d-6b19187ca866

Quantization - Hugging Face

ef9b4928-01c6-4631-8659-d35f0caa3775

https://huggingface.co/docs/transformers/en/main_classes/quantization

## Hugging Face

## Enterprise

## Transformers documentation

## Quantization

## Transformers

## Join the Hugging Face community

and get access to the augmented documentation experience

Collaborate on models, datasets and Spaces  Faster examples with accelerated inference  Switch between documentation themes

to get started

## Quantization

Quantization techniques reduce memory and computational costs by representing weights and activations with lower-precision data types like 8-bit integers (int8). This enables loading larger models you normally wouldn’t be able to fit into memory, and speeding up inference. Transformers supports the AWQ and GPTQ quantization algorithms and it supports 8-bit and 4-bit quantization with bitsandbytes.

Quantization techniques that aren’t supported in Transformers can be added with the  HfQuantizer  class.

## Learn how to quantize models in the

## Quantization

https://huggingface.co/quantization

## QuantoConfig

class   transformers. QuantoConfig

<   source   >

https://huggingface.co/quantization

(   weights  = 'int8'   activations  = None   modules_to_not_convert : typing.Optional[list] = None   **kwargs   )

## Parameters

, defaults to  "int8" ) — The target dtype for the weights after quantization. Supported values are (“float8”,“int8”,“int4”,“int2”)

activations

) — The target dtype for the activations after quantization. Supported values are (None,“int8”,“float8”)

modules_to_not_convert

, default to  None ) — The list of modules to not quantize, useful for quantizing models that explicitly require to have some modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers).

This is a wrapper class about all possible attributes and features that you can play with a model that has been loaded using  quanto .

<   source   >

## Safety checker that arguments are correct

class   transformers. AqlmConfig

<   source   >

(   in_group_size : int = 8   out_group_size : int = 1   num_codebooks : int = 1   nbits_per_codebook : int = 16   linear_weights_not_to_quantize : typing.Optional[list[str]] = None   **kwargs   )

## Parameters

in_group_size

, defaults to 8) — The group size along the input dimension.

out_group_size

, defaults to 1) — The group size along the output dimension. It’s recommended to always use 1.

num_codebooks

, defaults to 1) — Number of codebooks for the Additive Quantization procedure.

nbits_per_codebook

, defaults to 16) — Number of bits encoding a single codebook vector. Codebooks size is 2**nbits_per_codebook.

linear_weights_not_to_quantize

( Optional[list[str]] ,

) — List of full paths of  nn.Linear  weight parameters that shall not be quantized.

( dict[str, Any] ,

) — Additional parameters from which to initialize the configuration object.

This is a wrapper class about  aqlm  parameters.

<   source   >

Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.

class   transformers. VptqConfig

<   source   >

(   enable_proxy_error : bool = False   config_for_layers : dict = {}   shared_layer_config : dict = {}   modules_to_not_convert : typing.Optional[list] = None   **kwargs   )

## Parameters

enable_proxy_error

, defaults to  False ) — calculate proxy error for each layer

config_for_layers

, defaults to  {} ) — quantization params for each layer

shared_layer_config

, defaults to  {} ) — shared quantization params among layers

modules_to_not_convert

, default to  None ) — The list of modules to not quantize, useful for quantizing models that explicitly require to have some modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers).

( dict[str, Any] ,

) — Additional parameters from which to initialize the configuration object.

This is a wrapper class about  vptq  parameters.

<   source   >

## Safety checker that arguments are correct

class   transformers. AwqConfig

<   source   >

(   bits : int = 4   group_size : int = 128   zero_point : bool = True   version : AWQLinearVersion = <AWQLinearVersion.GEMM: 'gemm'>   backend : AwqBackendPackingMethod = <AwqBackendPackingMethod.AUTOAWQ: 'autoawq'>   do_fuse : typing.Optional[bool] = None   fuse_max_seq_len : typing.Optional[int] = None   modules_to_fuse : typing.Optional[dict] = None   modules_to_not_convert : typing.Optional[list] = None   exllama_config : typing.Optional[dict[str, int]] = None   **kwargs   )

## Parameters

, defaults to 4) — The number of bits to quantize to.

, defaults to 128) — The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.

, defaults to  True ) — Whether to use zero point quantization.

( AWQLinearVersion ,

, defaults to  AWQLinearVersion.GEMM ) — The version of the quantization algorithm to use. GEMM is better for big batch_size (e.g. >= 8) otherwise, GEMV is better (e.g. < 8 ). GEMM models are compatible with Exllama kernels.

( AwqBackendPackingMethod ,

, defaults to  AwqBackendPackingMethod.AUTOAWQ ) — The quantization backend. Some models might be quantized using  llm-awq  backend. This is useful for users that quantize their own models using  llm-awq  library.

, defaults to  False ) — Whether to fuse attention and mlp layers together for faster inference

fuse_max_seq_len

) — The Maximum sequence length to generate when using fusing.

modules_to_fuse

, default to  None ) — Overwrite the natively supported fusing scheme with the one specified by the users.

modules_to_not_convert

, default to  None ) — The list of modules to not quantize, useful for quantizing models that explicitly require to have some modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers). Note you cannot quantize directly with transformers, please refer to  AutoAWQ  documentation for quantizing HF models.

exllama_config

( dict[str, Any] ,

) — You can specify the version of the exllama kernel through the  version  key, the maximum sequence length through the  max_input_len  key, and the maximum batch size through the  max_batch_size  key. Defaults to  {"version": 2, "max_input_len": 2048, "max_batch_size": 8}  if unset.

This is a wrapper class about all possible attributes and features that you can play with a model that has been loaded using  auto-awq  library awq quantization relying on auto_awq backend.

<   source   >

## Safety checker that arguments are correct

class   transformers. EetqConfig

<   source   >

(   weights : str = 'int8'   modules_to_not_convert : typing.Optional[list] = None   **kwargs   )

## Parameters

, defaults to  "int8" ) — The target dtype for the weights. Supported value is only “int8”

modules_to_not_convert

, default to  None ) — The list of modules to not quantize, useful for quantizing models that explicitly require to have some modules left in their original precision.

This is a wrapper class about all possible attributes and features that you can play with a model that has been loaded using  eetq .

<   source   >

## Safety checker that arguments are correct

class   transformers. GPTQConfig

<   source   >

(   bits : int   tokenizer : typing.Any = None   dataset : typing.Union[list[str], str, NoneType] = None   group_size : int = 128   damp_percent : float = 0.1   desc_act : bool = False   sym : bool = True   true_sequential : bool = True   checkpoint_format : str = 'gptq'   meta : typing.Optional[dict[str, typing.Any]] = None   backend : typing.Optional[str] = None   use_cuda_fp16 : bool = False   model_seqlen : typing.Optional[int] = None   block_name_to_quantize : typing.Optional[str] = None   module_name_preceding_first_block : typing.Optional[list[str]] = None   batch_size : int = 1   pad_token_id : typing.Optional[int] = None   use_exllama : typing.Optional[bool] = None   max_input_length : typing.Optional[int] = None   exllama_config : typing.Optional[dict[str, typing.Any]] = None   cache_block_outputs : bool = True   modules_in_block_to_quantize : typing.Optional[list[list[str]]] = None   **kwargs   )

## Parameters

( int ) — The number of bits to quantize to, supported numbers are (2, 3, 4, 8).

( str  or  PreTrainedTokenizerBase ,

) — The tokenizer used to process the dataset. You can pass either:

A custom tokenizer object.

A string, the

of a predefined tokenizer hosted inside a model repo on huggingface.co.

## A path to a

containing vocabulary files required by the tokenizer, for instance saved using the

save_pretrained()

https://huggingface.co/docs/transformers/v4.57.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.save_pretrained

method, e.g.,  ./my_model_directory/ .

( Union[list[str]] ,

) — The dataset used for quantization. You can provide your own dataset in a list of string or just use the original datasets used in GPTQ paper [‘wikitext2’,‘c4’,‘c4-new’]

, defaults to 128) — The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.

damp_percent

( float ,

, defaults to 0.1) — The percent of the average Hessian diagonal to use for dampening. Recommended value is 0.1.

, defaults to  False ) — Whether to quantize columns in order of decreasing activation size. Setting it to False can significantly speed up inference but the perplexity may become slightly worse. Also known as act-order.

, defaults to  True ) — Whether to use symmetric quantization.

true_sequential

, defaults to  True ) — Whether to perform sequential quantization even within a single Transformer block. Instead of quantizing the entire block at once, we perform layer-wise quantization. As a result, each layer undergoes quantization using inputs that have passed through the previously quantized layers.

checkpoint_format

, defaults to  "gptq" ) — GPTQ weight format.  gptq (v1) is supported by both gptqmodel and auto-gptq.  gptq_v2  is gptqmodel only.

( dict[str, any] ,

) — Properties, such as tooling:version, that do not directly contributes to quantization or quant inference are stored in meta. i.e.  meta.quantizer : [“optimum:

”, “gptqmodel:

) — Controls which gptq kernel to be used. Valid values for gptqmodel are  auto ,  auto_trainable  and more. For auto-gptq, only valid value is None and  auto_trainable . Ref gptqmodel backends:

https://github.com/ModelCloud/GPTQModel/blob/main/gptqmodel/utils/backend.py

https://github.com/ModelCloud/GPTQModel/blob/main/gptqmodel/utils/backend.py

use_cuda_fp16

, defaults to  False ) — Whether or not to use optimized cuda kernel for fp16 model. Need to have model in fp16. Auto-gptq only.

model_seqlen

) — The maximum sequence length that the model can take.

block_name_to_quantize

) — The transformers block name to quantize. If None, we will infer the block name using common patterns (e.g. model.layers)

module_name_preceding_first_block

( list[str] ,

) — The layers that are preceding the first Transformer block.

, defaults to 1) — The batch size used when processing the dataset

pad_token_id

) — The pad token id. Needed to prepare the dataset when  batch_size  > 1.

use_exllama

) — Whether to use exllama backend. Defaults to  True  if unset. Only works with  bits  = 4.

max_input_length

) — The maximum input length. This is needed to initialize a buffer that depends on the maximum expected input length. It is specific to the exllama backend with act-order.

exllama_config

( dict[str, Any] ,

) — The exllama config. You can specify the version of the exllama kernel through the  version  key. Defaults to  {"version": 1}  if unset.

cache_block_outputs

, defaults to  True ) — Whether to cache block outputs to reuse as inputs for the succeeding block.

modules_in_block_to_quantize

( list[list[str]] ,

) — List of list of module names to quantize in the specified block. This argument is useful to exclude certain linear modules from being quantized. The block to quantize can be specified by setting  block_name_to_quantize . We will quantize each list sequentially. If not set, we will quantize all linear layers. Example:  modules_in_block_to_quantize =[["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"], ["self_attn.o_proj"]] . In this example, we will first quantize the q,k,v layers simultaneously since they are independent. Then, we will quantize  self_attn.o_proj  layer with the q,k,v layers quantized. This way, we will get better results since it reflects the real input  self_attn.o_proj  will get when the model is quantized.

This is a wrapper class about all possible attributes and features that you can play with a model that has been loaded using  optimum  api for gptq quantization relying on auto_gptq backend.

from_dict_optimum

<   source   >

(   config_dict   )

## Get compatible class with optimum gptq config dict

<   source   >

## Safety checker that arguments are correct

to_dict_optimum

<   source   >

## Get compatible dict for optimum gptq config

## BitsAndBytesConfig

class   transformers. BitsAndBytesConfig

<   source   >

(   load_in_8bit  = False   load_in_4bit  = False   llm_int8_threshold  = 6.0   llm_int8_skip_modules  = None   llm_int8_enable_fp32_cpu_offload  = False   llm_int8_has_fp16_weight  = False   bnb_4bit_compute_dtype  = None   bnb_4bit_quant_type  = 'fp4'   bnb_4bit_use_double_quant  = False   bnb_4bit_quant_storage  = None   **kwargs   )

## Parameters

load_in_8bit

, defaults to  False ) — This flag is used to enable 8-bit quantization with LLM.int8().

load_in_4bit

, defaults to  False ) — This flag is used to enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from  bitsandbytes .

llm_int8_threshold

( float ,

, defaults to 6.0) — This corresponds to the outlier threshold for outlier detection as described in  LLM.int8() : 8-bit Matrix Multiplication for Transformers at Scale  paper:

https://huggingface.co/papers/2208.07339

https://huggingface.co/papers/2208.07339

Any hidden states value that is above this threshold will be considered an outlier and the operation on those values will be done in fp16. Values are usually normally distributed, that is, most values are in the range [-3.5, 3.5], but there are some exceptional systematic outliers that are very differently distributed for large models. These outliers are often in the interval [-60, -6] or [6, 60]. Int8 quantization works well for values of magnitude ~5, but beyond that, there is a significant performance penalty. A good default threshold is 6, but a lower threshold might be needed for more unstable models (small models, fine-tuning).

llm_int8_skip_modules

( list[str] ,

) — An explicit list of the modules that we do not want to convert in 8-bit. This is useful for models such as Jukebox that has several heads in different places and not necessarily at the last position. For example for  CausalLM  models, the last  lm_head  is kept in its original  dtype .

llm_int8_enable_fp32_cpu_offload

, defaults to  False ) — This flag is used for advanced use cases and users that are aware of this feature. If you want to split your model in different parts and run some parts in int8 on GPU and some parts in fp32 on CPU, you can use this flag. This is useful for offloading large models such as  google/flan-t5-xxl . Note that the int8 operations will not be run on CPU.

llm_int8_has_fp16_weight

, defaults to  False ) — This flag runs LLM.int8() with 16-bit main weights. This is useful for fine-tuning as the weights do not have to be converted back and forth for the backward pass.

bnb_4bit_compute_dtype

( torch.dtype  or str,

, defaults to  torch.float32 ) — This sets the computational type which might be different than the input type. For example, inputs might be fp32, but computation can be set to bf16 for speedups.

bnb_4bit_quant_type

, defaults to  "fp4" ) — This sets the quantization data type in the bnb.nn.Linear4Bit layers. Options are FP4 and NF4 data types which are specified by  fp4  or  nf4 .

bnb_4bit_use_double_quant

, defaults to  False ) — This flag is used for nested quantization where the quantization constants from the first quantization are quantized again.

bnb_4bit_quant_storage

( torch.dtype  or str,

, defaults to  torch.uint8 ) — This sets the storage type to pack the quantized 4-bit params.

( dict[str, Any] ,

) — Additional parameters from which to initialize the configuration object.

This is a wrapper class about all possible attributes and features that you can play with a model that has been loaded using  bitsandbytes .

This replaces  load_in_8bit  or  load_in_4bit therefore both options are mutually exclusive.

Currently only supports  LLM.int8() ,  FP4 , and  NF4  quantization. If more methods are added to  bitsandbytes , then more arguments will be added to this class.

is_quantizable

<   source   >

Returns  True  if the model is quantizable,  False  otherwise.

<   source   >

Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.

quantization_method

<   source   >

This method returns the quantization method used for the model. If the model is not quantizable, it returns  None .

to_diff_dict

<   source   >

(   )   →   dict[str, Any]

dict[str, Any]

Dictionary of all the attributes that make up this configuration instance,

Removes all attributes from config which correspond to the default config attributes for better readability and serializes to a Python dictionary.

## HfQuantizer

class   transformers.quantizers. HfQuantizer

<   source   >

(   quantization_config : QuantizationConfigMixin   **kwargs   )

Abstract class of the HuggingFace quantizer. Supports for now quantizing HF transformers models for inference and/or quantization. This class is used only for transformers.PreTrainedModel.from_pretrained and cannot be easily used outside the scope of that method yet.

Attributes quantization_config ( transformers.utils.quantization_config.QuantizationConfigMixin ): The quantization config that defines the quantization parameters of your model that you want to quantize. modules_to_not_convert ( list[str] ,

): The list of module names to not convert when quantizing the model. required_packages ( list[str] ,

): The list of required pip packages to install prior to using the quantizer requires_calibration ( bool ): Whether the quantization method requires to calibrate the model before using it. requires_parameters_quantization ( bool ): Whether the quantization method requires to create a new Parameter. For example, for bitsandbytes, it is required to create a new xxxParameter in order to properly quantize the model.

adjust_max_memory

<   source   >

(   max_memory : dict   )

adjust max_memory argument for infer_auto_device_map() if extra memory is needed for quantization

adjust_target_dtype

<   source   >

(   dtype : torch.dtype   )

## Parameters

( torch.dtype ,

) — The dtype that is used to compute the device_map.

Override this method if you want to adjust the  target_dtype  variable used in  from_pretrained  to compute the device_map in case the device_map is a  str . E.g. for bitsandbytes we force-set  target_dtype  to  torch.int8  and for 4-bit we pass a custom enum  accelerate.CustomDtype.int4 .

check_quantized_param

<   source   >

(   *args   **kwargs   )

DEPRECATED -> remove in v5

create_quantized_param

<   source   >

(   *args   **kwargs   )

Take needed components from state_dict (those from which  param_needs_quantization  is True) and create quantized param. It usually also load the new param directly in the  model . Note: only applicable if requires_parameters_quantization == True.

<   source   >

(   model   )

Potentially dequantize the model to retrieve the original model, with some loss in accuracy / performance. Note not all quantization schemes support this.

get_accelerator_warm_up_factor

<   source   >

The factor to be used in  caching_allocator_warmup  to get the number of bytes to pre-allocate to warm up accelerator. A factor of 2 means we allocate all bytes in the empty model (since we allocate in fp16), a factor of 4 means we allocate half the memory of the weights residing in the empty model, etc…

get_param_name

<   source   >

(   param_name : str   )

Override this method if you want to adjust the  param_name .

get_special_dtypes_update

<   source   >

(   model   dtype : torch.dtype   )

## Parameters

( ~transformers.PreTrainedModel ) — The model to quantize

( torch.dtype ) — The dtype passed in  from_pretrained  method.

returns dtypes for modules that are not quantized - used for the computation of the device_map in case one passes a str as a device_map. The method will use the  modules_to_not_convert  that is modified in  _process_model_before_weight_loading .

get_state_dict_and_metadata

<   source   >

(   model   safe_serialization  = False   )

Get state dict and metadata. Useful when we need to modify a bit the state dict due to quantization

param_needs_quantization

<   source   >

(   model : PreTrainedModel   param_name : str   **kwargs   )

Check whether a given param needs quantization as defined by  create_quantized_param .

postprocess_model

<   source   >

(   model : PreTrainedModel   **kwargs   )

## Parameters

( ~transformers.PreTrainedModel ) — The model to quantize

) — The keyword arguments that are passed along  _process_model_after_weight_loading .

Post-process the model post weights loading. Make sure to override the abstract method  _process_model_after_weight_loading .

preprocess_model

<   source   >

(   model : PreTrainedModel   **kwargs   )

## Parameters

( ~transformers.PreTrainedModel ) — The model to quantize

) — The keyword arguments that are passed along  _process_model_before_weight_loading .

Setting model attributes and/or converting model before weights loading. At this point the model should be initialized on the meta device so you can freely manipulate the skeleton of the model in order to replace modules in-place. Make sure to override the abstract method  _process_model_before_weight_loading .

remove_quantization_config

<   source   >

(   model   )

Remove the quantization config from the model.

update_device_map

<   source   >

(   device_map : typing.Optional[dict[str, typing.Any]]   )

## Parameters

( Union[dict, str] ,

) — The device_map that is passed through the  from_pretrained  method.

Override this method if you want to pass a override the existing device map with a new one. E.g. for bitsandbytes, since  accelerate  is a hard requirement, if no device_map is passed, the device_map is set to `“auto”“

update_dtype

<   source   >

(   dtype : torch.dtype   )

## Parameters

( torch.dtype ) — The input dtype that is passed in  from_pretrained

Some quantization methods require to explicitly set the dtype of the model to a target dtype. You need to override this method in case you want to make sure that behavior is preserved

update_ep_plan

<   source   >

(   config   )

updates the tp plan for the scales

update_expected_keys

<   source   >

(   model   expected_keys : list   loaded_keys : list   )

## Parameters

expected_keys

( list[str] ,

) — The list of the expected keys in the initialized model.

loaded_keys

( list[str] ,

) — The list of the loaded keys in the checkpoint.

Override this method if you want to adjust the  update_expected_keys .

update_missing_keys

<   source   >

(   model   missing_keys : list   prefix : str   )

## Parameters

missing_keys

( list[str] ,

) — The list of missing keys in the checkpoint compared to the state dict of the model

Override this method if you want to adjust the  missing_keys .

update_state_dict_with_metadata

<   source   >

(   state_dict   metadata   )

Update state dict with metadata. Default behaviour returns state_dict

update_torch_dtype

<   source   >

(   dtype : torch.dtype   )

## Parameters

( torch.dtype ) — The input dtype that is passed in  from_pretrained

Deprecared in favor of  update_dtype !

update_tp_plan

<   source   >

(   config   )

updates the tp plan for the scales

validate_environment

<   source   >

(   *args   **kwargs   )

This method is used to potentially check for potential conflicts with arguments that are passed in  from_pretrained . You need to define it for all future quantizers that are integrated with transformers. If no explicit check are needed, simply return nothing.

## HiggsConfig

class   transformers. HiggsConfig

<   source   >

(   bits : int = 4   p : int = 2   modules_to_not_convert : typing.Optional[list[str]] = None   hadamard_size : int = 512   group_size : int = 256   tune_metadata : typing.Optional[dict[str, typing.Any]] = None   **kwargs   )

## Parameters

, defaults to 4) — Number of bits to use for quantization. Can be 2, 3 or 4. Default is 4.

, defaults to 2) — Quantization grid dimension. 1 and 2 are supported. 2 is always better in practice. Default is 2.

modules_to_not_convert

, default to [“lm_head”]) — List of linear layers that should not be quantized.

hadamard_size

, defaults to 512) — Hadamard size for the HIGGS method. Default is 512. Input dimension of matrices is padded to this value. Decreasing this below 512 will reduce the quality of the quantization.

, defaults to 256) — Group size for the HIGGS method. Can be 64, 128 or 256. Decreasing it barely affects the performance. Default is 256. Must be a divisor of hadamard_size.

tune_metadata

, defaults to {}) — Module-wise metadata (gemm block shapes, GPU metadata, etc.) for saving the kernel tuning results. Default is an empty dictionary. Is set automatically during tuning.

HiggsConfig is a configuration class for quantization using the HIGGS method.

<   source   >

Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.

class   transformers. HqqConfig

<   source   >

(   nbits : int = 4   group_size : int = 64   view_as_float : bool = False   axis : typing.Optional[int] = None   dynamic_config : typing.Optional[dict] = None   skip_modules : list = ['lm_head']   **kwargs   )

## Parameters

, defaults to 4) — Number of bits. Supported values are (8, 4, 3, 2, 1).

, defaults to 64) — Group-size value. Supported values are any value that is divisible by weight.shape[axis]).

view_as_float

, defaults to  False ) — View the quantized weight as float (used in distributed training) if set to  True .

( Optional[int] ,

) — Axis along which grouping is performed. Supported values are 0 or 1.

dynamic_config

) — Parameters for dynamic configuration. The key is the name tag of the layer and the value is a quantization config. If set, each layer specified by its id will use its dedicated quantization configuration.

skip_modules

( list[str] ,

, defaults to  ['lm_head'] ) — List of  nn.Linear  layers to skip.

( dict[str, Any] ,

) — Additional parameters from which to initialize the configuration object.

This is wrapper around hqq’s BaseQuantizeConfig.

<   source   >

(   config : dict   )

Override from_dict, used in AutoQuantizationConfig.from_dict in quantizers/auto.py

<   source   >

Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.

to_diff_dict

<   source   >

(   )   →   dict[str, Any]

dict[str, Any]

Dictionary of all the attributes that make up this configuration instance,

Removes all attributes from config which correspond to the default config attributes for better readability and serializes to a Python dictionary.

Mxfp4Config

class   transformers. Mxfp4Config

<   source   >

(   modules_to_not_convert : typing.Optional[list] = None   dequantize : bool = False   **kwargs   )

## Parameters

modules_to_not_convert

, default to  None ) — The list of modules to not quantize, useful for quantizing models that explicitly require to have some modules left in their original precision.

, default to  False ) — Whether we dequantize the model to bf16 precision or not

This is a wrapper class about all possible attributes and features that you can play with a model that has been loaded using mxfp4 quantization.

FbgemmFp8Config

class   transformers. FbgemmFp8Config

<   source   >

(   activation_scale_ub : float = 1200.0   modules_to_not_convert : typing.Optional[list] = None   **kwargs   )

## Parameters

activation_scale_ub

( float ,

, defaults to 1200.0) — The activation scale upper bound. This is used when quantizing the input activation.

modules_to_not_convert

, default to  None ) — The list of modules to not quantize, useful for quantizing models that explicitly require to have some modules left in their original precision.

This is a wrapper class about all possible attributes and features that you can play with a model that has been loaded using fbgemm fp8 quantization.

## CompressedTensorsConfig

class   transformers. CompressedTensorsConfig

<   source   >

(   config_groups : typing.Optional[dict[str, typing.Union[ForwardRef('QuantizationScheme'), list[str]]]] = None   format : str = 'dense'   quantization_status : QuantizationStatus = 'initialized'   kv_cache_scheme : typing.Optional[ForwardRef('QuantizationArgs')] = None   global_compression_ratio : typing.Optional[float] = None   ignore : typing.Optional[list[str]] = None   sparsity_config : typing.Optional[dict[str, typing.Any]] = None   quant_method : str = 'compressed-tensors'   run_compressed : bool = True   **kwargs   )

## Parameters

config_groups

( typing.dict[str, typing.Union[ForwardRef('QuantizationScheme'), typing.list[str]]] ,

) — dictionary mapping group name to a quantization scheme definition

, defaults to  "dense" ) — format the model is represented as. Set  run_compressed  True to execute model as the compressed format if not  dense

quantization_status

( QuantizationStatus ,

, defaults to  "initialized" ) — status of model in the quantization lifecycle, ie ‘initialized’, ‘calibration’, ‘frozen’

kv_cache_scheme

( typing.Union[QuantizationArgs, NoneType] ,

) — specifies quantization of the kv cache. If None, kv cache is not quantized.

global_compression_ratio

( typing.Union[float, NoneType] ,

) — 0-1 float percentage of model compression

( typing.Union[typing.list[str], NoneType] ,

) — layer names or types to not quantize, supports regex prefixed by ‘re:’

sparsity_config

( typing.dict[str, typing.Any] ,

) — configuration for sparsity compression

quant_method

, defaults to  "compressed-tensors" ) — do not override, should be compressed-tensors

run_compressed

, defaults to  True ) — alter submodules (usually linear) in order to emulate compressed model execution if True, otherwise use default submodule

This is a wrapper class that handles compressed-tensors quantization config options. It is a wrapper around  compressed_tensors.QuantizationConfig

<   source   >

(   config_dict   return_unused_kwargs  = False   **kwargs   )   →   QuantizationConfigMixin

## Parameters

config_dict

( dict[str, Any] ) — Dictionary that will be used to instantiate the configuration object.

return_unused_kwargs

, defaults to  False ) — Whether or not to return a list of unused keyword arguments. Used for  from_pretrained  method in  PreTrainedModel .

( dict[str, Any] ) — Additional parameters from which to initialize the configuration object.

## QuantizationConfigMixin

The configuration object instantiated from those parameters.

## Instantiates a

## CompressedTensorsConfig

https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/quantization#transformers.CompressedTensorsConfig

from a Python dictionary of parameters. Optionally unwraps any args from the nested quantization_config

<   source   >

https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/quantization#transformers.CompressedTensorsConfig

Quantization config to be added to config.json

Serializes this instance to a Python dictionary. Returns:  dict[str, Any] : Dictionary of all the attributes that make up this configuration instance.

to_diff_dict

<   source   >

https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/quantization#transformers.CompressedTensorsConfig

(   )   →   dict[str, Any]

dict[str, Any]

Dictionary of all the attributes that make up this configuration instance,

Removes all attributes from config which correspond to the default config attributes for better readability and serializes to a Python dictionary.

## TorchAoConfig

class   transformers. TorchAoConfig

<   source   >

https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/quantization#transformers.CompressedTensorsConfig

(   quant_type : typing.Union[str, ForwardRef('AOBaseConfig')]   modules_to_not_convert : typing.Optional[list] = None   include_input_output_embeddings : bool = False   untie_embedding_weights : bool = False   **kwargs   )

<   source   >

https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/quantization#transformers.CompressedTensorsConfig

(   config_dict   return_unused_kwargs  = False   **kwargs   )

Create configuration from a dictionary.

get_apply_tensor_subclass

<   source   >

https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/quantization#transformers.CompressedTensorsConfig

Create the appropriate quantization method based on configuration.

<   source   >

https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/quantization#transformers.CompressedTensorsConfig

Validate configuration and set defaults.

<   source   >

https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/quantization#transformers.CompressedTensorsConfig

Convert configuration to a dictionary.

## BitNetQuantConfig

class   transformers. BitNetQuantConfig

<   source   >

https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/quantization#transformers.CompressedTensorsConfig

(   modules_to_not_convert : typing.Optional[list] = None   linear_class : str = 'bitlinear'   quantization_mode : str = 'offline'   use_rms_norm : bool = False   rms_norm_eps : typing.Optional[float] = 1e-06   **kwargs   )

## Parameters

modules_to_not_convert

( Optional[List] ,

) — Optionally, provides a list of full paths of  nn.Linear  weight parameters that shall not be quantized. Defaults to None.

linear_class

, defaults to  "bitlinear" ) — The type of linear class to use. Can be either  bitlinear  or  autobitlinear .

quantization_mode

, defaults to  "offline" ) — The quantization mode to use. Can be either  online  or  offline . In  online  mode, the weight quantization parameters are calculated dynamically during each forward pass (e.g., based on the current weight values). This can adapt to weight changes during training (Quantization-Aware Training - QAT). In  offline  mode, quantization parameters are pre-calculated

inference. These parameters are then fixed and loaded into the quantized model. This generally results in lower runtime overhead compared to online quantization.

use_rms_norm

, defaults to  False ) — Whether to apply RMSNorm on the activations before quantization. This matches the original BitNet paper’s approach of normalizing activations before quantization/packing.

rms_norm_eps

( float ,

, defaults to 1e-06) — The epsilon value used in the RMSNorm layer for numerical stability.

( dict[str, Any] ,

) — Additional keyword arguments that may be used by specific quantization backends or future versions.

Configuration class for applying BitNet quantization.

<   source   >

## Safety checker that arguments are correct

class   transformers. SpQRConfig

<   source   >

(   bits : int = 3   beta1 : int = 16   beta2 : int = 16   shapes : typing.Optional[dict[str, int]] = None   modules_to_not_convert : typing.Optional[list[str]] = None   **kwargs   )

## Parameters

, defaults to 3) — Specifies the bit count for the weights and first order zero-points and scales. Currently only bits = 3 is supported.

, defaults to 16) — SpQR tile width. Currently only beta1 = 16 is supported.

, defaults to 16) — SpQR tile height. Currently only beta2 = 16 is supported.

( Optional ,

) — A dictionary holding the shape of each object. We need this because it’s impossible to deduce the exact size of the parameters just from bits, beta1, beta2.

modules_to_not_convert

( Optional[list[str]] ,

) — Optionally, provides a list of full paths of  nn.Linear  weight parameters that shall not be quantized. Defaults to None.

( dict[str, Any] ,

) — Additional parameters from which to initialize the configuration object.

This is a wrapper class about  spqr  parameters. Refer to the original publication for more details.

<   source   >

Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.

FineGrainedFP8Config

class   transformers. FineGrainedFP8Config

<   source   >

(   activation_scheme : str = 'dynamic'   weight_block_size : tuple = (128, 128)   modules_to_not_convert : typing.Optional[list] = None   **kwargs   )

## Parameters

activation_scheme

, defaults to  "dynamic" ) — The scheme used for activation, the defaults and only support scheme for now is “dynamic”.

weight_block_size

( typing.tuple[int, int] ,

, defaults to  (128, 128) ) — The size of the weight blocks for quantization, default is (128, 128).

modules_to_not_convert

) — A list of module names that should not be converted during quantization.

FineGrainedFP8Config is a configuration class for fine-grained FP8 quantization used mainly for deepseek models.

<   source   >

## Safety checker that arguments are correct

## QuarkConfig

class   transformers. QuarkConfig

<   source   >

(   **kwargs   )

## FPQuantConfig

class   transformers. FPQuantConfig

<   source   >

(   forward_dtype : str = 'nvfp4'   forward_method : str = 'abs_max'   backward_dtype : str = 'bf16'   store_master_weights : bool = False   hadamard_group_size : typing.Optional[int] = None   pseudoquantization : bool = False   transform_init : str = 'hadamard'   modules_to_not_convert : typing.Optional[list[str]] = None   **kwargs   )

## Parameters

forward_dtype

, defaults to  "nvfp4" ) — The dtype to use for the forward pass.

forward_method

, defaults to  "abs_max" ) — The scaling to use for the forward pass. Can be  "abs_max"  or  "quest" .  "abs_max"  is better for PTQ,  "quest"  is better for QAT.

backward_dtype

, defaults to  "bf16" ) — The dtype to use for the backward pass.

store_master_weights

, defaults to  False ) — Whether to store the master weights. Needed for QAT over layer weights.

hadamard_group_size

) — The group size for the hadamard transform before quantization for  "quest"  it matches the MXFP4 group size (32). If  None , it will be set to 16 for  "nvfp4"  and 32 for  "mxfp4" .

pseudoquantization

, defaults to  False ) — Whether to use Triton-based pseudo-quantization. Is mandatory for non-Blackwell GPUs. Doesn’t provide any speedup. For debugging purposes.

transform_init

, defaults to  "hadamard" ) — a method to initialize the pre-processing matrix with. Can be  "hadamard" ,  "identity"  or  "gsr" .

modules_to_not_convert

) — The list of modules to not quantize, useful for quantizing models that explicitly require to have some modules left in their original precision.

FPQuantConfig is a configuration class for quantization using the FPQuant method.

<   source   >

Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.

## AutoRoundConfig

class   transformers. AutoRoundConfig

<   source   >

(   bits : int = 4   group_size : int = 128   sym : bool = True   backend : str = 'auto'   **kwargs   )

## Parameters

, defaults to 4) — The number of bits to quantize to, supported numbers are (2, 3, 4, 8).

, defaults to 128) — Group-size value

, defaults to  True ) — Symmetric quantization or not

, defaults to  "auto" ) — The kernel to use, e.g., ipex,marlin, exllamav2, triton, etc. Ref.

https://github.com/intel/auto-round?tab=readme-ov-file#specify-backend

https://github.com/intel/auto-round?tab=readme-ov-file#specify-backend

This is a wrapper class about all possible attributes and features that you can play with a model that has been loaded AutoRound quantization.

<   source   >

https://github.com/intel/auto-round?tab=readme-ov-file#specify-backend

Safety checker that arguments are correct.

## Update  on GitHub

https://github.com/intel/auto-round?tab=readme-ov-file#specify-backend

