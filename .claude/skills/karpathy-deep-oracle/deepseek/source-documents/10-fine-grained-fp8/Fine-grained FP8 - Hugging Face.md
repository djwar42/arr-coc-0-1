---
sourceFile: "Fine-grained FP8 - Hugging Face"
exportedBy: "Kortex"
exportDate: "2025-10-29T02:37:36.106Z"
---

# Fine-grained FP8 - Hugging Face

e4c490d2-6efe-408f-ad64-e0a6fcd0e02d

Fine-grained FP8 - Hugging Face

c66f34d9-7477-4e42-bb58-3abcb743aa5a

https://huggingface.co/docs/transformers/en/quantization/finegrained_fp8

## Hugging Face

## Enterprise

## Transformers documentation

Fine-grained FP8

## Transformers

## Join the Hugging Face community

and get access to the augmented documentation experience

Collaborate on models, datasets and Spaces  Faster examples with accelerated inference  Switch between documentation themes

to get started

Fine-grained FP8

Fine-grained FP8 quantization quantizes the weights and activations to fp8.

The weights are quantized to 8-bits for each 2D block ( weight_block_size=(128, 128) ).

The activations are quantized to 8-bits for each group per token. The group value matches the weights in the input channel (128 by default).

FP8 quantization enables support for

DeepSeek-V3

https://hf.co/papers/2412.19437

and DeepSeek-R1.

You need a GPU with Compute Capability>=9 (H100), and install a PyTorch version compatible with the CUDA version of your GPU.

Install Accelerate and upgrade to the latest version of PyTorch.

pip install --upgrade accelerate torch

FineGrainedFP8Config

https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/quantization#transformers.FineGrainedFP8Config

class and pass it to

from_pretrained()

https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained

to quantize it. The weights are loaded in full precision ( torch.float32 ) by default regardless of the actual data type the weights are stored in. Set  dtype="auto"  to load the weights in the data type defined in a models  config.json  file to automatically load the most memory-optiomal data type.

from  transformers  import  FineGrainedFP8Config, AutoModelForCausalLM, AutoTokenizer model_name =  "meta-llama/Meta-Llama-3-8B"  quantization_config = FineGrainedFP8Config() quantized_model = AutoModelForCausalLM.from_pretrained(model_name, dtype= "auto" , device_map= "auto" , quantization_config=quantization_config) tokenizer = AutoTokenizer.from_pretrained(model_name) input_text =  "What are we having for dinner?"  input_ids = tokenizer(input_text, return_tensors= "pt" ).to(quantized_model.device. type ) output = quantized_model.generate(**input_ids, max_new_tokens= 10 )  print (tokenizer.decode(output[ 0 ], skip_special_tokens= True ))

save_pretrained()

https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/model#transformers.PreTrainedModel.save_pretrained

to save the quantized model and reload it with

from_pretrained()

https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained

quant_path =  "/path/to/save/quantized/model"  model.save_pretrained(quant_path) model = AutoModelForCausalLM.from_pretrained(quant_path, device_map= "auto" )

## Update  on GitHub

https://github.com/huggingface/transformers/blob/main/docs/source/en/quantization/finegrained_fp8.md

