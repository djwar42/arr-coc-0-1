# LLM Post-Training Quantization Optimization - Study

**Source**: Article (Optimizing LLMs for Performance and Accuracy with Post-Training Quantization)
**Date Processed**: 2025-10-29
**Category**: LOW - General PTQ Guide

---

## 📝 TL;DR

General article on post-training quantization (PTQ) for LLMs. Quantize after training to reduce size/speed up inference. Standard techniques: calibration, weight-only quant, activation quant. Not DeepSeek-specific - general PTQ overview.

---

## 🔗 Connections

- **05-fp8-lm-paper**: DeepSeek's training-time FP8 (not PTQ)
- **70-quantization-hf**: HF quantization docs

---

## 💭 Karpathy Take

PTQ = take trained model, quantize it for deployment. DeepSeek does TRAINING in FP8, which is harder but better quality than PTQ. This doc is about the easier/standard approach: train in FP32, quantize later.
