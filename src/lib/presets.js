// Reference specs for common inference hardware and model architectures.
// Approximate public figures; edit freely for your actual deployment target.

export const HARDWARE_PRESETS = {
  'DGX Spark': { memGB: 128, bandwidthGBs: 273 },
  'RTX 4090 24GB': { memGB: 24, bandwidthGBs: 1008 },
  'RTX 5090 32GB': { memGB: 32, bandwidthGBs: 1792 },
  'A100 80GB': { memGB: 80, bandwidthGBs: 2039 },
  'H100 80GB': { memGB: 80, bandwidthGBs: 3350 },
  custom: null,
}

export const MODEL_PRESETS = {
  'Qwen2.5-7B': { paramsB: 7.6, layers: 28, kvHeads: 4, headDim: 128 },
  'Qwen2.5-14B': { paramsB: 14.7, layers: 48, kvHeads: 8, headDim: 128 },
  'Qwen2.5-32B': { paramsB: 32.5, layers: 64, kvHeads: 8, headDim: 128 },
  'Qwen2.5-72B': { paramsB: 72.7, layers: 80, kvHeads: 8, headDim: 128 },
  'Llama-3.1-8B': { paramsB: 8.0, layers: 32, kvHeads: 8, headDim: 128 },
  'Llama-3.1-70B': { paramsB: 70.6, layers: 80, kvHeads: 8, headDim: 128 },
  custom: null,
}

// Bytes per element for each supported numeric format.
export const DTYPE_BYTES = {
  fp32: 4,
  fp16: 2,
  bf16: 2,
  fp8: 1,
  int4: 0.5,
}
