# capacity-planner

A GPU inference concurrency calculator: given a GPU's memory and memory bandwidth, a model's
architecture (params, layers, KV heads, head dim), and a weight/KV-cache precision, it computes
how much memory the model weights consume, how much is left over for KV cache after a reserve
margin, how many concurrent requests that KV budget can hold, and — from a token-throughput curve
over batch size — whether the deployment is bandwidth-bound (throughput plateaus early, so scale
horizontally with more replicas) or capacity-bound (throughput keeps climbing until memory runs
out, so scale vertically with more memory per replica).

The four core formulas:

```
weightMemGB      = paramsB * weightBytesPerParam
kvBudgetGB       = totalMemGB * (1 - reservePct) - weightMemGB
bytesPerTokenKV  = 2 * layers * kvHeads * headDim * kvBytesPerElem
maxConcurrent    = floor(kvBudgetGB * 1e9 / (bytesPerTokenKV * avgSeqLen))
```

## Run it

```
npm install && npm run dev
```

`npm run check` runs a plain-Node sanity check (`src/lib/calc.check.js`, no test framework)
against known DGX Spark + Qwen2.5-32B numbers.
