# Methodology

How `capacity-planner` turns hardware + model specs into a concurrency verdict, and what each
number actually depends on. Read this if a result surprises you before assuming it's wrong.

## 1. Memory split: weights vs. KV cache

Every request needs two pools of GPU memory:

- **Weights** — loaded once, shared by every concurrent request.
- **KV cache** — grows per request, per token generated.

```
weightMemGB = paramsB * weightBytesPerParam
kvBudgetGB  = totalMemGB * (1 - reservePct) - weightMemGB
```

`reservePct` (a fraction, e.g. `0.08` for 8%) sets aside headroom for activations, CUDA context,
fragmentation, etc. Whatever's left after weights and the reserve is what concurrent requests'
KV caches have to share. If `weightMemGB` alone exceeds `totalMemGB * (1 - reservePct)`,
`kvBudgetGB` goes negative — the model doesn't fit at all, and the UI shows that state directly
instead of a chart.

## 2. Per-token KV cost

```
bytesPerTokenKV = 2 * layers * kvHeads * headDim * kvBytesPerElem
```

The `2` is for K and V (one tensor each, same shape). This number is **per token, per request** —
it doesn't depend on batch size or sequence length. A request holding `avgSeqLen` tokens of
context costs `bytesPerTokenKV * avgSeqLen` in KV memory.

## 3. How many requests fit: `maxConcurrentCapacity`

```
maxConcurrent = floor(kvBudgetGB * 1e9 / (bytesPerTokenKV * avgSeqLen))
```

This is a pure **memory-capacity** number. It has nothing to do with GPU speed or bandwidth —
it only asks "how many `avgSeqLen`-sized KV caches fit in the leftover memory pool." Doubling
`avgSeqLen` halves `maxConcurrent`; doubling `kvBudgetGB` doubles it.

## 4. The throughput curve — and where bandwidth actually enters

Every decode step moves the same fixed set of weight bytes (once) plus one KV-cache read/write
per token in the batch:

```
bytesMoved   = weightMemGB * 1e9 + batch * bytesPerTokenKV * avgSeqLen
timePerStep  = bytesMoved / (bandwidthGBs * 1e9)
tokPerSec    = batch / timePerStep
```

At `batch = 1`, almost all the bytes moved are the fixed weight load — the KV term is tiny by
comparison. As `batch` grows, the KV term grows linearly while the weight term stays fixed, so
each additional unit of batch buys a shrinking *marginal* increase in `tokPerSec`. That's the
"knee" the chart looks for.

**`bandwidthGBs` is a single scalar multiplier applied identically to every batch size.** It sets
how fast the whole curve moves (10x the bandwidth → 10x `tokPerSec` at every point on the chart),
but it cancels out of any *ratio between two points on the curve*. Concretely, run the same model
and memory config through four wildly different GPUs:

| bandwidth | knee (batch) | maxConcurrent | verdict | tok/s @ maxConcurrent |
|---|---|---|---|---|
| 100 GB/s   | 158 | 158 | capacity | 134.7 |
| 273 GB/s   | 158 | 158 | capacity | 367.6 |
| 1000 GB/s  | 158 | 158 | capacity | 1346.7 |
| 3350 GB/s  | 158 | 158 | capacity | 4511.4 |

The knee batch and the verdict never move — only the absolute `tok/s` scale does. This isn't a
bug; it falls straight out of the algebra: `tokPerSec = batch * bandwidthGBs*1e9 / bytesMoved`, so
`bandwidthGBs` is a common factor in both the numerator and denominator of the marginal-gain ratio
`findKnee` computes, and it cancels.

**What actually moves the knee** is the *ratio* between the fixed weight-load cost and the
per-batch KV-load cost — i.e. `weightMemGB` vs. `bytesPerTokenKV * avgSeqLen`. A model with huge
weights relative to its KV footprint (few KV heads, small head dim, short sequences) plateaus
early — you're paying for the weight load on every step regardless of batch, so more concurrency
buys little. A model with a small weight footprint relative to a large KV footprint (long
sequences, many KV heads) keeps gaining from more batch almost all the way to the memory ceiling.

## 5. Knee detection

```
findKnee(points, marginalThreshold = 0.15):
  return first point[i] where (tokPerSec[i] - tokPerSec[i-1]) / tokPerSec[i-1] < 0.15
```

The default 15% threshold means: once doubling (or stepping up) the batch size buys less than a
15% throughput gain, we call that the point of diminishing returns — the "knee."

## 6. The verdict: bandwidth-bound vs. capacity-bound

```
bottleneck(knee, maxConcurrent):
  return knee < 0.6 * maxConcurrent ? "bandwidth" : "capacity"
```

- **`knee` well below `maxConcurrent`** (< 60% of it) → throughput already plateaued long before
  memory ran out. You're **bandwidth-bound**: the GPU's memory bus is the limiting factor, not
  how much KV cache fits. Scale **horizontally** (more replicas) — a bigger GPU with more memory
  won't help, since you were never using all of it productively.
- **`knee` close to or past `maxConcurrent`** (≥ 60% of it) → throughput kept climbing right up to
  the memory ceiling. You're **capacity-bound**: KV memory ran out before bandwidth did. Scale
  **vertically** (more memory per replica, or a lower-precision KV cache) — that directly raises
  the ceiling and lets throughput keep climbing.

Because bandwidth cancels out of `knee` (see §4), this classification is really answering: *"does
the weight/KV byte ratio for this model, at this sequence length, cause throughput to saturate
well inside the memory budget, or does it keep scaling almost all the way to the wall?"* — a
property of the model architecture and workload shape, not of which specific GPU you plugged in.
