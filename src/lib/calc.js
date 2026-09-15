// Pure functions for GPU inference concurrency planning. No React, no I/O.

export function weightMemGB(paramsB, weightBytesPerParam) {
  return paramsB * weightBytesPerParam
}

export function kvBudgetGB(totalMemGB, weightMemGB, reservePct) {
  return totalMemGB * (1 - reservePct) - weightMemGB
}

export function bytesPerTokenKV(layers, kvHeads, headDim, kvBytesPerElem) {
  return 2 * layers * kvHeads * headDim * kvBytesPerElem
}

export function maxConcurrentCapacity(kvBudgetGB, bytesPerTokenKV, avgSeqLen) {
  if (kvBudgetGB <= 0) return 0
  return Math.floor((kvBudgetGB * 1e9) / (bytesPerTokenKV * avgSeqLen))
}

const BATCH_STEPS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

export function throughputCurve(weightMemGB, bytesPerTokenKV, avgSeqLen, bandwidthGBs, maxConcurrent) {
  if (maxConcurrent <= 0) return []

  const batches = [...new Set([...BATCH_STEPS.filter((b) => b <= maxConcurrent), maxConcurrent])].sort(
    (a, b) => a - b,
  )

  return batches.map((batch) => {
    const bytesMoved = weightMemGB * 1e9 + batch * bytesPerTokenKV * avgSeqLen
    const timePerStep = bytesMoved / (bandwidthGBs * 1e9)
    const tokPerSec = batch / timePerStep
    return { batch, tokPerSec }
  })
}

export function findKnee(points, marginalThreshold = 0.15) {
  for (let i = 1; i < points.length; i++) {
    const prev = points[i - 1].tokPerSec
    const curr = points[i].tokPerSec
    const marginalGain = (curr - prev) / prev
    if (marginalGain < marginalThreshold) {
      return points[i].batch
    }
  }
  return points.length ? points[points.length - 1].batch : 0
}

export function bottleneck(knee, maxConcurrentCapacity) {
  return knee < 0.6 * maxConcurrentCapacity ? 'bandwidth' : 'capacity'
}
