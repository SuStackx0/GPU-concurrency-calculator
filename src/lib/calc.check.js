// Plain sanity check, no test framework. Run with: node src/lib/calc.check.js
import {
  weightMemGB,
  kvBudgetGB,
  bytesPerTokenKV,
  maxConcurrentCapacity,
  throughputCurve,
} from './calc.js'

const approx = (actual, expected, tol) => Math.abs(actual - expected) < tol

// DGX Spark (128GB, 273GB/s) + Qwen2.5-32B (32.5B params, 64 layers, 8 kv_heads,
// 128 head_dim) at fp8/fp8, 4096 seq len, 8% reserve.
const w = weightMemGB(32.5, 1)
console.assert(approx(w, 32.5, 0.1), `weightMemGB expected ~32.5, got ${w}`)

const kvBudget = kvBudgetGB(128, w, 0.08)
console.assert(approx(kvBudget, 85.3, 0.1), `kvBudgetGB expected ~85.3, got ${kvBudget}`)

const bptkv = bytesPerTokenKV(64, 8, 128, 1)
console.assert(bptkv === 131072, `bytesPerTokenKV expected 131072, got ${bptkv}`)

const maxConcurrent = maxConcurrentCapacity(kvBudget, bptkv, 4096)
console.assert(approx(maxConcurrent, 158, 1), `maxConcurrentCapacity expected ~158, got ${maxConcurrent}`)

const curve = throughputCurve(w, bptkv, 4096, 273, maxConcurrent)
const atMax = curve.find((p) => p.batch === maxConcurrent)
console.assert(
  approx(atMax.tokPerSec, 367, 5),
  `tokPerSec at batch=${maxConcurrent} expected ~367, got ${atMax.tokPerSec}`,
)

console.log('calc.check.js: all assertions passed')
console.log({ weightMemGB: w, kvBudgetGB: kvBudget, bytesPerTokenKV: bptkv, maxConcurrent, tokPerSecAtMax: atMax.tokPerSec })
