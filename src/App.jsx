import { useMemo, useState } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  bottleneck,
  bytesPerTokenKV,
  findKnee,
  kvBudgetGB,
  maxConcurrentCapacity,
  throughputCurve,
  weightMemGB,
} from './lib/calc.js'
import { DTYPE_BYTES, HARDWARE_PRESETS, MODEL_PRESETS } from './lib/presets.js'
import './App.css'

const fmt = (n, digits = 1) =>
  Number.isFinite(n) ? n.toLocaleString(undefined, { maximumFractionDigits: digits }) : '—'

function App() {
  const [hardwarePreset, setHardwarePreset] = useState('DGX Spark')
  const [memGB, setMemGB] = useState(HARDWARE_PRESETS['DGX Spark'].memGB)
  const [bandwidthGBs, setBandwidthGBs] = useState(HARDWARE_PRESETS['DGX Spark'].bandwidthGBs)

  const [modelPreset, setModelPreset] = useState('Qwen2.5-32B')
  const [paramsB, setParamsB] = useState(MODEL_PRESETS['Qwen2.5-32B'].paramsB)
  const [layers, setLayers] = useState(MODEL_PRESETS['Qwen2.5-32B'].layers)
  const [kvHeads, setKvHeads] = useState(MODEL_PRESETS['Qwen2.5-32B'].kvHeads)
  const [headDim, setHeadDim] = useState(MODEL_PRESETS['Qwen2.5-32B'].headDim)

  const [weightDtype, setWeightDtype] = useState('fp8')
  const [kvDtype, setKvDtype] = useState('fp8')

  const [avgSeqLen, setAvgSeqLen] = useState(4096)
  const [reservePct, setReservePct] = useState(8)

  const handleHardwarePreset = (name) => {
    setHardwarePreset(name)
    const preset = HARDWARE_PRESETS[name]
    if (preset) {
      setMemGB(preset.memGB)
      setBandwidthGBs(preset.bandwidthGBs)
    }
  }

  const handleModelPreset = (name) => {
    setModelPreset(name)
    const preset = MODEL_PRESETS[name]
    if (preset) {
      setParamsB(preset.paramsB)
      setLayers(preset.layers)
      setKvHeads(preset.kvHeads)
      setHeadDim(preset.headDim)
    }
  }

  const results = useMemo(() => {
    const wMem = weightMemGB(paramsB, DTYPE_BYTES[weightDtype])
    const kvBudget = kvBudgetGB(memGB, wMem, reservePct / 100)
    const bptkv = bytesPerTokenKV(layers, kvHeads, headDim, DTYPE_BYTES[kvDtype])
    const maxConcurrent = maxConcurrentCapacity(kvBudget, bptkv, avgSeqLen)
    const curve = throughputCurve(wMem, bptkv, avgSeqLen, bandwidthGBs, maxConcurrent)
    const knee = curve.length ? findKnee(curve) : 0
    const verdict = maxConcurrent > 0 ? bottleneck(knee, maxConcurrent) : null
    return { wMem, kvBudget, bptkv, maxConcurrent, curve, knee, verdict }
  }, [paramsB, weightDtype, memGB, reservePct, layers, kvHeads, headDim, kvDtype, avgSeqLen, bandwidthGBs])

  const { wMem, kvBudget, bptkv, maxConcurrent, curve, knee, verdict } = results
  const kneePct = maxConcurrent > 0 ? (knee / maxConcurrent) * 100 : 0

  return (
    <div className="app">
      <header className="app-header">
        <h1>capacity-planner</h1>
        <p className="subtitle">GPU inference concurrency calculator</p>
      </header>

      <div className="layout">
        <aside className="panel controls">
          <section className="control-group">
            <h2>Hardware</h2>
            <label>
              Preset
              <select value={hardwarePreset} onChange={(e) => handleHardwarePreset(e.target.value)}>
                {Object.keys(HARDWARE_PRESETS).map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Memory (GB)
              <input
                type="number"
                className="mono"
                value={memGB}
                onChange={(e) => {
                  setHardwarePreset('custom')
                  setMemGB(Number(e.target.value))
                }}
              />
            </label>
            <label>
              Bandwidth (GB/s)
              <input
                type="number"
                className="mono"
                value={bandwidthGBs}
                onChange={(e) => {
                  setHardwarePreset('custom')
                  setBandwidthGBs(Number(e.target.value))
                }}
              />
            </label>
          </section>

          <section className="control-group">
            <h2>Model</h2>
            <label>
              Preset
              <select value={modelPreset} onChange={(e) => handleModelPreset(e.target.value)}>
                {Object.keys(MODEL_PRESETS).map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Params (B)
              <input
                type="number"
                className="mono"
                value={paramsB}
                onChange={(e) => {
                  setModelPreset('custom')
                  setParamsB(Number(e.target.value))
                }}
              />
            </label>
            <label>
              Layers
              <input
                type="number"
                className="mono"
                value={layers}
                onChange={(e) => {
                  setModelPreset('custom')
                  setLayers(Number(e.target.value))
                }}
              />
            </label>
            <label>
              KV Heads
              <input
                type="number"
                className="mono"
                value={kvHeads}
                onChange={(e) => {
                  setModelPreset('custom')
                  setKvHeads(Number(e.target.value))
                }}
              />
            </label>
            <label>
              Head Dim
              <input
                type="number"
                className="mono"
                value={headDim}
                onChange={(e) => {
                  setModelPreset('custom')
                  setHeadDim(Number(e.target.value))
                }}
              />
            </label>
          </section>

          <section className="control-group">
            <h2>Precision</h2>
            <label>
              Weight dtype
              <select value={weightDtype} onChange={(e) => setWeightDtype(e.target.value)}>
                {Object.keys(DTYPE_BYTES).map((dt) => (
                  <option key={dt} value={dt}>
                    {dt}
                  </option>
                ))}
              </select>
            </label>
            <label>
              KV cache dtype
              <select value={kvDtype} onChange={(e) => setKvDtype(e.target.value)}>
                {Object.keys(DTYPE_BYTES).map((dt) => (
                  <option key={dt} value={dt}>
                    {dt}
                  </option>
                ))}
              </select>
            </label>
          </section>

          <section className="control-group">
            <h2>Workload</h2>
            <label>
              Avg sequence length
              <span className="mono slider-value">{avgSeqLen}</span>
              <input
                type="range"
                min="128"
                max="32768"
                step="128"
                value={avgSeqLen}
                onChange={(e) => setAvgSeqLen(Number(e.target.value))}
              />
            </label>
            <label>
              Memory reserve
              <span className="mono slider-value">{reservePct}%</span>
              <input
                type="range"
                min="0"
                max="50"
                step="1"
                value={reservePct}
                onChange={(e) => setReservePct(Number(e.target.value))}
              />
            </label>
          </section>
        </aside>

        <main className="results">
          <section className="metric-cards">
            <div className="metric-card">
              <span className="metric-label">Weight Memory</span>
              <span className="metric-value mono">{fmt(wMem)} GB</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">KV Budget</span>
              <span className="metric-value mono">{fmt(kvBudget)} GB</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Bytes / Token (KV)</span>
              <span className="metric-value mono">{fmt(bptkv, 0)} B</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Max Concurrent</span>
              <span className="metric-value mono">{fmt(maxConcurrent, 0)}</span>
            </div>
          </section>

          {maxConcurrent <= 0 ? (
            <section className="panel verdict-panel verdict-bandwidth">
              <h2>Model does not fit</h2>
              <p>
                Weight memory (<span className="mono">{fmt(wMem)} GB</span>) exceeds the memory
                available after reserving <span className="mono">{reservePct}%</span> of{' '}
                <span className="mono">{fmt(memGB)} GB</span>. There is no room left for a KV
                cache — reduce the reserve, pick a smaller model, or add memory.
              </p>
            </section>
          ) : (
            <>
              <section className="panel chart-panel">
                <h2>Throughput vs. Concurrent Batch</h2>
                <ResponsiveContainer width="100%" height={320}>
                  <LineChart data={curve} margin={{ top: 10, right: 24, bottom: 10, left: 0 }}>
                    <CartesianGrid stroke="#1f2b2b" strokeDasharray="3 3" />
                    <XAxis
                      dataKey="batch"
                      stroke="#6b8a8a"
                      tick={{ fill: '#6b8a8a', fontFamily: 'var(--mono)', fontSize: 12 }}
                      label={{ value: 'batch', position: 'insideBottom', offset: -4, fill: '#6b8a8a' }}
                    />
                    <YAxis
                      stroke="#6b8a8a"
                      tick={{ fill: '#6b8a8a', fontFamily: 'var(--mono)', fontSize: 12 }}
                      label={{
                        value: 'tok/s',
                        angle: -90,
                        position: 'insideLeft',
                        fill: '#6b8a8a',
                      }}
                    />
                    <Tooltip
                      contentStyle={{
                        background: '#0d1414',
                        border: '1px solid #2a3a3a',
                        fontFamily: 'var(--mono)',
                        fontSize: 12,
                      }}
                      labelStyle={{ color: '#9fe8c8' }}
                      itemStyle={{ color: '#9fe8c8' }}
                    />
                    <Line
                      type="monotone"
                      dataKey="tokPerSec"
                      stroke="#39d98a"
                      strokeWidth={2}
                      dot={{ r: 2, fill: '#39d98a' }}
                      activeDot={{ r: 4 }}
                    />
                    <ReferenceLine
                      x={knee}
                      stroke="#e8c339"
                      strokeDasharray="4 4"
                      label={{ value: `knee (${knee})`, fill: '#e8c339', fontSize: 12, position: 'top' }}
                    />
                    <ReferenceLine
                      x={maxConcurrent}
                      stroke="#e85f5f"
                      strokeDasharray="4 4"
                      label={{
                        value: `max (${maxConcurrent})`,
                        fill: '#e85f5f',
                        fontSize: 12,
                        position: 'top',
                      }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </section>

              <section
                className={`panel verdict-panel ${verdict === 'bandwidth' ? 'verdict-bandwidth' : 'verdict-capacity'}`}
              >
                <h2>
                  {verdict === 'bandwidth'
                    ? 'bandwidth-bound → scale horizontally'
                    : 'capacity-bound → scale vertically'}
                </h2>
                <p>
                  The throughput curve's marginal gain drops below the 15% knee threshold at{' '}
                  <span className="mono">batch={knee}</span>, which is{' '}
                  <span className="mono">{fmt(kneePct, 0)}%</span> of the{' '}
                  <span className="mono">{fmt(maxConcurrent, 0)}</span>-request KV-memory ceiling.
                  {verdict === 'bandwidth' ? (
                    <>
                      {' '}
                      Since <span className="mono">{fmt(kneePct, 0)}%</span> is below the{' '}
                      <span className="mono">60%</span> threshold, throughput plateaus well before
                      memory runs out — this deployment is limited by HBM bandwidth (
                      <span className="mono">{fmt(bandwidthGBs, 0)} GB/s</span>), not KV capacity.
                      Adding more memory won't raise throughput; run more replicas instead.
                    </>
                  ) : (
                    <>
                      {' '}
                      Since <span className="mono">{fmt(kneePct, 0)}%</span> is at or above the{' '}
                      <span className="mono">60%</span> threshold, the curve keeps climbing almost
                      all the way to the KV-memory ceiling — this deployment is limited by KV cache
                      capacity (<span className="mono">{fmt(kvBudget)} GB</span> budget), not
                      bandwidth. A bigger or additional GPU per replica raises the ceiling directly.
                    </>
                  )}
                </p>
              </section>
            </>
          )}
        </main>
      </div>
    </div>
  )
}

export default App
