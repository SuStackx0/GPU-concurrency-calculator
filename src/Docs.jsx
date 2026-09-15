import './App.css'
import './Docs.css'

function Docs() {
  return (
    <div className="app docs">
      <header className="app-header">
        <h1>capacity-planner</h1>
        <p className="subtitle">
          methodology &mdash; <a href="/">back to calculator</a>
        </p>
      </header>

      <article className="panel docs-body">
        <p>
          <code>capacity-planner</code> turns a GPU's memory + bandwidth spec and a model's
          architecture into a concurrency verdict. This page explains what each number depends on
          &mdash; read it if a result surprises you before assuming it's wrong.
        </p>

        <h2>1. Memory split: weights vs. KV cache</h2>
        <p>
          Every request needs two pools of GPU memory: <strong>weights</strong>, loaded once and
          shared by every concurrent request, and <strong>KV cache</strong>, which grows per
          request, per token generated.
        </p>
        <pre className="mono formula">{`weightMemGB = paramsB * weightBytesPerParam
kvBudgetGB  = totalMemGB * (1 - reservePct) - weightMemGB`}</pre>
        <p>
          <code className="mono">reservePct</code> (a fraction, e.g. 0.08 for 8%) sets aside
          headroom for activations, CUDA context, and fragmentation. Whatever's left after weights
          and the reserve is what concurrent requests' KV caches share. If weights alone exceed the
          post-reserve memory, <code className="mono">kvBudgetGB</code> goes negative &mdash; the
          model doesn't fit, and the app shows that state directly instead of a chart.
        </p>

        <h2>2. Per-token KV cost</h2>
        <pre className="mono formula">{`bytesPerTokenKV = 2 * layers * kvHeads * headDim * kvBytesPerElem`}</pre>
        <p>
          The <code className="mono">2</code> is for K and V (one tensor each, same shape). This is
          per token, per request &mdash; it doesn't depend on batch size. A request holding{' '}
          <code className="mono">avgSeqLen</code> tokens of context costs{' '}
          <code className="mono">bytesPerTokenKV * avgSeqLen</code> in KV memory.
        </p>

        <h2>3. How many requests fit</h2>
        <pre className="mono formula">{`maxConcurrent = floor(kvBudgetGB * 1e9 / (bytesPerTokenKV * avgSeqLen))`}</pre>
        <p>
          A pure <strong>memory-capacity</strong> number &mdash; it has nothing to do with GPU
          speed or bandwidth. It only asks how many <code className="mono">avgSeqLen</code>-sized
          KV caches fit in the leftover memory pool. Doubling{' '}
          <code className="mono">avgSeqLen</code> halves it; doubling{' '}
          <code className="mono">kvBudgetGB</code> doubles it.
        </p>

        <h2>4. The throughput curve &mdash; and where bandwidth actually enters</h2>
        <p>
          Every decode step moves the same fixed set of weight bytes (once) plus one KV-cache
          read/write per token in the batch:
        </p>
        <pre className="mono formula">{`bytesMoved  = weightMemGB * 1e9 + batch * bytesPerTokenKV * avgSeqLen
timePerStep = bytesMoved / (bandwidthGBs * 1e9)
tokPerSec   = batch / timePerStep`}</pre>
        <p>
          At <code className="mono">batch = 1</code>, almost all bytes moved are the fixed weight
          load. As batch grows, the KV term grows linearly while the weight term stays fixed, so
          each extra unit of batch buys a shrinking marginal increase in throughput &mdash; that's
          the "knee" the chart looks for.
        </p>
        <p>
          <strong>
            Bandwidth is a single scalar multiplier applied identically to every batch size.
          </strong>{' '}
          It sets how fast the whole curve moves (10x the bandwidth &rarr; 10x tok/s at every
          point), but it cancels out of any ratio between two points on the curve. Run the same
          model and memory config through four different GPUs:
        </p>
        <div className="table-wrap">
          <table className="mono docs-table">
            <thead>
              <tr>
                <th>bandwidth</th>
                <th>knee (batch)</th>
                <th>maxConcurrent</th>
                <th>verdict</th>
                <th>tok/s @ maxConcurrent</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>100 GB/s</td>
                <td>158</td>
                <td>158</td>
                <td>capacity</td>
                <td>134.7</td>
              </tr>
              <tr>
                <td>273 GB/s</td>
                <td>158</td>
                <td>158</td>
                <td>capacity</td>
                <td>367.6</td>
              </tr>
              <tr>
                <td>1000 GB/s</td>
                <td>158</td>
                <td>158</td>
                <td>capacity</td>
                <td>1346.7</td>
              </tr>
              <tr>
                <td>3350 GB/s</td>
                <td>158</td>
                <td>158</td>
                <td>capacity</td>
                <td>4511.4</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p>
          The knee batch and the verdict never move &mdash; only the absolute tok/s scale does.
          That's not a bug: <code className="mono">tokPerSec = batch * bandwidthGBs*1e9 / bytesMoved</code>,
          so <code className="mono">bandwidthGBs</code> is a common factor in both sides of the
          marginal-gain ratio the knee test computes, and it cancels.
        </p>
        <p>
          <strong>What actually moves the knee</strong> is the ratio between the fixed weight-load
          cost and the per-batch KV-load cost &mdash; i.e.{' '}
          <code className="mono">weightMemGB</code> vs.{' '}
          <code className="mono">bytesPerTokenKV * avgSeqLen</code>. A model with huge weights
          relative to its KV footprint (few KV heads, small head dim, short sequences) plateaus
          early &mdash; you pay for the weight load on every step regardless of batch, so more
          concurrency buys little. A model with a small weight footprint relative to a large KV
          footprint (long sequences, many KV heads) keeps gaining from more batch almost all the
          way to the memory ceiling.
        </p>

        <h2>5. Knee detection</h2>
        <pre className="mono formula">{`findKnee(points, marginalThreshold = 0.15):
  return first point[i] where (tokPerSec[i] - tokPerSec[i-1]) / tokPerSec[i-1] < 0.15`}</pre>
        <p>
          The default 15% threshold means: once stepping up the batch size buys less than a 15%
          throughput gain, that's the point of diminishing returns &mdash; the "knee."
        </p>

        <h2>6. The verdict: bandwidth-bound vs. capacity-bound</h2>
        <pre className="mono formula">{`bottleneck(knee, maxConcurrent):
  return knee < 0.6 * maxConcurrent ? "bandwidth" : "capacity"`}</pre>
        <ul>
          <li>
            <strong>knee well below maxConcurrent</strong> (&lt; 60% of it) &rarr; throughput
            already plateaued long before memory ran out. <strong>Bandwidth-bound</strong>: the
            memory bus is the limit, not KV capacity. Scale <strong>horizontally</strong> (more
            replicas) &mdash; a bigger GPU won't help, you were never using all of it productively.
          </li>
          <li>
            <strong>knee close to or past maxConcurrent</strong> (&ge; 60% of it) &rarr; throughput
            kept climbing right up to the memory ceiling. <strong>Capacity-bound</strong>: KV
            memory ran out before bandwidth did. Scale <strong>vertically</strong> (more memory per
            replica, or a lower-precision KV cache) &mdash; that directly raises the ceiling.
          </li>
        </ul>
        <p>
          Because bandwidth cancels out of the knee (see &sect;4), this classification really
          answers: does the weight/KV byte ratio for this model, at this sequence length, cause
          throughput to saturate well inside the memory budget, or does it keep scaling almost all
          the way to the wall? That's a property of the model architecture and workload shape, not
          of which specific GPU you plugged in.
        </p>
      </article>
    </div>
  )
}

export default Docs
