import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  Activity,
  AlertCircle,
  BrainCircuit,
  CheckCircle2,
  Code2,
  FileJson,
  LoaderCircle,
  Play,
  RotateCcw,
  Server,
  Workflow,
  RefreshCw,
  BarChart3,
  CircleHelp,
} from "lucide-react";
import {
  API_BASE_URL,
  checkHealth,
  getCapabilityTelemetry,
  runCognitive,
} from "../api/client.js";
import JsonPanel from "../components/JsonPanel.jsx";
import { exampleRequest } from "../data/exampleRequest.js";
import "./telemetry.css";

const pretty = (value) => JSON.stringify(value, null, 2);

function List({ items }) {
  return Array.isArray(items) && items.length ? (
    <ul className="list">
      {items.map((item, index) => (
        <li key={index}>{String(item)}</li>
      ))}
    </ul>
  ) : (
    <p className="muted">Tidak ada data.</p>
  );
}

function formatPercent(successes, failures) {
  const knownOutcomes = successes + failures;
  if (!knownOutcomes) return "—";
  return `${Math.round((successes / knownOutcomes) * 100)}%`;
}

function TelemetrySection({
  telemetry,
  loading,
  error,
  onRefresh,
  refreshedAt,
}) {
  const snapshot = telemetry?.snapshot;
  const capabilities = Object.entries(snapshot?.capabilities || {});
  const events = telemetry?.events || [];

  return (
    <section id="telemetry" className="telemetry-section">
      <div className="section-head">
        <BarChart3 size={18} />
        <div>
          <small>OBSERVABILITY</small>
          <h2>Capability telemetry</h2>
          <p>
            Statistik yang dibaca dari instance Agent pada backend saat ini.
          </p>
        </div>
        <button
          type="button"
          className="ghost telemetry-refresh"
          onClick={onRefresh}
          disabled={loading}
        >
          {loading ? (
            <LoaderCircle className="spin" size={14} />
          ) : (
            <RefreshCw size={14} />
          )}
          {loading ? "Memuat…" : "Refresh"}
        </button>
      </div>

      {error && (
        <div className="alert error telemetry-error">
          <AlertCircle size={17} />
          <div>
            <b>Telemetry tidak dapat dimuat</b>
            <p>{error}</p>
          </div>
        </div>
      )}

      {loading && !telemetry && (
        <div className="empty telemetry-empty">
          <LoaderCircle className="spin" size={24} />
          <h3>Memuat telemetry</h3>
          <p>Mengambil snapshot dari API backend.</p>
        </div>
      )}

      {!loading && !telemetry && !error && (
        <div className="empty telemetry-empty">
          <CircleHelp size={24} />
          <h3>Telemetry belum tersedia</h3>
          <p>Belum ada respons telemetry dari backend.</p>
        </div>
      )}

      {telemetry && (
        <>
          <div className="telemetry-metrics">
            <article className="metric">
              <small><Activity size={14} /> EVENTS RECORDED</small>
              <b>{snapshot?.event_count ?? 0}</b>
              <span>Total event pada instance saat ini</span>
            </article>
            <article className="metric">
              <small><Workflow size={14} /> ATTEMPTS</small>
              <b>{snapshot?.total_attempts ?? 0}</b>
              <span>Semua percobaan tercatat</span>
            </article>
            <article className="metric">
              <small><CheckCircle2 size={14} /> SUCCESSES</small>
              <b>{snapshot?.total_successes ?? 0}</b>
              <span>Outcome sukses yang diketahui</span>
            </article>
            <article className="metric">
              <small><AlertCircle size={14} /> FAILURES</small>
              <b>{snapshot?.total_failures ?? 0}</b>
              <span>Outcome gagal yang diketahui</span>
            </article>
          </div>

          {!snapshot?.has_evidence && (
            <div className="telemetry-note">
              <CircleHelp size={17} />
              <p>
                Belum ada bukti kapabilitas yang tercatat. Ini bukan error,
                tetapi belum ada data untuk menyimpulkan performa atau
                pembelajaran agent. Endpoint cognitive sendiri tidak
                mengeksekusi tindakan secara otomatis.
              </p>
            </div>
          )}

          <div className="panel telemetry-table-panel">
            <div className="panel-head">
              <h3>Ringkasan per kapabilitas</h3>
              <span className="tag">{capabilities.length} kapabilitas</span>
            </div>
            {capabilities.length ? (
              <div className="table-scroll">
                <table className="telemetry-table">
                  <thead>
                    <tr>
                      <th>Kapabilitas</th>
                      <th>Percobaan</th>
                      <th>Berhasil</th>
                      <th>Gagal</th>
                      <th>Rasio sukses*</th>
                    </tr>
                  </thead>
                  <tbody>
                    {capabilities.map(([name, values]) => (
                      <tr key={name}>
                        <td><code>{name}</code></td>
                        <td>{values.attempts ?? 0}</td>
                        <td>{values.successes ?? 0}</td>
                        <td>{values.failures ?? 0}</td>
                        <td>{formatPercent(values.successes ?? 0, values.failures ?? 0)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="muted telemetry-no-data">
                Belum ada kapabilitas yang tercatat.
              </p>
            )}
            <p className="telemetry-footnote">
              * Rasio dihitung dari outcome sukses dan gagal yang diketahui;
              percobaan tanpa outcome tidak masuk denominator.
            </p>
          </div>

          <div className="panel telemetry-table-panel">
            <div className="panel-head">
              <h3>Event terbaru</h3>
              <span className="tag">
                {telemetry.events_returned ?? events.length} event
              </span>
            </div>
            {events.length ? (
              <div className="table-scroll">
                <table className="telemetry-table">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Kapabilitas</th>
                      <th>Outcome</th>
                      <th>Sumber</th>
                    </tr>
                  </thead>
                  <tbody>
                    {events.map((event, index) => (
                      <tr key={`${event.event_id ?? "event"}-${index}`}>
                        <td>{event.event_id ?? "—"}</td>
                        <td><code>{event.capability ?? "—"}</code></td>
                        <td>
                          {event.success === true ? (
                            <span className="telemetry-outcome success-text">Berhasil</span>
                          ) : event.success === false ? (
                            <span className="telemetry-outcome failure-text">Gagal</span>
                          ) : (
                            <span className="telemetry-outcome unknown-text">Belum diketahui</span>
                          )}
                        </td>
                        <td>{event.source ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="muted telemetry-no-data">
                Belum ada event telemetry untuk ditampilkan.
              </p>
            )}
          </div>

          <p className="telemetry-updated">
            {refreshedAt
              ? `Terakhir diperbarui: ${refreshedAt.toLocaleTimeString()}`
              : "Waktu pembaruan belum tersedia."}
            {" · "}Data runtime saat ini; tidak dijamin persisten setelah server restart.
          </p>
        </>
      )}
    </section>
  );
}

export default function Dashboard() {
  const [text, setText] = useState(() => pretty(exampleRequest));
  const [status, setStatus] = useState("checking");
  const [healthError, setHealthError] = useState("");
  const [response, setResponse] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [tab, setTab] = useState("overview");
  const [telemetry, setTelemetry] = useState(null);
  const [telemetryLoading, setTelemetryLoading] = useState(false);
  const [telemetryError, setTelemetryError] = useState("");
  const [telemetryUpdatedAt, setTelemetryUpdatedAt] = useState(null);

  const parsed = useMemo(() => {
    try {
      return { value: JSON.parse(text), error: "" };
    } catch (exception) {
      return { value: null, error: exception.message };
    }
  }, [text]);

  async function refreshTelemetry(signal) {
    setTelemetryLoading(true);
    setTelemetryError("");
    try {
      const data = await getCapabilityTelemetry({ limit: 20, signal });
      setTelemetry(data);
      setTelemetryUpdatedAt(new Date());
    } catch (exception) {
      if (exception.name !== "AbortError") {
        setTelemetryError(
          `${exception.message}${exception.body ? `\n${pretty(exception.body)}` : ""}`,
        );
      }
    } finally {
      if (!signal?.aborted) setTelemetryLoading(false);
    }
  }

  useEffect(() => {
    const controller = new AbortController();

    checkHealth(controller.signal)
      .then(() => {
        setStatus("online");
        setHealthError("");
      })
      .catch((exception) => {
        if (exception.name !== "AbortError") {
          setStatus("offline");
          setHealthError(exception.message);
        }
      });

    refreshTelemetry(controller.signal);
    return () => controller.abort();
  }, []);

  async function submit(event) {
    event.preventDefault();
    setError("");
    setResponse(null);

    if (parsed.error) {
      setError(`JSON tidak valid: ${parsed.error}`);
      return;
    }

    setBusy(true);
    try {
      setResponse(await runCognitive(parsed.value));
      setTab("overview");
      // Refresh telemetry after a cognitive request, but do not imply that
      // this endpoint itself executed actions or necessarily recorded events.
      await refreshTelemetry();
    } catch (exception) {
      setError(
        `${exception.message}${exception.body ? `\n${pretty(exception.body)}` : ""}`,
      );
    } finally {
      setBusy(false);
    }
  }

  const result = response?.result;
  const interpretation = result?.interpretation;
  const reasoning = result?.reasoning;
  const plan = result?.plan;
  const meta = result?.metacognition;

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-icon"><BrainCircuit size={22} /></span>
          <div><b>hAI Agent</b><small>COGNITIVE SYSTEM</small></div>
        </div>
        <p className="nav-label">WORKSPACE</p>
        <a className="nav active" href="#top"><Activity size={16} /> Cognitive dashboard</a>
        <a className="nav" href="#request"><FileJson size={16} /> API request</a>
        <a className="nav" href="#results"><Workflow size={16} /> Cognitive output</a>
        <a className="nav" href="#telemetry"><BarChart3 size={16} /> Capability telemetry</a>
        <div className="side-bottom">
          <span className={`dot ${status}`} />
          <div>
            <b>Backend API</b>
            <small>{status === "online" ? "Connected" : status === "checking" ? "Checking…" : "Unavailable"}</small>
          </div>
          <code>{API_BASE_URL}</code>
        </div>
      </aside>

      <main className="main" id="top">
        <header className="topbar">
          <div>
            <p className="eyebrow">AGENT WORKSPACE / OVERVIEW</p>
            <h1>Cognitive Dashboard</h1>
          </div>
          <span className={`pill ${status}`}>
            <span className={`dot ${status}`} />
            {status === "online" ? "API connected" : status === "checking" ? "Connecting" : "API offline"}
          </span>
        </header>

        <motion.section className="hero" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <div>
            <p className="mint-label">✦ AGENT COGNITION</p>
            <h2>From observation<br />to a structured plan.</h2>
            <p>
              Jalankan cognitive pass dari API hAI Agent. Periksa interpretation,
              reasoning, planning, metacognition, trace, dan telemetry dalam satu workspace.
            </p>
            <a href="#request">Mulai cognitive run ↗</a>
          </div>
          <div className="hero-graphic">
            <div className="ring r1" />
            <div className="ring r2" />
            <div className="brain"><BrainCircuit size={42} /></div>
          </div>
        </motion.section>

        <section className="metrics">
          <div className="metric"><small><Server size={14} /> API STATUS</small><b>{status === "online" ? "Online" : status === "checking" ? "Checking" : "Offline"}</b><span>GET /health</span></div>
          <div className="metric"><small><Workflow size={14} /> CURRENT CYCLE</small><b>{response?.cycle ?? "—"}</b><span>Cycle terakhir</span></div>
          <div className="metric"><small><Activity size={14} /> PLAN STEPS</small><b>{plan?.steps?.length ?? "—"}</b><span>Langkah dari planner</span></div>
          <div className="metric"><small><CheckCircle2 size={14} /> PLAN CONFIDENCE</small><b>{typeof plan?.confidence === "number" ? `${Math.round(plan.confidence * 100)}%` : "—"}</b><span>Nilai dari output model</span></div>
        </section>

        {status === "offline" && (
          <div className="alert"><AlertCircle size={17} /><span>Backend belum dapat dijangkau: {healthError}</span></div>
        )}

        <section className="workspace">
          <div id="request">
            <div className="section-head"><Code2 size={18} /><div><small>INPUT</small><h2>Cognitive request</h2><p>Edit payload JSON dan kirim ke endpoint API.</p></div></div>
            <form className="panel" onSubmit={submit}>
              <div className="panel-head">
                <h3>Request payload</h3>
                <button type="button" className="ghost" onClick={() => { setText(pretty(exampleRequest)); setError(""); setResponse(null); }}>
                  <RotateCcw size={14} /> Reset
                </button>
              </div>
              <textarea className="editor" value={text} onChange={(event) => setText(event.target.value)} spellCheck="false" aria-label="Cognitive request JSON" />
              {parsed.error && <p className="bad">JSON tidak valid: {parsed.error}</p>}
              {error && <pre className="alert error">{error}</pre>}
              <div className="form-foot">
                <code>POST /api/v1/cognitive/run</code>
                <button className="primary" disabled={busy || Boolean(parsed.error)}>
                  {busy ? <LoaderCircle className="spin" size={15} /> : <Play size={15} />}
                  {busy ? "Memproses…" : "Run cognitive cycle"}
                </button>
              </div>
            </form>
            <p className="footnote">Endpoint ini menghasilkan rencana; tidak mengeksekusi aksi secara otomatis.</p>
          </div>

          <div id="results">
            <div className="section-head"><BrainCircuit size={18} /><div><small>OUTPUT</small><h2>Cognitive result</h2><p>Hasil aktual dari backend.</p></div></div>
            {busy && <div className="empty"><LoaderCircle className="spin" size={25} /><h3>Agent sedang berpikir</h3><p>Menunggu cognitive runtime dan LLM provider.</p></div>}
            {!response && !busy && <div className="empty"><Workflow size={27} /><h3>Belum ada hasil</h3><p>Kirim request untuk melihat hasil cognitive pass.</p></div>}
            {response && (
              <>
                <div className="success"><CheckCircle2 size={18} /><div><b>Cognitive run berhasil</b><small>Response diterima dari API.</small></div></div>
                <div className="tabs">
                  {[["overview", "Overview"], ["plan", "Plan"], ["meta", "Metacognition"], ["trace", "Trace"], ["raw", "Raw JSON"]].map(([key, label]) => (
                    <button type="button" key={key} className={tab === key ? "tab selected" : "tab"} onClick={() => setTab(key)}>{label}</button>
                  ))}
                </div>
                {tab === "overview" && (
                  <div className="stack">
                    <section className="panel">
                      <div className="panel-head"><h3>Interpretation</h3><span className="tag">{interpretation?.provider || "LLM"}</span></div>
                      <p className="summary">{interpretation?.summary || "Tidak tersedia."}</p>
                      <details><summary>Facts</summary><List items={interpretation?.facts} /></details>
                      <details><summary>Implications</summary><List items={interpretation?.implications} /></details>
                      <details><summary>Uncertainties</summary><List items={interpretation?.uncertainties} /></details>
                    </section>
                    <section className="panel">
                      <div className="panel-head"><h3>Reasoning</h3><span className="tag">{reasoning?.provider || "LLM"}</span></div>
                      <p className="summary">{reasoning?.summary || "Tidak tersedia."}</p>
                      <details open><summary>Conclusions</summary><List items={reasoning?.conclusions} /></details>
                      <details><summary>Risks</summary><List items={reasoning?.risks} /></details>
                    </section>
                  </div>
                )}
                {tab === "plan" && (
                  <div className="stack">
                    <section className="panel">
                      <div className="panel-head"><h3>Generated plan</h3><span className="tag">{plan?.steps?.length || 0} steps</span></div>
                      <p className="summary">{plan?.rationale || "Rationale tidak tersedia."}</p>
                      {(plan?.steps || []).map((step, index) => (
                        <div className="step" key={`${step.index || index}-${step.action}`}>
                          <span className="step-number">{step.index || index + 1}</span>
                          <div>
                            <b>{step.action}</b><p>{step.purpose}</p>
                            {step.preconditions?.length > 0 && <details><summary>Preconditions</summary><List items={step.preconditions} /></details>}
                            {step.success_criteria && <p className="good">✓ {step.success_criteria}</p>}
                            {step.risk && <p className="risk">⚠ {step.risk}</p>}
                          </div>
                        </div>
                      ))}
                    </section>
                    <JsonPanel title="Plan assumptions" value={plan?.assumptions} />
                    <JsonPanel title="Stop conditions" value={plan?.stop_conditions} />
                  </div>
                )}
                {tab === "meta" && <div className="stack"><JsonPanel title="Deterministic reflection" value={meta?.deterministic_reflection} /><JsonPanel title="LLM reflection" value={meta?.llm_reflection} /></div>}
                {tab === "trace" && <JsonPanel title="Runtime trace" value={response.trace} />}
                {tab === "raw" && <JsonPanel title="Full API response" value={response} />}
              </>
            )}
          </div>
        </section>

        <TelemetrySection
          telemetry={telemetry}
          loading={telemetryLoading}
          error={telemetryError}
          onRefresh={() => refreshTelemetry()}
          refreshedAt={telemetryUpdatedAt}
        />

        <footer>hAI Agent · Cognitive API <span><Activity size={12} /> Local development workspace</span></footer>
      </main>
    </div>
  );
}
