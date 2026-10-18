import "./execution.css";
import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  Activity,
  AlertCircle,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  CircleHelp,
  Code2,
  FileJson,
  FlaskConical,
  LoaderCircle,
  Play,
  RefreshCw,
  RotateCcw,
  Server,
  Workflow,
  ShieldCheck,
  Terminal,
  Zap,
} from "lucide-react";

import {
  API_BASE_URL,
  checkHealth,
  getCapabilityTelemetrySources,
  getExecutionState,
  getStrategies,
  resetExecution,
  runCognitive,
  runExecution,
} from "../api/client.js";

import JsonPanel from "../components/JsonPanel.jsx";
import AssistantComposer from "../components/AssistantComposer.jsx";
import StrategyLab from "./StrategyLab.jsx";
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

function StateView({ title, state }) {
  const entries = Object.entries(state || {});

  return (
    <section className="panel">
      <div className="panel-head">
        <h3>{title}</h3>
        <span className="tag">{entries.length} properti</span>
      </div>

      {entries.length ? (
        <div className="table-scroll">
          <table className="telemetry-table">
            <thead>
              <tr>
                <th>Properti</th>
                <th>Nilai</th>
              </tr>
            </thead>
            <tbody>
              {entries.map(([key, value]) => (
                <tr key={key}>
                  <td><code>{key}</code></td>
                  <td>{String(value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="muted">State belum tersedia.</p>
      )}
    </section>
  );
}

function ExecutionSection({
  stateData,
  loadingState,
  busy,
  resetting,
  error,
  result,
  selectedActions,
  setSelectedActions,
  goalDescription,
  setGoalDescription,
  targetStateText,
  setTargetStateText,
  onRun,
  onReset,
  onRefresh,
}) {
  const allowedActions = stateData?.allowed_actions || [];
  const currentState = stateData?.state || {};

  const targetParsed = useMemo(() => {
    try {
      const value = JSON.parse(targetStateText);

      if (
        value === null ||
        Array.isArray(value) ||
        typeof value !== "object" ||
        Object.keys(value).length === 0
      ) {
        return {
          value: null,
          error: "Target state harus berupa object JSON yang tidak kosong.",
        };
      }

      return { value, error: "" };
    } catch (exception) {
      return { value: null, error: exception.message };
    }
  }, [targetStateText]);

  function toggleAction(action) {
    setSelectedActions((previous) =>
      previous.includes(action)
        ? previous.filter((item) => item !== action)
        : [...previous, action],
    );
  }

  function moveAction(index, direction) {
    setSelectedActions((previous) => {
      const next = [...previous];
      const destination = index + direction;

      if (destination < 0 || destination >= next.length) {
        return previous;
      }

      [next[index], next[destination]] = [
        next[destination],
        next[index],
      ];

      return next;
    });
  }

  const canRun =
    !busy &&
    !resetting &&
    selectedActions.length > 0 &&
    selectedActions.length <= 4 &&
    Boolean(goalDescription.trim()) &&
    !targetParsed.error &&
    allowedActions.length > 0;

  return (
    <section id="execution" className="execution-section">
      <div className="section-head">
        <Zap size={18} />
        <div>
          <small>CLOSED-LOOP EXECUTION</small>
          <h2>Execution Workspace</h2>
          <p>
            Pilih aksi yang diizinkan, jalankan simulasi, lalu periksa
            perubahan state dan pencapaian goal.
          </p>
        </div>

        <button
          type="button"
          className="ghost"
          onClick={onRefresh}
          disabled={loadingState || busy || resetting}
        >
          {loadingState ? (
            <LoaderCircle className="spin" size={14} />
          ) : (
            <RefreshCw size={14} />
          )}
          Refresh state
        </button>
      </div>

      {error && (
        <div className="alert error">
          <AlertCircle size={17} />
          <div>
            <b>Execution API error</b>
            <pre>{error}</pre>
          </div>
        </div>
      )}

      <div className="execution-grid">
        <section className="panel">
          <div className="panel-head">
            <h3>Action selection</h3>
            <span className="tag">
              {selectedActions.length}/4 dipilih
            </span>
          </div>

          <p className="muted">
            Pilih satu aksi atau susun beberapa aksi sesuai urutan
            eksekusi yang diinginkan.
          </p>

          {allowedActions.length ? (
            <div className="execution-action-list">
              {allowedActions.map((action) => (
                <label className="execution-action" key={action}>
                  <input
                    type="checkbox"
                    checked={selectedActions.includes(action)}
                    disabled={
                      busy ||
                      resetting ||
                      (!selectedActions.includes(action) &&
                        selectedActions.length >= 4)
                    }
                    onChange={() => toggleAction(action)}
                  />
                  <span>
                    <b>{action}</b>
                    <small>
                      {selectedActions.includes(action)
                        ? "Termasuk dalam urutan eksekusi"
                        : "Aksi yang diizinkan backend"}
                    </small>
                  </span>
                </label>
              ))}
            </div>
          ) : (
            <div className="empty">
              <CircleHelp size={24} />
              <h3>Aksi belum tersedia</h3>
              <p>Periksa koneksi Execution API.</p>
            </div>
          )}

          {selectedActions.length > 0 && (
            <div className="execution-order">
              <div className="panel-head">
                <h3>Urutan eksekusi</h3>
                <button
                  type="button"
                  className="ghost"
                  disabled={busy || resetting}
                  onClick={() => setSelectedActions([])}
                >
                  Hapus pilihan
                </button>
              </div>

              {selectedActions.map((action, index) => (
                <div className="execution-order-item" key={action}>
                  <span className="step-number">{index + 1}</span>
                  <code>{action}</code>
                  <div className="execution-order-controls">
                    <button
                      type="button"
                      className="ghost"
                      aria-label={`Naikkan ${action}`}
                      disabled={busy || resetting || index === 0}
                      onClick={() => moveAction(index, -1)}
                    >
                      ↑
                    </button>
                    <button
                      type="button"
                      className="ghost"
                      aria-label={`Turunkan ${action}`}
                      disabled={
                        busy ||
                        resetting ||
                        index === selectedActions.length - 1
                      }
                      onClick={() => moveAction(index, 1)}
                    >
                      ↓
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}

          <label className="execution-field">
            <span>Goal description</span>
            <input
              value={goalDescription}
              onChange={(event) => setGoalDescription(event.target.value)}
              maxLength={500}
              disabled={busy || resetting}
              placeholder="Contoh: Unlock the door"
            />
          </label>

          <label className="execution-field">
            <span>Target state (JSON)</span>
            <textarea
              className="editor execution-target-editor"
              value={targetStateText}
              onChange={(event) => setTargetStateText(event.target.value)}
              disabled={busy || resetting}
              spellCheck="false"
              aria-label="Execution target state JSON"
            />
          </label>

          {targetParsed.error && (
            <p className="bad">Target state: {targetParsed.error}</p>
          )}

          <div className="form-foot execution-form-foot">
            <code>POST /api/v1/execution/run</code>
            <button
              type="button"
              className="primary"
              disabled={!canRun}
              onClick={() => onRun(targetParsed.value)}
            >
              {busy ? (
                <LoaderCircle className="spin" size={15} />
              ) : (
                <Play size={15} />
              )}
              {busy ? "Menjalankan..." : "Execute actions"}
            </button>
          </div>

          <p className="footnote">
            Hanya aksi simulasi yang terdaftar pada backend yang dapat
            dijalankan. Urutan aksi memengaruhi hasil.
          </p>
        </section>

        <div className="stack">
          <div className="panel">
            <div className="panel-head">
              <h3>Environment controls</h3>
              <span className="tag">
                {stateData?.status || "belum terhubung"}
              </span>
            </div>

            <p className="muted">
              State saat ini dibaca langsung dari simulation runtime.
            </p>

            <div className="execution-control-buttons">
              <button
                type="button"
                className="ghost"
                onClick={onRefresh}
                disabled={loadingState || busy || resetting}
              >
                <RefreshCw size={14} /> Refresh
              </button>
              <button
                type="button"
                className="ghost"
                onClick={onReset}
                disabled={busy || resetting || loadingState}
              >
                {resetting ? (
                  <LoaderCircle className="spin" size={14} />
                ) : (
                  <RotateCcw size={14} />
                )}
                {resetting ? "Mereset..." : "Reset simulation"}
              </button>
            </div>
          </div>

          <StateView title="Current environment state" state={currentState} />

          {result && (
            <>
              <section className="panel">
                <div className="panel-head">
                  <h3>Latest execution result</h3>
                  <span
                    className={`tag ${
                      result.goal_satisfied ? "execution-tag-success" : ""
                    }`}
                  >
                    {result.goal_satisfied
                      ? "Goal tercapai"
                      : result.success
                        ? "Aksi selesai"
                        : "Belum tercapai"}
                  </span>
                </div>

                <div className="execution-result-metrics">
                  <div>
                    <small>EXECUTION</small>
                    <b>{result.success ? "Berhasil" : "Gagal"}</b>
                  </div>
                  <div>
                    <small>GOAL VERIFICATION</small>
                    <b>{result.goal_satisfied ? "Lulus" : "Belum lulus"}</b>
                  </div>
                  <div>
                    <small>MODE</small>
                    <b>{result.mode || "—"}</b>
                  </div>
                </div>

                <p className="summary">
                  {result.goal?.description ||
                    result.reason ||
                    "Eksekusi selesai."}
                </p>

                {result.steps?.length > 0 && (
                  <>
                    <h4>Step outcomes</h4>
                    <div className="execution-steps">
                      {result.steps.map((step, index) => (
                        <div
                          className="execution-order-item"
                          key={`${step.action || "step"}-${index}`}
                        >
                          {step.success ? (
                            <CheckCircle2
                              size={17}
                              className="execution-success-icon"
                            />
                          ) : (
                            <AlertCircle
                              size={17}
                              className="execution-failure-icon"
                            />
                          )}
                          <div>
                            <b>{step.action || `Step ${index + 1}`}</b>
                            <small>
                              {step.message ||
                                (step.success ? "Berhasil" : "Gagal")}
                            </small>
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                )}

                <details>
                  <summary>Detail respons execution</summary>
                  <pre className="execution-json">
                    {pretty(result)}
                  </pre>
                </details>
              </section>

              <StateView
                title="State after execution"
                state={result.state_after}
              />
            </>
          )}
        </div>
      </div>
    </section>
  );
}

function TelemetrySourcePanel({ title, description, sourceData }) {
  const isUnavailable = sourceData?.status === "unavailable";
  const isError = sourceData?.status === "error";
  const snapshot = sourceData?.snapshot;
  const capabilities = Object.entries(snapshot?.capabilities || {});
  const events = sourceData?.events || [];

  return (
    <section className="panel telemetry-source-panel">
      <div className="panel-head telemetry-source-heading">
        <div>
          <h3>{title}</h3>
          <p className="muted telemetry-source-description">{description}</p>
        </div>
        <span className={`tag ${sourceData?.status === "ok" ? "telemetry-status-ok" : "telemetry-status-muted"}`}>
          {sourceData?.status || "unknown"}
        </span>
      </div>

      {isUnavailable && (
        <div className="telemetry-note">
          <CircleHelp size={17} />
          <p>Runtime ini tidak tersedia. Kode: <code>{sourceData?.error || "unknown"}</code>.</p>
        </div>
      )}
      {isError && (
        <div className="alert error telemetry-error">
          <AlertCircle size={17} />
          <div><b>Telemetry source error</b><p>{sourceData?.message || sourceData?.error || "Gagal membaca telemetry."}</p></div>
        </div>
      )}
      {!sourceData && (
        <p className="muted telemetry-no-data">Data sumber belum tersedia.</p>
      )}

      {sourceData?.status === "ok" && (
        <>
          <div className="telemetry-metrics">
            <article className="metric"><small><Activity size={14} /> EVENTS RECORDED</small><b>{snapshot?.event_count ?? 0}</b><span>Total event pada runtime ini</span></article>
            <article className="metric"><small><Workflow size={14} /> ATTEMPTS</small><b>{snapshot?.total_attempts ?? 0}</b><span>Semua percobaan tercatat</span></article>
            <article className="metric"><small><CheckCircle2 size={14} /> SUCCESSES</small><b>{snapshot?.total_successes ?? 0}</b><span>Outcome sukses diketahui</span></article>
            <article className="metric"><small><AlertCircle size={14} /> FAILURES</small><b>{snapshot?.total_failures ?? 0}</b><span>Outcome gagal diketahui</span></article>
          </div>

          {!snapshot?.has_evidence && (
            <div className="telemetry-note"><CircleHelp size={17} /><p>Belum ada bukti kapabilitas yang tercatat pada sumber ini. Data kosong bukan bukti bahwa agent gagal atau telah belajar.</p></div>
          )}

          <div className="table-scroll">
            <table className="telemetry-table">
              <thead><tr><th>Kapabilitas</th><th>Percobaan</th><th>Berhasil</th><th>Gagal</th><th>Rasio sukses*</th></tr></thead>
              <tbody>
                {capabilities.map(([name, values]) => (
                  <tr key={name}><td><code>{name}</code></td><td>{values.attempts ?? 0}</td><td>{values.successes ?? 0}</td><td>{values.failures ?? 0}</td><td>{formatPercent(values.successes ?? 0, values.failures ?? 0)}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
          {!capabilities.length && <p className="muted telemetry-no-data">Belum ada kapabilitas yang tercatat.</p>}
          <p className="telemetry-footnote">* Rasio dihitung dari outcome sukses dan gagal yang diketahui.</p>

          <div className="panel-head telemetry-events-heading"><h3>Event terbaru</h3><span className="tag">{sourceData.events_returned ?? events.length} event</span></div>
          {events.length ? (
            <div className="table-scroll"><table className="telemetry-table">
              <thead><tr><th>ID</th><th>Kapabilitas</th><th>Outcome</th><th>Sumber event</th></tr></thead>
              <tbody>{events.map((event, index) => (
                <tr key={`${event.event_id ?? "event"}-${index}`}><td>{event.event_id ?? "—"}</td><td><code>{event.capability ?? "—"}</code></td><td>{event.success === true ? <span className="telemetry-outcome success-text">Berhasil</span> : event.success === false ? <span className="telemetry-outcome failure-text">Gagal</span> : <span className="telemetry-outcome unknown-text">Belum diketahui</span>}</td><td>{event.source ?? "—"}</td></tr>
              ))}</tbody>
            </table></div>
          ) : <p className="muted telemetry-no-data">Belum ada event telemetry pada runtime ini.</p>}
        </>
      )}
    </section>
  );
}

function TelemetrySection({ telemetrySources, loading, error, onRefresh, refreshedAt }) {
  const sources = telemetrySources?.sources || {};
  return (
    <section id="telemetry" className="telemetry-section">
      <div className="section-head">
        <BarChart3 size={18} />
        <div><small>OBSERVABILITY</small><h2>Runtime telemetry</h2><p>Data kognitif dan eksekusi ditampilkan terpisah dari sumber API masing-masing.</p></div>
        <button type="button" className="ghost telemetry-refresh" onClick={onRefresh} disabled={loading}>
          {loading ? <LoaderCircle className="spin" size={14} /> : <RefreshCw size={14} />}
          {loading ? "Memuat..." : "Refresh"}
        </button>
      </div>
      <p className="telemetry-contract-note"><code>GET /api/v1/telemetry/capabilities/sources</code> · Snapshot tidak digabungkan antar-runtime.</p>
      {error && <div className="alert error telemetry-error"><AlertCircle size={17} /><div><b>Telemetry tidak dapat dimuat</b><p>{error}</p></div></div>}
      {loading && !telemetrySources && <div className="empty telemetry-empty"><LoaderCircle className="spin" size={24} /><h3>Memuat telemetry</h3><p>Mengambil sumber telemetry dari API backend.</p></div>}
      {!loading && !telemetrySources && !error && <div className="empty telemetry-empty"><CircleHelp size={24} /><h3>Telemetry belum tersedia</h3><p>Belum ada respons telemetry dari backend.</p></div>}
      {telemetrySources && <div className="telemetry-sources-grid">
        <TelemetrySourcePanel title="Cognitive Runtime" description="Penalaran dan kapabilitas runtime kognitif." sourceData={sources.cognitive_runtime} />
        <TelemetrySourcePanel title="Execution Runtime" description="Aksi simulasi dan outcome runtime eksekusi." sourceData={sources.execution_runtime} />
      </div>}
      <p className="telemetry-updated">{refreshedAt ? `Terakhir diperbarui: ${refreshedAt.toLocaleTimeString()}` : "Waktu pembaruan belum tersedia."} · Data runtime saat ini; tidak dijamin persisten setelah server restart.</p>
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

  const [telemetrySources, setTelemetrySources] = useState(null);
  const [telemetryLoading, setTelemetryLoading] = useState(false);
  const [telemetryError, setTelemetryError] = useState("");
  const [telemetryUpdatedAt, setTelemetryUpdatedAt] = useState(null);

  const [strategiesData, setStrategiesData] = useState(null);
  const [strategiesLoading, setStrategiesLoading] = useState(false);
  const [strategiesError, setStrategiesError] = useState("");
  const [strategiesUpdatedAt, setStrategiesUpdatedAt] = useState(null);

  const [executionState, setExecutionState] = useState(null);
  const [executionLoading, setExecutionLoading] = useState(false);
  const [executionBusy, setExecutionBusy] = useState(false);
  const [executionResetting, setExecutionResetting] = useState(false);
  const [executionError, setExecutionError] = useState("");
  const [executionResult, setExecutionResult] = useState(null);
  const [selectedActions, setSelectedActions] = useState(["open_box"]);
  const [goalDescription, setGoalDescription] = useState("Open the box");
  const [targetStateText, setTargetStateText] = useState(
    pretty({ box: "open" }),
  );

  const [activeHash, setActiveHash] = useState(
    window.location.hash || "#top",
  );

  useEffect(() => {
    const updateActiveHash = () => {
      setActiveHash(window.location.hash || "#top");
    };

    window.addEventListener("hashchange", updateActiveHash);

    return () => {
      window.removeEventListener("hashchange", updateActiveHash);
    };
  }, []);

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
      const data = await getCapabilityTelemetrySources({ limit: 20, signal });
      setTelemetrySources(data);
      setTelemetryUpdatedAt(new Date());
    } catch (exception) {
      if (exception.name !== "AbortError") {
        setTelemetryError(
          `${exception.message}${
            exception.body ? `\n${pretty(exception.body)}` : ""
          }`,
        );
      }
    } finally {
      if (!signal?.aborted) setTelemetryLoading(false);
    }
  }

  async function refreshStrategies(signal) {
    setStrategiesLoading(true);
    setStrategiesError("");

    try {
      const data = await getStrategies(signal);
      setStrategiesData(data);
      setStrategiesUpdatedAt(new Date());
    } catch (exception) {
      if (exception.name !== "AbortError") {
        setStrategiesError(
          `${exception.message}${
            exception.body ? `\\n${pretty(exception.body)}` : ""
          }`,
        );
      }
    } finally {
      if (!signal?.aborted) setStrategiesLoading(false);
    }
  }

  async function refreshExecutionState(signal) {
    setExecutionLoading(true);
    setExecutionError("");

    try {
      const data = await getExecutionState(signal);
      setExecutionState(data);

      if (Array.isArray(data.allowed_actions)) {
        setSelectedActions((previous) =>
          previous.filter((action) => data.allowed_actions.includes(action)),
        );
      }
    } catch (exception) {
      if (exception.name !== "AbortError") {
        setExecutionError(
          `${exception.message}${
            exception.body ? `\n${pretty(exception.body)}` : ""
          }`,
        );
      }
    } finally {
      if (!signal?.aborted) setExecutionLoading(false);
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
    refreshExecutionState(controller.signal);
    refreshStrategies(controller.signal);

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
      await Promise.all([refreshTelemetry(), refreshStrategies()]);
    } catch (exception) {
      setError(
        `${exception.message}${
          exception.body ? `\n${pretty(exception.body)}` : ""
        }`,
      );
    } finally {
      setBusy(false);
    }
  }

  async function executeSelectedActions(targetState) {
    setExecutionError("");
    setExecutionResult(null);
    setExecutionBusy(true);

    try {
      const data = await runExecution({
        actions: selectedActions,
        goal_description: goalDescription.trim(),
        target_state: targetState,
      });

      setExecutionResult(data);

      await Promise.all([
        refreshExecutionState(),
        refreshTelemetry(),
        refreshStrategies(),
      ]);
    } catch (exception) {
      setExecutionError(
        `${exception.message}${
          exception.body ? `\n${pretty(exception.body)}` : ""
        }`,
      );

      await Promise.all([
        refreshExecutionState(),
        refreshTelemetry(),
        refreshStrategies(),
      ]);
    } finally {
      setExecutionBusy(false);
    }
  }

  async function resetSimulation() {
    setExecutionError("");
    setExecutionBusy(false);
    setExecutionResetting(true);

    try {
      const data = await resetExecution();
      setExecutionResult(null);
      setExecutionState((previous) => ({
        ...previous,
        status: data.status || "ok",
        state: data.state,
      }));

      await Promise.all([refreshTelemetry(), refreshStrategies()]);
    } catch (exception) {
      setExecutionError(
        `${exception.message}${
          exception.body ? `\n${pretty(exception.body)}` : ""
        }`,
      );
    } finally {
      setExecutionResetting(false);
      await refreshExecutionState();
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
        <a
          className={`nav ${activeHash === "#top" ? "active" : ""}`}
          href="#top"
        >
          <Activity size={16} /> Cognitive dashboard
        </a>
        <a
          className={`nav ${activeHash === "#request" ? "active" : ""}`}
          href="#request"
        >
          <FileJson size={16} /> API request
        </a>
        <a
          className={`nav ${activeHash === "#results" ? "active" : ""}`}
          href="#results"
        >
          <Workflow size={16} /> Cognitive output
        </a>
        <a
          className={`nav ${activeHash === "#execution" ? "active" : ""}`}
          href="#execution"
        >
          <Zap size={16} /> Execution workspace
        </a>
        <a
          className={`nav ${activeHash === "#telemetry" ? "active" : ""}`}
          href="#telemetry"
        >
          <BarChart3 size={16} /> Capability telemetry
        </a>
        <a
          className={`nav ${activeHash === "#strategies" ? "active" : ""}`}
          href="#strategies"
        >
          <FlaskConical size={16} /> Strategy Lab
        </a>

        <div className="side-bottom">
          <span className={`dot ${status}`} />
          <div>
            <b>Backend API</b>
            <small>
              {status === "online"
                ? "Connected"
                : status === "checking"
                  ? "Checking..."
                  : "Unavailable"}
            </small>
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
            {status === "online"
              ? "API connected"
              : status === "checking"
                ? "Connecting"
                : "API offline"}
          </span>
        </header>

        <motion.section
          className="hero"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div>
            <p className="mint-label">✦ AGENT COGNITION</p>
            <h2>From observation<br />to a structured plan.</h2>
            <p>
              Jalankan cognitive pass, eksekusi aksi simulasi yang diizinkan,
              verifikasi goal, dan tinjau telemetry dalam satu workspace.
            </p>
            <a href="#request">Mulai cognitive run ↗</a>
          </div>
          <div className="hero-graphic">
            <div className="ring r1" />
            <div className="ring r2" />
            <div className="brain"><BrainCircuit size={42} /></div>
          </div>
        </motion.section>

        <AssistantComposer />

        <section className="metrics">
          <div className="metric">
            <small><Server size={14} /> API STATUS</small>
            <b>{status === "online" ? "Online" : status === "checking" ? "Checking" : "Offline"}</b>
            <span>GET /health</span>
          </div>
          <div className="metric">
            <small><Workflow size={14} /> CURRENT CYCLE</small>
            <b>{response?.cycle ?? "—"}</b>
            <span>Cycle terakhir</span>
          </div>
          <div className="metric">
            <small><Activity size={14} /> PLAN STEPS</small>
            <b>{plan?.steps?.length ?? "—"}</b>
            <span>Langkah dari planner</span>
          </div>
          <div className="metric">
            <small><CheckCircle2 size={14} /> PLAN CONFIDENCE</small>
            <b>
              {typeof plan?.confidence === "number"
                ? `${Math.round(plan.confidence * 100)}%`
                : "—"}
            </b>
            <span>Nilai dari output model</span>
          </div>
        </section>

        {status === "offline" && (
          <div className="alert">
            <AlertCircle size={17} />
            <span>Backend belum dapat dijangkau: {healthError}</span>
          </div>
        )}

        <section className="workspace">
          <div id="request">
            <div className="section-head">
              <Code2 size={18} />
              <div>
                <small>INPUT</small>
                <h2>Cognitive request</h2>
                <p>Edit payload JSON dan kirim ke endpoint API.</p>
              </div>
            </div>

            <form className="panel" onSubmit={submit}>
              <div className="panel-head">
                <h3>Request payload</h3>
                <button
                  type="button"
                  className="ghost"
                  onClick={() => {
                    setText(pretty(exampleRequest));
                    setError("");
                    setResponse(null);
                  }}
                >
                  <RotateCcw size={14} /> Reset
                </button>
              </div>

              <textarea
                className="editor"
                value={text}
                onChange={(event) => setText(event.target.value)}
                spellCheck="false"
                aria-label="Cognitive request JSON"
              />

              {parsed.error && (
                <p className="bad">JSON tidak valid: {parsed.error}</p>
              )}
              {error && <pre className="alert error">{error}</pre>}

              <div className="form-foot">
                <code>POST /api/v1/cognitive/run</code>
                <button
                  className="primary"
                  disabled={busy || Boolean(parsed.error)}
                >
                  {busy ? (
                    <LoaderCircle className="spin" size={15} />
                  ) : (
                    <Play size={15} />
                  )}
                  {busy ? "Memproses..." : "Run cognitive cycle"}
                </button>
              </div>
            </form>
            <p className="footnote">
              Cognitive run menghasilkan output kognitif; eksekusi aksi
              simulasi dilakukan secara eksplisit melalui Execution Workspace.
            </p>
          </div>

          <div id="results">
            <div className="section-head">
              <BrainCircuit size={18} />
              <div>
                <small>OUTPUT</small>
                <h2>Cognitive result</h2>
                <p>Hasil aktual dari backend.</p>
              </div>
            </div>

            {busy && (
              <div className="empty">
                <LoaderCircle className="spin" size={25} />
                <h3>Agent sedang berpikir</h3>
                <p>Menunggu cognitive runtime dan LLM provider.</p>
              </div>
            )}

            {!response && !busy && (
              <div className="empty">
                <Workflow size={27} />
                <h3>Belum ada hasil</h3>
                <p>Kirim request untuk melihat hasil cognitive pass.</p>
              </div>
            )}

            {response && (
              <>
                <div className="success">
                  <CheckCircle2 size={18} />
                  <div>
                    <b>Cognitive run berhasil</b>
                    <small>Response diterima dari API.</small>
                  </div>
                </div>

                <div className="tabs">
                  {[
                    ["overview", "Overview"],
                    ["plan", "Plan"],
                    ["meta", "Metacognition"],
                    ["trace", "Trace"],
                    ["raw", "Raw JSON"],
                  ].map(([key, label]) => (
                    <button
                      type="button"
                      key={key}
                      className={tab === key ? "tab selected" : "tab"}
                      onClick={() => setTab(key)}
                    >
                      {label}
                    </button>
                  ))}
                </div>

                {tab === "overview" && (
                  <div className="stack">
                    <section className="panel">
                      <div className="panel-head">
                        <h3>Interpretation</h3>
                        <span className="tag">{interpretation?.provider || "LLM"}</span>
                      </div>
                      <p className="summary">{interpretation?.summary || "Tidak tersedia."}</p>
                      <details><summary>Facts</summary><List items={interpretation?.facts} /></details>
                      <details><summary>Implications</summary><List items={interpretation?.implications} /></details>
                      <details><summary>Uncertainties</summary><List items={interpretation?.uncertainties} /></details>
                    </section>
                    <section className="panel">
                      <div className="panel-head">
                        <h3>Reasoning</h3>
                        <span className="tag">{reasoning?.provider || "LLM"}</span>
                      </div>
                      <p className="summary">{reasoning?.summary || "Tidak tersedia."}</p>
                      <details open><summary>Conclusions</summary><List items={reasoning?.conclusions} /></details>
                      <details><summary>Risks</summary><List items={reasoning?.risks} /></details>
                    </section>
                  </div>
                )}

                {tab === "plan" && (
                  <div className="stack">
                    <section className="panel">
                      <div className="panel-head">
                        <h3>Generated plan</h3>
                        <span className="tag">{plan?.steps?.length || 0} steps</span>
                      </div>
                      <p className="summary">{plan?.rationale || "Rationale tidak tersedia."}</p>
                      {(plan?.steps || []).map((step, index) => (
                        <div className="step" key={`${step.index || index}-${step.action}`}>
                          <span className="step-number">{step.index || index + 1}</span>
                          <div>
                            <b>{step.action}</b>
                            <p>{step.purpose}</p>
                            {step.preconditions?.length > 0 && (
                              <details><summary>Preconditions</summary><List items={step.preconditions} /></details>
                            )}
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

                {tab === "meta" && (
                  <div className="stack">
                    <JsonPanel title="Deterministic reflection" value={meta?.deterministic_reflection} />
                    <JsonPanel title="LLM reflection" value={meta?.llm_reflection} />
                  </div>
                )}
                {tab === "trace" && <JsonPanel title="Runtime trace" value={response.trace} />}
                {tab === "raw" && <JsonPanel title="Full API response" value={response} />}
              </>
            )}
          </div>
        </section>

        <ExecutionSection
          stateData={executionState}
          loadingState={executionLoading}
          busy={executionBusy}
          resetting={executionResetting}
          error={executionError}
          result={executionResult}
          selectedActions={selectedActions}
          setSelectedActions={setSelectedActions}
          goalDescription={goalDescription}
          setGoalDescription={setGoalDescription}
          targetStateText={targetStateText}
          setTargetStateText={setTargetStateText}
          onRun={executeSelectedActions}
          onReset={resetSimulation}
          onRefresh={() => refreshExecutionState()}
        />

        <StrategyLab
          data={strategiesData}
          loading={strategiesLoading}
          error={strategiesError}
          updatedAt={strategiesUpdatedAt}
          onRefresh={() => refreshStrategies()}
        />

        <TelemetrySection
          telemetrySources={telemetrySources}
          loading={telemetryLoading}
          error={telemetryError}
          onRefresh={() => refreshTelemetry()}
          refreshedAt={telemetryUpdatedAt}
        />

        <footer>
          hAI Agent · Cognitive API
          <span><Activity size={12} /> Local development workspace</span>
        </footer>
      </main>
    </div>
  );
}
