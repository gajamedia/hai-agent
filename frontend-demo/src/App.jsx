
import { useCallback, useEffect, useState } from "react";
import {
  Activity, ArrowDown, ArrowRight, BrainCircuit, Check, CheckCircle2,
  ChevronRight, CircleHelp, Clock3, Command, Database, DoorClosed,
  FileText, FlaskConical, Globe2, LoaderCircle, LockKeyhole, Play,
  RefreshCw, ShieldCheck, Sparkles, Terminal, XCircle
} from "lucide-react";

const API_BASE = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

const GOALS = {
  unlock: {
    label: "Buka pintu yang terkunci",
    description: "Buka pintu yang terkunci menggunakan kunci yang tersedia.",
    target_state: { door: "unlocked" }
  },
  box: {
    label: "Buka kotak",
    description: "Buka kotak yang masih tertutup.",
    target_state: { box: "open" }
  },
  key: {
    label: "Ambil kunci",
    description: "Pastikan kunci berada pada agent.",
    target_state: { key: "agent" }
  }
};

/**
 * Convert arbitrary API values to safe, readable text.
 * React must not receive a plain object as a child.
 */
function formatValue(value, depth = 0) {
  if (value === undefined || value === null) {
    return "Tidak tersedia";
  }

  if (typeof value === "string") {
    return value.replaceAll("_", " ");
  }

  if (typeof value === "number") {
    return Number.isFinite(value) ? String(value) : "Tidak valid";
  }

  if (typeof value === "boolean") {
    return value ? "Ya" : "Tidak";
  }

  if (Array.isArray(value)) {
    if (value.length === 0) return "—";

    return value
      .map((item) => formatValue(item, depth + 1))
      .join(", ");
  }

  if (typeof value === "object") {
    if (depth >= 2) {
      try {
        return JSON.stringify(value);
      } catch {
        return "[Object]";
      }
    }

    try {
      return JSON.stringify(value, null, 2);
    } catch {
      return "[Object tidak dapat ditampilkan]";
    }
  }

  return String(value);
}

function StateValue({ value }) {
  return <>{formatValue(value)}</>;
}

function SectionTitle({ icon: Icon, title, trailing }) {
  return (
    <div className="section-title">
      <div className="section-heading">
        <Icon size={17} />
        <span>{title}</span>
      </div>
      {trailing}
    </div>
  );
}

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...options.headers
    }
  });

  const text = await response.text();
  let body = {};

  try {
    body = text ? JSON.parse(text) : {};
  } catch {
    body = { message: text || "Respons tidak dapat dibaca." };
  }

  if (!response.ok) {
    const detail = body?.detail;

    let message = "Permintaan gagal.";

    if (typeof detail === "string") {
      message = detail;
    } else if (detail && typeof detail === "object") {
      message =
        detail.message ||
        detail.error ||
        detail.msg ||
        formatValue(detail);
    } else if (body?.message) {
      message = formatValue(body.message);
    } else if (body?.error) {
      message = formatValue(body.error);
    } else {
      message = `HTTP ${response.status}`;
    }

    throw new Error(message);
  }

  return body;
}

function getErrorMessage(error) {
  if (error instanceof Error) {
    return error.message;
  }

  return formatValue(error);
}

function getActionName(action) {
  if (typeof action === "string" || typeof action === "number") {
    return String(action);
  }

  if (action && typeof action === "object") {
    return String(action.id || action.name || action.action || "Aksi");
  }

  return "Aksi";
}

export default function App() {
  const [health, setHealth] = useState("checking");
  const [world, setWorld] = useState(null);
  const [goalKey, setGoalKey] = useState("unlock");
  const [customGoal, setCustomGoal] = useState("");
  const [useCustom, setUseCustom] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [report, setReport] = useState(null);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [stateDraft, setStateDraft] = useState(null);
  const [savingState, setSavingState] = useState(false);
  const [stateSaveMessage, setStateSaveMessage] = useState("");

  const selectedGoal = GOALS[goalKey] || GOALS.unlock;
  const goalDescription = useCustom
    ? customGoal.trim()
    : selectedGoal.description;

  const refresh = useCallback(async () => {
    setError("");

    try {
      const [healthResponse, stateResponse] = await Promise.all([
        api("/health"),
        api("/api/v1/execution/state")
      ]);

      setHealth(
        healthResponse?.status === "ok" ? "online" : "unknown"
      );

      setWorld(stateResponse);
      setLastUpdated(new Date());
    } catch (err) {
      setHealth("offline");
      setError(getErrorMessage(err));
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    if (world?.state && typeof world.state === "object") {
      setStateDraft({ ...world.state });
    }
  }, [world?.state]);

  function requestRun() {
    setError("");
    setReport(null);

    if (!world?.state || typeof world.state !== "object") {
      setError(
        "State simulasi belum tersedia. Periksa koneksi backend."
      );
      return;
    }

    if (!goalDescription) {
      setError("Masukkan tujuan sebelum melanjutkan.");
      return;
    }

    if (useCustom) {
      setError(
        "Tujuan bebas belum diaktifkan karena target state terverifikasi harus ditentukan. Pilih salah satu contoh tujuan yang tersedia."
      );
      return;
    }

    setConfirmOpen(true);
  }

  async function executeGoal() {
    setConfirmOpen(false);
    setBusy(true);
    setError("");
    setReport(null);

    try {
      const result = await api("/api/v1/execution/plan-run", {
        method: "POST",
        body: JSON.stringify({
          observation: {
            state: world.state,
            available_actions: world.allowed_actions,
            source: "hai_workbench_demo"
          },
          goal_description: selectedGoal.description,
          target_state: selectedGoal.target_state
        })
      });

      setReport(result);

      try {
        const latest = await api("/api/v1/execution/state");
        setWorld(latest);
        setLastUpdated(new Date());
      } catch (refreshError) {
        // Preserve the execution report if refreshing state fails.
        console.error(
          "Gagal memperbarui state setelah eksekusi:",
          getErrorMessage(refreshError)
        );
      }
    } catch (err) {
      setError(getErrorMessage(err));

      try {
        const latest = await api("/api/v1/execution/state");
        setWorld(latest);
        setLastUpdated(new Date());
      } catch {
        // Keep the last known state.
      }
    } finally {
      setBusy(false);
    }
  }

  async function saveWorldState() {
    if (!stateDraft) return;

    setSavingState(true);
    setStateSaveMessage("");
    setError("");
    setReport(null);

    try {
      const result = await api("/api/v1/execution/state", {
        method: "PUT",
        body: JSON.stringify(stateDraft)
      });

      const latest = await api("/api/v1/execution/state");
      setWorld(latest);
      setStateDraft({ ...latest.state });
      setLastUpdated(new Date());
      setStateSaveMessage("World state berhasil disimpan ke backend.");
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setSavingState(false);
    }
  }

  async function resetSimulation() {
    const approved = window.confirm(
      "Reset akan mengembalikan state Room simulation ke kondisi awal. Lanjutkan?"
    );

    if (!approved) return;

    setBusy(true);
    setError("");
    setReport(null);

    try {
      const result = await api("/api/v1/execution/reset", { method: "POST" });

      setWorld((previous) => ({
        status: "ok",
        state: result?.state || {},
        allowed_actions: previous?.allowed_actions || []
      }));
      setStateDraft({ ...(result?.state || {}) });
      setStateSaveMessage("Simulasi dikembalikan ke kondisi awal.");

      setLastUpdated(new Date());
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  const stateEntries =
    world?.state && typeof world.state === "object"
      ? Object.entries(world.state)
      : [];

  const allowedActions = Array.isArray(world?.allowed_actions)
    ? world.allowed_actions
    : [];

  const execution = report?.execution;
  const planSteps = Array.isArray(report?.plan?.steps)
    ? report.plan.steps
    : [];

  const executionSteps = Array.isArray(execution?.steps)
    ? execution.steps
    : [];

  const feedbackItems = Array.isArray(report?.runtime_feedback?.items)
    ? report.runtime_feedback.items
    : [];

  const goalSatisfied = report?.goal?.satisfied === true;
  const executedStepCount = Number(execution?.executed_step_count ?? executionSteps.length);
  const plannedStepCount = Number(execution?.planned_step_count ?? planSteps.length);
  const targetState = report?.goal?.target_state;
  const stateBefore = report?.state_before;
  const targetSatisfiedBeforeExecution =
    targetState &&
    typeof targetState === "object" &&
    Object.keys(targetState).length > 0 &&
    stateBefore &&
    Object.entries(targetState).every(([key, value]) => stateBefore[key] === value);

  function getPlanExecutionStatus(step, index) {
    const executedStep = executionSteps[index];
    if (!executedStep) {
      return {
        kind: "not-run",
        label: execution?.stop_reason === "goal_satisfied"
          ? "Tidak dijalankan: runtime berhenti setelah target terpenuhi"
          : "Belum dijalankan oleh runtime"
      };
    }

    const plannedAction = getActionName(step?.action || step?.name);
    const actualAction = getActionName(executedStep?.action);
    if (plannedAction !== actualAction) {
      return {
        kind: "mismatch",
        label: `Trace runtime berbeda: ${actualAction}`
      };
    }

    return executedStep.success
      ? { kind: "done", label: "Dijalankan · runtime melaporkan berhasil" }
      : { kind: "failed", label: "Dijalankan · hasil tidak berhasil" };
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <BrainCircuit size={22} />
          </div>
          <div>
            <div className="brand-name">hAI Workbench</div>
            <div className="brand-subtitle">AGENT LAB · PROTOTYPE</div>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>

        <div className="nav-item active">
          <Command size={17} />
          Agent workspace
          <span className="nav-dot" />
        </div>

        <div className="nav-item muted">
          <Database size={17} />
          Memory &amp; learning
          <span className="soon">ROADMAP</span>
        </div>

        <div className="nav-item muted">
          <Activity size={17} />
          Activity history
          <span className="soon">ROADMAP</span>
        </div>

        <div className="sidebar-bottom">
          <div className="environment-card">
            <div className="environment-icon">
              <FlaskConical size={17} />
            </div>
            <div>
              <div className="environment-title">
                Sandbox environment
              </div>
              <div className="environment-copy">
                Room simulation only
              </div>
            </div>
          </div>

          <div className="sidebar-note">
            <ShieldCheck size={15} />
            <span>
              Eksekusi dibatasi oleh allowlist dan audit backend.
            </span>
          </div>

          <div className="version">hAI Agent · v0.1 prototype</div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <ChevronRight size={14} />{" "}
            <strong>Agent demo</strong>
          </div>

          <div className={`connection ${health}`}>
            <span className="connection-dot" />
            {health === "checking"
              ? "Menghubungkan"
              : health === "online"
                ? "Backend terhubung"
                : health === "unknown"
                  ? "Status backend tidak diketahui"
                  : "Backend offline"}
          </div>
        </header>

        <div className="content">
          <div className="eyebrow">
            <Sparkles size={14} />
            AUTONOMOUS AGENT WORKSPACE
          </div>

          <h1>
            Ubah tujuan menjadi <span>aksi nyata.</span>
          </h1>

          <p className="hero-copy">
            Berikan tujuan, tinjau permintaan, lalu amati bagaimana
            agent merencanakan, mengeksekusi, dan memverifikasi hasil
            di lingkungan simulasi.
          </p>

          <div className="prototype-banner">
            <FlaskConical size={18} />
            <div>
              <strong>Mode prototipe terbatas</strong>
              <span>
                Agent hanya dapat menggunakan aksi Room simulation
                yang diizinkan. Belum ada aksi internet atau sistem
                eksternal.
              </span>
            </div>
          </div>

          <div className="workspace-grid">
            <section className="panel goal-panel">
              <SectionTitle
                icon={Command}
                title="01 · Define your goal"
                trailing={<span className="step-chip">INPUT</span>}
              />

              <label className="field-label" htmlFor="goal-select">
                Contoh tujuan
              </label>

              <select
                id="goal-select"
                value={goalKey}
                onChange={(event) => {
                  setGoalKey(event.target.value);
                  setUseCustom(false);
                }}
                disabled={busy}
              >
                {Object.entries(GOALS).map(([key, goal]) => (
                  <option key={key} value={key}>
                    {goal.label}
                  </option>
                ))}
              </select>

              <div className="goal-preview">
                <div className="preview-icon">
                  <Sparkles size={18} />
                </div>
                <div>
                  <div className="preview-label">
                    TUJUAN TERPILIH
                  </div>
                  <div className="preview-goal">
                    {selectedGoal.description}
                  </div>
                </div>
              </div>

              <button
                className="text-button"
                onClick={() => setUseCustom((value) => !value)}
                disabled={busy}
              >
                {useCustom
                  ? "Kembali ke tujuan contoh"
                  : "Tulis instruksi sendiri"}
                <ArrowRight size={14} />
              </button>

              {useCustom && (
                <div className="custom-goal">
                  <label className="field-label" htmlFor="custom-goal">
                    Instruksi natural language
                  </label>
                  <textarea
                    id="custom-goal"
                    rows={3}
                    maxLength={500}
                    value={customGoal}
                    onChange={(event) =>
                      setCustomGoal(event.target.value)
                    }
                    placeholder="Jelaskan apa yang ingin dicapai..."
                  />
                  <p className="helper">
                    Input bebas ditampilkan sebagai preview. Eksekusi
                    menunggu pemetaan target state yang terverifikasi.
                  </p>
                </div>
              )}

              <div className="goal-meta">
                <div>
                  <LockKeyhole size={14} />
                  <span>Target terverifikasi</span>
                </div>
                <div>
                  <ShieldCheck size={14} />
                  <span>Allowlist aktif</span>
                </div>
              </div>

              <button
                className="primary-button"
                onClick={requestRun}
                disabled={
                  busy || health !== "online" || !world?.state
                }
              >
                {busy ? (
                  <LoaderCircle className="spin" size={17} />
                ) : (
                  <Play size={17} />
                )}
                {busy ? "Memproses..." : "Tinjau & jalankan"}
                {!busy && <ArrowRight size={16} />}
              </button>
            </section>

            <section className="panel state-panel">
              <SectionTitle
                icon={Database}
                title="02 · World state"
                trailing={
                  <button
                    className="icon-button"
                    title="Refresh state"
                    onClick={refresh}
                    disabled={busy}
                  >
                    <RefreshCw size={15} />
                  </button>
                }
              />

              <div className="state-subtitle">
                State yang dilaporkan oleh backend
              </div>

              {!world?.state ? (
                <div className="empty-state">
                  <LoaderCircle
                    size={22}
                    className={health === "checking" ? "spin" : ""}
                  />
                  <span>
                    {health === "offline"
                      ? "State belum tersedia."
                      : "Mengambil state simulasi..."}
                  </span>
                </div>
              ) : (
                <>
                  <div className="state-list">
                    {stateEntries.map(([key, value]) => (
                      <div className="state-row" key={key}>
                        <div className="state-key">
                          {key.includes("door") ? (
                            <DoorClosed size={16} />
                          ) : (
                            <Database size={16} />
                          )}
                          <span>{key.replaceAll("_", " ")}</span>
                        </div>

                        <span
                          className={`state-value ${
                            typeof value === "string" &&
                            (value.toLowerCase().includes("unlock") ||
                              value === "open")
                              ? "positive"
                              : ""
                          }`}
                        >
                          <StateValue value={value} />
                        </span>
                      </div>
                    ))}
                  </div>

                  <div className="state-footer">
                    <span>
                      <span className="live-dot" /> Live snapshot
                    </span>
                    <span>
                      {lastUpdated
                        ? lastUpdated.toLocaleTimeString("id-ID")
                        : "—"}
                    </span>
                  </div>
                </>
              )}

              {stateDraft && (
                <div className="world-state-editor" style={{ marginTop: 18, padding: 14, border: "1px solid var(--border, #293241)", borderRadius: 12 }}>
                  <div className="allowed-title">
                    <FlaskConical size={14} />
                    EDIT WORLD STATE · SANDBOX
                  </div>
                  <p className="helper">
                    Atur kondisi awal simulasi. Perubahan ini menyiapkan lingkungan,
                    bukan action yang dilakukan agent.
                  </p>

                  <div className="state-editor-grid" style={{ display: "grid", gridTemplateColumns: "minmax(90px, 0.7fr) minmax(130px, 1fr)", alignItems: "center", gap: 10, marginTop: 12 }}>
                    <label className="field-label" htmlFor="state-door">Door</label>
                    <select id="state-door" value={stateDraft.door || "locked"}
                      disabled={busy || savingState}
                      onChange={(event) => setStateDraft((current) => ({ ...current, door: event.target.value }))}>
                      <option value="locked">Locked</option>
                      <option value="unlocked">Unlocked</option>
                    </select>

                    <label className="field-label" htmlFor="state-box">Box</label>
                    <select id="state-box" value={stateDraft.box || "closed"}
                      disabled={busy || savingState}
                      onChange={(event) => setStateDraft((current) => ({ ...current, box: event.target.value }))}>
                      <option value="closed">Closed</option>
                      <option value="open">Open</option>
                    </select>

                    <label className="field-label" htmlFor="state-key">Key</label>
                    <select id="state-key" value={stateDraft.key || "inside_box"}
                      disabled={busy || savingState}
                      onChange={(event) => setStateDraft((current) => ({ ...current, key: event.target.value }))}>
                      <option value="inside_box">Inside box</option>
                      <option value="agent">Held by agent</option>
                      <option value="missing">Missing</option>
                    </select>

                    <label className="field-label" htmlFor="state-spare-key">Spare key</label>
                    <select id="state-spare-key" value={stateDraft.spare_key || "available"}
                      disabled={busy || savingState}
                      onChange={(event) => setStateDraft((current) => ({ ...current, spare_key: event.target.value }))}>
                      <option value="available">Available</option>
                      <option value="found">Found</option>
                      <option value="unavailable">Unavailable</option>
                    </select>

                    <label className="field-label" htmlFor="state-agent-location">Agent location</label>
                    <select id="state-agent-location" value={stateDraft.agent_location || "room"}
                      disabled={busy || savingState} onChange={(event) => setStateDraft((current) => ({ ...current, agent_location: event.target.value }))}>
                      <option value="room">Room</option>
                    </select>
                  </div>

                  <button className="primary-button state-save-button" style={{ width: "100%", marginTop: 14 }}
                    onClick={saveWorldState} disabled={busy || savingState || health !== "online"}>
                    {savingState ? <LoaderCircle className="spin" size={16} /> : <Check size={16} />}
                    {savingState ? "Menyimpan state..." : "Terapkan world state"}
                  </button>
                  {stateSaveMessage && <p className="helper state-save-message" role="status">{stateSaveMessage}</p>}
                </div>
              )}

              <div className="allowed-actions">
                <div className="allowed-title">
                  <ShieldCheck size={14} />
                  AKSI YANG TERDAFTAR
                </div>

                <div className="action-chips">
                  {allowedActions.map((action, index) => (
                    <span
                      className="action-chip"
                      key={`${getActionName(action)}-${index}`}
                    >
                      {getActionName(action)}
                    </span>
                  ))}

                  {!allowedActions.length && (
                    <span className="helper">Belum tersedia</span>
                  )}
                </div>
              </div>

              <button
                className="reset-button"
                onClick={resetSimulation}
                disabled={busy || !world?.state}
              >
                <RefreshCw size={14} />
                Reset simulation
              </button>
            </section>
          </div>

          <section className="workflow panel">
            <SectionTitle
              icon={Activity}
              title="03 · Agent lifecycle"
              trailing={
                <span className="step-chip">LIVE WORKFLOW</span>
              }
            />

            <div className="workflow-track">
              <div className="workflow-step">
                <div className="workflow-icon">
                  <Command size={18} />
                </div>
                <div className="workflow-name">Understand</div>
                <div className="workflow-desc">Tujuan &amp; konteks</div>
              </div>

              <div className="workflow-arrow">
                <ArrowRight size={16} />
              </div>

              <div className="workflow-step">
                <div className="workflow-icon">
                  <BrainCircuit size={18} />
                </div>
                <div className="workflow-name">Plan</div>
                <div className="workflow-desc">Rencana kognitif</div>
              </div>

              <div className="workflow-arrow">
                <ArrowRight size={16} />
              </div>

              <div className="workflow-step">
                <div className="workflow-icon">
                  <ShieldCheck size={18} />
                </div>
                <div className="workflow-name">Authorize</div>
                <div className="workflow-desc">Allowlist &amp; audit</div>
              </div>

              <div className="workflow-arrow">
                <ArrowRight size={16} />
              </div>

              <div className="workflow-step">
                <div className="workflow-icon">
                  <Terminal size={18} />
                </div>
                <div className="workflow-name">Execute</div>
                <div className="workflow-desc">Room simulation</div>
              </div>

              <div className="workflow-arrow">
                <ArrowRight size={16} />
              </div>

              <div className="workflow-step">
                <div className="workflow-icon">
                  <CheckCircle2 size={18} />
                </div>
                <div className="workflow-name">Verify</div>
                <div className="workflow-desc">Target state</div>
              </div>
            </div>

            <p className="workflow-footnote">
              Diagram ini menjelaskan alur yang dimaksud; status
              keberhasilan hanya ditampilkan dari respons runtime.
            </p>
          </section>

          {error && (
            <div className="error-banner" role="alert">
              <XCircle size={18} />
              <div>
                <strong>Permintaan belum selesai</strong>
                <span>{formatValue(error)}</span>
              </div>
              <button
                className="icon-button"
                onClick={() => setError("")}
                aria-label="Tutup pesan"
              >
                <XCircle size={16} />
              </button>
            </div>
          )}

          {report && (
            <section className="panel report-panel">
              <SectionTitle
                icon={FileText}
                title="04 · Execution report"
                trailing={
                  <span
                    className={`result-badge ${
                      goalSatisfied ? "success" : "warning"
                    }`}
                  >
                    {goalSatisfied ? (
                      <CheckCircle2 size={14} />
                    ) : (
                      <CircleHelp size={14} />
                    )}
                    {goalSatisfied
                      ? "Target terverifikasi"
                      : "Target belum terverifikasi"}
                  </span>
                }
              />

              <div className="report-summary">
                <div>
                  <div className="report-label">GOAL</div>
                  <div className="report-value">
                    {formatValue(
                      report.goal?.description ||
                        selectedGoal.description
                    )}
                  </div>
                </div>

                <div>
                  <div className="report-label">EXECUTION</div>
                  <div className="report-value">
                    {executedStepCount} / {plannedStepCount} langkah dijalankan
                  </div>
                </div>

                <div>
                  <div className="report-label">RUNTIME RESULT</div>
                  <div className="report-value">
                    {targetSatisfiedBeforeExecution && executedStepCount === 0
                      ? "Target sudah terpenuhi sebelum eksekusi"
                      : execution?.success === true && goalSatisfied
                        ? "Eksekusi berhasil dan target terverifikasi"
                        : execution?.success === true
                          ? "Eksekusi berhasil; target belum terverifikasi"
                          : "Eksekusi belum berhasil sepenuhnya"}
                  </div>
                </div>
              </div>

              <div className="report-columns">
                <div>
                  <div className="report-section-title">
                    Rencana agent
                  </div>

                  {planSteps.length ? (
                    <div className="timeline">
                      {planSteps.map((step, index) => (
                        <div
                          className="timeline-row"
                          key={`${getActionName(step?.action || step?.name)}-${index}`}
                        >
                          <div className={`timeline-marker ${
                            getPlanExecutionStatus(step, index).kind === "done"
                              ? "done"
                              : getPlanExecutionStatus(step, index).kind === "failed"
                                ? "failed"
                                : ""
                          }`}>
                            {getPlanExecutionStatus(step, index).kind === "done" ? (
                              <Check size={13} />
                            ) : index + 1}
                          </div>
                          <div>
                            <strong>
                              {formatValue(
                                step?.action ||
                                  step?.name ||
                                  `Langkah ${index + 1}`
                              )}
                            </strong>
                            <span>
                              {formatValue(
                                step?.description ||
                                  step?.reason ||
                                  "Langkah dari rencana runtime"
                              )}
                            </span>
                            <span className="helper">
                              {getPlanExecutionStatus(step, index).label}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="helper">
                      Runtime tidak mengembalikan daftar langkah
                      rencana.
                    </p>
                  )}
                </div>

                <div>
                  <div className="report-section-title">
                    Hasil eksekusi
                  </div>

                  {executionSteps.length ? (
                    <div className="timeline">
                      {executionSteps.map((step, index) => (
                        <div
                          className="timeline-row"
                          key={`${getActionName(step?.action)}-${index}`}
                        >
                          <div
                            className={`timeline-marker ${
                              step?.success ? "done" : "failed"
                            }`}
                          >
                            {step?.success ? (
                              <Check size={13} />
                            ) : (
                              index + 1
                            )}
                          </div>

                          <div>
                            <strong>
                              {formatValue(
                                step?.action || `Langkah ${index + 1}`
                              )}
                            </strong>
                            <span>
                              {step?.success
                                ? "Runtime melaporkan langkah berhasil"
                                : formatValue(
                                    step?.error ||
                                      step?.observation ||
                                      "Keberhasilan langkah tidak terkonfirmasi"
                                  )}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="helper">
                      Tidak ada langkah eksekusi yang dilaporkan.
                    </p>
                  )}
                </div>
              </div>

              {execution?.stop_reason === "goal_satisfied" && executedStepCount < plannedStepCount && (
                <div className="verification-box">
                  <div className="verification-icon">
                    <CheckCircle2 size={19} />
                  </div>
                  <div>
                    <strong>Runtime menghentikan rencana lebih awal</strong>
                    <span>
                      {executedStepCount} dari {plannedStepCount} langkah rencana dijalankan. Langkah sisanya tidak dieksekusi karena runtime menilai target telah terpenuhi; daftar rencana tidak berarti semua langkah dijalankan.
                    </span>
                  </div>
                </div>
              )}

              <div className="verification-box">
                <div className="verification-icon">
                  {goalSatisfied ? (
                    <CheckCircle2 size={19} />
                  ) : (
                    <CircleHelp size={19} />
                  )}
                </div>

                <div>
                  <strong>
                    {goalSatisfied
                      ? "Target state cocok dengan hasil observasi"
                      : "Target state belum cocok atau belum terbukti"}
                  </strong>
                  <span>
                    State akhir berasal dari runtime. Keberhasilan
                    rencana saja tidak dianggap sebagai bukti
                    tercapainya tujuan.
                  </span>
                </div>
              </div>

              {feedbackItems.length > 0 && (
                <div className="feedback-box">
                  <div className="report-section-title">
                    Runtime feedback
                  </div>

                  {feedbackItems.map((item, index) => (
                    <div
                      className="feedback-row"
                      key={`${item?.step_index ?? "step"}-${index}`}
                    >
                      <span>
                        Langkah {formatValue(item?.step_index)}
                      </span>
                      <span>
                        {formatValue(
                          item?.diagnosis || "Feedback tersedia"
                        )}
                      </span>
                      <span
                        className={`risk ${
                          item?.risk_level === "high" ? "high" : ""
                        }`}
                      >
                        {formatValue(item?.risk_level || "unknown")} risk
                      </span>
                    </div>
                  ))}
                </div>
              )}

              <details className="raw-details">
                <summary>
                  Inspect raw runtime response <ArrowDown size={14} />
                </summary>
                <pre>{JSON.stringify(report, null, 2)}</pre>
              </details>
            </section>
          )}

          <footer className="page-footer">
            <div>
              <Globe2 size={14} />
              hAI Workbench · Local prototype
            </div>
            <div>
              <ShieldCheck size={14} />
              Simulation only · No external side effects
            </div>
          </footer>
        </div>
      </main>

      {confirmOpen && (
        <div
          className="modal-backdrop"
          role="presentation"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              setConfirmOpen(false);
            }
          }}
        >
          <div
            className="confirm-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="confirm-title"
          >
            <div className="modal-icon">
              <ShieldCheck size={22} />
            </div>

            <h2 id="confirm-title">
              Tinjau permintaan eksekusi
            </h2>

            <p>
              Agent akan mencoba mencapai target berikut di Room
              simulation.
            </p>

            <div className="confirm-detail">
              <span>Tujuan</span>
              <strong>{selectedGoal.description}</strong>
            </div>

            <div className="confirm-detail">
              <span>Target state</span>
              <code>
                {JSON.stringify(selectedGoal.target_state)}
              </code>
            </div>

            <div className="confirm-detail">
              <span>Lingkungan</span>
              <strong>Room simulation · lokal</strong>
            </div>

            <div className="modal-warning">
              Konfirmasi ini adalah langkah UI prototipe, bukan
              persetujuan manusia yang ditegakkan sebagai kontrol
              terpisah di server. Backend tetap menerapkan allowlist
              dan audit yang tersedia.
            </div>

            <div className="modal-actions">
              <button
                className="secondary-button"
                onClick={() => setConfirmOpen(false)}
                disabled={busy}
              >
                Batal
              </button>

              <button
                className="primary-button modal-run"
                onClick={executeGoal}
                disabled={busy}
              >
                {busy ? (
                  <LoaderCircle className="spin" size={16} />
                ) : (
                  <Play size={16} />
                )}
                {busy ? "Memproses..." : "Konfirmasi & jalankan"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
