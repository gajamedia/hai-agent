import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertCircle,
  ArrowLeft,
  CheckCircle2,
  Clock3,
  FlaskConical,
  LoaderCircle,
  Play,
  RefreshCw,
  ShieldCheck,
  XCircle,
} from "lucide-react";
import { Link } from "react-router-dom";
import {
  createAutonomousTestBatch,
  getAutonomousTestBatch,
  getAutonomousTestBatchResults,
  getAutonomousTestBatches,
  startAutonomousTestBatch,
} from "../api/client.js";
import "./autonomous-testing.css";

const pretty = (value) => JSON.stringify(value ?? {}, null, 2);
const TERMINAL = new Set(["completed", "failed", "cancelled"]);
const STATUS_LABELS = {
  pending: "Menunggu",
  running: "Berjalan",
  passed: "Lulus",
  failed: "Gagal",
  inconclusive: "Tidak konklusif",
  blocked: "Terblokir",
  completed: "Selesai",
  cancelled: "Dibatalkan",
};

function ErrorBox({ children }) {
  if (!children) return null;
  return <div className="at-alert"><AlertCircle size={17} /><span>{children}</span></div>;
}

function StatusPill({ status }) {
  return <span className={`at-status at-status-${status || "unknown"}`}>{STATUS_LABELS[status] || status || "—"}</span>;
}

function Metric({ label, value, tone = "" }) {
  return <article className={`at-metric ${tone}`}><small>{label}</small><b>{value ?? 0}</b></article>;
}

function CaseDetail({ item }) {
  const result = item.result || item;
  const evaluation = result.evaluation || {};
  return (
    <details className="at-case-detail">
      <summary>
        <span><code>{item.case_id || result.case_id}</code></span>
        <StatusPill status={item.status || result.status} />
      </summary>
      <div className="at-case-content">
        <div className="at-detail-grid">
          <div><small>GOAL</small><p>{evaluation.reason || "Tidak ada ringkasan evaluasi."}</p></div>
          <div><small>RECOVERY ATTEMPTS</small><p>{result.recovery_attempts ?? 0}</p></div>
          <div><small>GOAL SATISFIED</small><p>{String(evaluation.goal_satisfied ?? "unknown")}</p></div>
          <div><small>ERROR</small><p>{result.error || "—"}</p></div>
        </div>
        <div className="at-json-grid">
          <div><small>INITIAL STATE</small><pre>{pretty(evaluation.initial_state)}</pre></div>
          <div><small>FINAL STATE</small><pre>{pretty(evaluation.final_state)}</pre></div>
          <div><small>EVALUATION EVIDENCE</small><pre>{pretty(evaluation.evidence || [])}</pre></div>
          <div><small>CASE METADATA</small><pre>{pretty(result.metadata)}</pre></div>
        </div>
      </div>
    </details>
  );
}

export default function AutonomousTesting() {
  const [form, setForm] = useState({
    count: 100,
    seed: 42,
    difficulty: "medium",
    inject_fault_every: 0,
    recovery_enabled: true,
    max_attempts: 2,
  });
  const [history, setHistory] = useState([]);
  const [selectedId, setSelectedId] = useState("");
  const [detail, setDetail] = useState(null);
  const [caseResults, setCaseResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [starting, setStarting] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const selectedStatus = detail?.status;
  const isActive = selectedStatus === "running" && detail?.active_in_process === true;
  const interrupted = detail?.interrupted_requires_reconciliation === true;
  const counts = useMemo(() => {
    const fromResult = detail?.result?.counts;
    if (fromResult) return fromResult;
    return caseResults.reduce((acc, item) => {
      const status = item.status || item.result?.status || "unknown";
      acc[status] = (acc[status] || 0) + 1;
      return acc;
    }, {});
  }, [detail, caseResults]);

  const refreshHistory = useCallback(async (signal) => {
    const data = await getAutonomousTestBatches({ limit: 50, signal });
    setHistory(Array.isArray(data.items) ? data.items : []);
  }, []);

  const refreshSelected = useCallback(async (batchId, signal) => {
    if (!batchId) return;
    const [batchData, resultData] = await Promise.all([
      getAutonomousTestBatch(batchId, signal),
      getAutonomousTestBatchResults(batchId, signal),
    ]);
    setDetail(batchData);
    setCaseResults(Array.isArray(resultData.items) ? resultData.items : []);
  }, []);

  const refreshAll = useCallback(async (batchId = selectedId, signal) => {
    setRefreshing(true);
    setError("");
    try {
      await refreshHistory(signal);
      if (batchId) await refreshSelected(batchId, signal);
    } catch (exception) {
      if (exception.name !== "AbortError") setError(`${exception.message}${exception.body ? `\n${pretty(exception.body)}` : ""}`);
    } finally {
      if (!signal?.aborted) {
        setLoading(false);
        setRefreshing(false);
      }
    }
  }, [refreshHistory, refreshSelected, selectedId]);

  useEffect(() => {
    const controller = new AbortController();
    refreshAll("", controller.signal);
    return () => controller.abort();
  }, [refreshAll]);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      setCaseResults([]);
      return undefined;
    }
    const controller = new AbortController();
    refreshSelected(selectedId, controller.signal).catch((exception) => {
      if (exception.name !== "AbortError") setError(exception.message);
    });
    return () => controller.abort();
  }, [selectedId, refreshSelected]);

  useEffect(() => {
    if (!selectedId || TERMINAL.has(selectedStatus) || interrupted) return undefined;
    const timer = window.setInterval(() => {
      refreshSelected(selectedId).catch((exception) => setError(exception.message));
      refreshHistory().catch((exception) => setError(exception.message));
    }, 2000);
    return () => window.clearInterval(timer);
  }, [selectedId, selectedStatus, interrupted, refreshSelected, refreshHistory]);

  function updateField(name, value) {
    setForm((previous) => ({ ...previous, [name]: value }));
  }

  async function handleCreate(event) {
    event.preventDefault();
    setCreating(true);
    setError("");
    setNotice("");
    try {
      const created = await createAutonomousTestBatch({
        count: Number(form.count),
        seed: Number(form.seed),
        difficulty: form.difficulty,
        inject_fault_every: Number(form.inject_fault_every),
        recovery_policy: {
          enabled: form.recovery_enabled,
          max_attempts: form.recovery_enabled ? Number(form.max_attempts) : 0,
          stop_on_unknown_outcome: true,
        },
        environment_config: {
          cognitive_runtime_mode: "isolated",
          sequential_only: true,
          simulation_memory_mode: "in_memory",
        },
      });
      await refreshHistory();
      setSelectedId(created.batch_id);
      setNotice(`Batch ${created.batch_id} berhasil dibuat.`);
    } catch (exception) {
      setError(`${exception.message}${exception.body ? `\n${pretty(exception.body)}` : ""}`);
    } finally {
      setCreating(false);
    }
  }

  async function handleStart() {
    if (!selectedId || detail?.status !== "pending") return;
    setStarting(true);
    setError("");
    setNotice("");
    try {
      const response = await startAutonomousTestBatch(selectedId);
      setNotice(response.message || "Batch diterima oleh backend.");
      await refreshAll(selectedId);
    } catch (exception) {
      setError(`${exception.message}${exception.body ? `\n${pretty(exception.body)}` : ""}`);
      await refreshAll(selectedId);
    } finally {
      setStarting(false);
    }
  }

  const total = detail?.requested_cases ?? detail?.batch?.requested_cases ?? caseResults.length;
  const completed = caseResults.length;

  return (
    <div className="at-shell">
      <header className="at-header">
        <div>
          <Link to="/" className="at-back"><ArrowLeft size={15} /> Kembali ke Dashboard</Link>
          <p className="at-eyebrow">AT-02 · QUALITY & RELIABILITY</p>
          <h1>Autonomous Testing Workspace</h1>
          <p className="at-subtitle">Bangun batch skenario, jalankan pengujian secara sequential, dan evaluasi hasil secara independen.</p>
        </div>
        <button className="at-button at-button-secondary" onClick={() => refreshAll()} disabled={refreshing}>
          {refreshing ? <LoaderCircle className="spin" size={15} /> : <RefreshCw size={15} />}
          Refresh
        </button>
      </header>

      <div className="at-contract-note">
        <ShieldCheck size={17} />
        <span>Isolated per-case environment · Sequential execution · Independent evaluator · Learning integration disabled</span>
      </div>
      <ErrorBox>{error}</ErrorBox>
      {notice && <div className="at-notice"><CheckCircle2 size={16} />{notice}</div>}

      <div className="at-layout">
        <section className="at-panel">
          <div className="at-section-heading"><FlaskConical size={18} /><div><small>CONFIGURATION</small><h2>Buat batch pengujian</h2></div></div>
          <form onSubmit={handleCreate} className="at-form">
            <label>Jumlah test case
              <input type="number" min="1" max="1000" required value={form.count} onChange={(e) => updateField("count", e.target.value)} />
            </label>
            <div className="at-form-row">
              <label>Seed
                <input type="number" min="0" required value={form.seed} onChange={(e) => updateField("seed", e.target.value)} />
              </label>
              <label>Difficulty
                <select value={form.difficulty} onChange={(e) => updateField("difficulty", e.target.value)}>
                  <option value="easy">Easy</option><option value="medium">Medium</option><option value="hard">Hard</option>
                </select>
              </label>
            </div>
            <label>Inject fault setiap N case
              <input type="number" min="0" max="1000" value={form.inject_fault_every} onChange={(e) => updateField("inject_fault_every", e.target.value)} />
              <small>0 = nonaktif; angka lebih besar dari 0 mengaktifkan injeksi berkala.</small>
            </label>
            <label className="at-check">
              <input type="checkbox" checked={form.recovery_enabled} onChange={(e) => updateField("recovery_enabled", e.target.checked)} />
              Aktifkan recovery policy
            </label>
            <label>Maximum recovery attempts
              <input type="number" min="0" max="10" disabled={!form.recovery_enabled} value={form.max_attempts} onChange={(e) => updateField("max_attempts", e.target.value)} />
            </label>
            <div className="at-config-summary">
              <b>Kontrak eksekusi</b>
              <span>Sequential only</span><span>Memory: in-memory</span><span>Unknown outcome: stop & reconcile</span>
            </div>
            <button className="at-button at-button-primary" type="submit" disabled={creating}>
              {creating ? <LoaderCircle className="spin" size={16} /> : <FlaskConical size={16} />}
              {creating ? "Membuat batch..." : "Generate batch"}
            </button>
          </form>
        </section>

        <section className="at-main-column">
          <section className="at-panel">
            <div className="at-section-heading"><Activity size={18} /><div><small>RUN CONTROL</small><h2>Batch aktif</h2></div></div>
            {!detail ? (
              <div className="at-empty"><Clock3 size={25} /><b>Belum ada batch dipilih</b><p>Buat batch baru atau pilih riwayat untuk melihat detail.</p></div>
            ) : (
              <>
                <div className="at-batch-heading">
                  <div><code>{detail.batch_id}</code><p>{detail.difficulty || detail.batch?.difficulty} · Seed {detail.seed ?? detail.batch?.seed} · {total} cases</p></div>
                  <StatusPill status={selectedStatus} />
                </div>
                {interrupted && <div className="at-warning"><AlertCircle size={17} /><div><b>Batch terhenti dan perlu rekonsiliasi</b><p>Backend melaporkan batch berstatus running tetapi tidak aktif pada proses ini. Workspace tidak akan menjalankan ulang batch secara otomatis.</p></div></div>}
                <div className="at-metrics">
                  <Metric label="TOTAL CASES" value={total} />
                  <Metric label="RECORDED" value={completed} />
                  <Metric label="PASSED" value={counts.passed || 0} tone="positive" />
                  <Metric label="FAILED" value={counts.failed || 0} tone="negative" />
                  <Metric label="INCONCLUSIVE" value={counts.inconclusive || 0} />
                  <Metric label="BLOCKED" value={counts.blocked || 0} />
                </div>
                <div className="at-progress"><span style={{ width: `${total ? Math.min(100, (completed / total) * 100) : 0}%` }} /></div>
                <div className="at-run-actions">
                  <span>{isActive ? "Sedang berjalan · status diperbarui otomatis" : `${completed} dari ${total} hasil tercatat`}</span>
                  <button className="at-button at-button-primary" onClick={handleStart} disabled={starting || detail.status !== "pending" || interrupted}>
                    {starting ? <LoaderCircle className="spin" size={15} /> : <Play size={15} />}
                    {starting ? "Mengirim..." : "Start batch"}
                  </button>
                </div>
                {detail.result?.metadata && <pre className="at-meta">{pretty(detail.result.metadata)}</pre>}
              </>
            )}
          </section>

          <section className="at-panel">
            <div className="at-section-heading"><CheckCircle2 size={18} /><div><small>INDEPENDENT EVALUATION</small><h2>Hasil per test case</h2></div></div>
            {caseResults.length ? (
              <div className="at-case-list">{caseResults.map((item) => <CaseDetail key={item.case_id || item.result?.case_id} item={item} />)}</div>
            ) : <p className="at-muted">Belum ada hasil kasus yang tersimpan.</p>}
          </section>
        </section>
      </div>

      <section className="at-panel at-history-panel">
        <div className="at-section-heading"><Clock3 size={18} /><div><small>PERSISTED HISTORY</small><h2>Riwayat batch</h2></div></div>
        {loading ? <p className="at-muted">Memuat riwayat...</p> : history.length ? (
          <div className="at-history-table-wrap">
            <table className="at-history-table">
              <thead><tr><th>Batch ID</th><th>Status</th><th>Difficulty</th><th>Cases</th><th>Seed</th><th>Dibuat</th></tr></thead>
              <tbody>{history.map((item) => (
                <tr key={item.batch_id} className={item.batch_id === selectedId ? "selected" : ""}>
                  <td><button className="at-link-button" onClick={() => setSelectedId(item.batch_id)}>{item.batch_id}</button></td>
                  <td><StatusPill status={item.status} /></td><td>{item.difficulty}</td><td>{item.requested_cases}</td><td>{item.seed}</td><td>{item.created_at || "—"}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        ) : <p className="at-muted">Belum ada riwayat batch.</p>}
      </section>
      <footer className="at-footer">hAI Agent · AT-02 Autonomous Testing · API-backed workspace</footer>
    </div>
  );
}
