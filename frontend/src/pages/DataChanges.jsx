import { useCallback, useEffect, useState } from "react";
import { AlertCircle, ArrowLeft, LoaderCircle, ShieldCheck } from "lucide-react";
import { Link } from "react-router-dom";
import {
  cancelDataChangePlan,
  createDataChangePlan,
  decideDataChange,
  getDataChangeConfig,
  getDataChangePlan,
  listDataChangeApprovals,
  listDatabaseConnections,
  runDataChangePlan,
} from "../api/client.js";
import { signInAgain } from "../auth/keycloak.js";
import { describeCondition } from "./conditions.js";
import "./autonomous-testing.css";

const PENDING_KEY = "hai.dataChange.pending";
const OPERATION_LABEL = { insert: "Tambah data", update: "Ubah data", delete: "Hapus data" };

const rememberPending = (planId) => {
  try { sessionStorage.setItem(PENDING_KEY, planId); } catch { /* private window: the plan is just asked again */ }
};
const takePending = () => {
  try {
    const value = sessionStorage.getItem(PENDING_KEY);
    sessionStorage.removeItem(PENDING_KEY);
    return value;
  } catch { return null; }
};

const errorCode = (error) => error?.body?.detail?.error;
const errorText = (error) => error?.body?.detail?.message || error?.message || "Terjadi kesalahan.";

function RowsTable({ title, rows }) {
  if (!rows?.length) return null;
  const headers = Object.keys(rows[0]);
  return (
    <div style={{ overflowX: "auto" }}>
      <b>{title}</b>
      <table aria-label={title} style={{ width: "100%", fontSize: 12, borderCollapse: "collapse" }}>
        <thead>
          <tr>{headers.map((h) => <th key={h} style={{ textAlign: "left", padding: 6 }}>{h}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>{headers.map((h) => <td key={h} style={{ padding: 6 }}>{String(row[h] ?? "")}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ChangeSummary({ change }) {
  const { plan, preview } = change;
  return (
    <div className="at-config-summary">
      <span>Jenis: <b>{OPERATION_LABEL[plan.operation]}</b> pada tabel <b>{plan.table}</b></span>
      {plan.set && (
        <span>Nilai baru: {Object.entries(plan.set).map(([k, v]) => `${k} = ${v}`).join(", ")}</span>
      )}
      {plan.conditions && (
        <span>Baris yang dipilih: {plan.conditions.map(describeCondition).join(" DAN ")}</span>
      )}
      <span>Jumlah baris terdampak: <b>{change.expected_rows}</b> (batas Anda: {plan.max_rows})</span>
      {preview?.note && <small>{preview.note}</small>}
      <RowsTable title="Sebelum" rows={preview?.before} />
      <RowsTable title={plan.operation === "insert" ? "Baris yang ditambahkan" : "Sesudah"} rows={preview?.after} />
    </div>
  );
}

export default function DataChanges() {
  const [config, setConfig] = useState(null);
  const [databases, setDatabases] = useState([]);
  const [database, setDatabase] = useState("");
  const [question, setQuestion] = useState("");
  const [maxRows, setMaxRows] = useState(10);
  const [plan, setPlan] = useState(null);
  const [phrase, setPhrase] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState(null);
  const [result, setResult] = useState(null);
  const [approvals, setApprovals] = useState(null); // null: this person is not an approver
  const [notice, setNotice] = useState("");

  const showError = (error) => setMessage({ kind: "error", text: errorText(error) });

  const refreshApprovals = useCallback(async () => {
    try {
      setApprovals((await listDataChangeApprovals()).approvals);
    } catch {
      setApprovals(null);
    }
  }, []);

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const [cfg, list] = await Promise.all([getDataChangeConfig().catch(() => null), listDatabaseConnections()]);
        if (!active) return;
        setConfig(cfg);
        if (cfg) setMaxRows(Math.min(cfg.default_max_rows, cfg.max_rows_ceiling));
        const able = (list.connections || []).filter((c) => c.writable_tables?.length);
        setDatabases(able);
        if (able.length === 1) setDatabase(able[0].name);
      } catch (error) {
        if (active) showError(error);
      }
      const pending = takePending();
      if (pending && active) {
        try {
          const restored = await getDataChangePlan(pending);
          if (active && restored.status === "planned") {
            setPlan(restored);
            setNotice("Anda sudah login ulang. Tekan Jalankan untuk melanjutkan perubahan ini.");
          }
        } catch { /* the plan expired: ask again */ }
      }
      if (active) refreshApprovals();
    })();
    return () => { active = false; };
  }, [refreshApprovals]);

  const guarded = async (work) => {
    setBusy(true);
    setMessage(null);
    try {
      await work();
    } catch (error) {
      showError(error);
    } finally {
      setBusy(false);
    }
  };

  const makePlan = (event) => {
    event.preventDefault();
    return guarded(async () => {
      setResult(null);
      setPlan(null);
      setNotice("");
      setPhrase("");
      const body = await createDataChangePlan({
        question, max_rows: Number(maxRows), ...(database ? { database } : {}),
      });
      if (body.status === "planned") setPlan(body);
      else setMessage({ kind: "warning", text: body.reason });
    });
  };

  // A fresh login is required: if the server says so, go to Keycloak and come back to this plan.
  const needsSignIn = (error, planId) => {
    if (errorCode(error) !== "step_up_required") return false;
    if (planId) rememberPending(planId);
    setMessage({ kind: "warning", text: "Untuk keamanan, Anda akan diminta login ulang ke Keycloak..." });
    signInAgain();
    return true;
  };

  const run = () =>
    guarded(async () => {
      try {
        const body = await runDataChangePlan(plan.plan_id, phrase, plan.plan_id);
        setResult(body);
        setPlan(null);
        setNotice("");
      } catch (error) {
        if (!needsSignIn(error, plan.plan_id)) throw error;
      }
    });

  const cancel = () =>
    guarded(async () => {
      await cancelDataChangePlan(plan.plan_id);
      setPlan(null);
      setNotice("");
    });

  const decide = (planId, decision) =>
    guarded(async () => {
      try {
        await decideDataChange(planId, decision);
        await refreshApprovals();
        setMessage({ kind: "info", text: decision === "reject" ? "Perubahan ditolak." : "Perubahan disetujui." });
      } catch (error) {
        if (!needsSignIn(error, null)) throw error;
      }
    });

  const phraseOk = plan && phrase.trim() === plan.confirm_phrase;
  const waiting = plan?.requires_approval && plan.state !== "approved";

  return (
    <main className="at-page">
      <header className="at-header">
        <Link to="/data-reports" className="at-back"><ArrowLeft size={14} /> Data Reports</Link>
        <h1>Ubah data</h1>
        <p>
          Tambah, ubah, atau hapus data dengan perintah biasa. Agent tidak menulis SQL: ia hanya memilih
          perubahan terstruktur yang Anda lihat dulu, dan hanya pada tabel dan kolom yang disetujui untuk diubah.
        </p>
      </header>

      {message && (
        <div className="at-warning" role="alert">
          <AlertCircle size={16} /><div>{message.text}</div>
        </div>
      )}

      <section className="at-panel">
        {databases.length === 0 ? (
          <p>
            Belum ada koneksi yang tabelnya disetujui untuk diubah. Di <Link to="/data-reports">Data Reports</Link>,
            buka <b>Ubah persetujuan</b> dan centang <b>boleh diubah</b> serta izin Tambah/Ubah/Hapus. Server juga
            butuh akun penulis di profil database (lihat docs/TRY_DATA_REPORTS_LOCALLY.md).
          </p>
        ) : (
          <form className="at-form" onSubmit={makePlan} aria-label="Buat rencana perubahan">
            {databases.length > 1 && (
              <label>
                Database
                <select value={database} onChange={(e) => setDatabase(e.target.value)} required>
                  <option value="">Pilih...</option>
                  {databases.map((d) => <option key={d.name} value={d.name}>{d.name}</option>)}
                </select>
              </label>
            )}
            <label>
              Perintah
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Contoh: naikkan harga beli bahan gula menjadi 15 ribu"
                rows={3}
                required
              />
            </label>
            <label>
              Maksimal baris yang boleh berubah
              <input
                type="number" min={1} max={config?.max_rows_ceiling || 10000} value={maxRows}
                onChange={(e) => setMaxRows(e.target.value)} required
              />
              <small>Jika perubahan mengenai lebih banyak baris, rencana ditolak. Batas server: {config?.max_rows_ceiling ?? "-"}.</small>
            </label>
            <button className="at-button at-button-primary" disabled={busy || !question.trim()}>
              {busy ? <LoaderCircle size={14} className="spin" /> : null} Buat rencana
            </button>
          </form>
        )}
      </section>

      {plan && (
        <section className="at-panel" aria-label="Rencana perubahan">
          <h2>Periksa dulu. Belum ada yang diubah.</h2>
          <ChangeSummary change={plan} />
          {notice && <p role="status">{notice}</p>}
          {waiting && (
            <div className="at-warning">
              <ShieldCheck size={16} />
              <div>
                Perubahan ini perlu disetujui <b>orang lain</b> (peran data-change-approver) sebelum bisa dijalankan.
                Minta mereka membuka halaman ini; setelah disetujui, tekan Jalankan.
              </div>
            </div>
          )}
          <label>
            Ketik persis: <code>{plan.confirm_phrase}</code>
            <input value={phrase} onChange={(e) => setPhrase(e.target.value)} autoComplete="off" aria-label="Frasa konfirmasi" />
          </label>
          <small>Saat Jalankan ditekan, Anda mungkin diminta login ulang ke Keycloak untuk memastikan ini benar Anda.</small>
          <div className="at-form-row">
            <button type="button" className="at-button at-button-primary" disabled={busy || !phraseOk} onClick={run}>
              Jalankan
            </button>
            <button type="button" className="at-button at-button-secondary" disabled={busy} onClick={cancel}>
              Batal
            </button>
          </div>
        </section>
      )}

      {result && (
        <section className="at-panel" aria-live="polite">
          <b>Berhasil.</b> {result.rows_affected} baris {{ insert: "ditambahkan", update: "diubah", delete: "dihapus" }[result.operation]} pada tabel {result.table}.
          {result.approved_by && <> Disetujui oleh {result.approved_by}.</>}
        </section>
      )}

      {approvals && (
        <section className="at-panel" aria-label="Menunggu persetujuan">
          <h2>Menunggu persetujuan Anda</h2>
          {approvals.length === 0 && <p>Tidak ada.</p>}
          {approvals.map((item) => (
            <div key={item.plan_id} className="at-config-summary">
              <span>Diajukan oleh <b>{item.owner}</b> pada database <b>{item.database}</b>: “{item.question}”</span>
              <ChangeSummary change={item} />
              <div className="at-form-row">
                <button type="button" className="at-button at-button-primary" disabled={busy}
                  onClick={() => decide(item.plan_id, "approve")}>Setujui</button>
                <button type="button" className="at-button at-button-secondary" disabled={busy}
                  onClick={() => decide(item.plan_id, "reject")}>Tolak</button>
              </div>
            </div>
          ))}
        </section>
      )}
    </main>
  );
}
