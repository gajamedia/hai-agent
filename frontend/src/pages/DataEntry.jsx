import { useCallback, useEffect, useState } from "react";
import { AlertCircle, CheckCircle2, LoaderCircle } from "lucide-react";
import {
  cancelDataEntry, createDataEntry, getDataEntryDatabases, listRecentEntries, runDataEntry, undoDataEntry,
} from "../api/client.js";
import "./autonomous-testing.css";

const errorText = (error) => error?.body?.detail?.message || error?.message || "Terjadi kesalahan.";

function RowsTable({ rows }) {
  if (!rows?.length) return null;
  const headers = Object.keys(rows[0]);
  return (
    <div style={{ overflowX: "auto" }}>
      <table aria-label="Baris yang akan dicatat" style={{ width: "100%", fontSize: 12, borderCollapse: "collapse" }}>
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

const rowStyle = {
  display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", padding: "10px 12px", borderRadius: 10,
  border: "1px solid rgba(148,163,184,0.22)", background: "rgba(15,23,42,0.55)", fontSize: 13,
};
const formatTime = (seconds) => new Date(seconds * 1000).toLocaleString("id-ID", { dateStyle: "medium", timeStyle: "short" });
const formatWindow = (minutes) =>
  minutes % 1440 === 0 ? `${minutes / 1440} hari` : minutes % 60 === 0 ? `${minutes / 60} jam` : `${minutes} menit`;

const fieldsetStyle = { border: 0, padding: 0, margin: 0, minWidth: 0 };
const legendStyle = { fontSize: 12, fontWeight: 600, letterSpacing: "0.04em", color: "#94a3b8", padding: 0, marginBottom: 10 };
const gridStyle = { display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))", gap: 12 };
const VISIBLE_TABLES = 3;

// One database as a card: its name, how many tables accept new rows and the first few of them.
// (A list of every table in a dropdown ran off the screen.)
function DatabaseCard({ item, selected, onSelect }) {
  const shown = item.tables.slice(0, VISIBLE_TABLES);
  const hidden = item.tables.length - shown.length;
  const chip = {
    fontSize: 11, padding: "2px 8px", borderRadius: 999, color: "#cbd5e1",
    background: "rgba(148,163,184,0.12)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", maxWidth: "100%",
  };
  return (
    <label
      style={{
        position: "relative", display: "flex", flexDirection: "column", gap: 8, padding: "14px 16px",
        borderRadius: 14, cursor: "pointer", transition: "border-color .15s, background .15s, box-shadow .15s",
        border: `1px solid ${selected ? "#5eead4" : "rgba(148,163,184,0.25)"}`,
        background: selected ? "rgba(94,234,212,0.08)" : "rgba(15,23,42,0.6)",
        boxShadow: selected ? "0 0 0 3px rgba(94,234,212,0.15)" : "none",
      }}
    >
      <input
        type="radio" name="entry-database" value={item.name} checked={selected} onChange={onSelect}
        aria-label={item.name} style={{ position: "absolute", opacity: 0, inset: 0, cursor: "pointer" }}
      />
      <span style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
        <b style={{ fontSize: 15, color: "#f1f5f9", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {item.name}
        </b>
        <span aria-hidden="true" style={{
          width: 16, height: 16, borderRadius: "50%", flex: "none",
          border: `2px solid ${selected ? "#5eead4" : "rgba(148,163,184,0.5)"}`,
          background: selected ? "radial-gradient(#5eead4 45%, transparent 50%)" : "transparent",
        }} />
      </span>
      <small style={{ color: "#94a3b8" }}>{item.tables.length} tabel dapat diisi</small>
      <span style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
        {shown.map((t) => <span key={t} style={chip} title={t}>{t}</span>)}
        {hidden > 0 && <span style={{ ...chip, color: "#5eead4" }}>+{hidden} lainnya</span>}
      </span>
    </label>
  );
}

// Everyday recording: say it, check the rows, press Simpan. Nothing more to type or confirm.
export default function DataEntry() {
  const [databases, setDatabases] = useState(null);
  const [database, setDatabase] = useState("");
  const [question, setQuestion] = useState("");
  const [entry, setEntry] = useState(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState(null);
  const [result, setResult] = useState(null);
  const [recent, setRecent] = useState([]);
  const [undoMinutes, setUndoMinutes] = useState(null);
  const [confirming, setConfirming] = useState(null);
  const [notice, setNotice] = useState("");

  const loadRecent = useCallback(async () => {
    try {
      const body = await listRecentEntries();
      setRecent(body.entries || []);
      setUndoMinutes(body.undo_minutes ?? null);
    } catch {
      setRecent([]);
    }
  }, []);

  useEffect(() => { loadRecent(); }, [loadRecent]);

  useEffect(() => {
    let active = true;
    getDataEntryDatabases()
      .then((body) => {
        if (!active) return;
        setDatabases(body.databases || []);
        if (body.databases?.length === 1) setDatabase(body.databases[0].name);
      })
      .catch((error) => {
        if (!active) return;
        setDatabases([]);
        setMessage({ text: errorText(error) });
      });
    return () => { active = false; };
  }, []);

  const guarded = async (work) => {
    setBusy(true);
    setMessage(null);
    try {
      await work();
    } catch (error) {
      setMessage({ text: errorText(error) });
    } finally {
      setBusy(false);
    }
  };

  const prepare = (event) => {
    event.preventDefault();
    return guarded(async () => {
      setResult(null);
      setEntry(null);
      const body = await createDataEntry({ question, ...(database ? { database } : {}) });
      if (body.status === "planned") setEntry(body);
      else setMessage({ text: body.reason });
    });
  };

  // The plan id is the idempotency key: pressing Simpan twice (or a retry after a lost answer) records once.
  const save = () =>
    guarded(async () => {
      const body = await runDataEntry(entry.plan_id, entry.plan_id);
      setResult(body);
      setNotice("");
      setEntry(null);
      setQuestion("");
      await loadRecent();
    });

  const undo = (entryId) =>
    guarded(async () => {
      const body = await undoDataEntry(entryId);
      setConfirming(null);
      setResult(null);
      setNotice(`Dibatalkan: ${body.rows_removed} baris di tabel ${body.table} dihapus.`);
      await loadRecent();
    });

  const cancel = () =>
    guarded(async () => {
      await cancelDataEntry(entry.plan_id);
      setEntry(null);
    });

  return (
    <main className="at-page">
      <header className="at-header">
        <h1>Catat data</h1>
        <p>
          Tulis apa yang ingin dicatat dengan kalimat biasa, misalnya penjualan atau pengeluaran. Anda melihat dulu
          baris yang akan dicatat, lalu menekan Simpan. Halaman ini hanya menambah data baru; mengubah atau
          menghapus data ada di menu Ubah data.
        </p>
      </header>

      {message && (
        <div className="at-warning" role="alert">
          <AlertCircle size={16} /><div>{message.text}</div>
        </div>
      )}

      <section className="at-panel">
        {databases && databases.length === 0 ? (
          <p>
            Belum ada tabel yang disetujui untuk pencatatan. Minta administrator mencentang izin Tambah
            pada tabel yang boleh diisi.
          </p>
        ) : (
          <form className="at-form" onSubmit={prepare} aria-label="Siapkan pencatatan">
            {databases && databases.length > 1 && (
              <fieldset style={fieldsetStyle}>
                <legend style={legendStyle}>Database</legend>
                <div style={gridStyle} role="radiogroup" aria-label="Database">
                  {databases.map((d) => (
                    <DatabaseCard key={d.name} item={d} selected={database === d.name}
                      onSelect={() => setDatabase(d.name)} />
                  ))}
                </div>
              </fieldset>
            )}
            <label>
              Yang dicatat
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Contoh: catat penjualan 10 pcs kopi 25 ribu dan 5 pcs teh 15 ribu"
                rows={3}
                required
              />
            </label>
            <button className="at-button at-button-primary" disabled={busy || !question.trim() || (databases?.length > 1 && !database)}>
              {busy && !entry ? <LoaderCircle size={14} className="spin" /> : null} Siapkan
            </button>
          </form>
        )}
      </section>

      {entry && (
        <section className="at-panel" aria-label="Yang akan dicatat">
          <h2>Periksa dulu. Belum ada yang tersimpan.</h2>
          <div className="at-config-summary">
            <span>Ke tabel <b>{entry.plan.table}</b> pada database <b>{entry.database}</b>: <b>{entry.expected_rows}</b> baris</span>
            {entry.preview?.note && <small>{entry.preview.note}</small>}
            <RowsTable rows={entry.preview?.after} />
          </div>
          <div className="at-form-row">
            <button type="button" className="at-button at-button-primary" disabled={busy} onClick={save}>
              Simpan
            </button>
            <button type="button" className="at-button at-button-secondary" disabled={busy} onClick={cancel}>
              Batal
            </button>
          </div>
        </section>
      )}

      {result && (
        <section className="at-panel" aria-live="polite">
          <CheckCircle2 size={16} /> <b>Tersimpan.</b> {result.rows_affected} baris dicatat pada tabel {result.table}.
          {result.verification !== "verified" && (
            <> Hasilnya belum dapat dipastikan; periksa datanya sebelum mengulang.</>
          )}
        </section>
      )}

      {notice && <section className="at-panel" role="status">{notice}</section>}

      {recent.length > 0 && (
        <section className="at-panel" aria-label="Pencatatan terbaru">
          <h2>Pencatatan terbaru</h2>
          <small>
            Salah catat? Anda bisa membatalkan pencatatan Anda sendiri{undoMinutes ? ` dalam ${formatWindow(undoMinutes)}` : ""},
            selama barisnya belum diubah orang lain.
          </small>
          <ul style={{ listStyle: "none", margin: "10px 0 0", padding: 0, display: "flex", flexDirection: "column", gap: 8 }}>
            {recent.map((item) => (
              <li key={item.entry_id} style={rowStyle}>
                <span style={{ flex: 1, minWidth: 0 }}>
                  <b>{item.rows} baris</b> di {item.table} <small style={{ color: "#94a3b8" }}>· {formatTime(item.recorded_at)}</small>
                </span>
                {item.undone && <small style={{ color: "#94a3b8" }}>dibatalkan</small>}
                {!item.undone && !item.undoable && <small style={{ color: "#94a3b8" }}>batas waktu lewat</small>}
                {item.undoable && confirming !== item.entry_id && (
                  <button type="button" className="at-button at-button-secondary" disabled={busy}
                    onClick={() => setConfirming(item.entry_id)}>Batalkan</button>
                )}
                {item.undoable && confirming === item.entry_id && (
                  <>
                    <small>Hapus {item.rows} baris ini?</small>
                    <button type="button" className="at-button at-button-primary" disabled={busy}
                      onClick={() => undo(item.entry_id)}>Ya, batalkan</button>
                    <button type="button" className="at-button at-button-secondary" disabled={busy}
                      onClick={() => setConfirming(null)}>Tidak</button>
                  </>
                )}
              </li>
            ))}
          </ul>
        </section>
      )}
    </main>
  );
}
