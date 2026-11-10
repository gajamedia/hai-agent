import { useCallback, useEffect, useState } from "react";
import { AlertCircle, ArrowLeft, Download, ListChecks, LoaderCircle, RefreshCw } from "lucide-react";
import { Link } from "react-router-dom";
import { downloadReport, getTask, listTasks } from "../api/client.js";
import "./autonomous-testing.css";

export const STATUS_LABEL = {
  planned: "Direncanakan",
  awaiting_approval: "Menunggu persetujuan orang lain",
  approved: "Siap dijalankan",
  executing: "Sedang berjalan",
  verifying: "Sedang diverifikasi",
  completed: "Selesai",
  unverified: "Selesai, hasil belum terverifikasi",
  failed: "Gagal",
  cancelled: "Dibatalkan",
  rejected: "Ditolak",
};

export const VERIFICATION_LABEL = {
  verified: "Terverifikasi",
  unverified: "Belum terverifikasi",
  failed: "Verifikasi gagal",
  waived: "Tidak dapat diverifikasi (dinyatakan di kontrak)",
  not_applicable: "Tidak perlu verifikasi",
};

export const EVENT_LABEL = {
  task_planned: "Rencana dibuat",
  task_approved: "Disetujui",
  task_rejected: "Ditolak",
  task_executing: "Mulai dijalankan",
  task_executed: "Selesai dijalankan",
  task_completed: "Terverifikasi dan selesai",
  task_unverified: "Selesai, belum terverifikasi",
  task_failed: "Gagal",
  task_cancelled: "Dibatalkan",
  task_expired: "Kedaluwarsa",
  task_recovered: "Dipulihkan setelah server dimulai ulang",
};

const FILTERS = [
  { value: "", label: "Semua" },
  { value: "waiting", label: "Menunggu" },
  { value: "running", label: "Berjalan" },
  { value: "done", label: "Selesai" },
  { value: "problem", label: "Gagal / dibatalkan" },
];

// Where a person goes to act on a task that is still open.
const OPERATION_PAGE = {
  "database.report": "/database-agent/reports",
  "database.change": "/database-agent/changes",
  "database.create": "/database-agent/create",
};

const LIVE = new Set(["awaiting_approval", "executing", "verifying"]);
const POLL_MS = 10000;

const when = (seconds) => (seconds ? new Date(seconds * 1000).toLocaleString("id-ID") : "");
const errorText = (error) => error?.body?.detail?.message || error?.message || "Terjadi kesalahan.";
const badgeColor = (status) =>
  ({ completed: "#5eead4", unverified: "#f4d19b", failed: "#fda4af", cancelled: "#91a2b9", rejected: "#fda4af" })[status] ||
  "#aab8ca";

export default function Tasks() {
  const [filter, setFilter] = useState("");
  const [tasks, setTasks] = useState([]);
  const [hasMore, setHasMore] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState(null);
  const [detail, setDetail] = useState(null);
  const [detailError, setDetailError] = useState("");

  const load = useCallback(
    async (signal) => {
      try {
        const body = await listTasks({ status: filter || undefined, signal });
        setTasks(body.tasks);
        setHasMore(Boolean(body.has_more));
        setError("");
      } catch (e) {
        if (e?.name !== "AbortError") setError(errorText(e));
      } finally {
        setLoading(false);
      }
    },
    [filter],
  );

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    load(controller.signal);
    return () => controller.abort();
  }, [load]);

  // Open work changes by itself (someone approves, a run finishes): look again while there is any.
  const anyLive = tasks.some((t) => LIVE.has(t.status));
  useEffect(() => {
    if (!anyLive) return undefined;
    const timer = setInterval(() => {
      if (document.visibilityState !== "hidden") load();
    }, POLL_MS);
    return () => clearInterval(timer);
  }, [anyLive, load]);

  useEffect(() => {
    if (!selected) {
      setDetail(null);
      return undefined;
    }
    const controller = new AbortController();
    setDetailError("");
    getTask(selected, controller.signal)
      .then(setDetail)
      .catch((e) => {
        if (e?.name !== "AbortError") setDetailError(errorText(e));
      });
    return () => controller.abort();
  }, [selected]);

  const save = async (info) => {
    try {
      const { blob, filename } = await downloadReport(info.url);
      const href = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = href;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(href);
    } catch (e) {
      setDetailError(errorText(e));
    }
  };

  return (
    <main className="at-page" style={{ padding: "24px clamp(18px,3vw,48px)", background: "#0b1220", minHeight: "100vh" }}>
      <Link className="at-back" to="/">
        <ArrowLeft size={14} /> Kembali
      </Link>
      <header className="at-header">
        <div>
          <span className="at-eyebrow">PLATFORM</span>
          <h1>Riwayat tugas</h1>
          <p className="at-subtitle">
            Semua yang Anda minta dari agen, dengan statusnya. Hasil dinyatakan selesai hanya bila terverifikasi;
            data bisnis yang sudah tidak diperlukan dihapus dari catatan ini.
          </p>
        </div>
      </header>

      {error && (
        <div className="at-alert" role="alert">
          <AlertCircle size={15} /> {error}
        </div>
      )}

      <section className="at-panel" aria-label="Daftar tugas">
        <div className="at-form-row" style={{ alignItems: "end", marginBottom: 12 }}>
          <form className="at-form" onSubmit={(e) => e.preventDefault()}>
            <label>
              Tampilkan
              <select value={filter} onChange={(e) => { setFilter(e.target.value); setSelected(null); }}>
                {FILTERS.map((f) => (
                  <option key={f.value} value={f.value}>{f.label}</option>
                ))}
              </select>
            </label>
          </form>
          <button type="button" className="at-button at-button-secondary" onClick={() => load()} disabled={loading}>
            {loading ? <LoaderCircle size={14} className="spin" /> : <RefreshCw size={14} />} Muat ulang
          </button>
        </div>

        {!loading && !error && tasks.length === 0 && (
          <div className="at-empty">
            <ListChecks size={22} />
            <b>Belum ada tugas</b>
            <p>Tugas muncul di sini setelah Anda membuat rencana laporan atau perubahan data.</p>
          </div>
        )}

        {tasks.length > 0 && (
          <div style={{ overflowX: "auto" }}>
            <table aria-label="Tugas" style={{ width: "100%", fontSize: 12, borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  {["Waktu", "Tugas", "Status", "Verifikasi", ""].map((h) => (
                    <th key={h} style={{ textAlign: "left", padding: 6 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {tasks.map((t) => (
                  <tr key={t.task_id} aria-selected={selected === t.task_id}>
                    <td style={{ padding: 6, whiteSpace: "nowrap" }}>{when(t.created_at)}</td>
                    <td style={{ padding: 6 }}>
                      <b>{t.label}</b>
                      <div className="at-muted">{t.title}</div>
                    </td>
                    <td style={{ padding: 6 }}>
                      <span className="at-status" style={{ color: badgeColor(t.status) }}>
                        {STATUS_LABEL[t.status] || t.status}
                      </span>
                    </td>
                    <td style={{ padding: 6 }}>{VERIFICATION_LABEL[t.verification] || "-"}</td>
                    <td style={{ padding: 6 }}>
                      <button
                        type="button"
                        className="at-button at-button-secondary"
                        aria-label={`Lihat rincian: ${t.title}`}
                        onClick={() => setSelected(t.task_id)}
                      >
                        Rincian
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {hasMore && <p className="at-muted">Menampilkan tugas terbaru saja. Persempit dengan filter.</p>}
      </section>

      {selected && (
        <section className="at-panel" aria-label="Rincian tugas">
          {detailError && (
            <div className="at-alert" role="alert">
              <AlertCircle size={15} /> {detailError}
            </div>
          )}
          {!detail && !detailError && <LoaderCircle size={16} className="spin" aria-label="Memuat rincian" />}
          {detail && (
            <>
              <div className="at-batch-heading">
                <div>
                  <h2 style={{ margin: 0 }}>{detail.title}</h2>
                  <p>
                    {detail.label} · {STATUS_LABEL[detail.status] || detail.status}
                    {detail.verification ? ` · ${VERIFICATION_LABEL[detail.verification] || detail.verification}` : ""}
                  </p>
                </div>
                <button type="button" className="at-button at-button-secondary" onClick={() => setSelected(null)}>
                  Tutup
                </button>
              </div>

              {detail.status === "unverified" && (
                <div className="at-warning" role="status">
                  <AlertCircle size={15} />
                  <div>
                    Tugas sudah dijalankan tetapi hasilnya belum bisa dipastikan dari keadaan yang teramati.
                    Periksa datanya sebelum dianggap selesai.
                  </div>
                </div>
              )}
              {detail.status === "failed" && detail.error && (
                <div className="at-alert" role="status">
                  <AlertCircle size={15} /> {detail.error}
                </div>
              )}
              {["planned", "awaiting_approval", "approved"].includes(detail.status) && OPERATION_PAGE[detail.operation] && (
                <p>
                  <Link to={OPERATION_PAGE[detail.operation]}>Buka halamannya untuk melanjutkan</Link>
                  {detail.expires_in_seconds != null && ` (kedaluwarsa dalam ${Math.ceil(detail.expires_in_seconds / 60)} menit)`}
                </p>
              )}

              {detail.result && (
                <div className="at-config-summary" aria-label="Hasil">
                  {Object.entries(detail.result).map(([key, value]) => (
                    <span key={key}>{key}: <b>{String(value)}</b></span>
                  ))}
                </div>
              )}
              {detail.download && (
                <button type="button" className="at-button at-button-primary" onClick={() => save(detail.download)}>
                  <Download size={14} /> Unduh {detail.download.filename}
                </button>
              )}

              <h3>Jejak</h3>
              <ol aria-label="Jejak tugas" style={{ paddingLeft: 18, fontSize: 12, lineHeight: 1.9 }}>
                {detail.timeline.map((e, i) => (
                  <li key={i}>
                    <b>{EVENT_LABEL[e.event] || e.event}</b>{" "}
                    <span className="at-muted">
                      {when(e.at)}{e.actor && e.actor !== "system" ? ` · oleh ${e.actor}` : ""}
                      {e.self_approved ? " · disetujui sendiri" : ""}
                    </span>
                  </li>
                ))}
              </ol>
            </>
          )}
        </section>
      )}
    </main>
  );
}
