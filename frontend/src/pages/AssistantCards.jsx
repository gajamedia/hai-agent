import { useState } from "react";
import { AlertCircle, ShieldCheck } from "lucide-react";
import { downloadReport } from "../api/client.js";
import { describeCondition } from "./conditions.js";
import { ChangeSummary, RowsTable } from "./DataChanges.jsx";
import { ResultTable } from "./DataReports.jsx";
import "./autonomous-testing.css";

// The cards the chat shows for each pipeline. They only display a plan and report what the person pressed;
// the calls themselves are made by the Assistant page, with the same routes the other pages use.

const formatTime = (seconds) => new Date(seconds * 1000).toLocaleString("id-ID", { dateStyle: "medium", timeStyle: "short" });

function Buttons({ busy, disabled, onRun, runLabel, onCancel, extra }) {
  return (
    <div className="at-form-row">
      <button type="button" className="at-button at-button-primary" disabled={busy || disabled} onClick={onRun}>
        {runLabel}
      </button>
      <button type="button" className="at-button at-button-secondary" disabled={busy} onClick={onCancel}>Batal</button>
      {extra}
    </div>
  );
}

function PhraseField({ phrase, value, onChange }) {
  return (
    <label>
      Ketik persis: <code>{phrase}</code>
      <input value={value} onChange={(e) => onChange(e.target.value)} autoComplete="off" aria-label="Frasa konfirmasi" />
    </label>
  );
}

export function DatabaseCard({ view, busy, onRun, onCancel }) {
  const [phrase, setPhrase] = useState("");
  return (
    <section className="at-panel" aria-label="Rencana database">
      <b>Periksa dulu. Belum ada yang dibuat.</b>
      <div className="at-config-summary">
        {view.mode === "add_tables" ? (
          <span>Menambah tabel ke database yang <b>sudah ada</b>: <b>{view.plan.database}</b> (server {view.profile}).</span>
        ) : (
          <span>Database baru: <b>{view.plan.database}</b> (server {view.profile})</span>
        )}
        {view.file?.sent_to_model && <small>Gambar {view.file.name} dikirim ke model AI untuk dibaca. Periksa tabel dan kolomnya.</small>}
        {view.file?.sent_to_model_rows != null && (
          <small>Isi file ({view.file.sent_to_model_rows} baris pertama) dikirim ke model AI untuk merancang ini.</small>
        )}
        {view.plan.tables.map((t) => (
          <span key={t.name}>
            Tabel <b>{t.name}</b>: {t.columns.map((c) => `${c.name} (${c.type}${c.nullable ? "" : ", wajib"})`).join(", ")}
          </span>
        ))}
      </div>
      <PhraseField phrase={view.confirm_phrase} value={phrase} onChange={setPhrase} />
      <small>Saat Jalankan ditekan, Anda mungkin diminta login ulang ke Keycloak.</small>
      <Buttons busy={busy} disabled={phrase.trim() !== view.confirm_phrase} onRun={() => onRun({})}
        runLabel="Jalankan" onCancel={onCancel} />
    </section>
  );
}

// Recording new rows: look, then Simpan. No phrase, as on the Catat data page.
export function EntryCard({ view, busy, onRun, onCancel }) {
  const [again, setAgain] = useState(false);
  const duplicate = view.file?.already_recorded_at;
  return (
    <section className="at-panel" aria-label="Yang akan dicatat">
      <b>Periksa dulu. Belum ada yang tersimpan.</b>
      <div className="at-config-summary">
        <span>Ke tabel <b>{view.plan.table}</b> pada database <b>{view.database}</b>: <b>{view.expected_rows}</b> baris</span>
        {view.file && (
          <small>File {view.file.name}{view.file.sheet ? ` (lembar ${view.file.sheet})` : ""}: {view.file.rows} baris, kolom {view.file.columns.join(", ")}.</small>
        )}
        {view.file?.left_empty?.length > 0 && (
          <small>Dibiarkan kosong (tidak ada aturan isian, atau data pribadi): {view.file.left_empty.join(", ")}.</small>
        )}
        {view.preview?.note && <small>{view.preview.note}</small>}
        <RowsTable title="Baris yang akan dicatat" rows={view.preview?.after} order={view.column_order} />
      </div>
      {duplicate && (
        <div className="at-warning" role="alert">
          <AlertCircle size={16} />
          <div>
            File yang sama sudah pernah dicatat pada {formatTime(duplicate)}.
            <label style={{ display: "block", marginTop: 6 }}>
              <input type="checkbox" checked={again} onChange={(e) => setAgain(e.target.checked)} />
              {" "}Saya tahu, catat lagi
            </label>
          </div>
        </div>
      )}
      <Buttons busy={busy} disabled={Boolean(duplicate) && !again}
        onRun={() => onRun({ confirmDuplicate: Boolean(duplicate) })} runLabel="Simpan" onCancel={onCancel} />
    </section>
  );
}

// Changing or deleting rows: preview, exact phrase, a fresh login, and a second person where the server wants one.
export function ChangeCard({ view, busy, onRun, onCancel, onRefresh }) {
  const [phrase, setPhrase] = useState("");
  const waiting = view.requires_approval && view.state !== "approved";
  return (
    <section className="at-panel" aria-label="Rencana perubahan">
      <b>Periksa dulu. Belum ada yang diubah.</b>
      <ChangeSummary change={view} />
      {waiting && (
        <div className="at-warning">
          <ShieldCheck size={16} />
          <div>
            Perubahan ini perlu disetujui dulu (peran data-change-approver), di tab Ubah data pada kotak
            <b> Menunggu persetujuan</b>. Setelah disetujui, tekan Segarkan status lalu Jalankan.
          </div>
        </div>
      )}
      <PhraseField phrase={view.confirm_phrase} value={phrase} onChange={setPhrase} />
      <small>Saat Jalankan ditekan, Anda mungkin diminta login ulang ke Keycloak untuk memastikan ini benar Anda.</small>
      <Buttons busy={busy} disabled={waiting || phrase.trim() !== view.confirm_phrase} onRun={() => onRun({})}
        runLabel="Jalankan" onCancel={onCancel}
        extra={waiting && (
          <button type="button" className="at-button at-button-secondary" disabled={busy} onClick={onRefresh}>
            Segarkan status
          </button>
        )} />
    </section>
  );
}

export function ReportCard({ view, busy, onRun, onCancel }) {
  const p = view.plan;
  const filters = [...Object.entries(p.filters || {}).map(([k, v]) => `${k} = ${v}`), ...(p.conditions || []).map(describeCondition)];
  return (
    <section className="at-panel" aria-label="Rencana laporan">
      <b>Rencana (belum ada data yang dibaca)</b>
      <div className="at-config-summary">
        <span>Tabel: {p.table}</span>
        {p.proposed_relations?.length > 0 && (
          <div className="at-warning" aria-label="Relasi usulan AI">
            <AlertCircle size={16} />
            <div>
              <b>Periksa: AI menduga hubungan antar tabel</b>
              {p.proposed_relations.map((r) => (
                <p key={`${r.table}.${r.column}`}>
                  Isi kolom <b>{r.column}</b> di tabel <b>{r.table}</b> sama dengan kolom <b>{r.ref_column}</b> di tabel <b>{r.ref_table}</b>.
                  Bila benar, tekan Jalankan: hubungan ini diingat untuk pertanyaan berikutnya.
                </p>
              ))}
            </div>
          </div>
        )}
        {p.joins?.length > 0 && (
          <span>Gabungkan tabel: {p.joins.map((j) => `${j.table} (${j.on.join(" = ")})`).join("; ")}</span>
        )}
        {p.kind === "list" ? (
          <>
            <span>Daftar baris (data apa adanya). Kolom: {p.columns.join(", ")}</span>
            <span>Maksimal: {p.limit} baris</span>
          </>
        ) : (
          <>
            <span>Kelompokkan per: {p.group_by.join(", ")}</span>
            <span>Hitung: {p.metrics.map((m) => `${m.operation}(${m.column || "*"})`).join(", ")}</span>
          </>
        )}
        <span>Filter: {filters.length ? filters.join(" DAN ") : "tidak ada"}</span>
      </div>
      <Buttons busy={busy} onRun={() => onRun({})} runLabel="Jalankan" onCancel={onCancel} />
    </section>
  );
}

export function ReportResult({ result, onError }) {
  const save = async () => {
    try {
      const { blob, filename } = await downloadReport(result.download.url);
      const href = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = href;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(href);
    } catch (error) { onError(error); }
  };
  if (result.status === "succeeded") {
    return (
      <section className="at-panel" aria-label="Hasil laporan">
        <b>
          {result.kind === "list" ? `Daftar baris dari ${result.plan?.table}` : `Laporan selesai: ${result.plan?.table}`}
          : {result.rows?.length} dari {result.row_count} baris.
        </b>
        {result.row_count > (result.rows?.length || 0) && <small>Layar menampilkan baris pertama; file yang diunduh memuat semuanya.</small>}
        <ResultTable rows={result.rows} />
        {result.download && (
          <div className="at-form-row">
            <button type="button" className="at-button at-button-secondary" onClick={save}>
              Unduh {String(result.download.filename || "file").split(".").pop().toUpperCase()}
            </button>
          </div>
        )}
      </section>
    );
  }
  return (
    <section className="at-panel" role="alert">
      {result.status === "failed"
        ? `Gagal (${result.error_code}): ${result.reason || "lihat log server"}`
        : result.reason || "Pertanyaan itu belum bisa dijawab."}
    </section>
  );
}

// A change somebody else (or this person, where the server allows it) planned and that needs a second yes.
export function ApprovalCard({ view, busy, onDecide }) {
  return (
    <section className="at-panel" aria-label="Menunggu persetujuan">
      <span>Diajukan oleh <b>{view.own ? "Anda sendiri" : view.owner}</b> pada database <b>{view.database}</b>: “{view.question}”</span>
      <ChangeSummary change={view} />
      <div className="at-form-row">
        <button type="button" className="at-button at-button-primary" disabled={busy} onClick={() => onDecide("approve")}>Setujui</button>
        <button type="button" className="at-button at-button-secondary" disabled={busy} onClick={() => onDecide("reject")}>Tolak</button>
      </div>
    </section>
  );
}
