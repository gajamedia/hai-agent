import { useState } from "react";
import "./autonomous-testing.css";

// Which table column each column of an uploaded file goes into. The proposal (if any) was made from column names
// only; the person corrects it, and a file column set to "(abaikan)" is left out. Used wherever a file is uploaded.
export default function FileMapping({ state, busy, onApply, onCancel, note, required }) {
  const [choice, setChoice] = useState(state.mapping || {});
  // A table column that some file column must go into (the column that finds the rows to update).
  const missing = required && !Object.values(choice).includes(required);
  return (
    <section className="at-panel" aria-label="Cocokkan kolom file">
      <h2>Cocokkan kolom file dengan kolom tabel</h2>
      <small>
        {state.proposed_by_model
          ? "Usulan ini dibuat AI dari nama kolom saja; isi file tidak dikirim ke AI. Periksa dan ubah bila perlu."
          : "Pilih kolom tabel untuk tiap kolom file."}
        {" "}Kolom yang tidak dipilih diabaikan, tidak dicatat.{note ? ` ${note}` : ""}
      </small>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: 10, marginTop: 10 }}>
        {state.headers.map((header) => (
          <label key={header}>
            {header}
            <select value={choice[header] || ""} aria-label={`Kolom untuk ${header}`}
              onChange={(e) => setChoice({ ...choice, [header]: e.target.value || null })}>
              <option value="">(abaikan)</option>
              {state.columns.map((c) => <option key={c.name} value={c.name}>{c.name} ({c.type})</option>)}
            </select>
          </label>
        ))}
      </div>
      {missing && (
        <div className="at-warning" role="alert">
          Belum ada kolom file yang dipilih untuk <b>{required}</b>. Kolom itu dipakai untuk menemukan baris yang
          diubah. Pilih kolom file yang berisi nilainya, atau batal dan pilih kolom pencari lain.
        </div>
      )}
      <div className="at-form-row">
        <button type="button" className="at-button at-button-primary"
          disabled={busy || missing || !Object.values(choice).some(Boolean)}
          onClick={() => onApply(Object.fromEntries(state.headers.map((h) => [h, choice[h] || null])))}>
          Pakai pemetaan ini
        </button>
        <button type="button" className="at-button at-button-secondary" disabled={busy} onClick={onCancel}>Batal</button>
      </div>
    </section>
  );
}
