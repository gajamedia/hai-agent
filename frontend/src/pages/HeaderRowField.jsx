// Which spreadsheet row holds the column names. Rows above it (a title, say) are ignored.
export default function HeaderRowField({ value, onChange }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", fontSize: 13 }}>
      <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
        Nama kolom ada di baris ke-
        <input type="number" min="1" max="100" value={value} style={{ width: 70 }}
          aria-label="Baris nama kolom"
          onChange={(e) => onChange(Math.max(1, Math.min(100, Number(e.target.value) || 1)))} />
      </label>
      <span>Baris di atasnya (misalnya judul) diabaikan.</span>
    </div>
  );
}
