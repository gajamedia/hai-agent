import { useCallback, useEffect, useState } from "react";
import { Database, FileText, LoaderCircle, RefreshCw } from "lucide-react";
import { listDataFiles, listProfileDatabases } from "../api/client.js";

const errorText = (error) => error?.body?.detail?.message || error?.message || "Tidak bisa memuat daftar.";

// A short connection name made from a database or file name ("hai_toko.sqlite3" -> "hai_toko").
export function suggestName(raw) {
  const base = String(raw).split("/").pop().replace(/\.(sqlite3|sqlite|db)$/i, "");
  const clean = base.toLowerCase().replace(/[^a-z0-9_]/g, "_").replace(/^[^a-z]+/, "").slice(0, 32);
  return clean.length >= 2 ? clean : "";
}

const listStyle = {
  maxHeight: 232, overflowY: "auto", display: "flex", flexDirection: "column", gap: 6, margin: 0, padding: 0,
  listStyle: "none",
};

function Badge({ children, tone = "muted" }) {
  const colors = {
    muted: ["#94a3b8", "rgba(148,163,184,0.14)"],
    good: ["#5eead4", "rgba(94,234,212,0.14)"],
    warn: ["#fbbf24", "rgba(251,191,36,0.14)"],
  }[tone];
  return (
    <span style={{ fontSize: 10, padding: "2px 8px", borderRadius: 999, color: colors[0], background: colors[1], whiteSpace: "nowrap" }}>
      {children}
    </span>
  );
}

// Lists what can be connected from the chosen source, so nothing has to be typed from memory:
// the SQLite files of the data folder when no server is chosen, otherwise the databases of that server.
export default function SourceChoices({ source, selected, onPick }) {
  const [items, setItems] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const files = !source;

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      if (files) {
        setItems(((await listDataFiles()).files || []).map((f) => ({ name: f.name, allowed: true, connected: f.connected })));
      } else {
        setItems((await listProfileDatabases(source)).databases || []);
      }
    } catch (err) {
      setItems([]);
      setError(errorText(err));
    } finally {
      setLoading(false);
    }
  }, [source, files]);

  useEffect(() => { setItems(null); load(); }, [load]);

  return (
    <div aria-label="Pilihan sumber" style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span style={{ fontSize: 11, fontWeight: 600, color: "#b8c7d8" }}>
          {files ? "File SQLite di folder data" : `Database di ${source}`}
          {items ? ` (${items.length})` : ""}
        </span>
        <button type="button" className="at-button at-button-secondary" onClick={load} disabled={loading}
          aria-label="Muat ulang daftar" style={{ padding: "4px 8px" }}>
          {loading ? <LoaderCircle size={12} className="spin" /> : <RefreshCw size={12} />}
        </button>
      </div>
      {error && <small role="status" style={{ color: "#fbbf24" }}>{error}</small>}
      {items && items.length === 0 && !error && (
        <small style={{ color: "#94a3b8" }}>
          {files ? "Tidak ada file SQLite di folder data server." : "Tidak ada database yang bisa dilihat akun baca-saja server ini."}
        </small>
      )}
      {items && items.length > 0 && (
        <ul style={listStyle}>
          {items.map((item) => {
            const disabled = !item.allowed || item.connected;
            const chosen = selected === item.name;
            return (
              <li key={item.name}>
                <button
                  type="button" disabled={disabled} onClick={() => onPick(item)}
                  aria-label={`Pilih ${item.name}`} aria-pressed={chosen}
                  style={{
                    width: "100%", display: "flex", alignItems: "center", gap: 10, padding: "9px 12px", textAlign: "left",
                    borderRadius: 10, cursor: disabled ? "not-allowed" : "pointer", color: "#e2e8f0", font: "inherit",
                    fontSize: 13, opacity: disabled ? 0.55 : 1,
                    border: `1px solid ${chosen ? "#5eead4" : "rgba(148,163,184,0.22)"}`,
                    background: chosen ? "rgba(94,234,212,0.09)" : "rgba(15,23,42,0.55)",
                  }}
                >
                  {files ? <FileText size={14} color="#5eead4" /> : <Database size={14} color="#5eead4" />}
                  <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{item.name}</span>
                  {item.default && <Badge>bawaan profil</Badge>}
                  {item.connected && <Badge tone="good">terhubung</Badge>}
                  {!item.allowed && <Badge tone="warn">tidak diizinkan</Badge>}
                </button>
              </li>
            );
          })}
        </ul>
      )}
      {!files && items?.some((i) => !i.allowed) && (
        <small style={{ color: "#94a3b8" }}>
          Database berlabel "tidak diizinkan" bukan bawaan profil dan tidak berawalan database buatan agent; admin
          server bisa mengizinkannya lewat HAI_ALLOW_ANY_DATABASE_NAME.
        </small>
      )}
    </div>
  );
}
