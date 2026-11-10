import { ArrowLeft } from "lucide-react";
import { Link, NavLink, Outlet } from "react-router-dom";
import "./autonomous-testing.css";

// One menu for everything that touches databases: connecting, creating databases, recording, changing and viewing data.
export const DATABASE_AGENT_TABS = [
  { to: "/database-agent/create", label: "Buat database" },
  { to: "/database-agent/connections", label: "Koneksi" },
  { to: "/database-agent/entry", label: "Catat data" },
  { to: "/database-agent/changes", label: "Ubah data" },
  { to: "/database-agent/view", label: "Lihat data" },
];

const barStyle = { display: "flex", gap: "8px", flexWrap: "wrap", margin: "0 0 4px" };
const tabStyle = (isActive) => ({
  padding: "6px 14px",
  borderRadius: "999px",
  fontSize: "12px",
  textDecoration: "none",
  color: isActive ? "#0b1220" : "#5eead4",
  background: isActive ? "#5eead4" : "transparent",
  border: "1px solid #5eead4",
});

export default function DatabaseAgent() {
  return (
    <div style={{ background: "#0b1220", minHeight: "100vh" }}>
      <div style={{ padding: "24px clamp(18px,3vw,48px) 0", fontFamily: "'DM Sans',sans-serif" }}>
        <Link className="at-back" to="/"><ArrowLeft size={14} /> Kembali</Link>
        <nav style={barStyle} aria-label="Database Agent">
          {DATABASE_AGENT_TABS.map((tab) => (
            <NavLink key={tab.to} to={tab.to} style={({ isActive }) => tabStyle(isActive)}>
              {tab.label}
            </NavLink>
          ))}
        </nav>
      </div>
      <Outlet />
    </div>
  );
}
