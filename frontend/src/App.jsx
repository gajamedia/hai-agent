import { Navigate, Route, Routes } from "react-router-dom";
import Dashboard from "./pages/Dashboard.jsx";
import AutonomousTesting from "./pages/AutonomousTesting.jsx";
import DataChanges from "./pages/DataChanges.jsx";
import DatabaseCreation from "./pages/DatabaseCreation.jsx";
import DataReports from "./pages/DataReports.jsx";
import DatabaseAgent from "./pages/DatabaseAgent.jsx";
import Tasks from "./pages/Tasks.jsx";
import { AuthProvider, useAuth } from "./auth/AuthProvider.jsx";

function AuthenticatedApplication() {
  const { status, user, error, login, logout } = useAuth();

  if (status === "initializing") {
    return (
      <main style={screenStyle}>
        <section style={panelStyle}>
          <div className="spin" aria-hidden="true" />
          <h1>Connecting to Keycloak</h1>
          <p>Menyiapkan sesi autentikasi sebelum mengakses hAI Agent.</p>
        </section>
      </main>
    );
  }

  if (status === "error") {
    return (
      <main style={screenStyle}>
        <section style={panelStyle} role="alert">
          <h1>Autentikasi gagal</h1>
          <p>{error}</p>
          <button type="button" className="primary" onClick={login}>
            Coba login kembali
          </button>
        </section>
      </main>
    );
  }

  return (
    <>
      <div style={authBarStyle}>
        <span>
          Signed in as <strong>{user?.username || "User"}</strong>
          {user?.email ? ` · ${user.email}` : ""}
        </span>
        <button type="button" className="ghost" onClick={logout}>
          Logout
        </button>
      </div>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/tasks" element={<Tasks />} />
        <Route path="/autonomous-testing" element={<AutonomousTesting />} />
        <Route path="/database-agent" element={<DatabaseAgent />}>
          <Route index element={<Navigate to="reports" replace />} />
          <Route path="reports" element={<DataReports />} />
          <Route path="changes" element={<DataChanges />} />
          <Route path="create" element={<DatabaseCreation />} />
        </Route>
        {/* Old addresses keep working (bookmarks). */}
        <Route path="/data-reports" element={<Navigate to="/database-agent/reports" replace />} />
        <Route path="/data-changes" element={<Navigate to="/database-agent/changes" replace />} />
        <Route path="/database-creation" element={<Navigate to="/database-agent/create" replace />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <AuthenticatedApplication />
    </AuthProvider>
  );
}

const screenStyle = {
  minHeight: "100vh",
  display: "grid",
  placeItems: "center",
  padding: "24px",
  background: "#0b1220",
  color: "#e5e7eb",
};

const panelStyle = {
  width: "min(100%, 440px)",
  padding: "28px",
  border: "1px solid rgba(148, 163, 184, 0.25)",
  borderRadius: "16px",
  background: "#111827",
};

const authBarStyle = {
  position: "relative",
  zIndex: 20,
  display: "flex",
  alignItems: "center",
  justifyContent: "flex-end",
  gap: "16px",
  padding: "8px 20px",
  color: "var(--text, #e5e7eb)",
  background: "var(--surface, #111827)",
  borderBottom: "1px solid rgba(148, 163, 184, 0.2)",
  fontSize: "12px",
};
