# Try Data Reports locally (Windows, PowerShell)

Runs the dashboard against your local Keycloak, a sample SQLite database and the real
agent. Everything stays on your machine; the database is opened read-only.

This exact flow (real Keycloak tokens, real backend, real Claude) is what was verified
end to end; the steps below are the same ones, with your own Keycloak.

## 0. One-time: give your user the data-report roles

The dev realm keeps `hai.dev` read-only on purpose, so grant the roles explicitly. Use the
Keycloak **admin** account (the one from `keycloak/.env.keycloak`):

```powershell
cd C:\Projects\Python\AI\new-hAI-agent
$env:KC_ADMIN_PASSWORD = "<Keycloak admin password>"
.\venv\Scripts\python.exe scripts\keycloak_data_report_roles.py --user hai.dev
```

It grants `data-reports-user` and `database-connection-manager` (creating the roles if your
realm predates them). `database-engine-reviewer` is only granted if you ask for it with
`--roles database-engine-reviewer`. Sign out and in again afterwards so your token has the roles.

**If every request answers 401** even after signing in, your realm's tokens have no `sub`
claim (Keycloak 25+ puts it in the `basic` client scope). In the admin console:
*Clients -> hai-agent-web -> Client scopes -> Add client scope -> basic -> Default*, then
sign in again. The repository's realm file already includes `basic`.

## 1. Create the sample database

```powershell
.\venv\Scripts\python.exe scripts\make_sample_sales_db.py data\reports
```

Creates `data\reports\sales.sqlite3` (120 synthetic orders; no personal data).

## 2. Start the backend

```powershell
$env:HAI_DATABASE_ROOT = "$PWD\data\reports"   # the only folder users can pick databases from
$env:HAI_LLM_PROVIDER  = "auto"                # OpenCode first, Claude Code CLI as the fallback
.\venv\Scripts\python.exe -m uvicorn api.main:app --env-file .env --port 8000
```

`.env` supplies your Keycloak settings (`HAI_AUTH_ENABLED=true`, ...). Data Reports refuses to
start without authentication enabled.

## 3. Start the frontend

```powershell
cd frontend
npm install        # first time only
npm run dev
```

Open http://localhost:5173, sign in, and choose **Data Reports** in the sidebar.

## 4. Walk through it

1. **Koneksi**: name `sales`, file `sales.sqlite3` -> *Hubungkan*.
2. **Setujui**: tick the `sales` table and its columns; tick *boleh difilter* for `region`
   and `product`. Leave out any column you would not want the agent to see -> *Setujui*.
3. **Tanya laporan**: pick *Hasil* (Excel / PDF / ...), ask for example
   *"Total penjualan per wilayah untuk produk widget"* -> *Buat rencana*.
   The plan is shown first and **no data has been read yet**.
4. *Jalankan* -> the table appears, then *Unduh XLSX* / *Unduh PDF*.
5. Open the downloaded file in Excel or Acrobat. Check that the number columns are right-aligned
   (real numbers), the header is bold, and there is no "file is corrupt" message.

## Troubleshooting

| You see | Meaning | Fix |
|---|---|---|
| 401 on everything | Token rejected (often no `sub` claim, or expired) | See step 0 note; sign out and in |
| 403 `permission_denied` / `authorization_route_unmapped` | Missing role | Run the step 0 script, then sign in again |
| 503 `database_connections_unavailable` | Backend started without `HAI_DATABASE_ROOT` | Set it (step 2) and restart |
| 422 "database file not found" | File is not inside `HAI_DATABASE_ROOT` | Put it in that folder; use only its name |
| Report fails `no_matching_rows` | The filter value matched nothing (case matters) | Rephrase; the agent learns the right spelling after one success |
| 503 `table_report_failed` | No LLM available | Check `OPENCODE_*` in `.env`, or install/log in to the Claude Code CLI |

Stop everything with `Ctrl+C` in each terminal.
