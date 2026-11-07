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

Do not skip this step: the backend refuses to start if `HAI_DATABASE_ROOT` is not an existing folder,
and the page answers "database file not found" if `sales.sqlite3` is not inside it.

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

### What you can ask

The agent never writes SQL. It picks one of two plan kinds from the tables and columns you approved, and
our own code builds the (parameterized, read-only) query from that plan. You always see the plan first.

| You say | Plan kind | What you get |
|---|---|---|
| "total penjualan per wilayah", "jumlah order per produk" | **aggregate** | a summary: totals, counts, averages per group |
| "tampilkan data penjualan", "tampilkan 10 penjualan terbesar", "daftar order di wilayah west urut dari id terkecil", "show the first 20 widget sales with region and amount" | **list** | the rows themselves: the columns you ask for (or all), equality filters, up to 3 sort keys, and a row limit |

For a list: 100 rows unless you ask for another number, at most 1000. If more rows exist than the limit, the
page and the file say so ("Dibatasi N baris pertama"). The page shows the first 200 rows; the downloaded
Excel, PDF, CSV or HTML file holds all of them. Columns appear in the order you asked for.

Not supported yet, and refused with a reason instead of being silently ignored: comparisons and ranges
("di atas 100", "bulan Maret"), OR, "contains", several tables or joins, anything that changes data.
Only columns you approved can be shown, sorted by or filtered on.

## 5. Use your own MySQL instead of the sample file

MySQL is chosen **by profile name**, never by typing a host or password into the page: those live on the
server. The page only shows a **Sumber** list once at least one profile exists.

### Quick way (recommended): one command creates the account and fills in the files

1. Create `data\reports\profiles.json` as in step **c** below, with your database name. The `user` field may
   say anything (even `root`); the tool replaces it.
2. Make sure `.env` has `HAI_DATABASE_PROFILES=...` (step **d**). Leave `HAI_MYSQL_LOCAL_PASSWORD=` empty.
3. Run:

   ```powershell
   python scripts\setup_mysql_readonly_user.py local_mysql
   ```

   It shows the plan and asks for a yes, then asks for the password of a MySQL administrator (`root` by
   default; typed hidden, never stored or printed). It creates a read-only account `hai_reader` that can
   only `SELECT` from your one database, gives it a random password, writes that password into `.env`,
   sets `"user": "hai_reader"` in `profiles.json`, and runs the same check as `check_database_profile.py`.
   Your other databases stay out of its reach. Running it again just resets the password.
4. Restart the backend, reload the page, choose **Sumber -> local_mysql**, **Hubungkan**.

If the final check says `Access denied for 'hai_reader'@'<host>'`, run the tool again with that host,
for example `--hosts localhost,127.0.0.1,::1`. A different administrator: `--admin-user <name>`.

The manual way follows, if you prefer to run the SQL yourself.

**a. The driver** is installed with the project (`pip install ".[mysql]"` if it is missing).

**b. Create a read-only account.** In MySQL Shell or Workbench, as an administrator:

```sql
CREATE USER 'hai_reader'@'127.0.0.1' IDENTIFIED BY '<a password>';
CREATE USER 'hai_reader'@'localhost' IDENTIFIED BY '<the same password>';
GRANT SELECT ON your_database.* TO 'hai_reader'@'127.0.0.1';
GRANT SELECT ON your_database.* TO 'hai_reader'@'localhost';
```

Rules the server enforces (it refuses the connection otherwise): the account may hold **only SELECT**
(`root` and any account with INSERT, UPDATE, DELETE, CREATE, ... are refused), privileges must be granted
to the account directly (not through a role), and the link is always encrypted (MySQL 8 offers TLS by
default; a server without TLS cannot be used).

**c. Describe the connection** in `data\reports\profiles.json` (git-ignored). It holds no password, only
the *name* of the variable that does:

```json
{
  "local_mysql": {
    "type": "mysql",
    "host": "127.0.0.1",
    "port": 3306,
    "database": "your_database",
    "user": "hai_reader",
    "password_env": "HAI_MYSQL_LOCAL_PASSWORD",
    "ssl": "require"
  }
}
```

**d. Add two lines to your `.env`:**

```
HAI_DATABASE_PROFILES=C:\Projects\Python\AI\new-hAI-agent\data\reports\profiles.json
HAI_MYSQL_LOCAL_PASSWORD=<the password of hai_reader>
```

**e. Check it before opening the browser.** This reads your `.env` the way the server does and, unlike
the web page, prints the real reason when something is wrong (and never the password):

```powershell
python scripts\check_database_profile.py local_mysql
```

You want `Result: this profile will work in the web app.` Common failures it explains:

| It says | Fix |
|---|---|
| `Access denied for user 'hai_reader'@'<host>'` | The account exists for a different host than the one named in the message. Create it for that host (list accounts with `SELECT user, host FROM mysql.user;`). |
| `refused by the safety rules ... more than SELECT` | Use a dedicated account that has only `GRANT SELECT`, not `root`. |
| `environment variable ... is not set` | Add `HAI_MYSQL_LOCAL_PASSWORD=...` to `.env`. |
| `not reachable` | MySQL is not running, or the host/port in the profile is wrong. |
| `TLS handshake failed` | The server does not offer encryption; enable TLS on the server. |
| `database driver is not installed` | `pip install ".[mysql]"` |

**f. Restart the backend** (it reads `.env` at start), reload the page, open **Koneksi**, choose
**Sumber -> local_mysql (mysql)**, give the connection a name, and press **Hubungkan**. The rest is the
same as for the sample file: approve tables and columns, ask, review the plan, run, download.

Tables or columns whose names contain characters other than letters, digits and underscores (a space or a
hyphen) are hidden, and the page tells you how many. Reserved words such as `order` or `group` work.

## Troubleshooting

| You see | Meaning | Fix |
|---|---|---|
| 401 on everything | Token rejected (often no `sub` claim, or expired) | See step 0 note; sign out and in |
| 403 `permission_denied` / `authorization_route_unmapped` | Missing role | Run the step 0 script, then sign in again |
| 503 `database_connections_unavailable` | Backend started without `HAI_DATABASE_ROOT` | Set it (step 2) and restart |
| 422 "database file not found" | File is not inside `HAI_DATABASE_ROOT` | Put it in that folder; use only its name |
| Report fails `no_matching_rows` | The filter value matched nothing (case matters) | Rephrase; the agent learns the right spelling after one success |
| "Agent sedang menyusun rencana..." for minutes | The first LLM provider (OpenCode with `HAI_LLM_PROVIDER=auto`) is stuck; the page waits for its whole `OPENCODE_TIMEOUT` before trying Claude | Run `python scripts\check_llm_provider.py`: it times each provider and says which one is slow. Then set `HAI_LLM_PROVIDER=claude-code` (or lower `OPENCODE_TIMEOUT`) in `.env` and restart the backend. The **Batalkan** button stops the wait. |
| 503 `table_report_failed` | No LLM available | Run `python scripts\check_llm_provider.py`; check `OPENCODE_*` in `.env`, or install/log in to the Claude Code CLI |

Stop everything with `Ctrl+C` in each terminal.

## Which LLM answers (OpenCode, Claude Code)

Planning a report sends the question and the approved table catalog (never rows) to an LLM.

- `HAI_LLM_PROVIDER=auto` (default): OpenCode first, Claude Code only if OpenCode fails. A provider that
  *hangs* only fails when its timeout ends, so the first provider is capped at `HAI_LLM_FAILOVER_SECONDS`
  (default 60, never more than `OPENCODE_TIMEOUT`). Worst case before the second answers: about one minute.
  A provider that just failed (an account that hit its usage limit, a stalled model) is then skipped for
  `HAI_LLM_PROVIDER_COOLDOWN_SECONDS` (default 300), so only the first request pays that minute; it is
  still tried as a last resort if the other provider also fails, and one success clears it.
- **If your OpenCode account is rate limited**, do not rely on `auto`: it still spends a minute per cooldown
  window and keeps sending requests to a limited account. Set `HAI_LLM_PROVIDER=claude-code`.
- `HAI_LLM_PROVIDER=claude-code` or `opencode`: use one provider only.
- If OpenCode never answers, check its model: with `OPENCODE_MODEL` empty it uses its built-in free model,
  which can stall. List the models you can use with `opencode models`, put one in `OPENCODE_MODEL`
  (for example `opencode/<name>`), and test with `python scripts\check_llm_provider.py --provider opencode`.
  Some models are paid; choose deliberately. Claude Code answers in roughly 5 seconds when it is logged in.
- The agent command line tools are started without this server's secrets in their environment (no
  `HAI_*`, `KC_*`, `*_PASSWORD`, `*_SECRET` variables), with no keyboard attached, and are killed together
  with everything they started when the timeout ends.
