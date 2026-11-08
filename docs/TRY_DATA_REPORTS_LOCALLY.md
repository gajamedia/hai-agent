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

Both kinds accept **conditions** beyond equality, on columns you ticked *boleh difilter*: `<`, `<=`, `>`, `>=`,
`!=`, between (dates and numbers, inclusive), in (up to 20 values), contains / starts with / ends with (text),
and empty / not empty. Up to 5, all combined with AND. Examples: *"tampilkan bahan yang harga belinya kurang dari
30 ribu"*, *"order bulan Maret 2026"*, *"nama mengandung gula"*. "30 ribu" is turned into 30000 and a month into
its first and last day. A condition on a column that is not filterable is refused with a reason, so tick
*boleh difilter* for the price/date columns you want to compare.

Not supported, and refused with a reason instead of being silently ignored: OR, several tables or joins,
calculations inside a condition, anything that changes data. Only approved columns can be shown, sorted by or
filtered on. The agent still never writes SQL: free-form SQL from an LLM is deliberately not accepted.

### Change what is approved, or disconnect

An active connection has an **Ubah persetujuan** button: the form opens with the current tables, columns and
"boleh difilter" ticks already set. Tick another table, add a column, or tick *boleh difilter* on a price or
date column, then **Setujui**. The connection stays active; the new selection replaces the old one. The
plug icon (**Putuskan**) removes the connection and what the agent learned about it; the database itself is
untouched. Several connections can exist at once; each question uses the one you pick.

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

### MySQL on another computer (for example 10.62.10.35)

Everything above works for a server on another computer. Only the profile and the account host change.
The setup tools connect to the `host` and `port` in the profile, so no extra option is needed for that.

**On the MySQL server (a DBA does this once):**

1. Let MySQL listen on the network: in `my.ini` / `my.cnf` set `bind-address = 0.0.0.0` (or the server's own
   IP) and restart MySQL. By default it listens on `127.0.0.1` only.
2. Allow port 3306 in the firewall for this computer only.
3. Have an administrator account that may connect **from this computer** (for example
   `'setup_admin'@'<IP of this computer>'`). `root@localhost` cannot be used from outside. Remove or lock this
   account after the setup; the tools use it once and never store its password.

**On this computer:**

1. Add the profile to `data\reports\profiles.json`, next to the others. The name must be different from the
   existing ones; the password variable can be empty for now:

   ```json
   "kantor_mysql": {
     "type": "mysql", "host": "10.62.10.35", "port": 3306, "database": "dbsales",
     "user": "hai_reader", "password_env": "HAI_MYSQL_KANTOR_PASSWORD", "ssl": "require"
   }
   ```

   and put `HAI_MYSQL_KANTOR_PASSWORD=` (empty) in `.env`.
2. Create the accounts. **`--hosts` is the important option**: a MySQL account belongs to the address the
   connection comes from, and the default (`localhost,127.0.0.1`) is useless from another computer. Give the
   IP of this computer as the MySQL server sees it:

   ```powershell
   python scripts\setup_mysql_readonly_user.py kantor_mysql --admin-user setup_admin --hosts "10.62.10.20"
   python scripts\setup_mysql_creator_user.py  kantor_mysql --admin-user setup_admin --hosts "10.62.10.20"
   python scripts\check_database_profile.py kantor_mysql
   ```

   (`setup_mysql_writer_user.py` takes the same options, for section 6.) If the check says
   `Access denied for 'hai_reader'@'<host>'`, that `<host>` is the address to put in `--hosts`; run the tool
   again.
3. Restart the backend. `kantor_mysql` now appears in **Sumber** (Data Reports) and in the **Server** list of
   *Buat database*, next to `local_mysql` and any SQL Server profile.

Good to know:

* The link is always encrypted, but `"ssl": "require"` does not check the server's certificate. Outside a
  trusted LAN use `"ssl": "verify"`, which needs a certificate this computer trusts.
* The rules are the same as on this computer: the reader holds only `SELECT` on its one database, the creator
  only `CREATE` on `hai\_%` databases (enforced by MySQL itself), and the writer is a separate account.
* A server that cannot be reached shows a generic message in the page; the real reason (timeout, refused,
  access denied) is in the backend log and in the output of `check_database_profile.py`.
* Not yet automated: unlike SQL Server (`connect_sqlserver.py`), nothing writes the MySQL profile for you.

## 6. Changing data (insert, update, delete)

Reports only read. Changing data is a separate, deliberately heavier path ("Ubah data" in the sidebar). The
agent still never writes SQL: it picks a structured change, our code builds the parameterized statement, and
**nothing runs until you have seen exactly what it would do**.

Protections, in the order you meet them:

| Layer | What it means |
|---|---|
| Own roles | `data-writer` may plan and run; `data-change-approver` may approve. Report users and connection managers can do neither. |
| Own database account | A second account (`writer` in the profile) with only SELECT, INSERT, UPDATE, DELETE on one database. The server refuses a writer that can DROP, ALTER, CREATE, GRANT or reach other databases. The read-only account used by reports is never used for writing. |
| Per-table approval | In **Ubah persetujuan** tick *boleh diubah* on the columns that may be written and the operations (Tambah / Ubah / Hapus) per table. The key column can never be written. UPDATE and DELETE select rows with the *boleh difilter* columns. |
| Your row limit | You type how many rows the change may touch (default 100, server ceiling `HAI_DATA_CHANGE_MAX_ROWS`, default 10000). More rows than that and the plan is refused before anything runs. |
| Preview | The number of rows, a sample before and after, and the exact confirmation phrase, for example `UBAH 12 BARIS tb_bahan`. UPDATE and DELETE without a condition are impossible. |
| Fresh login | When you press **Jalankan**, Keycloak asks you to sign in again (the page remembers the plan and shows it again afterwards; you press Jalankan once more). The backend checks the token's `auth_time` is at most `HAI_STEPUP_MAX_AGE_SECONDS` old (default 120). This application never sees your password, and it works with MFA. |
| Second person | `HAI_DATA_CHANGE_APPROVAL=delete` (default) makes every DELETE wait for someone *else* with `data-change-approver` (who must also sign in again); `all` for every change; `none` for no approval. An unknown value is treated as `delete`. You can never approve your own change. |
| All or nothing | One transaction: the affected rows are locked and compared with what you were shown; if the data changed meanwhile, or the statement touches a different number of rows, everything is rolled back. |
| Audit | `<work dir>/audit/changes-<connection>.jsonl` records who planned, who approved, who ran, the parameterized SQL and, for UPDATE and DELETE, the old values of every row. Treat it as business data. |

### Set it up (MySQL)

1. Reports first: follow section 5 so the read-only account works.
2. Create the writer account (asks for the MySQL administrator password, never stores it):

   ```powershell
   python scripts\setup_mysql_writer_user.py local_mysql --tables tb_bahan
   ```

   Without `--tables` the writer may change every table of the database. It writes the new password into `.env`
   and a `"writer": {"user": "hai_writer", "password_env": "..."}` block into `profiles.json`.
3. Give people the roles (sign out and in afterwards):

   ```powershell
   python scripts\keycloak_data_report_roles.py --user hai-admin --roles data-writer,data-change-approver
   ```

   Use two different people in real life: a change needs a second person for DELETE.
4. Restart the backend. In Data Reports open **Ubah persetujuan** for the connection, tick the tables, the
   *boleh diubah* columns and the operations, **Setujui**. Then open **Ubah data**.

Notes: inserting needs a table whose key is generated by the database (AUTO_INCREMENT), because the key column
cannot be set by the agent. Only MySQL can change data for now; PostgreSQL and SQLite stay read-only. If your
Keycloak token has no `auth_time` claim, every change is refused with "sign in again" (it fails closed).

> **Order matters for databases the agent creates.** `setup_mysql_creator_user.py` gives the writer account access
> to the `hai_%` databases only if the profile already has a writer. If you add the writer afterwards (section 6),
> run `python scripts\setup_mysql_creator_user.py <profile>` once more (it is safe to repeat); otherwise approving
> "boleh diubah" on a `hai_...` database fails with "the writer account has no access to this database".
> `setup_mysql_writer_user.py --tables ...` only covers the profile's own database (for example `dbsales`).

## 7. Creating a new database

"Buat database" in the sidebar lets the agent create a database that does not exist yet, with its tables. As
everywhere, the agent never writes SQL: it picks a structured design (a name, tables, columns of a closed set
of types: integer, decimal, text, long_text, date, datetime, boolean; an `id` key that the database numbers
itself is added to every table), you review it, and our code builds the `CREATE` statements.

| Protection | What it means |
|---|---|
| Own role | `data-schema-creator`. Nothing else grants it, and it grants nothing else. |
| Own account | A third account that holds only `CREATE` on the pattern `hai\_%`. It cannot DROP, ALTER, read, or touch an existing database such as `dbsales`. The server refuses an account that can do more. |
| Name prefix | Only names starting with the profile's prefix (default `hai_`) can be created or even connected this way. |
| Never overwrite | If the database does not exist it is created. If it already exists (made earlier, name starting with the prefix), only **new tables are added** to it, with the phrase `TAMBAH TABEL hai_xxx`; existing tables and data are never touched, and a table name that already exists is refused. Both are re-checked at run time. There is no drop. |
| Preview, phrase, fresh login | The tables are shown first; you type `BUAT DATABASE hai_xxx`; Keycloak asks you to sign in again (same as changing data). No second approver is needed because nothing existing can be harmed. |
| Audit | `<work dir>/audit/creations-<profile>.jsonl` records who created what and the statements. |

Set it up (after sections 5 and 6):

```powershell
python scripts\setup_mysql_creator_user.py local_mysql
python scripts\keycloak_data_report_roles.py --user hai-admin --roles data-schema-creator
```

The first command asks for the MySQL administrator password (never stored), creates the creator account, lets
your read-only and writer accounts use `hai_%` databases, and writes the creator block into `profiles.json`.
Restart the backend, sign in again, open **Buat database**, then **Hubungkan database ini** after it is created.
A database created this way is then approved for reading (and, if you wish, writing) in **Ubah persetujuan**
like any other. Only MySQL, and only the server profiles that have a creator account, can do this for now.

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

## 8. Find the databases installed on this computer

```powershell
python scripts\discover_databases.py                 # look and report only
python scripts\discover_databases.py --add           # also register new MySQL / PostgreSQL servers
python scripts\discover_databases.py --sqlite-dir D:\data --host 10.0.0.5
```

It probes only `127.0.0.1` and the addresses you pass with `--host` (never a network range), recognises
MySQL/MariaDB, PostgreSQL and SQL Server by their first bytes on the standard ports, and lists Windows
database services and SQLite files. It sends no login and no query.

* `--add` writes a profile into `profiles.json` (host, port, database name, account names, the name of the
  password variable). It never asks for or writes a password. The next step it prints creates the read-only
  account (`setup_mysql_readonly_user.py` asks for the administrator password once and does not keep it).
* SQL Server is only reported: the agent has no SQL Server engine yet.
* A SQLite file can be used only inside `HAI_DATABASE_ROOT`; the report says which ones are.
* A server on a non-standard port is not found; add it by hand as in section 5.

## 9. SQL Server (read-only reports)

```powershell
pip install pyodbc                       # or: pip install -e .[sqlserver]  (and Microsoft ODBC Driver 17 or 18)
python scripts\connect_sqlserver.py      # diagnose, changes nothing
python scripts\connect_sqlserver.py --create-reader --database <name>
python scripts\check_database_profile.py local_sqlserver
```

The diagnosis lists the installed instances (service running or not, TCP/IP on or off, accepted login types),
checks the ODBC driver and pyodbc, and connects with your own Windows login to show the version, whether you
administer the instance and the user databases. SQL Server Express usually has TCP/IP off: it is then reachable
only from this computer, by the server name `(local)\SQLEXPRESS` (shared memory). The script never turns TCP/IP
on and never changes server settings.

`--create-reader` creates a SQL login with a random password that can only read the one database you name
(writes and schema changes are denied), stores the password in `.env`, adds the profile (no password) to
`profiles.json`, then checks that a write is refused. It needs mixed-mode authentication and you being an
administrator; otherwise it stops and says what to change. Restart the backend: the profile
`local_sqlserver` then appears in Data Reports like a MySQL or PostgreSQL profile.

A SQL Server profile in `profiles.json` looks like this (the password is never in the file):

```json
"local_sqlserver": {"type": "sqlserver", "server": "(local)\SQLEXPRESS", "database": "dbsales_sql",
                    "user": "hai_sql_reader", "password_env": "HAI_LOCAL_SQLSERVER_PASSWORD", "ssl": "require"}
```

For another computer use `"server": "host.example.com", "port": 1433` (TCP/IP must be enabled there) and
`"ssl": "verify"` when its certificate is trusted.

How it stays read-only: SQL Server has no read-only session flag, so the engine refuses a login that is an
administrator or database owner, that holds anything beyond CONNECT and SELECT (or server-wide rights), or that
can INSERT, UPDATE, DELETE or ALTER any table. Reports use only SELECT statements the agent builds itself.

### SQL Server on another computer (for example 10.62.10.34)

Prepare on the SQL Server machine (a DBA does this once): enable TCP/IP in *SQL Server Configuration Manager*
(Network Configuration > Protocols > TCP/IP, set a fixed port such as 1433 under IPAll), restart the service, and
allow that port in the firewall for this computer only. The server must accept SQL logins (mixed mode).

Then, from this computer, with an administrator SQL login such as `sa` (its password is asked in the terminal,
never shown, never stored):

```powershell
python scripts\discover_databases.py --host 10.62.10.34          # does it answer on 1433?
python scripts\connect_sqlserver.py --server "10.62.10.34,1433" --admin-user sa --create-reader --database <db>
python scripts\connect_sqlserver.py --server "10.62.10.34,1433" --admin-user sa --create-creator
python scripts\check_database_profile.py local_sqlserver
```

The profile then holds `"server": "10.62.10.34", "port": 1433` (use `--profile name` for a second server, so it
does not replace `local_sqlserver`). Connections are encrypted; the certificate is not verified unless you set
`"ssl": "verify"` in the profile, which needs a certificate this computer trusts (do that for anything beyond a
trusted LAN). If you would rather not give this tool an administrator login, the DBA can create the read-only login
(`db_datareader` and `db_denydatawriter` on one database) and give you its name; put its password in `.env` and
write the profile by hand as shown above. A named instance is written `"server": "10.62.10.34\SQL2019"` (the SQL
Browser service must be running) or, better, with its fixed port.

### Creating databases and tables on SQL Server (the "Buat database" page)

```powershell
python scripts\connect_sqlserver.py --create-creator          # after --create-reader made the profile
```

This creates a login that holds ONE server permission, `CREATE ANY DATABASE` (never `sysadmin`, never the
`dbcreator` role, which also allows altering other databases), stores its password in `.env` and adds a
`creator` block to the profile. Restart the backend; the page lists the server as `local_sqlserver (sqlserver)`
and works exactly like MySQL: a plan to review, the phrase `BUAT DATABASE hai_xxx` (or `TAMBAH TABEL hai_xxx`
for tables in a database the agent made earlier), a fresh Keycloak login, an audit line.

Differences from MySQL you should know:

* SQL Server cannot limit `CREATE ANY DATABASE` to names starting with a prefix. The application enforces the
  prefix (default `hai_`); a name outside it is refused before anything runs. The login owns only the databases
  it creates and cannot open, alter or drop any other one (checked by the tests).
* A new database automatically gives the profile's read-only login read access (`db_datareader`, nothing more),
  so **Hubungkan database ini** works at once. The tables get an `IDENTITY` numeric key `id`.
* Columns: `text` = `NVARCHAR(255)`, `long_text` = `NVARCHAR(MAX)`, `decimal` = `DECIMAL(14,2)`,
  `datetime` = `DATETIME2`, `boolean` = `BIT`. Up to 500 columns per table; the 65,000-byte MySQL row cap does
  not apply.
* Adding tables to a database the agent did not make is refused (the creator has no access to it).

Not available for SQL Server yet: changing data (insert, update, delete) from the agent (MySQL only).
