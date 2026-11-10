# Manual Instalasi hAI Agent di Komputer Lain (Windows)

Panduan ini memasang hAI Agent dari nol di komputer baru sampai menu **Database Agent** (tab **Laporan**,
**Ubah data**, **Buat database**) bisa dipakai. Perintah memakai **PowerShell**. Semua jalan di satu komputer (localhost).

> Untuk panduan memakai fitur (tanya laporan, ubah data, MySQL jarak jauh), lihat
> [TRY_DATA_REPORTS_LOCALLY.md](TRY_DATA_REPORTS_LOCALLY.md). Dokumen ini fokus pada **pemasangan**.

## Daftar isi

1. [Gambaran komponen](#1-gambaran-komponen)
2. [Prasyarat](#2-prasyarat)
3. [Ambil kode](#3-ambil-kode)
4. [Backend Python](#4-backend-python)
5. [Keycloak (login)](#5-keycloak-login)
6. [File `.env` backend](#6-file-env-backend)
7. [Provider LLM](#7-provider-llm-opencode-atau-claude-code)
8. [Database contoh](#8-database-contoh)
9. [Menjalankan backend](#9-menjalankan-backend)
10. [Frontend](#10-frontend)
11. [Beri hak akses (role) ke user](#11-beri-hak-akses-role-ke-user)
12. [Menghubungkan database sungguhan](#12-menghubungkan-database-sungguhan-opsional)
13. [Verifikasi pemasangan](#13-verifikasi-pemasangan)
14. [Pemecahan masalah](#14-pemecahan-masalah)
15. [Akses dari komputer lain di jaringan / produksi](#15-akses-dari-komputer-lain-di-jaringan--produksi)
16. [Menjalankan sehari-hari & backup](#16-menjalankan-sehari-hari--backup)

---

## 1. Gambaran komponen

```text
Browser ──► Frontend React (Vite, :5173)
              │  login (PKCE)                    ┌──────────────────────┐
              ├────────────────────────────────► │ Keycloak (Docker,    │
              │                                  │ :8080) + PostgreSQL  │
              │  Bearer token                    └──────────────────────┘
              ▼                                           ▲ verifikasi JWT (JWKS)
        Backend FastAPI (Uvicorn, :8000) ─────────────────┘
              ├── LLM: OpenCode CLI  atau  Claude Code CLI  (menyusun rencana laporan)
              └── Database target: SQLite / MySQL / PostgreSQL / SQL Server
```

Empat hal yang harus ada: **backend**, **frontend**, **Keycloak** (autentikasi wajib untuk Data Reports),
dan **satu provider LLM** (OpenCode atau Claude Code). Database target bersifat opsional untuk uji coba
karena ada database contoh SQLite.

## 2. Prasyarat

| Perangkat lunak | Versi | Untuk apa | Cek |
|---|---|---|---|
| Windows | 10/11 | Panduan ini ditulis untuk Windows | |
| Python | **3.10.11 atau lebih baru** | Backend | `python --version` |
| Node.js + npm | Node 22 (disarankan; sama dengan CI) | Frontend | `node --version` |
| Git | terbaru | Mengambil kode | `git --version` |
| Docker Desktop | terbaru | Keycloak + PostgreSQL-nya | `docker --version` |
| OpenCode CLI **atau** Claude Code CLI | terbaru | Provider LLM | lihat [bagian 7](#7-provider-llm-opencode-atau-claude-code) |

Hanya jika memakai database tertentu:

| Database | Tambahan |
|---|---|
| MySQL | Server MySQL 8 dengan TLS aktif |
| PostgreSQL | Server PostgreSQL |
| SQL Server | **ODBC Driver 17/18 for SQL Server** (Microsoft) dan instance yang mengizinkan SQL login (mixed mode) |

Instal cepat dengan winget (opsional):

```powershell
winget install Python.Python.3.10
winget install OpenJS.NodeJS.LTS
winget install Git.Git
winget install Docker.DockerDesktop
```

Setelah memasang Docker Desktop, **jalankan aplikasinya** dan tunggu sampai statusnya *Running*
(`docker info` tidak boleh error). Tutup dan buka lagi PowerShell setelah memasang apa pun agar `PATH` terbaca.

Jika PowerShell menolak menjalankan skrip aktivasi venv:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## 3. Ambil kode

```powershell
cd C:\Projects
git clone https://github.com/gajamedia/hai-agent.git new-hAI-agent
cd new-hAI-agent
```

(Repositori privat: login GitHub dulu, atau salin foldernya lewat ZIP/flashdisk **tanpa** `venv`,
`node_modules`, dan `.env`.) Branch fitur terbaru saat ini `feat/dynamic-database-reports`:

```powershell
git checkout feat/dynamic-database-reports
```

> Path di panduan ini `C:\Projects\new-hAI-agent`. Jika Anda memilih folder lain, ganti di semua perintah.
> Hindari path dengan spasi.

## 4. Backend Python

Dari root project:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[mysql,postgresql,sqlserver]"
```

`pyproject.toml` memasang FastAPI, Uvicorn, PyJWT, httpx. Bagian dalam kurung siku adalah driver database
opsional; **pasang hanya yang dipakai** (mis. `".[mysql]"`). `sqlserver` memerlukan ODBC Driver terpasang.

Untuk menjalankan pengujian (opsional):

```powershell
python -m pip install pytest openpyxl pypdf
python -m pytest -q
```

## 5. Keycloak (login)

Data Reports menolak start jika autentikasi mati, jadi Keycloak wajib. Repositori sudah membawa
`docker-compose.keycloak.yml` (Keycloak 26.8.1 + PostgreSQL 17) dan realm siap impor
`keycloak/realm/hai-agent-realm.json` (client `hai-agent-web` dan `hai-agent-api`, semua role).

### 5.1 Buat file rahasia Keycloak

File ini **tidak ada di Git** (sengaja). Buat `keycloak\.env.keycloak`:

```powershell
@"
KC_DB_PASSWORD=<password-acak-untuk-database-keycloak>
KC_BOOTSTRAP_ADMIN_USERNAME=admin
KC_BOOTSTRAP_ADMIN_PASSWORD=<password-admin-keycloak>
HAI_KEYCLOAK_DEV_USER_PASSWORD=<password-sementara-user-hai.dev>
"@ | Set-Content -Encoding ascii keycloak\.env.keycloak
```

Ganti semua `<...>` dengan nilai Anda sendiri (jangan memakai tanda kutip; hindari karakter `$`).
Password `hai.dev` bersifat **sementara**: Keycloak memaksa menggantinya saat login pertama.

### 5.2 Jalankan Keycloak

```powershell
docker compose --env-file keycloak\.env.keycloak -f docker-compose.keycloak.yml up -d
docker compose -f docker-compose.keycloak.yml ps
```

Tunggu 30–90 detik pada start pertama (mengunduh image, mengimpor realm). Cek:

```powershell
powershell -File scripts\Check-Keycloak.ps1
```

atau buka <http://localhost:8080> (konsol admin: login dengan `KC_BOOTSTRAP_ADMIN_*`).
Realm `hai-agent` harus ada, dengan user `hai.dev`.

> Port 8080 hanya terikat ke `127.0.0.1` (komputer ini saja). Untuk akses dari komputer lain lihat
> [bagian 15](#15-akses-dari-komputer-lain-di-jaringan--produksi).

### 5.3 Buat user Anda (disarankan)

Selain `hai.dev`, buat user sendiri di konsol admin: *Realm hai-agent → Users → Add user*, lalu
*Credentials → Set password* (matikan *Temporary* jika tidak ingin dipaksa ganti). Role diberikan di
[bagian 11](#11-beri-hak-akses-role-ke-user).

## 6. File `.env` backend

Buat `.env` di **root project** (sudah masuk `.gitignore`; jangan di-commit). Isi minimal:

```dotenv
# --- Autentikasi (wajib untuk Data Reports) ---
HAI_AUTH_ENABLED=true
HAI_AUTH_PROVIDER=keycloak
HAI_KEYCLOAK_BASE_URL=http://localhost:8080
HAI_KEYCLOAK_REALM=hai-agent
HAI_KEYCLOAK_API_CLIENT_ID=hai-agent-api
HAI_KEYCLOAK_ALLOW_INSECURE_HTTP=true      # HTTP hanya untuk loopback/lokal

HAI_AUTH_ALGORITHMS=RS256
HAI_AUTH_LEEWAY_SECONDS=0
HAI_AUTH_JWKS_CACHE_TTL_SECONDS=300
HAI_AUTH_HTTP_TIMEOUT_SECONDS=5
HAI_AUTH_MAX_JWKS_BYTES=1048576
HAI_AUTH_MAX_TOKEN_LENGTH=32768

# --- Data Reports ---
HAI_DATABASE_ROOT=C:\Projects\new-hAI-agent\data\reports
HAI_LLM_PROVIDER=auto                      # OpenCode dulu, Claude Code sebagai cadangan

# --- LLM (isi sesuai bagian 7) ---
OPENCODE_TIMEOUT=120
HAI_AGENT_NAME=hAI
```

Blok autentikasi sama dengan [keycloak/api-auth.env.example](../keycloak/api-auth.env.example).

Variabel opsional yang sering dipakai:

| Variabel | Default | Keterangan |
|---|---|---|
| `CORS_ORIGINS` | `http://127.0.0.1:5173,http://localhost:5173` | Origin frontend, dipisah koma |
| `HAI_DATABASE_PROFILES` | – | Path `profiles.json` untuk MySQL/PostgreSQL/SQL Server |
| `HAI_DATABASE_WORK_DIR` | `<HAI_DATABASE_ROOT>\.hai-work` | Folder kerja & audit |
| `HAI_DATA_CHANGE_APPROVAL` | `delete` | `delete` / `all` / `none` – siapa perlu persetujuan orang kedua |
| `HAI_DATA_CHANGE_SELF_APPROVAL` | mati | `true` agar pembuat rencana boleh menyetujui sendiri (laptop/uji saja) |
| `HAI_DATA_CHANGE_MAX_ROWS` | `10000` | Batas atas baris per perubahan |
| `HAI_STEPUP_MAX_AGE_SECONDS` | `120` | Umur maksimal login ulang untuk menjalankan perubahan |
| `OPENCODE_EXECUTABLE`, `OPENCODE_MODEL`, `OPENCODE_AGENT`, `OPENCODE_CWD` | otomatis | Pengaturan OpenCode |
| `CLAUDE_CODE_EXECUTABLE`, `CLAUDE_CODE_MODEL`, `CLAUDE_CODE_TIMEOUT` | otomatis / 120 | Pengaturan Claude Code |

> Backend harus **dimulai ulang** setiap `.env` berubah.

## 7. Provider LLM (OpenCode atau Claude Code)

Agent tidak menulis SQL; ia meminta LLM menyusun *rencana* terstruktur. Perlu **minimal satu** provider
yang sudah terpasang **dan sudah login/terkonfigurasi** di komputer ini.

**Opsi A – OpenCode CLI**
Pasang dan login sesuai dokumentasi OpenCode, lalu cek `opencode --version`. Jika tidak ada di `PATH`,
isi `OPENCODE_EXECUTABLE=C:\path\ke\opencode.cmd` di `.env`. Model/agent dipilih lewat `OPENCODE_MODEL`
dan `OPENCODE_AGENT` (opsional).

**Opsi B – Claude Code CLI**
Pasang Claude Code, jalankan `claude` sekali dan login. Backend mencari `claude` di `PATH`, atau
isi `CLAUDE_CODE_EXECUTABLE=...`.

Dengan `HAI_LLM_PROVIDER=auto`, OpenCode dicoba dulu dan Claude Code dipakai jika OpenCode gagal/timeout.
Jika hanya satu yang dipasang, tidak masalah, tetapi bila yang pertama macet, halaman menunggu
`OPENCODE_TIMEOUT` penuh. Uji tiap provider:

```powershell
python scripts\check_llm_provider.py
```

Anda ingin melihat `pong` dari minimal satu provider beserta waktunya.

## 8. Database contoh

Backend **tidak mau start** jika `HAI_DATABASE_ROOT` bukan folder yang ada. Buat folder dan database contoh
(120 pesanan sintetis, tanpa data pribadi):

```powershell
New-Item -ItemType Directory -Force data\reports | Out-Null
python scripts\make_sample_sales_db.py data\reports
```

Hasil: `data\reports\sales.sqlite3`.

## 9. Menjalankan backend

Terminal 1, dari root project, venv aktif:

```powershell
python -m uvicorn api.main:app --env-file .env --port 8000
```

Cek: <http://127.0.0.1:8000/health> dan dokumentasi API di <http://127.0.0.1:8000/docs>.
Tambahkan `--reload` hanya untuk pengembangan.

Jika log menyebut `HAI_DATABASE_ROOT` atau autentikasi, periksa [bagian 14](#14-pemecahan-masalah).

## 10. Frontend

Terminal 2:

```powershell
cd C:\Projects\new-hAI-agent\frontend
npm install
Copy-Item .env.example .env
npm run dev
```

`frontend\.env` default sudah benar untuk satu komputer:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
# Opsional (default ditampilkan):
# VITE_KEYCLOAK_URL=http://localhost:8080
# VITE_KEYCLOAK_REALM=hai-agent
# VITE_KEYCLOAK_CLIENT_ID=hai-agent-web
```

Buka **<http://localhost:5173>** (atau `http://127.0.0.1:5173`). Anda diarahkan ke Keycloak untuk login.
Port yang diizinkan realm hanya **5173 dan 5174**; jika Vite berpindah port (karena 5173 terpakai),
login akan ditolak ("Invalid redirect uri"). Bebaskan port 5173 atau tambahkan URI di client
`hai-agent-web`.

## 11. Beri hak akses (role) ke user

Realm sengaja memberi `hai.dev` hak baca saja. Beri role Data Reports memakai skrip bawaan, dengan akun
**admin** Keycloak:

```powershell
$env:KC_ADMIN_PASSWORD = "<password-admin-keycloak>"
python scripts\keycloak_data_report_roles.py --user hai.dev
```

Default memberi `data-reports-user` dan `database-connection-manager`. Role lain lewat `--roles`:

| Role | Hak |
|---|---|
| `data-reports-user` | Membuat & menjalankan laporan |
| `database-connection-manager` | Menghubungkan database & menyetujui tabel/kolom |
| `database-engine-reviewer` | Meninjau permintaan engine database baru |
| `data-schema-creator` | Membuat database/tabel baru (awalan nama `hai_`) |
| `data-writer` | Merencanakan & menjalankan perubahan data (insert/update/delete) |
| `data-entry` | Mencatat data baru saja (insert), mis. transaksi harian: ringkasan lalu "Simpan", tanpa frasa, login ulang, atau persetujuan. Tidak bisa ubah/hapus |
| `data-change-approver` | Menyetujui perubahan buatan orang lain |

```powershell
python scripts\keycloak_data_report_roles.py --user hai.dev --roles data-writer,data-change-approver
```

Setelah itu **sign out lalu sign in lagi** agar token membawa role baru.

Isi file audit perubahan data secara bawaan tidak memuat data bisnis (hanya kunci baris dan jumlah). `HAI_AUDIT_DETAIL=full` menyimpan juga data sebelum perubahan dan nilai yang ditulis.

Satu orang yang memegang `data-writer` dan `data-change-approver` hanya bisa menyetujui perubahannya
sendiri jika `HAI_DATA_CHANGE_SELF_APPROVAL=true` (hanya untuk uji/laptop).

## 12. Menghubungkan database sungguhan (opsional)

Database dipilih **berdasarkan nama profil**; host dan password tidak pernah diketik di halaman web.
Profil ada di `data\reports\profiles.json` (tidak di Git) dan password di `.env`.

1. Tambahkan ke `.env`:
   ```dotenv
   HAI_DATABASE_PROFILES=C:\Projects\new-hAI-agent\data\reports\profiles.json
   ```
2. Pilih salah satu:

**MySQL** – buat `data\reports\profiles.json`:

```json
{
  "local_mysql": {
    "type": "mysql", "host": "127.0.0.1", "port": 3306,
    "database": "nama_database", "user": "hai_reader",
    "password_env": "HAI_MYSQL_LOCAL_PASSWORD", "ssl": "require"
  }
}
```

Kosongkan `HAI_MYSQL_LOCAL_PASSWORD=` di `.env`, lalu skrip membuat akun **hanya-SELECT** dan mengisi password:

```powershell
python scripts\setup_mysql_readonly_user.py local_mysql          # akun pembaca (laporan)
python scripts\setup_mysql_creator_user.py  local_mysql          # opsional: boleh buat database hai_%
python scripts\setup_mysql_writer_user.py   local_mysql --tables nama_tabel   # opsional: ubah data
python scripts\check_database_profile.py local_mysql             # harus: "this profile will work"
```

Skrip meminta password administrator MySQL (diketik tersembunyi, tidak disimpan). Untuk server di
komputer lain tambahkan `--admin-user ... --hosts "<IP komputer ini>"`; lihat bagian
*MySQL on another computer* di [TRY_DATA_REPORTS_LOCALLY.md](TRY_DATA_REPORTS_LOCALLY.md).

**SQL Server** – skrip mendiagnosis dan menulis profil + password untuk Anda:

```powershell
python scripts\connect_sqlserver.py                                  # hanya diagnosis
python scripts\connect_sqlserver.py --create-reader --database dbsales
python scripts\connect_sqlserver.py --create-writer --database dbsales   # opsional
```

Perlu ODBC Driver terpasang, instance mixed-mode, dan Anda administrator instance tersebut.

**PostgreSQL** – tulis profil `"type": "postgresql"` dengan akun hanya-SELECT (lihat
[ADDING_A_DATABASE_ENGINE.md](ADDING_A_DATABASE_ENGINE.md)); PostgreSQL dan SQLite hanya baca.

3. **Mulai ulang backend**, muat ulang halaman, buka *Koneksi* → pilih **Sumber** → *Hubungkan*.

## 13. Verifikasi pemasangan

Centang berurutan:

- [ ] `docker compose -f docker-compose.keycloak.yml ps` → kedua container *running/healthy*.
- [ ] <http://localhost:8080/realms/hai-agent/.well-known/openid-configuration> menampilkan JSON.
- [ ] <http://127.0.0.1:8000/health> menjawab OK.
- [ ] `python scripts\check_llm_provider.py` → minimal satu provider menjawab.
- [ ] <http://localhost:5173> → diarahkan ke login Keycloak → masuk ke dashboard.
- [ ] **Database Agent → Laporan → Koneksi**: nama `sales`, file `sales.sqlite3` → *Hubungkan*.
- [ ] **Setujui**: centang tabel `sales` dan kolomnya (centang *boleh difilter* untuk `region`, `product`) → *Setujui*.
- [ ] **Tanya laporan**: *"Total penjualan per wilayah untuk produk widget"* → *Buat rencana* → *Jalankan* → tabel muncul → *Unduh XLSX/PDF*.

Jika semuanya lolos, pemasangan selesai.

## 14. Pemecahan masalah

| Gejala | Penyebab & solusi |
|---|---|
| `docker compose` mengatakan `Set KC_DB_PASSWORD ...` | Lupa `--env-file keycloak\.env.keycloak`, atau variabel di file itu kosong. |
| `error during connect` / `cannot find the file specified` (Docker) | Docker Desktop belum berjalan. Buka dan tunggu *Running*. |
| Keycloak tidak naik / realm kosong | Lihat `docker compose -f docker-compose.keycloak.yml logs keycloak`. Realm hanya diimpor saat start pertama; untuk mengulang dari nol: `docker compose -f docker-compose.keycloak.yml down -v` (**menghapus semua data Keycloak**). |
| Backend: gagal start soal `HAI_DATABASE_ROOT` | Folder belum dibuat atau path salah (bagian 8). |
| Backend: gagal start soal autentikasi | `HAI_AUTH_ENABLED=true` dan `HAI_KEYCLOAK_*` harus terisi; Keycloak harus bisa dijangkau dari backend. |
| Semua permintaan **401** padahal sudah login | Token tanpa klaim `sub`. Konsol admin: *Clients → hai-agent-web → Client scopes → Add client scope → basic → Default*, lalu login ulang. Alat bantu: `python scripts\inspect_keycloak_token.py`. |
| **403** di Database Agent | User belum punya role; jalankan bagian 11 lalu sign out/in. |
| Login: *Invalid redirect uri* | Frontend jalan di port selain 5173/5174, atau memakai host lain. Tambahkan di client `hai-agent-web`. |
| Halaman: *Failed to fetch* / CORS | Origin frontend tidak ada di `CORS_ORIGINS`; ubah dan mulai ulang backend. Pastikan `VITE_API_BASE_URL` benar. |
| Buat rencana lama lalu gagal | Provider pertama macet sampai `OPENCODE_TIMEOUT`. Jalankan `check_llm_provider.py`; atur `HAI_LLM_PROVIDER` ke provider yang sehat. |
| *database file not found* | `sales.sqlite3` tidak ada di dalam `HAI_DATABASE_ROOT`. |
| `Access denied for 'hai_reader'@'<host>'` | Akun MySQL dibuat untuk host lain; jalankan ulang skrip dengan `--hosts <host itu>`. |
| `database driver is not installed` | `python -m pip install -e ".[mysql]"` (atau `postgresql` / `sqlserver`). |
| SQL Server: driver tidak ditemukan | Pasang *ODBC Driver 17/18 for SQL Server* dari Microsoft. |
| Perubahan data ditolak "sign in again" | Token tanpa `auth_time` atau login lebih tua dari `HAI_STEPUP_MAX_AGE_SECONDS`; login ulang saat diminta. |
| `npm install` gagal | Pakai Node 22 LTS; hapus `node_modules` dan coba lagi (`npm ci` jika ada `package-lock.json`). |

## 15. Akses dari komputer lain di jaringan / produksi

Setup di atas **khusus localhost**. Jika pengguna lain mengakses dari browser mereka, ubah semuanya
berpasangan; jangan menjalankan produksi dengan `start-dev`, HTTP tanpa TLS, atau password contoh.

1. **Keycloak**: jalankan dengan hostname & HTTPS sungguhan (bukan `start-dev`). Di compose, `KC_HOSTNAME`
   dan pengikatan port `127.0.0.1:8080` perlu disesuaikan. Di realm, tambahkan URL frontend ke
   *Valid redirect URIs*, *Web origins*, dan *Post logout redirect URIs* client `hai-agent-web`.
2. **Backend `.env`**: `HAI_KEYCLOAK_BASE_URL=https://<host-keycloak>`, **hapus**
   `HAI_KEYCLOAK_ALLOW_INSECURE_HTTP`, set `CORS_ORIGINS=https://<host-frontend>`.
3. **Frontend**: set `VITE_API_BASE_URL`, `VITE_KEYCLOAK_URL` ke alamat sebenarnya lalu build:
   ```powershell
   cd frontend; npm run build      # hasil statis di frontend\dist, sajikan lewat IIS/nginx
   ```
   (variabel `VITE_*` dibakukan saat build; ubah → build ulang.)
4. **Backend**: jalankan sebagai layanan (NSSM/Task Scheduler) di belakang reverse proxy HTTPS,
   tanpa `--reload`.
5. **Rahasia**: password admin Keycloak, `KC_DB_PASSWORD`, dan semua `HAI_*_PASSWORD` harus kuat dan unik;
   `profiles.json`, `.env`, `keycloak\.env.keycloak` jangan masuk Git atau dikirim lewat chat.
6. Pakai `"ssl": "verify"` untuk MySQL di luar LAN tepercaya, dan hindari `HAI_DATA_CHANGE_SELF_APPROVAL=true`.

## 16. Menjalankan sehari-hari & backup

Setelah terpasang, urutan start:

```powershell
# 1. Keycloak (otomatis hidup lagi bersama Docker Desktop karena restart: unless-stopped)
docker compose --env-file keycloak\.env.keycloak -f docker-compose.keycloak.yml up -d

# 2. Backend (terminal 1)
cd C:\Projects\new-hAI-agent
.\venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --env-file .env --port 8000

# 3. Frontend (terminal 2)
cd C:\Projects\new-hAI-agent\frontend
npm run dev
```

Memperbarui ke versi baru:

```powershell
git pull
python -m pip install -e ".[mysql]"        # sesuaikan extras
cd frontend; npm install
```

Lalu mulai ulang backend dan frontend.

**Yang perlu dicadangkan** (semuanya di luar Git):

| Apa | Lokasi |
|---|---|
| Konfigurasi & rahasia | `.env`, `keycloak\.env.keycloak`, `data\reports\profiles.json` |
| Data Keycloak (user, role) | volume Docker `hai_keycloak_pgdata` (atau ekspor realm dari konsol admin) |
| Persetujuan tabel & audit perubahan | `<HAI_DATABASE_ROOT>\.hai-work\` (audit: `audit\changes-<koneksi>.jsonl`, berisi nilai lama; perlakukan sebagai data bisnis) |
| Database contoh / SQLite | `data\reports\*.sqlite3` |

Memindahkan ke komputer lain: ulangi bagian 2–11, lalu salin file konfigurasi di atas. Jangan menyalin
folder `venv` atau `node_modules`; buat ulang di komputer baru.
