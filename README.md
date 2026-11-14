# hAI Agent

hAI Agent adalah **platform AI Agent otonom umum**. Pengguna menyatakan tujuan dalam bahasa natural; agen menafsirkan, merencanakan, menjalankan tindakan yang diizinkan kebijakan, memverifikasi hasil, dan menjelaskan apa yang berhasil, gagal, atau belum selesai.

Platform ini disusun sebagai **pondasi bersama + fitur agent**:

- **Pondasi**: kognisi (interpretasi, penalaran, perencanaan), memori dan pembelajaran, registry kapabilitas tepercaya, kebijakan risiko dan approval, autentikasi/otorisasi (Keycloak), audit/trace, idempotency, dan pemulihan.
- **Fitur agent**: modul yang memakai pondasi itu untuk satu bidang pekerjaan. Fitur pertama dan satu-satunya yang sudah berfungsi di dunia nyata adalah **Database Agent**. Fitur lain direncanakan dan akan dibangun di atas pondasi yang sama.

> Status: **pra-produksi, dalam pengembangan aktif**. Rincian cakupan terverifikasi dan celah yang diketahui ada di [Product Contract](docs/product/PRODUCT_CONTRACT_V0.1.md). Jangan menganggap sistem ini siap produksi.

## Daftar isi

- [Status fitur](#status-fitur)
- [Gambaran sistem](#gambaran-sistem)
- [Struktur direktori](#struktur-direktori)
- [Prasyarat](#prasyarat)
- [Instalasi](#instalasi)
- [Konfigurasi](#konfigurasi)
- [Menjalankan aplikasi](#menjalankan-aplikasi)
- [API](#api)
- [Menjalankan pengujian](#menjalankan-pengujian)
- [Penyimpanan dan data eksperimen](#penyimpanan-dan-data-eksperimen)
- [Pengembangan](#pengembangan)
- [Roadmap](#roadmap)
- [Dokumentasi lain](#dokumentasi-lain)

## Status fitur

| Bagian | Status | Keterangan |
| --- | --- | --- |
| **Database Agent: Laporan** | Berfungsi | Laporan tabel dari bahasa natural dalam dua fase (rencana untuk ditinjau, lalu dijalankan). LLM tidak pernah menulis SQL. Mendukung filter, perbandingan, daftar baris, dan join pada relasi yang disetujui. Keluaran Excel, PDF, CSV, HTML. |
| **Database Agent: Ubah data** | Berfungsi | Insert/update/delete dengan akun penulis terpisah, pratinjau, batas baris, login ulang, dan approver kedua (self-approval opsional, default mati). |
| **Database Agent: Buat database** | Berfungsi | Membuat database/tabel baru (prefix wajib) atau menambah tabel ke database yang ada. |
| Engine database | MySQL, PostgreSQL, SQL Server, SQLite | Tes langsung ke server nyata bersifat opt-in (lihat [Menjalankan pengujian](#menjalankan-pengujian)). |
| Pondasi kognitif | Berfungsi, teruji di simulasi | Orchestrator LLM, memori, pembelajaran, replanning adaptif, telemetry. Bukti benchmark hanya pada fixture simulasi Room. |
| Keamanan | Berfungsi | Keycloak (JWT/JWKS), otorisasi per peran, audit, approval berlapis. |
| Pengujian otonom | Berfungsi | Skenario, injeksi kegagalan, dan evaluasi pemulihan (halaman *Autonomous Testing*). |
| Riwayat tugas (menu *Riwayat tugas*) | Berfungsi | Daftar tugas milik Anda dari semua fitur di pipeline platform (laporan, ubah data), dengan status, verifikasi, jejak audit, dan unduhan berkas laporan. Hanya membaca; tugas dibuat dan dilanjutkan di halaman fiturnya. |
| Siklus tugas terpadu di UI | **Sebagian** | Riwayat sudah terpadu. Membuat, menyetujui, dan menjalankan masih di halaman tiap fitur; buat database belum memakai pipeline. |
| Tugas persisten, penjadwal, pemicu | **Belum** | Agen belum bekerja sendiri tanpa permintaan pengguna. |
| Kontrak "fitur agent" yang baku | **Belum** | Fitur baru belum punya cara standar untuk mendaftarkan diri ke pondasi. |
| Fitur agent lain (berkas, HTTP/API, email, dan sebagainya) | **Belum** | Direncanakan. |
| Kebijakan versi API/SDK publik | **Belum** | |
| Paket deploy (Docker) | **Belum** | Hanya ada compose untuk Keycloak. |

Lihat [roadmap](#roadmap) untuk urutan pengerjaan.

## Gambaran sistem

```text
Dashboard React (frontend/)
        │ HTTP (token Keycloak)
        ▼
FastAPI (api/)  ── autentikasi, otorisasi, audit
   ├── Fitur: Database Agent ── laporan / ubah data / buat database
   │       └── engine MySQL · PostgreSQL · SQL Server · SQLite
   ├── Cognitive runtime ── Orchestrator ── LLMService ── provider LLM
   │       └── Agent: kognisi, memori, pembelajaran, refleksi
   ├── Execution runtime ── Agent + RoomEnvironment (simulasi lokal)
   ├── Registry & siklus hidup kapabilitas (usul → approval → aktif)
   └── Telemetry kapabilitas, strategi, autonomous testing
```

Prinsip yang dijaga di seluruh platform:

1. **Kognisi dan eksekusi terpisah.** Endpoint kognitif hanya menghasilkan interpretasi/rencana dan **tidak mengeksekusi** rencana itu.
2. **Otonomi yang diawasi.** Tindakan berisiko hanya berjalan dengan approval yang sah; kegagalan pemeriksaan otorisasi berarti ditolak (*fail closed*).
3. **LLM tidak menulis SQL atau perintah mentah.** LLM mengusulkan rencana terstruktur; backend memvalidasi dan menjalankannya.
4. **Rencana bukan bukti hasil.** Keberhasilan hanya dinyatakan jika diverifikasi dari state yang teramati.
5. **Jujur soal cakupan.** Bukti simulasi tidak diklaim sebagai bukti kinerja dunia nyata.

Simulasi Room (`open_box`, `take_key`, `find_spare_key`, `unlock_door`) adalah lingkungan contoh dan lahan uji pondasi, bukan konektor ke perangkat atau layanan eksternal. Statenya berjalan di memori proses.

Provider LLM dipilih lewat `HAI_LLM_PROVIDER` (`auto` mencoba OpenCode CLI dulu, lalu Claude Code sebagai cadangan); lihat [manual instalasi](docs/INSTALLATION.md).

## Struktur direktori

| Path | Isi |
| --- | --- |
| `src/agent/core/` | Agent, state, goal, action, experience, dan komposisi kemampuan inti. |
| `src/agent/cognition/` | Pengambilan keputusan, planner, eksplorasi, risiko, dan unified executor. |
| `src/agent/llm/` | Layanan/provider LLM, interpretasi, reasoning, planning, orchestrator, dan runtime integrasi. |
| `src/agent/memory/` | Working, episodic, semantic, causal, procedural, reflection, dan strategy memory beserta persistence. |
| `src/agent/learning/` | Pembelajaran strategi/kausal, adaptasi, telemetry, self-awareness, reflection, dan metacognition. |
| `src/agent/world/` | Model lingkungan dan simulasi ruangan (Room). |
| `src/agent/tools/` | Sistem aksi dan tool untuk lingkungan. |
| `src/agent/database/` | Manajer koneksi, deteksi, skema, dan gateway kapabilitas database. |
| `src/agent/workflow/` | Alur kerja Database Agent: rencana laporan, agregasi, perubahan data, pembuatan database, dan renderer (Excel/PDF/CSV/HTML). |
| `src/agent/autonomous_testing/` | Generator skenario, injeksi kegagalan, pemulihan, dan evaluator. |
| `api/` | FastAPI app, schema, dependency factory, keamanan (`api/security/`), engine database, dan routes. |
| `frontend/` | React + Vite dashboard (Dashboard, Database Agent, Strategy Lab, Autonomous Testing). |
| `docs/` | Manual instalasi, panduan fitur, dan [Product Contract](docs/product/PRODUCT_CONTRACT_V0.1.md). |
| `keycloak/`, `docker-compose.keycloak.yml` | Konfigurasi realm dan compose Keycloak untuk pengembangan. |
| `scripts/` | Skrip operasional (setup akun database, diagnosa, Keycloak) dan skrip benchmark `step41_*`. |
| `tests/` | Pengujian unit, integrasi, API, runtime, engine database, memori, dan pembelajaran. |
| `experiments/`, `data/` | Materi eksperimen historis dan data lokal. File database lokal di `data/` tidak dilacak Git. |

## Prasyarat

- Python **3.10.11 atau lebih baru**.
- Node.js dan npm untuk menjalankan dashboard.
- Keycloak (compose tersedia) untuk autentikasi; lihat [manual instalasi](docs/INSTALLATION.md).
- OpenCode CLI dan/atau Claude Code CLI untuk provider LLM. Simulasi execution tidak memerlukan LLM.
- Driver database opsional sesuai engine: `.[mysql]`, `.[postgresql]`, `.[sqlserver]` (SQL Server juga butuh ODBC Driver 17/18).

## Instalasi

> Memasang di komputer baru dari nol (Keycloak, `.env`, LLM, database, role)? Ikuti [manual instalasi lengkap](docs/INSTALLATION.md). Bagian di bawah hanya ringkasan.

Dari direktori root project, buat dan aktifkan virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

`pyproject.toml` memasang FastAPI dan Uvicorn. Untuk menjalankan suite API, pasang juga `httpx`; untuk seluruh tes, pasang `pytest`:

```powershell
python -m pip install httpx pytest
```

Untuk frontend, buka terminal baru:

```powershell
cd frontend
npm install
Copy-Item .env.example .env
```

Jangan commit file `.env` lokal atau kredensial. `.env` sudah dikecualikan oleh `.gitignore`.

## Konfigurasi

Backend membaca konfigurasi dari environment. Uvicorn dapat memuat file `.env` dengan opsi `--env-file`, tetapi file tersebut harus dibuat secara lokal.

| Variabel | Default | Keterangan |
| --- | --- | --- |
| `OPENCODE_EXECUTABLE` | Deteksi otomatis | Path executable OpenCode bila tidak tersedia di `PATH`. |
| `OPENCODE_MODEL` | Default OpenCode | Model dalam format yang dikenali instalasi OpenCode. |
| `OPENCODE_AGENT` | Default OpenCode | Agent OpenCode opsional. |
| `OPENCODE_CWD` | Direktori kerja proses | Direktori kerja untuk proses OpenCode. |
| `OPENCODE_TIMEOUT` | `120` | Batas waktu provider dalam detik; harus angka positif. |
| `HAI_AGENT_NAME` | `hAI` | Nama agent kognitif. |
| `HAI_LLM_PROVIDER` | `auto` | Pemilihan provider LLM (`auto` = OpenCode lalu Claude Code). |
| `HAI_AUTH_ENABLED`, `HAI_AUTH_REQUIRED`, `HAI_KEYCLOAK_*` | - | Autentikasi Keycloak; lihat manual instalasi. |
| `HAI_DATABASE_ROOT`, `HAI_DATABASE_PROFILES`, `HAI_DATABASE_WORK_DIR` | - | Lokasi database SQLite, profil server database, dan folder kerja Database Agent. |
| `HAI_DATA_CHANGE_APPROVAL`, `HAI_DATA_CHANGE_SELF_APPROVAL`, `HAI_DATA_CHANGE_MAX_ROWS`, `HAI_STEPUP_MAX_AGE_SECONDS` | - | Kebijakan approval dan batas perubahan data. Self-approval default mati. |
| `CORS_ORIGINS` | `http://127.0.0.1:5173,http://localhost:5173` | Daftar origin frontend yang dipisahkan koma. |
| `VITE_API_BASE_URL` | `http://127.0.0.1:8000` | Base URL backend yang dibaca saat Vite berjalan/build. Atur di `frontend/.env`. |

Contoh `frontend/.env`:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## Menjalankan aplikasi

Jalankan backend dari root project dengan virtual environment aktif:

```powershell
uvicorn api.main:app --env-file .env --reload
```

Jika belum menggunakan file `.env`, hilangkan `--env-file .env`. API tersedia di `http://127.0.0.1:8000`; dokumentasi interaktif FastAPI ada di `/docs`, dan pemeriksaan kesehatan di `/health`.

Jalankan frontend dari direktori `frontend`:

```powershell
npm run dev
```

Vite menampilkan URL dashboard di terminal (umumnya `http://127.0.0.1:5173`). Backend mengizinkan dua origin localhost tersebut secara default. Untuk origin lain, sesuaikan `CORS_ORIGINS`, lalu mulai ulang backend.

## API

Semua route versi berada di prefix `/api/v1`.

Referensi interaktif lengkap ada di `/docs` saat backend berjalan. Ringkasan per kelompok:

| Kelompok (prefix `/api/v1`) | Fungsi |
| --- | --- |
| `/cognitive/run` | Satu siklus kognitif (interpretasi, penalaran, rencana). **Tidak mengeksekusi** rencana. |
| `/execution/*` | Simulasi Room: `state`, `run`, `plan-run` (rencana + eksekusi + verifikasi, idempotent), `reset`, `reconcile`. |
| `/table-reports/*` | Database Agent, laporan: buat rencana, jalankan, unduh berkas, batalkan. |
| `/data-changes/*` | Database Agent, ubah data: rencana, jalankan, antrean approval (`approve`/`reject`). |
| `/database-creation/*` | Database Agent, buat database/tabel: rencana dan jalankan. |
| `/database-connections/*` | Profil dan server database: hubungkan, setujui tabel/kolom, segarkan, hapus. |
| `/database-onboarding/*`, `/database-engine-requests/*` | Allowlist awal dan tiket permintaan engine baru. |
| `/capability-proposals/*`, `/capabilities/*` | Siklus hidup kapabilitas: usul, approval, aktivasi, nonaktif, cabut. |
| `/capability-reads`, `/trusted-capability-invocations` | Eksekusi kapabilitas tepercaya lewat gateway. |
| `/tasks`, `/tasks/{id}` | Riwayat tugas Anda sendiri (filter `waiting`, `running`, `done`, `problem`) dan rincian dengan jejak audit. |
| `/autonomous-testing/*` | Batch pengujian otonom dan rekonsiliasinya. |
| `/telemetry/capabilities`, `/strategies` | Telemetry kapabilitas dan informasi strategi. |
| `GET /health` (tanpa prefix) | Pemeriksaan hidup. |

Sebagian besar endpoint memerlukan token Keycloak dengan peran yang sesuai.

Contoh request kognitif minimal:

```json
{
  "observation": { "location": "lab", "door": "closed" },
  "goal": "open the door",
  "include_trace": true
}
```

Untuk mengaktifkan metakognisi pada request, sertakan `include_metacognition: true` beserta `episode` dan `deterministic_reflection`. Lihat contoh lengkap yang digunakan dashboard di `frontend/src/data/exampleRequest.js`.

Contoh eksekusi simulasi:

```json
{
  "actions": ["open_box", "take_key", "unlock_door"],
  "goal_description": "Unlock the door",
  "target_state": { "door": "unlocked" }
}
```

Request yang dikirim ke execution API menentukan aksi secara eksplisit; pemanggilan endpoint ini bukan bukti bahwa agent secara otonom memilih aksi tersebut. Skema request dan batas validasi ada di `api/schemas.py`.

## Menjalankan pengujian

Jalankan semua tes dari root project:

```powershell
pytest -q
```

Atau jalankan tes API/simulasi saja:

```powershell
pytest -q tests/test_api.py tests/test_api_execution.py tests/test_api_telemetry.py
```

Tes API umumnya menggunakan runtime tiruan agar tidak butuh provider LLM eksternal. Pada pemeriksaan terakhir seluruh suite menghasilkan **5220 lulus, 82 dilewati, 0 gagal** (sekitar 6 menit).

Tes yang dilewati adalah tes integrasi *opt-in* yang butuh server nyata dan otomatis aktif jika variabel lingkungannya diisi (CI menjalankan MySQL dan PostgreSQL):

| Variabel | Mengaktifkan |
| --- | --- |
| `HAI_MYSQL_TEST_PORT` (+ `HAI_MYSQL_TEST_ROOT_PASSWORD`) | Tes MySQL langsung |
| `HAI_PG_TEST_PORT` (+ `HAI_PG_TEST_ADMIN_PASSWORD`) | Tes PostgreSQL langsung |
| `HAI_SQLSERVER_TEST_SERVER` | Tes SQL Server langsung (belum ada di CI) |
| `HAI_RUN_LIVE_LLM_TESTS=1` | Tes yang memanggil Claude sungguhan |

Untuk tes laporan Excel/PDF pasang juga `openpyxl` dan `pypdf`. Tes frontend di `frontend/`: `npm test` (unit) dan `npm run test:e2e` (Playwright).

## Penyimpanan dan data eksperimen

Beberapa modul mendukung penyimpanan memori atau model diri persisten. Lokasi/jenis penyimpanan ditentukan oleh repository/runtime yang digunakan; sebagian memakai SQLite dan sebagian dapat menggunakan penyimpanan file. Runtime simulasi API dibuat saat startup dan bersifat process-local: state simulasi serta telemetry-nya tidak dijamin bertahan setelah server berhenti atau dimulai ulang.

Direktori `data/` berisi database eksperimen lokal. `.gitignore` mengecualikan `data/*.db`, sehingga database tersebut tidak masuk commit secara default. Jaga salinan data eksperimen yang ingin dipertahankan.

## Pengembangan

- Kemampuan inti ditambahkan pada package yang sesuai di `src/agent/`, lalu diintegrasikan lewat runtime/orchestrator.
- Fitur agent baru harus memakai pondasi yang ada (registry kapabilitas, kebijakan risiko/approval, trace, autentikasi) dan tidak membuat jalur eksekusi sendiri di luar itu. Kontrak baku untuk fitur agent sedang disusun (lihat roadmap).
- Route API ada di `api/routes/`, schema di `api/schemas.py`, dan dependency/runtime di `api/dependencies.py` serta `api/execution_runtime.py`.
- Engine database baru: ikuti [docs/ADDING_A_DATABASE_ENGINE.md](docs/ADDING_A_DATABASE_ENGINE.md) dan jalankan suite konformansi bersama.
- Aksi simulasi Room didefinisikan di `api/execution_runtime.py`; pertahankan validasi allowlist.
- Setiap perubahan perilaku disertai tes di `tests/`. CI (`.github/workflows/ci.yml`) menjalankan tes backend, engine database, dan frontend pada setiap pull request.
- Skrip sekali pakai, cadangan lokal, dan hasil audit tidak dilacak Git (lihat `.gitignore`).

## Roadmap

1. **Fase 0 - Penyelarasan**: dokumentasi mengikuti arah platform, repo dirapikan, tes yang dilewati ditinjau.
2. **Fase 1 - Kontrak fitur**: kontrak baku "fitur agent" dan pemetaan seluruh jalur Database Agent ke pondasi umum; paket diatur ulang (inti, fitur, lingkungan).
3. **Fase 2 - Lapisan otonom**: tugas persisten dengan status siklus hidup, penjadwal/pemicu, verifikator hasil dan idempotency di jalur eksekusi umum, UI terpadu.
4. **Fase 3 - Fitur kedua**: membuktikan pondasi dengan fitur yang sifatnya berbeda dari database, plus router yang memilih fitur dari permintaan natural.
5. **Fase 4 - Pengerasan**: paket deploy, penyimpanan state bersama, kebijakan versi API, evaluasi di tugas nyata.

## Dokumentasi lain

- [Product Contract](docs/product/PRODUCT_CONTRACT_V0.1.md): cakupan terverifikasi, prinsip, kriteria akseptasi, dan celah.
- [Manual instalasi](docs/INSTALLATION.md) dan [mencoba Database Agent secara lokal](docs/TRY_DATA_REPORTS_LOCALLY.md).
- [Menambah engine database](docs/ADDING_A_DATABASE_ENGINE.md).
- [README_API.md](README_API.md) (catatan awal API kognitif) dan [frontend/README.md](frontend/README.md).
