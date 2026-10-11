# hAI Agent

hAI Agent adalah project eksperimen untuk membangun agen AI modular dengan kemampuan persepsi, penalaran, perencanaan, eksekusi aksi, memori, pembelajaran, refleksi, dan metakognisi. Project ini menyediakan library Python, API HTTP berbasis FastAPI, serta dashboard React untuk mencoba pipeline kognitif dan simulasi lingkungan.

> Status project: eksperimental. Banyak kemampuan tersedia sebagai modul yang dapat dirangkai; tidak semuanya aktif otomatis pada setiap request.

## Daftar isi

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

## Gambaran sistem

```text
Dashboard React (frontend/)
        │ HTTP
        ▼
FastAPI (api/)
   ├── Cognitive runtime ── Orchestrator ── LLMService ── OpenCode CLI
   │       └── Agent: kognisi, memori, pembelajaran, refleksi
   ├── Execution runtime ── Agent + RoomEnvironment (simulasi lokal)
   └── Capability telemetry
```

Ada dua runtime backend yang sengaja dipisahkan:

1. **Cognitive runtime** menyusun siklus interpretasi, penalaran, dan perencanaan melalui `CognitiveOrchestrator`, lalu menjalankannya dalam konteks `AgentCognitiveRuntime`. Provider LLM yang disiapkan saat ini adalah `OpenCodeProvider`, yang memanggil OpenCode CLI.
2. **Execution runtime** menjalankan aksi yang diminta secara eksplisit di `RoomEnvironment`. Aksi dibatasi pada allowlist (`open_box`, `take_key`, `find_spare_key`, `unlock_door`) dan state simulasi berjalan di memori proses.

Endpoint kognitif hanya menghasilkan hasil kognitif/rencana; endpoint tersebut **tidak mengeksekusi rencana**. Untuk simulasi aksi, gunakan endpoint execution. Simulasi bukan konektor ke perangkat atau lingkungan eksternal.

Dashboard saat ini menyediakan input request kognitif, ringkasan hasil/trace, telemetry kapabilitas, serta kontrol simulasi. Base URL API dapat diatur lewat `VITE_API_BASE_URL`.

## Struktur direktori

| Path | Isi |
| --- | --- |
| `src/agent/core/` | Agent, state, goal, action, experience, dan komposisi kemampuan inti. |
| `src/agent/cognition/` | Pengambilan keputusan, planner, eksplorasi, risiko, dan unified executor. |
| `src/agent/llm/` | Layanan/provider LLM, interpretasi, reasoning, planning, orchestrator, dan runtime integrasi. |
| `src/agent/memory/` | Working, episodic, semantic, causal, procedural, reflection, dan strategy memory beserta persistence. |
| `src/agent/learning/` | Pembelajaran strategi/kausal, adaptasi, telemetry, self-awareness, reflection, dan metacognition. |
| `src/agent/world/` | Model lingkungan dan simulasi ruangan. |
| `src/agent/tools/` | Sistem aksi dan tool untuk lingkungan. |
| `api/` | FastAPI app, schema, dependency factory, runtime simulasi, dan routes. |
| `frontend/` | React + Vite dashboard. |
| `tests/` | Pengujian unit, integrasi, API, runtime, memori, dan pembelajaran. |
| `experiments/`, `data/` | Materi eksperimen dan data lokal. File database lokal pada `data/*.db` tidak dilacak Git. |

## Prasyarat

- Python **3.10.11 atau lebih baru**.
- Node.js dan npm untuk menjalankan dashboard.
- OpenCode CLI yang sudah terpasang dan dikonfigurasi jika ingin memakai endpoint kognitif dengan provider LLM nyata. Simulasi execution tidak memerlukan provider LLM.

## Instalasi

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

| Method | Path | Fungsi |
| --- | --- | --- |
| `POST` | `/api/v1/cognitive/run` | Menjalankan satu siklus kognitif dan mengembalikan hasil serta trace opsional. Tidak mengeksekusi rencana. |
| `GET` | `/api/v1/execution/state` | Membaca state simulasi dan daftar aksi yang diizinkan. |
| `POST` | `/api/v1/execution/run` | Menjalankan satu atau beberapa aksi allowlist secara berurutan pada simulasi. |
| `POST` | `/api/v1/execution/reset` | Mengembalikan state simulasi ke awal; telemetry historis runtime tetap tersimpan selama proses hidup. |
| `GET` | `/api/v1/telemetry/capabilities?limit=20` | Membaca snapshot dan event telemetry kapabilitas. `limit` bernilai 1–100. |
| `GET` | `/health` | Memeriksa bahwa aplikasi API hidup. |

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

Tes API umumnya menggunakan runtime tiruan untuk menghindari kebutuhan provider LLM eksternal. Tes provider/runtime yang memanggil OpenCode dapat memiliki kebutuhan konfigurasi berbeda.

## Penyimpanan dan data eksperimen

Beberapa modul mendukung penyimpanan memori atau model diri persisten. Lokasi/jenis penyimpanan ditentukan oleh repository/runtime yang digunakan; sebagian memakai SQLite dan sebagian dapat menggunakan penyimpanan file. Runtime simulasi API dibuat saat startup dan bersifat process-local: state simulasi serta telemetry-nya tidak dijamin bertahan setelah server berhenti atau dimulai ulang.

Direktori `data/` berisi database eksperimen lokal. `.gitignore` mengecualikan `data/*.db`, sehingga database tersebut tidak masuk commit secara default. Jaga salinan data eksperimen yang ingin dipertahankan.

## Pengembangan

- Tambahkan kemampuan inti pada package yang sesuai di `src/agent/`, lalu integrasikan melalui runtime/orchestrator seperlunya.
- Route API didefinisikan di `api/routes/`, schema request/response di `api/schemas.py`, dan runtime dependency dibuat di `api/dependencies.py` atau `api/execution_runtime.py`.
- Komponen dashboard dan helper request berada di `frontend/src/`.
- Nama aksi simulasi yang dapat dieksekusi didefinisikan pada `api/execution_runtime.py`; pertahankan validasi allowlist saat menambah aksi.
- Untuk perubahan API, perbarui tes terkait di `tests/` dan contoh dokumentasi ini bila kontraknya berubah.

Dokumentasi pendukung: [README_API.md](README_API.md) dan [frontend/README.md](frontend/README.md).
