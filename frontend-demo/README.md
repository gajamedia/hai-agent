# hAI Workbench Demo

Frontend demo terpisah untuk Room simulation pada proyek hAI Agent.

## Prasyarat

- Node.js dan npm
- Backend hAI Agent berjalan di `http://127.0.0.1:8000`

## Menjalankan

```powershell
Copy-Item .env.example .env.local
npm install
npm run dev
```

Buka `http://127.0.0.1:5174`.

## CORS backend

Default CORS backend mengizinkan port 5173. Untuk development, jalankan backend
dengan environment `CORS_ORIGINS` yang mencakup kedua origin frontend:

```powershell
$env:CORS_ORIGINS = "http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:5174,http://localhost:5174"
```

Atur variabel itu pada terminal backend sebelum memulai ulang server. Jangan
mengubah konfigurasi produksi hanya untuk prototipe lokal.

## Batasan

- Eksekusi berlangsung di Room simulation.
- Goal contoh memakai target state yang telah ditentukan.
- Input bebas belum dieksekusi karena belum ada pemetaan target-state yang
  aman dan terverifikasi.
- Konfirmasi UI bukan approval manusia yang ditegakkan secara terpisah di server.
- Tidak ada konektor web atau integrasi eksternal pada tahap ini.
