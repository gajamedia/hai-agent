# Fase 1 - Audit pondasi dan usulan kontrak fitur agent

- **Tanggal:** 8 November 2026
- **Tujuan:** memeriksa apakah Database Agent (baca, ubah data, buat database) berjalan di atas pondasi umum platform, dan merumuskan kontrak "fitur agent" supaya fitur berikutnya bisa ditambahkan tanpa membuat jalur eksekusi baru.
- **Metode:** membaca kode (`api/`, `src/agent/workflow`, `src/agent/llm`, `src/agent/database`) dan riwayat commit. Tidak ada kode yang diubah dan tes tidak dijalankan ulang untuk dokumen ini.
- **Menjawab:** G-08 di [Product Contract](PRODUCT_CONTRACT_V0.1.md).

## 1. Ringkasan temuan

Hanya **jalur baca** (laporan) yang memakai pondasi umum. **Ubah data** dan **buat database** berjalan di jalur sendiri yang paralel, dengan mekanisme otorisasi, approval, audit, dan idempotency yang ditulis ulang untuk fitur itu. Pondasi umum untuk eksekusi (registry, gateway, trace) **hanya mendukung baca**, sehingga operasi tulis memang belum mungkin lewat jalur itu.

Akibatnya, fitur kedua saat ini hanya punya dua pilihan: menyalin pola Database Agent (jalur paralel ketiga) atau memakai gateway baca yang tidak cocok untuk tindakan yang mengubah dunia. Keduanya tidak memenuhi tujuan "pondasi bersama".

## 2. Peta jalur Database Agent terhadap pondasi

Keterangan: **Ya** = memakai komponen umum, **Sebagian** = memakai sebagian, **Tidak** = jalur sendiri.

| Aspek | Laporan (baca) | Ubah data (tulis) | Buat database |
| --- | --- | --- | --- |
| Autentikasi (Keycloak, principal) | Ya | Ya | Ya |
| Otorisasi (`AuthorizationPolicy`, permission) | Ya (`data-reports`) | Ya, permission sendiri (`data:change:plan/execute/approve`) | Ya, permission sendiri (`database:create:manage`) |
| Registry kapabilitas tepercaya | Ya, `business_database.select`; registry dibuat per koneksi di `build_business_database_gateway` | **Tidak** | **Tidak** |
| Gateway eksekusi berotentikasi | Ya (`TracedAuthenticatedCapabilityExecutionGateway`) | **Tidak** | **Tidak** |
| Kebijakan risiko (`ExecutionSecurityPolicy`) | Sebagian: risiko tetap 0.1 pada stand-in `SimpleNamespace` | **Tidak** | **Tidak** |
| Trace eksekusi (`ExecutionTraceStore`) | Ya, JSONL per koneksi | **Tidak** | **Tidak** |
| Audit | Lewat trace di atas | JSONL sendiri (`audit_path`, `DataChangeService._audit`) | JSONL sendiri (`DatabaseCreationService._audit`) |
| Approval | Tidak diperlukan; hanya persetujuan relasi hasil AI (`relation_approver`) | Alur sendiri di route: frasa persis, login ulang, approver kedua | Frasa persis dan login ulang |
| Siklus hidup kapabilitas (usul, approval, aktif) | Tidak: kapabilitas didaftarkan langsung | Tidak | Tidak |
| Penyimpanan rencana | `PlanStore` di memori | `ChangePlanStore` di memori | Memakai `ChangePlanStore` |
| Idempotency | Kunci dibuat server per run | Header `Idempotency-Key` di `ChangePlanStore.begin_run` (di memori) | Sama |
| Durable idempotency lintas restart | Tidak | **Tidak** (hilang saat restart) | Tidak |
| Workflow persisten (`SQLiteWorkflowStore`) | Ya | Tidak | Tidak |
| Telemetry kapabilitas | Tidak | Tidak | Tidak |
| Pembelajaran | `ConnectionExperience` khusus Database Agent | Tidak | Tidak |
| Planner LLM | `LLMService` langsung | `LLMService` langsung | `LLMService` langsung |
| Verifikator hasil | Tidak ada; hanya pengecekan jumlah baris dan truncation | Pratinjau dan jumlah baris yang diharapkan | Tidak ada |

Catatan: baris "Ubah data" dan "Buat database" sudah memiliki kontrol keamanan yang kuat dan teruji (fail closed, pratinjau, frasa konfirmasi, login ulang, approver kedua). Masalahnya bukan kelemahan keamanan, melainkan **duplikasi dan ketidakselarasan** dengan pondasi.

## 3. Temuan terperinci

### F-1. Pondasi eksekusi hanya mengenal baca
`traced_authenticated_capability_execution_gateway.py` secara eksplisit adalah batas baca ("The capability gateway is read-only"). `CapabilityReadResult` dan prefix route `/capability-reads` mengikuti hal yang sama. Operasi tulis tidak punya jalur umum: tidak ada kontrak untuk pratinjau, efek samping, kompensasi, atau verifikasi.

### F-2. Dua sistem approval dan dua sistem audit
`CapabilityApprovalManager` dan lifecycle manager hanya dipakai untuk usulan kapabilitas baru. Approval perubahan data adalah mesin status terpisah di `ChangePlanStore` dan `api/routes/data_changes.py`. Audit perubahan data dan pembuatan database adalah file JSONL sendiri, sedangkan baca memakai `ExecutionTraceStore`. Tidak ada satu tempat untuk menjawab "siapa menyetujui dan menjalankan apa, kapan, dengan hasil apa" lintas fitur.

### F-3. Kapabilitas Database Agent tidak lewat siklus hidup umum
`business_database.select` didaftarkan langsung per koneksi, bukan lewat usul, approval, dan aktivasi. Endpoint `/capabilities/*` (aktifkan, nonaktifkan, cabut) tidak mengendalikan kapabilitas Database Agent. Tidak ada tombol operasional untuk menonaktifkan seluruh Database Agent atau satu operasinya secara terpusat.

### F-4. Kebijakan risiko bersifat tempelan
Laporan memakai `SimpleNamespace` dengan `ExecutionSecurityPolicy(action_risks={READ_ACTION: 0.1})`. Ini stand-in agar kode lama bisa dipakai ulang. Tulis dan buat database tidak melewati kebijakan risiko umum, sehingga skor risiko dan ambang approval tidak berlaku seragam.

### F-5. Status tugas dan rencana hidup di memori proses
`PlanStore` dan `ChangePlanStore` menyimpan rencana di dict dalam proses, dengan masa berlaku singkat. Setelah restart, rencana yang menunggu approval hilang dan idempotency tulis tidak bertahan. Ini tidak aman untuk tugas jangka panjang (G-09) dan membatasi idempotency (G-01).

### F-6. Telemetry dan pembelajaran terpisah
Database Agent tidak mengirim telemetry kapabilitas (`capability_telemetry`) dan memiliki mekanisme pengalaman sendiri (`ConnectionExperience`). Modul pembelajaran umum belum mendapat sinyal dari fitur nyata.

### F-7. Tidak ada verifikator hasil umum
Setelah tulis atau buat database, hasil tidak diperiksa terhadap state yang teramati secara terstandar (G-10). Pada Room, verifikasi ada karena target state didefinisikan; di fitur lain konsep itu belum ada.

### F-8. Perencanaan LLM dipanggil langsung per fitur
Planner laporan (`TableReportPlanner`), ubah data (`DataChangePlanner`), dan pembuatan database memanggil `LLMService` langsung dengan prompt dan validasi sendiri. Prinsip "LLM tidak menulis SQL, hanya rencana terstruktur" konsisten di ketiganya, tetapi penegakannya diulang tiga kali. Orkestrator kognitif umum tidak terlibat, sehingga belum ada satu pintu untuk "maksud pengguna menjadi rencana fitur".

### F-9. Lokasi dan penamaan
Kode fitur tersebar: `api/routes/*`, `api/*_plans.py`, `api/database_runtime_registry.py`, `src/agent/workflow/*` (campuran generik dan khusus database), `src/agent/database/*`, serta `src/agent/llm/*` (100 file, bercampur dengan benchmark). Belum ada paket tempat "semua milik Database Agent".

## 4. Yang sudah baik dan sebaiknya dipertahankan

- Pola dua fase: **rencana untuk ditinjau, lalu dijalankan**, dengan server menyimpan rencana dan klien hanya mengirim id.
- LLM tidak menulis SQL; backend membangun pernyataan dengan identifier yang di-quote dan parameter terikat.
- Fail closed di otorisasi, approval, dan audit.
- Approval berlapis untuk tulis (frasa persis, login ulang, approver kedua).
- Suite konformansi bersama untuk engine database dan tiket permintaan engine.
- Workflow persisten (`SQLiteWorkflowStore`) dengan klaim tugas berbasis versi.

## 5. Usulan kontrak "fitur agent"

Prinsip desain: kontrak dibangun dari pola Database Agent yang sudah terbukti, bukan dari nol; kontrak **mengangkat** pola itu ke pondasi.

### 5.1 Konsep

Sebuah **fitur agent** mendeklarasikan satu atau lebih **operasi** (`Operation`). Setiap operasi memiliki:

| Bagian | Isi | Contoh Database Agent |
| --- | --- | --- |
| `id`, `feature` | Pengenal stabil | `database.report`, `database.change`, `database.create` |
| `effect` | `read`, `write`, atau `external` (menentukan kebijakan minimum) | read, write, write |
| `permission` | Permission yang wajib dimiliki principal | `data-reports-user`, `data:change:execute` |
| `risk` | Skor atau tingkat risiko, dievaluasi oleh kebijakan umum | 0.1 untuk baca; lebih tinggi untuk tulis |
| `plan(request, ctx)` | Mengubah permintaan menjadi **rencana terstruktur** yang divalidasi; tidak punya efek samping | planner laporan, ubah data, buat database |
| `preview(plan)` | Ringkasan dampak untuk ditinjau manusia (wajib untuk `write` dan `external`) | pratinjau baris terdampak |
| `approval` | Kebijakan: tidak perlu, konfirmasi pemilik, login ulang, approver kedua | tulis: frasa persis + login ulang + approver kedua |
| `execute(plan, ctx)` | Menjalankan **persis** rencana yang disetujui; idempotent terhadap kunci | `DataChangeService.execute` |
| `verify(plan, result)` | Memeriksa hasil terhadap state yang teramati; mengembalikan `verified`, `unverified`, atau `failed` dengan bukti | baru; mis. hitung ulang baris |
| `compensate` (opsional) | Cara membatalkan atau memperbaiki jika ada | tidak ada saat ini |

### 5.2 Yang disediakan pondasi (bukan ditulis ulang oleh fitur)

1. **Registry dan siklus hidup**: operasi terdaftar di registry yang sama dan bisa diaktifkan, dinonaktifkan, atau dicabut terpusat.
2. **Pipeline eksekusi tunggal**: otentikasi, otorisasi, kebijakan risiko, approval, idempotency, eksekusi, verifikasi, dan trace dalam satu urutan baku. Fitur tidak boleh memanggil `execute` tanpa lewat pipeline.
3. **Audit dan trace terpusat**: satu format, satu penyimpanan, mencakup keputusan non-eksekusi (setuju, tolak, tolak karena kebijakan).
4. **Penyimpanan tugas persisten** untuk rencana dan statusnya, sehingga tahan restart (dasar Fase 2).
5. **Telemetry dan pengalaman** dikirim otomatis dari pipeline ke modul pembelajaran.
6. **Skema status lifecycle** baku: `planned → awaiting_approval → approved → executing → verifying → completed | failed | unverified | cancelled`.

### 5.3 Aturan keras untuk fitur

- LLM hanya menghasilkan rencana terstruktur; tidak ada string perintah mentah (SQL, shell, HTTP) dari LLM.
- Operasi `write` dan `external` wajib punya `preview` dan kebijakan approval eksplisit.
- Tidak ada operasi yang boleh dieksekusi di luar pipeline.
- Hasil tidak boleh dilaporkan sebagai berhasil bila `verify` tidak mengonfirmasi (kecuali operasi menyatakan `verify = none` secara eksplisit dan itu tercatat).

### 5.4 Pemetaan Database Agent ke kontrak

| Operasi | `effect` | Perubahan yang dibutuhkan |
| --- | --- | --- |
| `database.report` | read | Kecil: sudah lewat gateway; tambah `verify` (mis. cek jumlah baris dan truncation sebagai bukti), ganti stand-in risiko dengan kebijakan nyata, kirim telemetry. |
| `database.change` | write | Besar: pindahkan approval, audit, dan idempotency ke pipeline umum; rencana jadi persisten; tambah `verify` (hitung ulang baris yang terdampak). |
| `database.create` | write | Sedang: sama seperti `database.change`; verifikasi dengan membaca skema hasil. |
| `database.connect` (profil, server, persetujuan tabel) | write ringan (konfigurasi) | Tentukan apakah masuk kontrak sebagai operasi atau tetap administrasi di luar agent. |

## 6. Usulan urutan pengerjaan Fase 1 (setelah audit disetujui)

Tiap langkah kecil, bisa di-commit terpisah, dan seluruh tes tetap hijau:

1. **1a. Kontrak dan kerangka**: definisikan tipe `Operation`/`Feature` dan registry operasi di paket baru (`src/agent/platform/` atau nama lain), tanpa mengubah perilaku.
2. **1b. Pipeline eksekusi umum** untuk `effect = write`: otorisasi, risiko, approval, idempotency persisten, audit terpusat, dengan tes kontrak (termasuk fail closed dan restart).
3. **1c. Migrasi `database.report`**: paling mudah; membuktikan kontrak pada jalur baca yang sudah memakai pondasi.
4. **1d. Migrasi `database.change`** lalu **`database.create`**: memindahkan approval, audit, dan idempotency ke pipeline; pertahankan seluruh perilaku dan tes yang ada.
5. **1e. Penataan paket**: pisahkan inti (`core`), fitur (`features/database`), dan lingkungan (`environments/room`); pindahkan benchmark P0 keluar dari `src/agent/llm/`.
6. **1f. Bukti pondasi**: tulis fitur contoh kecil (tanpa dunia nyata, mis. `echo` atau `notes`) hanya untuk memastikan fitur baru bisa dibuat tanpa menyentuh inti.

## 7. Risiko dan hal yang perlu diputuskan

1. **Risiko regresi keamanan.** Memindahkan approval dan audit tulis adalah perubahan paling sensitif. Mitigasi: tes karakterisasi dulu (pastikan perilaku saat ini terkunci), pindahkan bertahap, jangan melonggarkan satu pun kontrol.
2. **Kompatibilitas data.** Format audit JSONL yang ada tidak boleh hilang tanpa jalur baca ulang atau impor. Perlu keputusan: tetap menulis format lama berdampingan selama masa transisi atau menyediakan konverter.
3. **Cakupan `verify`.** Untuk setiap operasi perlu disepakati apa bukti keberhasilan; tanpa itu verifikator menjadi formalitas.
4. **Batas tanggung jawab LLM.** Apakah ketiga planner dipindahkan ke satu "pintu rencana" umum (F-8) pada Fase 1, atau ditunda ke Fase 2/3.
5. **Nama dan lokasi paket** (`platform`, `features`, `environments`) sebaiknya ditentukan sebelum 1a.

## 8. Keputusan yang diminta

- Setujui atau revisi bentuk kontrak di bagian 5 (terutama `effect`, `verify`, dan status lifecycle).
- Pilih penanganan format audit lama (konverter, atau tulis ganda sementara).
- Putuskan apakah unifikasi planner LLM masuk Fase 1 atau ditunda.
- Setelah itu langkah 1a dapat dimulai.

## 9. Kemajuan

| Langkah | Status | Catatan |
| --- | --- | --- |
| 1a Kontrak dan registry | Selesai (9 Nov 2026) | `src/agent/platform/contracts.py`, `registry.py` |
| 1b Pipeline dan penyimpanan tugas | Selesai (9 Nov 2026) | `pipeline.py`, `store.py`; audit dan status satu transaksi, replay hanya dengan `Idempotency-Key` yang sama |
| 1c `database.report` di pipeline | Selesai (9 Nov 2026) | `api/features/database_reports.py`, `api/platform_runtime.py`; rute `/table-reports/*` tidak berubah bagi frontend |
| 1d-1 `database.change` | Selesai (9 Nov 2026) | `api/features/database_changes.py`; audit JSONL lama tetap ditulis berdampingan |
| 1d-2 `database.create` | Belum | |
| 1e Penataan paket | Belum | |
| 1f Fitur contoh | Belum | |

Catatan 1c (perilaku yang sengaja dijaga atau diubah):

- Rencana laporan kini tersimpan di `<work_dir>/platform.sqlite3` dan tahan restart. Sebelumnya hanya di memori.
- **Baris laporan tidak pernah ditulis ke penyimpanan tugas.** Tugas hanya menyimpan metadata (status, id workflow, info berkas, jumlah baris). Baris ada di berkas laporan dan, untuk replay idempoten dalam proses yang sama, di cache memori. Replay setelah restart mengembalikan fakta yang sama dengan `rows: []`.
- Pemeriksaan "koneksi berubah" kini memakai token sidik jari persetujuan (`DatabaseRuntimeRegistry.service_token`), bukan identitas objek, sehingga tetap berlaku setelah restart. Menyetujui ulang dengan tabel/kolom yang sama tidak lagi membatalkan rencana; mengubah persetujuan tetap membatalkannya.
- Membatalkan (`DELETE`) rencana yang sudah selesai kini dijawab 409, bukan menghapus diam-diam.
- Verifikator laporan memeriksa status keberhasilan, id workflow, dan ukuran berkas. Pemeriksaan data dan penolakan pembacaan terpotong tetap dilakukan job runner seperti sebelumnya.

Catatan 1d-1 (ubah data):

- Kontrak platform diperluas karena ubah data membutuhkannya: kebijakan approval yang dihitung per rencana (hanya boleh lebih ketat dari baseline operasi), izin merencanakan berbeda dari izin menjalankan, `sensitive` (rencana dan pratinjau yang memuat data bisnis **dihapus dari penyimpanan tugas begitu tugas berakhir**, hanya hash yang tersisa), batas tugas per operasi, dan pengaturan server (self-approval, batas login segar) yang dibaca saat dibutuhkan.
- Verifikasi setelah commit: delete dicek dengan membaca ulang (tidak ada baris yang masih cocok), update dengan membaca ulang baris berisi nilai baru, insert dan update yang mengubah kolom pemilihnya memakai jumlah baris yang dilaporkan database. Kalau hasil baca ulang tidak cocok, status `unverified`, bukan `completed`.
- Perubahan perilaku yang disengaja: setelah `change_refused` (data berubah sejak pratinjau) rencana tidak bisa dijalankan ulang, harus direncanakan lagi (sebelumnya dilepas kembali). Membatalkan rencana yang sudah selesai dijawab 409. Respons run mendapat satu kolom baru, `verification`.
- Pemeriksaan "koneksi berubah" memakai `DatabaseRuntimeRegistry.change_token` (sidik jari tabel, kolom, dan operasi yang boleh ditulis).
- Audit JSONL lama (`audit/changes-<db>.jsonl`, termasuk before-image) tidak berubah dan tetap menjadi catatan forensik utama; audit platform menambah siapa, kapan, dan status tiap langkah tanpa memuat data.
