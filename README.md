# 📚 RAG Chatbot (FastAPI + LangChain + LangFlow + ChromaDB + Redis + PostgreSQL)

ระบบ **Retrieval-Augmented Generation** (RAG) Chatbot ที่พัฒนาโดยใช้ **FastAPI**, **LangChain**, **LangFlow**, **ChromaDB**, **Redis** สำหรับจัดการความจำการสนทนา และ **PostgreSQL** สำหรับ LangFlow/บันทึกแชท รองรับการแสดงผล Citation ใน API

---

## ✨ Features

- **Conversational RAG**: โต้ตอบพร้อมจดจำบริบทการสนทนา
- **Document Ingestion**: รองรับไฟล์ `.pdf`, `.txt`, `.md`
- **Vector Store**: ใช้ **ChromaDB** จัดเก็บ embeddings
- **LangFlow UI**: ออกแบบ/แก้ไข pipeline ได้แบบ no-code
- **Redis Memory**: เก็บประวัติการสนทนาแบบ per-session
- **PostgreSQL**: เก็บข้อมูล LangFlow และ (optional) log แชท

---

## 🗂 โครงสร้างโปรเจกต์

```text
.
├─ docker-compose.yml
├─ .env.example
├─ Makefile
├─ README.md
├─ data/docs/                 # วางไฟล์เอกสารที่ต้องการทำ ingestion
├─ api/                       # FastAPI + LangChain
│  ├─ app.py                  # API หลัก
│  ├─ pipelines.py            # Single-turn RAG
│  ├─ pipelines_chat.py       # Conversational RAG + citations
│  ├─ ingest.py               # สร้าง embeddings และบันทึกลง Chroma
│  ├─ memory.py               # Redis/In-memory store
│  ├─ db.py                   # บันทึกแชทลง Postgres (optional)
│  ├─ requirements.txt
│  └─ Dockerfile
├─ frontend/                  # Next.js UI (ในอนาตค!!!)
└─ ops/                       # ไฟล์ deploy/monitoring
```

---

## 🚀 การติดตั้งและใช้งาน

### 1. Clone โปรเจกต์

```bash
git clone https://github.com/yourname/rag-chatbot.git
cd rag-chatbot
```

### 2. คัดลอกไฟล์ Environment

```bash
cp .env.example .env
```

แก้ไขค่าตามต้องการ เช่น รหัสผ่าน PostgreSQL, HuggingFace token

### 3. รันระบบด้วย Docker Compose

```bash
docker compose up -d --build
```

### 4. เพิ่มเอกสารและทำ Ingestion

วางไฟล์ `.pdf` `.md` `.txt` ในโฟลเดอร์ `data/docs/` จากนั้นรัน:

```bash
curl -X POST http://localhost:8000/ingest
```

### 5. ทดสอบถามข้อมูล

```bash
curl -X POST http://localhost:8000/chat   -H "Content-Type: application/json"   -d '{"session_id":"test","message":"สรุปเอกสารทั้งหมด"}'
```

---

## 🌐 บริการที่รันอยู่

- **API (FastAPI)** → [http://localhost:8000](http://localhost:8000)
- **LangFlow UI** → [http://localhost:7860](http://localhost:7860)
- **Redis Commander** → [http://localhost:8081](http://localhost:8081)
- **Adminer (Postgres UI)** → [http://localhost:8082](http://localhost:8082)

---

## 🧩 Endpoints หลัก

| Method | Endpoint      | Description                             |
| ------ | ------------- | --------------------------------------- |
| POST   | `/ingest`     | สร้าง embeddings จากไฟล์ใน `data/docs/` |
| POST   | `/query`      | ถามคำถามแบบ single-turn                 |
| POST   | `/chat`       | ถามคำถามแบบมีความจำ session + citations |
| POST   | `/chat/clear` | ล้างความจำของ session                   |
| GET    | `/health`     | ตรวจสอบสถานะระบบ                        |

---

## 💻 Frontend

1. เข้าโฟลเดอร์ `frontend/`
2. ติดตั้ง dependencies
   ```bash
   npm install
   ```
3. รัน dev server
   ```bash
   npm run dev
   ```
4. เปิด [http://localhost:3000](http://localhost:3000)

---

## ⚙️ การตั้งค่าเพิ่มเติม

- **USE_REDIS**: เปิด/ปิดการใช้ Redis memory
- **ENABLE_CHATLOG_POSTGRES**: บันทึกแชทลง Postgres หรือไม่
- **HF_EMBEDDING_MODEL**: เลือกโมเดล embeddings
- **HF_GENERATOR_MODEL**: เลือกโมเดลสร้างคำตอบ

---

# เปรียบเทียบ Embedding Models และ LLM/SLM สำหรับเอกสาร PDF ภาษาไทย

## 1. Embedding Models (สำหรับ Retrieval)

### `mrp/simcse-model-m-bert-thai-cased`

- **ลักษณะ**: Sentence-Transformers (SimCSE) บน mBERT ฝึกจาก Thai Wikipedia
- **รายละเอียด**:
  - embedding dimension: 768 :contentReference[oaicite:2]{index=2}
  - ใช้ง่ายผ่าน sentence-transformers
- **ข้อดี**: ความเข้าใจภาษาไทยดี ใช้ RAG retrieval ได้ตรงจุด
- **ข้อควรระวัง**: context จำกัด ไม่รองรับเอกสารยาวเกิน token limit

### `mrp/simcse-model-roberta-base-thai`

- **ลักษณะ**: SimCSE บน XLM-Roberta สำหรับภาษาไทย
- **รายละเอียด**: embedding dimension: 768 :contentReference[oaicite:3]{index=3}
- **เหมาะกับ**: ความหลากหลายภาษา + ไทย แต่คล้ายกับ m-bert performance

### `mrp/simcse-model-wangchanberta`

- **ลักษณะ**: SimCSE บน WangchanBERTa (ไทย) :contentReference[oaicite:4]{index=4}
- **จุดเด่น**: ใช้ BERT ไทยที่ออกแบบเฉพาะ
- **สถานการณ์ใช้งาน**: หากต้องการ embed ภาษาไทยเฉพาะเจาะจง

---

## 2. LLM / SLM (สำหรับ Generation)

### `iapp/chinda-qwen3-4b`

- **ลักษณะ**: LLM ภาษาไทย ขนาด 4B บน Qwen3, ฟรี, Apache-2.0 :contentReference[oaicite:5]{index=5}
- **จุดเด่น**:
  - รู้จักคิดและตอบเป็นภาษาไทยแม่น (98.4% accuracy)
  - เหมาะสำหรับระบบ RAG ให้ context ช่วยตอบ
  - inference ใช้งบประมาณต่ำ (suitable for edge)
- **ข้อจำกัด**: อาจ hallucinate ถ้าไม่มี context

### `openthaigpt/openthaigpt-1.0.0-13b-chat`

- **ลักษณะ**: LLaMA-based 13B, instruction tuned ภาษาไทย :contentReference[oaicite:6]{index=6}
- **จุดเด่น**:
  - คะแนนสูงบนข้อสอบภาษาไทย (benchmark สูง)
  - รองรับ context ยาว (4096 Thai words)
- **ข้อจำกัด**: หน่วยความจำ GPU สูงกว่ารุ่น 4B

### `pythainlp/KhanomTanLLM-1B`

- **ลักษณะ**: Bilingual (Thai-English) 1B LLM, ได้รับ open source :contentReference[oaicite:7]{index=7}
- **ข้อดี**: ผสม data หลักภาษาไทย เหมาะสำหรับ edge / low-resource
- **ข้อจำกัด**: ความแม่นยำอาจต่ำกว่ารุ่นใหญ่

---

## 3. สรุปการใช้งาน

| ความต้องการ                  | Embedding Model                                 | LLM/SLM Model                             |
| ---------------------------- | ----------------------------------------------- | ----------------------------------------- |
| ภาษาไทยล้วน, ใช้ local, free | `mrp/simcse-model-m-bert-thai-cased`            | `iapp/chinda-qwen3-4b` (ประหยัด resource) |
| เอกสารภาษาไทยหลากหลายภาษา    | `mrp/simcse-roberta-base-thai`, `wangchanberta` | forecast for generation                   |
| ต้องการรองรับ context ยาว    | (อาจ chunk ก่อน)                                | `OpenThaiGPT 13b` (RAG + long context)    |
| edge device / low resource   | Compact embeddings                              | `KhanomTanLLM-1B` (1B model)              |

---

## 📜 License

MIT License – SatangThevalue

---

## 🙌 Credits

- [LangChain](https://www.langchain.com/)
- [LangFlow](https://www.langflow.org/)
- [ChromaDB](https://www.trychroma.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Redis](https://redis.io/)
- [PostgreSQL](https://www.postgresql.org/)
