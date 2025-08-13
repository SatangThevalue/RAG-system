# 📚 RAG Chatbot (FastAPI + LangChain + LangFlow + ChromaDB + Redis + PostgreSQL)

ระบบ **Retrieval-Augmented Generation** (RAG) Chatbot ที่พัฒนาโดยใช้ **FastAPI**, **LangChain**, **LangFlow**, **ChromaDB**, **Redis** สำหรับจัดการความจำการสนทนา และ **PostgreSQL** สำหรับ LangFlow/บันทึกแชท พร้อม Frontend (Next.js) สำหรับ UI และการแสดงผล Citation

---

## ✨ Features

- **Conversational RAG**: โต้ตอบพร้อมจดจำบริบทการสนทนา
- **Document Ingestion**: รองรับไฟล์ `.pdf`, `.txt`, `.md`
- **Vector Store**: ใช้ **ChromaDB** จัดเก็บ embeddings
- **LangFlow UI**: ออกแบบ/แก้ไข pipeline ได้แบบ no-code
- **Redis Memory**: เก็บประวัติการสนทนาแบบ per-session
- **PostgreSQL**: เก็บข้อมูล LangFlow และ (optional) log แชท
- **Frontend (Next.js)**: UI สวยงาม แสดง citation อ้างอิงจากเอกสาร

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
| Method | Endpoint       | Description |
|--------|---------------|-------------|
| POST   | `/ingest`     | สร้าง embeddings จากไฟล์ใน `data/docs/` |
| POST   | `/query`      | ถามคำถามแบบ single-turn |
| POST   | `/chat`       | ถามคำถามแบบมีความจำ session + citations |
| POST   | `/chat/clear` | ล้างความจำของ session |
| GET    | `/health`     | ตรวจสอบสถานะระบบ |

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

## 📜 License
MIT License – ใช้ได้อิสระ พร้อมระบุเครดิต

---

## 🙌 Credits
- [LangChain](https://www.langchain.com/)
- [LangFlow](https://www.langflow.org/)
- [ChromaDB](https://www.trychroma.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Redis](https://redis.io/)
- [PostgreSQL](https://www.postgresql.org/)
