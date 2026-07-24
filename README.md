# ⚡ CYCLUP — Campus Intelligence Vault & Grounded RAG Engine

> **Offline-first campus RAG assistant for Lagos State University (LASU).**  
> Ingests campus documents, notices, fees, and timetables; corroborates source facts; and streams accurate, hallucination-free answers using local LLMs (Gemma 4 / Ollama) or Cloud API fallbacks.

---

## 🌟 Key Features

- 📂 **Multi-Format Ingestion**: Ingest raw typed notes, `.txt`, `.docx`, digital `.pdf` files, and `.jpg`/`.png` image uploads (with automated OCR extraction).
- 🔍 **Double-Net RAG Retrieval Engine**: Smart keyword search with 7-category campus concept boosting (exams, fees, lecturers, venues, schedules, grading, health) and back-of-book index suppression.
- ⚡ **Zero-Leak Grounded Corroboration**: Answers are generated strictly from ingested vault dumps with multi-source factual verification flags (`✅ Confirmed by multiple dumps`).
- 🧠 **Multi-Mode SSE Streaming**:
  - **⚡ Fast Mode**: Instant direct answers.
  - **📖 Detailed Mode**: Structured paragraphs and bullet points.
  - **🧠 Deep Thinking Mode**: Streams step-by-step reasoning tokens live.
- 🎥 **Cloud API Demo Switch**: Toggle instant cloud inference (Google Gemini API / OpenAI API) via `.env` for recording demo videos smoothly.

---

## 🛠️ Project Structure

```text
cyclup/
├── core/                   # Main Django app
│   ├── gemma_client.py     # Local Ollama Gemma 4 & Cloud API client
│   ├── ingestion.py        # PDF, DOCX, TXT, and Image OCR parsers
│   ├── models.py           # Dump & ChatMessage models
│   ├── views.py            # RAG search pipeline & SSE streaming endpoints
│   ├── urls.py             # App route mapping
│   ├── tests.py            # Automated test suite
│   └── templates/core/     # Modern dark glassmorphism SPA UI
├── cyclup/                 # Project configuration & settings
│   ├── settings.py         # Django settings (auto-loads .env)
│   └── urls.py             # Root URL routing
├── .env                    # Local environment variables & API keys
├── .env.example            # Environment configuration template
├── manage.py               # Django CLI utility
└── requirements.txt        # Python dependencies
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) (Optional if using Cloud API for demo recording)

### 2. Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/olajireyy/cyclup.git
   cd cyclup
   ```

2. **Set up virtual environment**:
   ```bash
   # On Windows
   python -m venv .venv
   .venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run migrations**:
   ```bash
   python manage.py migrate
   ```

---

## 🔑 Environment Configuration (`.env`)

A `.env` file has been created at the root directory:

```env
# ── Cyclup Environment Configuration ──

# Toggle Cloud API mode for fast demo video recording (true/false)
USE_CLOUD_API=false

# Google Gemini API Key (Recommended for instant responses during recording)
GEMINI_API_KEY=your_gemini_api_key_here

# OpenAI API Key (Alternative option)
OPENAI_API_KEY=

# Django Settings
SECRET_KEY=cyclup-hackathon-demo-key-not-for-production
DEBUG=True
```

---

## 🎥 Recording Demo Videos (Cloud API Switch)

If local CPU/GPU inference is slow on your machine when recording a video demo:

1. Open your [.env](file:///.env) file.
2. Add your **Google Gemini API Key**:
   ```env
   USE_CLOUD_API=true
   GEMINI_API_KEY=AIzaSy...
   ```
3. Start the server:
   ```bash
   python manage.py runserver
   ```
4. Responses will now stream back in **<1 second** for recording!

---

## 🏃 Running the Application

Start the local Django server:
```bash
python manage.py runserver
```

Open your browser and navigate to:  
👉 **`http://127.0.0.1:8000/`**

---

## 🧪 Running Automated Tests

Run the test suite to verify system integrity:
```bash
python manage.py test
```

---

## 📡 Core API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Main Glassmorphism SPA Interface |
| `/api/dump/text/` | `POST` | Ingest raw typed notes |
| `/api/dump/file/` | `POST` | Ingest `.pdf`, `.docx`, `.txt` files |
| `/api/dump/image/` | `POST` | Ingest image files via OCR |
| `/api/dumps/` | `GET` | List recent dumps in vault |
| `/api/dump/<id>/delete/` | `POST/DELETE` | Delete single dump |
| `/api/dumps/delete/bulk/` | `POST` | Bulk delete selected dumps |
| `/api/ask/` | `POST/GET` | Synchronous RAG query endpoint |
| `/api/ask/stream/` | `GET` | Real-time SSE streaming RAG query endpoint |
| `/api/chat/history/` | `GET` | Retrieve saved chat history |
| `/api/gemma/status/` | `GET` | Ollama / Gemma health check |

---

## 📄 License

MIT License. Developed for LASU campus intelligence automation.
