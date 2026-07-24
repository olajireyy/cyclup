import json
import re
import time
import logging
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET
from django.db.models import Q, Max
from .models import Dump, ChatMessage
from .gemma_client import ask_gemma, ask_gemma_generator, get_available_model
from .ingestion import parse_txt, parse_docx, parse_pdf, parse_image, detect_file_type

logger = logging.getLogger(__name__)


# =============================================================================
# Page View
# =============================================================================
def index_view(request):
    """Render the main single-page application."""
    return render(request, "core/index.html")


# =============================================================================
# Gemma / Ollama Health Check
# =============================================================================
@require_GET
def gemma_status(request):
    """
    GET /api/gemma/status/

    Pings the local Ollama server and checks whether gemma4:latest is available.
    Returns JSON:
        {"connected": true/false, "model": "gemma4:latest", ...}
    """
    import requests as http_requests

    OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

    try:
        resp = http_requests.get(OLLAMA_TAGS_URL, timeout=5)
        if resp.status_code != 200:
            return JsonResponse({
                "connected": False,
                "model": None,
                "error": f"Ollama returned HTTP {resp.status_code}",
            })

        data = resp.json()
        models = data.get("models", [])
        model_names = [m.get("name", "") for m in models]

        active_tag = get_available_model()
        gemma_found = any("gemma" in name.lower() for name in model_names)

        gemma_info = {}
        for m in models:
            if m.get("name") == active_tag or (not gemma_info and "gemma" in m.get("name", "").lower()):
                size_bytes = m.get("size", 0)
                size_gb = round(size_bytes / (1024 ** 3), 1) if size_bytes else None
                gemma_info = {
                    "model": m.get("name"),
                    "size_gb": size_gb,
                    "parameter_size": m.get("details", {}).get("parameter_size", ""),
                    "quantization": m.get("details", {}).get("quantization_level", ""),
                }
                if m.get("name") == active_tag:
                    break

        return JsonResponse({
            "connected": True,
            "gemma_loaded": gemma_found,
            "available_models": model_names,
            **gemma_info,
        })

    except http_requests.ConnectionError:
        return JsonResponse({
            "connected": False,
            "model": None,
            "error": "Ollama is not running on localhost:11434",
        })
    except Exception as exc:
        logger.exception("Error checking Gemma status")
        return JsonResponse({
            "connected": False,
            "model": None,
            "error": str(exc),
        })


# =============================================================================
# Dump API Views
# =============================================================================
@csrf_exempt
def dump_text(request):
    """Ingest raw typed text."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required."}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    raw_text = data.get("raw_text", "").strip()
    if not raw_text:
        return JsonResponse({"error": "raw_text is required."}, status=400)

    source_name = data.get("source_name", "").strip() or "Typed_Dump"
    course_code = data.get("course_code", "").strip() or None
    tags = data.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    dump = Dump.objects.create(
        raw_text=raw_text,
        source_name=source_name,
        source_type="text",
        course_code=course_code,
        tags=tags,
    )

    return JsonResponse({
        "status": "success",
        "message": f"Text dump '{dump.source_name}' saved.",
        "dump_id": dump.id,
    })


@csrf_exempt
def dump_file(request):
    """Ingest .txt, .docx, or .pdf file uploads."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required."}, status=405)

    uploaded_file = request.FILES.get("file")
    if not uploaded_file:
        return JsonResponse({"error": "No file uploaded."}, status=400)

    filename = uploaded_file.name
    file_type = detect_file_type(filename)
    course_code = request.POST.get("course_code", "").strip() or None
    tags = request.POST.getlist("tags")
    if not tags:
        tags_raw = request.POST.get("tags", "")
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []

    try:
        if file_type == "txt_file":
            text = parse_txt(uploaded_file)
            Dump.objects.create(
                raw_text=text,
                source_name=filename,
                source_type="txt_file",
                course_code=course_code,
                tags=tags,
            )
            return JsonResponse({
                "status": "success",
                "message": f"Text file '{filename}' ingested.",
                "pages": 1,
            })

        elif file_type == "docx":
            text = parse_docx(uploaded_file)
            Dump.objects.create(
                raw_text=text,
                source_name=filename,
                source_type="docx",
                course_code=course_code,
                tags=tags,
            )
            return JsonResponse({
                "status": "success",
                "message": f"Word document '{filename}' ingested.",
                "pages": 1,
            })

        elif file_type == "pdf":
            pages = parse_pdf(uploaded_file)
            if not pages:
                return JsonResponse({
                    "status": "error",
                    "message": (
                        "Scanned PDF detected! This PDF contains no extractable text. "
                        "Please upload direct photos of the pages using the Image Upload tool."
                    ),
                }, status=422)

            for page_data in pages:
                Dump.objects.create(
                    raw_text=page_data["text"],
                    source_name=filename,
                    source_type="pdf",
                    page_number=page_data["page_number"],
                    course_code=course_code,
                    tags=tags,
                )
            return JsonResponse({
                "status": "success",
                "message": f"PDF '{filename}' ingested ({len(pages)} pages).",
                "pages": len(pages),
            })

        else:
            return JsonResponse({
                "status": "error",
                "message": f"Unsupported file type: {filename}. Supported: .txt, .docx, .pdf",
            }, status=400)

    except Exception as exc:
        logger.exception("Error processing file '%s'", filename)
        return JsonResponse({
            "status": "error",
            "message": f"Failed to process '{filename}': {str(exc)}",
        }, status=500)


@csrf_exempt
def dump_image(request):
    """Ingest .jpg/.png image via OCR."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required."}, status=405)

    uploaded_file = request.FILES.get("file")
    if not uploaded_file:
        return JsonResponse({"error": "No image uploaded."}, status=400)

    filename = uploaded_file.name
    course_code = request.POST.get("course_code", "").strip() or None
    tags = request.POST.getlist("tags")
    if not tags:
        tags_raw = request.POST.get("tags", "")
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []

    try:
        text = parse_image(uploaded_file)
        if not text.strip():
            return JsonResponse({
                "status": "error",
                "message": f"No text could be extracted from '{filename}'. Try a clearer image.",
            }, status=422)

        dump = Dump.objects.create(
            raw_text=text,
            source_name=filename,
            source_type="image",
            course_code=course_code,
            tags=tags,
        )
        return JsonResponse({
            "status": "success",
            "message": f"Image '{filename}' OCR'd and saved ({len(text)} chars extracted).",
            "dump_id": dump.id,
            "extracted_preview": text[:200],
        })

    except Exception as exc:
        logger.exception("Error processing image '%s'", filename)
        return JsonResponse({
            "status": "error",
            "message": f"Failed to process image '{filename}': {str(exc)}",
        }, status=500)


@require_GET
def list_dumps(request):
    """Return recent dumps as JSON for the sidebar."""
    dumps = Dump.objects.values(
        "id", "source_name", "source_type", "page_number",
        "course_code", "tags", "created_at"
    ).order_by("-created_at")[:50]

    seen = {}
    results = []
    for d in dumps:
        key = (d["source_name"], d["source_type"])
        if d["source_type"] == "pdf" and key in seen:
            seen[key]["page_count"] += 1
            continue

        entry = {
            "id": d["id"],
            "source_name": d["source_name"],
            "source_type": d["source_type"],
            "page_number": d["page_number"],
            "course_code": d["course_code"],
            "tags": d["tags"] if isinstance(d["tags"], list) else [],
            "created_at": d["created_at"].isoformat(),
            "page_count": 1,
        }
        seen[key] = entry
        results.append(entry)

    return JsonResponse({"dumps": results})


@csrf_exempt
def delete_dump(request, dump_id):
    """Delete a single dump or all chunks matching its source_name."""
    if request.method not in ["DELETE", "POST"]:
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        target = Dump.objects.filter(id=dump_id).first()
        if not target:
            return JsonResponse({"error": "Dump not found"}, status=404)

        source_name = target.source_name
        deleted_count, _ = Dump.objects.filter(source_name=source_name).delete()
        return JsonResponse({"status": "success", "message": f"Deleted '{source_name}' ({deleted_count} entries)."})
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)


@csrf_exempt
def bulk_delete_dumps(request):
    """Delete multiple or all dumps by IDs or source_names."""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8")) if request.body else {}
        dump_ids = data.get("ids", [])
        source_names = data.get("source_names", [])

        deleted_total = 0
        if dump_ids:
            names_to_delete = Dump.objects.filter(id__in=dump_ids).values_list("source_name", flat=True)
            cnt, _ = Dump.objects.filter(source_name__in=names_to_delete).delete()
            deleted_total += cnt
        elif source_names:
            cnt, _ = Dump.objects.filter(source_name__in=source_names).delete()
            deleted_total += cnt

        return JsonResponse({"status": "success", "message": f"Bulk deleted {deleted_total} dump entries."})
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)


@require_GET
def list_chat_history(request):
    """Return stored chat conversation history as JSON."""
    messages = ChatMessage.objects.values(
        "id", "session_id", "user_query", "assistant_response",
        "mode", "status", "sources", "latency", "created_at"
    ).order_by("created_at")

    history = []
    for m in messages:
        history.append({
            "id": m["id"],
            "user_query": m["user_query"],
            "assistant_response": m["assistant_response"],
            "mode": m["mode"],
            "status": m["status"],
            "sources": m["sources"] if isinstance(m["sources"], list) else [],
            "latency": m["latency"],
            "created_at": m["created_at"].isoformat(),
        })

    return JsonResponse({"messages": history})


@csrf_exempt
def delete_chat_message(request, message_id):
    """Delete an individual chat message entry."""
    if request.method not in ["DELETE", "POST"]:
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        cnt, _ = ChatMessage.objects.filter(id=message_id).delete()
        if cnt == 0:
            return JsonResponse({"error": "Message not found"}, status=404)
        return JsonResponse({"status": "success", "message": f"Deleted chat message #{message_id}."})
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)


@csrf_exempt
def clear_chat_history(request):
    """Clear all chat history."""
    if request.method not in ["DELETE", "POST"]:
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        cnt, _ = ChatMessage.objects.all().delete()
        return JsonResponse({"status": "success", "message": f"Cleared {cnt} chat history messages."})
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)


# =============================================================================
# 1. Conversational Chatter Guard (0.0s Latency)
# =============================================================================
CHATTER_PATTERNS = [
    r"^\s*(hi|hello|hey|greetings|good\s+morning|good\s+afternoon|good\s+evening)\b",
    r"^\s*(thanks|thank\s+you|thx|thankyou)\b",
    r"^\s*(oh\s+wow\s+fast|wow|awesome|nice|great|cool|amazing)\b",
]

def check_chatter_guard(query_text: str):
    """
    Returns instant greeting/praise response if query matches conversational chatter.
    """
    cleaned = query_text.strip().lower()
    for pattern in CHATTER_PATTERNS:
        if re.search(pattern, cleaned):
            return "Glad to help! Let me know if you have any campus questions."
    return None


# =============================================================================
# 2. Agentic Document Metadata Tool (0.0s Latency)
# =============================================================================
def handle_metadata_tool(query_text: str):
    """
    Directly queries SQLite metadata for page counts or file listings with 0.0s latency.
    """
    cleaned = query_text.lower().strip()

    # Query for page count
    if any(phrase in cleaned for phrase in ["how many pages", "page count", "number of pages", "pages in"]):
        filenames = list(Dump.objects.values_list("source_name", flat=True).distinct())
        target_file = None
        for fn in filenames:
            fn_base = fn.lower().split(".")[0]
            if fn.lower() in cleaned or (len(fn_base) > 2 and fn_base in cleaned):
                target_file = fn
                break

        if target_file:
            max_page = Dump.objects.filter(source_name__iexact=target_file).aggregate(Max("page_number"))["page_number__max"]
            if max_page is not None:
                return f"'{target_file}' has {max_page} page(s) logged in the vault."
            else:
                total_chunks = Dump.objects.filter(source_name__iexact=target_file).count()
                return f"'{target_file}' is in the vault with {total_chunks} entry/entries."
        else:
            max_page_overall = Dump.objects.aggregate(Max("page_number"))["page_number__max"]
            if max_page_overall:
                return f"The maximum page count recorded across dumps in the vault is {max_page_overall} pages."
            return "Document page metadata is currently not available for the specified query."

    # Query for file listing
    if any(phrase in cleaned for phrase in ["what files", "list files", "files in vault", "files available", "what documents", "show files"]):
        files = [fn for fn in dict.fromkeys(Dump.objects.values_list("source_name", flat=True)) if fn]
        if files:
            file_list_str = "\n".join([f"• {fn}" for fn in files])
            return f"The vault currently contains {len(files)} file(s):\n\n{file_list_str}"
        else:
            return "The vault is currently empty. No files or dumps have been ingested yet."

    return None


# =============================================================================
# 3. Agentic Query Analyzer & 7-Category Concept Dictionary
# =============================================================================
CONCEPT_DICTIONARY = {
    "lecturers": {
        "triggers": ["teaches", "lecturer", "prof", "professor", "instructor", "teacher", "doctor", "dr", "faculty"],
        "boost_terms": ["lecturer", "prof", "professor", "teaches", "instructor", "faculty", "doctor", "dr"],
    },
    "venues": {
        "triggers": ["where", "location", "venue", "room", "hall", "lab", "auditorium", "building", "center", "theatre", "class"],
        "boost_terms": ["venue", "room", "hall", "lab", "auditorium", "building", "location", "theater", "theatre"],
    },
    "schedules": {
        "triggers": ["when", "time", "timetable", "schedule", "hours", "period", "days", "slot", "when is"],
        "boost_terms": ["timetable", "schedule", "hours", "time", "when", "slot", "period"],
    },
    "exams": {
        "triggers": ["test", "exam", "examination", "quiz", "malpractice", "expulsion", "cheating", "assessment"],
        "boost_terms": ["exam", "test", "malpractice", "expulsion", "quiz", "examination", "assessment"],
    },
    "fees": {
        "triggers": ["cost", "fee", "fees", "price", "payment", "tuition", "caution", "deposit", "money", "pay", "dues"],
        "boost_terms": ["fee", "fees", "caution", "deposit", "cost", "tuition", "payment", "dues", "price"],
    },
    "grading": {
        "triggers": ["gpa", "cgpa", "grade", "marks", "mark", "score", "attendance", "75%", "75 percent", "pass", "fail", "units"],
        "boost_terms": ["gpa", "cgpa", "grade", "attendance", "75%", "marks", "score", "grading", "units"],
    },
    "health": {
        "triggers": ["doctor", "nurse", "hospital", "clinic", "health", "ambulance", "security", "emergency", "medical"],
        "boost_terms": ["doctor", "ambulance", "security", "clinic", "health", "hospital", "medical", "nurse"],
    },
}

STARTERS_TO_STRIP = [
    r"^\s*could\s+you\s+please\s+tell\s+me\s*",
    r"^\s*can\s+you\s+please\s+tell\s+me\s*",
    r"^\s*can\s+you\s+tell\s+me\s*",
    r"^\s*could\s+you\s+tell\s+me\s*",
    r"^\s*please\s+tell\s+me\s*",
    r"^\s*tell\s+me\s+about\s*",
    r"^\s*tell\s+me\s*",
    r"^\s*i\s+want\s+to\s+know\s*",
    r"^\s*do\s+you\s+know\s*",
    r"^\s*what\s+is\s*",
    r"^\s*where\s+is\s*",
    r"^\s*when\s+is\s*",
    r"^\s*who\s+is\s*",
]

FILLERS_TO_STRIP = [r"\blasu\b", r"\bcampus\b", r"\bschool\b", r"\buniversity\b"]

def analyze_user_query(user_question: str):
    """
    Strips conversational starters and domain fillers, and maps synonyms across 7 campus categories.
    """
    cleaned = user_question.strip()

    # Strip conversational starters
    for starter_pat in STARTERS_TO_STRIP:
        cleaned = re.sub(starter_pat, "", cleaned, flags=re.IGNORECASE).strip()

    # Strip domain fillers
    for filler_pat in FILLERS_TO_STRIP:
        cleaned = re.sub(filler_pat, "", cleaned, flags=re.IGNORECASE).strip()

    # Strip trailing punctuation marks
    cleaned = re.sub(r"[\?\!\.\,]+$", "", cleaned).strip()

    tokens = [t for t in re.split(r"\W+", cleaned.lower()) if len(t) > 1]

    matched_categories = []
    concept_boost_terms = set()

    raw_q_lower = user_question.lower()
    for cat_name, cat_data in CONCEPT_DICTIONARY.items():
        for trigger in cat_data["triggers"]:
            if trigger in raw_q_lower or trigger in tokens:
                matched_categories.append(cat_name)
                concept_boost_terms.update(cat_data["boost_terms"])
                break

    return {
        "cleaned_query": cleaned,
        "tokens": tokens,
        "matched_categories": matched_categories,
        "concept_boost_terms": list(concept_boost_terms),
    }


# =============================================================================
# 4. Double-Net Search with Concept Boosts
# =============================================================================
INDEX_PAGE_PATTERNS = [
    r"\bsee\s+also\b",
    r"\bindex\b",
    r"([A-Z][a-z]+,\s*\d+([\s,-]+\d+)+)",
]

def is_back_of_book_index(raw_text: str) -> bool:
    """
    Returns True if chunk represents a back-of-book index page.
    """
    text_lower = raw_text.lower()
    if re.search(r"\bsee\s+also\b", text_lower):
        return True

    if "index" in text_lower and len(re.findall(r"\b\d{1,3}\b", raw_text)) > 10:
        return True

    return False


def double_net_search(analysis_result: dict, raw_user_question: str):
    """
    Queries Django ORM using Q objects across raw_text, source_name, course_code, and tags.
    Applies back-of-book suppression and concept/filename boosts.
    """
    tokens = analysis_result["tokens"]
    concept_boost_terms = analysis_result["concept_boost_terms"]
    matched_categories = analysis_result["matched_categories"]

    if not tokens:
        tokens = [t for t in re.split(r"\W+", raw_user_question.lower()) if len(t) > 1]

    if not tokens:
        return []

    # Build Double-Net ORM Q filter
    q_filter = Q()
    for token in tokens:
        q_filter |= Q(raw_text__icontains=token)
        q_filter |= Q(source_name__icontains=token)
        q_filter |= Q(course_code__icontains=token)
        q_filter |= Q(tags__icontains=token)

    for boost_term in concept_boost_terms:
        q_filter |= Q(raw_text__icontains=boost_term)
        q_filter |= Q(tags__icontains=boost_term)

    # Execute DB Query
    candidate_dumps = Dump.objects.filter(q_filter).distinct()

    scored_chunks = []

    # Scoring logic per chunk
    for dump in candidate_dumps:
        # Suppress back-of-book index pages
        if is_back_of_book_index(dump.raw_text):
            continue

        score = 0
        text_lower = dump.raw_text.lower()
        source_lower = dump.source_name.lower()
        course_lower = (dump.course_code or "").lower()

        # Token matching points
        for token in tokens:
            if token in text_lower:
                score += 25
            if token in source_lower:
                score += 35
            if course_lower and token in course_lower:
                score += 40

        # +70 pts concept boost for lecturer / venue / fee concept categories
        boost_eligible_categories = {"lecturers", "venues", "fees", "schedules", "exams", "grading", "health"}
        if any(cat in boost_eligible_categories for cat in matched_categories):
            for b_term in concept_boost_terms:
                if b_term in text_lower or b_term in source_lower:
                    score += 70
                    break

        # +50 pts source_name filename match
        for token in tokens:
            if len(token) > 2 and token in source_lower:
                score += 50
                break

        if score > 0:
            scored_chunks.append((score, dump))

    # Sort descending by score
    scored_chunks.sort(key=lambda item: item[0], reverse=True)

    if scored_chunks:
        top_source = scored_chunks[0][1].source_name
        all_top_source_chunks = [item for item in scored_chunks if item[1].source_name == top_source]
        if len(all_top_source_chunks) > 1:
            all_top_source_chunks.sort(key=lambda x: (x[1].page_number or 0, x[1].id))
            return all_top_source_chunks[:10]

        top_score = scored_chunks[0][0]
        if top_score >= 80:
            return scored_chunks[:3]
        else:
            return scored_chunks[:5]

    return []


# =============================================================================
# 5. Master Grounded Corroborator Execution
# =============================================================================
MASTER_GROUNDED_SYSTEM_PROMPT_FAST = """You are an intelligent campus information AI assistant for LASU. Synthesize a natural, clear, and concise response to the user's question using ONLY the source context below.

TYPO & SYNTHESIS INSTRUCTIONS:
- Correct any minor spelling or typos in the user's question gracefully without mentioning them.
- Do NOT just copy raw text verbatim; explain naturally in your own words while staying 100% faithful to the context.

CORROBORATION RULES:
1. Agreement: If multiple sources provide the exact same answer, end your response by stating: "✅ Confirmed by multiple dumps."
2. Conflict: If sources contradict each other (e.g. one says exam is Monday, another says Wednesday), state BOTH versions and cite Source numbers (e.g. Source 1 vs Source 2). Do not guess.
3. Missing Data: If sources do not contain the answer, say EXACTLY: "I don't have that information in what's been dumped here."

Context:
{context}

Question: {question}

Answer:"""

MASTER_GROUNDED_SYSTEM_PROMPT_DETAILED = """You are an intelligent campus information AI assistant for LASU. Synthesize a thorough, detailed response using clear structured paragraphs and bullet points based ONLY on the source context below.

TYPO & SYNTHESIS INSTRUCTIONS:
- Correct any minor spelling or typos in the user's question gracefully without mentioning them.
- Explain the concepts fluidly and logically in your own words while preserving strict factual accuracy from the sources.

CORROBORATION RULES:
1. Agreement: If multiple sources provide the exact same answer, end your response by stating: "✅ Confirmed by multiple dumps."
2. Conflict: If sources contradict each other (e.g. one says exam is Monday, another says Wednesday), state BOTH versions and cite Source numbers (e.g. Source 1 vs Source 2). Do not guess.
3. Missing Data: If sources do not contain the answer, say EXACTLY: "I don't have that information in what's been dumped here."

Context:
{context}

Question: {question}

Detailed Answer:"""

MASTER_GROUNDED_SYSTEM_PROMPT = MASTER_GROUNDED_SYSTEM_PROMPT_FAST

def execute_master_corroborator(scored_chunks, user_question: str, provider: str = "auto"):
    """
    Formats retrieved chunks into Master Grounded Prompt and invokes Gemma 4 / Cloud API.
    """
    if not scored_chunks:
        return {
            "answer": "I don't have that information in what's been dumped here.",
            "status": "refusal",
            "top_score": 0,
            "sources": [],
        }

    top_score = scored_chunks[0][0]

    # Format context chunks
    context_blocks = []
    sources_meta = []
    for idx, (score, dump) in enumerate(scored_chunks, 1):
        src_label = f"Source {idx}: {dump.source_name}"
        if dump.page_number:
            src_label += f" (Page {dump.page_number})"
        if dump.course_code:
            src_label += f" [{dump.course_code}]"

        context_blocks.append(f"--- {src_label} ---\n{dump.raw_text}")
        sources_meta.append({
            "source_name": dump.source_name,
            "page_number": dump.page_number,
            "course_code": dump.course_code,
            "score": score,
        })

    context_str = "\n\n".join(context_blocks)
    prompt = MASTER_GROUNDED_SYSTEM_PROMPT.format(context=context_str, question=user_question)

    try:
        raw_answer = ask_gemma(prompt, max_tokens=150, provider=provider)
        if "Source 1 vs Source 2" in raw_answer or "vs" in raw_answer.lower():
            status = "conflict"
        elif "don't have that information" in raw_answer.lower() or "don't have" in raw_answer.lower():
            status = "refusal"
        else:
            status = "grounded"

        return {
            "answer": raw_answer,
            "status": status,
            "top_score": top_score,
            "sources": sources_meta,
        }
    except Exception as exc:
        logger.error("Error calling ask_gemma: %s", exc)
        return {
            "answer": f"System error calling model: {str(exc)}",
            "status": "error",
            "top_score": top_score,
            "sources": sources_meta,
        }


# =============================================================================
# Main Django View: ask_question(request)
# =============================================================================
@csrf_exempt
def ask_question(request):
    """
    Offline Q&A Engine without vector DBs.
    Follows 5-step execution pipeline:
    1. Conversational Chatter Guard (0.0s Latency)
    2. Agentic Document Metadata Tool (0.0s Latency)
    3. Agentic Query Analyzer & 7-Category Concept Dictionary
    4. Double-Net Search with Concept Boosts & Precision Rule
    5. Master Grounded Corroborator Execution
    """
    if request.method not in ["POST", "GET"]:
        return JsonResponse({"error": "Method not allowed. Use POST or GET."}, status=405)

    user_question = ""
    provider = "auto"
    if request.method == "POST":
        if request.content_type == "application/json":
            try:
                data = json.loads(request.body.decode("utf-8"))
                user_question = data.get("question", "") or data.get("q", "")
                provider = data.get("provider", "auto")
            except json.JSONDecodeError:
                return JsonResponse({"error": "Invalid JSON body."}, status=400)
        else:
            user_question = request.POST.get("question", "") or request.POST.get("q", "")
            provider = request.POST.get("provider", "auto")
    else:
        user_question = request.GET.get("question", "") or request.GET.get("q", "")
        provider = request.GET.get("provider", "auto")

    user_question = user_question.strip()
    if not user_question:
        return JsonResponse({"error": "Missing 'question' parameter."}, status=400)

    # STEP 1: Conversational Chatter Guard (0.0s Latency)
    chatter_response = check_chatter_guard(user_question)
    if chatter_response:
        return JsonResponse({
            "answer": chatter_response,
            "status": "chatter_guard",
            "top_score": 0,
            "sources": [],
            "latency": "0.0s",
        })

    # STEP 2: Agentic Document Metadata Tool (0.0s Latency)
    metadata_response = handle_metadata_tool(user_question)
    if metadata_response:
        return JsonResponse({
            "answer": metadata_response,
            "status": "metadata",
            "top_score": 0,
            "sources": [],
            "latency": "0.0s",
        })

    # STEP 3: Agentic Query Analyzer & 7-Category Concept Dictionary
    analysis_result = analyze_user_query(user_question)

    # STEP 4: Double-Net Search with Concept Boosts & Back-of-Book Suppression
    scored_chunks = double_net_search(analysis_result, user_question)

    # STEP 5: Master Grounded Corroborator Execution
    result = execute_master_corroborator(scored_chunks, user_question, provider=provider)
    result["analysis"] = {
        "cleaned_query": analysis_result["cleaned_query"],
        "matched_categories": analysis_result["matched_categories"],
    }

    return JsonResponse(result)


# =============================================================================
# Streaming Q&A Endpoint: ask_question_stream(request)
# =============================================================================
@csrf_exempt
def ask_question_stream(request):
    """
    Real-time Server-Sent Events (SSE) streaming endpoint.
    Yields chunks:
        event: metadata -> JSON with sources, status, top_score
        event: thinking -> JSON with thinking token
        event: answer   -> JSON with answer token
        event: done     -> JSON with final latency
    """
    start_time = time.time()
    user_question = request.GET.get("question", "") or request.GET.get("q", "")
    mode = request.GET.get("mode", "fast")
    provider = request.GET.get("provider", "auto")

    if not user_question and request.method == "POST":
        try:
            data = json.loads(request.body.decode("utf-8"))
            user_question = data.get("question", "") or data.get("q", "")
            mode = data.get("mode", "fast")
            provider = data.get("provider", "auto")
        except json.JSONDecodeError:
            pass

    user_question = user_question.strip()

    def event_stream():
        if not user_question:
            yield "event: error\ndata: {\"error\": \"Missing question parameter\"}\n\n"
            return

        # 1. Chatter Guard
        chatter_resp = check_chatter_guard(user_question)
        if chatter_resp:
            elapsed = round(time.time() - start_time, 3)
            chat_obj = ChatMessage.objects.create(
                user_query=user_question,
                assistant_response=chatter_resp,
                mode=mode,
                status="chatter_guard",
                sources=[],
                latency=f"{elapsed}s",
            )
            meta = json.dumps({"status": "chatter_guard", "top_score": 0, "sources": [], "msg_id": chat_obj.id})
            yield f"event: metadata\ndata: {meta}\n\n"
            ans = json.dumps({"content": chatter_resp})
            yield f"event: answer\ndata: {ans}\n\n"
            done_data = json.dumps({"latency": f"{elapsed}s", "msg_id": chat_obj.id})
            yield f"event: done\ndata: {done_data}\n\n"
            return

        # 2. Metadata Tool
        meta_resp = handle_metadata_tool(user_question)
        if meta_resp:
            elapsed = round(time.time() - start_time, 3)
            chat_obj = ChatMessage.objects.create(
                user_query=user_question,
                assistant_response=meta_resp,
                mode=mode,
                status="metadata",
                sources=[],
                latency=f"{elapsed}s",
            )
            meta = json.dumps({"status": "metadata", "top_score": 0, "sources": [], "msg_id": chat_obj.id})
            yield f"event: metadata\ndata: {meta}\n\n"
            ans = json.dumps({"content": meta_resp})
            yield f"event: answer\ndata: {ans}\n\n"
            done_data = json.dumps({"latency": f"{elapsed}s", "msg_id": chat_obj.id})
            yield f"event: done\ndata: {done_data}\n\n"
            return

        # 3. Analyze & Double-Net Search
        analysis_res = analyze_user_query(user_question)
        scored_chunks = double_net_search(analysis_res, user_question)

        if not scored_chunks:
            elapsed = round(time.time() - start_time, 2)
            refusal_txt = "I don't have that information in what's been dumped here."
            chat_obj = ChatMessage.objects.create(
                user_query=user_question,
                assistant_response=refusal_txt,
                mode=mode,
                status="refusal",
                sources=[],
                latency=f"{elapsed}s",
            )
            meta = json.dumps({"status": "refusal", "top_score": 0, "sources": [], "msg_id": chat_obj.id})
            yield f"event: metadata\ndata: {meta}\n\n"
            ans = json.dumps({"content": refusal_txt})
            yield f"event: answer\ndata: {ans}\n\n"
            done_data = json.dumps({"latency": f"{elapsed}s", "msg_id": chat_obj.id})
            yield f"event: done\ndata: {done_data}\n\n"
            return

        # Format sources metadata
        top_score = scored_chunks[0][0]
        context_blocks = []
        sources_meta = []
        for idx, (score, dump) in enumerate(scored_chunks, 1):
            src_label = f"Source {idx}: {dump.source_name}"
            if dump.page_number:
                src_label += f" (Page {dump.page_number})"
            if dump.course_code:
                src_label += f" [{dump.course_code}]"

            context_blocks.append(f"--- {src_label} ---\n{dump.raw_text}")
            sources_meta.append({
                "source_name": dump.source_name,
                "page_number": dump.page_number,
                "course_code": dump.course_code,
                "score": score,
            })

        context_str = "\n\n".join(context_blocks)
        if mode == "detailed":
            prompt_tmpl = MASTER_GROUNDED_SYSTEM_PROMPT_DETAILED
            think_flag = False
        elif mode == "thinking":
            prompt_tmpl = MASTER_GROUNDED_SYSTEM_PROMPT_DETAILED
            think_flag = True
        else:  # 'fast'
            prompt_tmpl = MASTER_GROUNDED_SYSTEM_PROMPT_FAST
            think_flag = False

        prompt = prompt_tmpl.format(context=context_str, question=user_question)

        meta = json.dumps({
            "status": "grounded",
            "top_score": top_score,
            "sources": sources_meta,
            "mode": mode,
            "provider": provider,
        })
        yield f"event: metadata\ndata: {meta}\n\n"

        # Stream tokens live from Gemma & accumulate response
        full_answer_acc = []
        thinking_acc = []
        try:
            for chunk in ask_gemma_generator(prompt, think=think_flag, max_tokens=1000, provider=provider):

                event_type = chunk.get("type", "answer")
                content = chunk.get("content", "")
                if event_type == "answer":
                    full_answer_acc.append(content)
                elif event_type == "thinking":
                    thinking_acc.append(content)
                elif event_type == "error":
                    logger.error("Error from gemma generator: %s", content)
                    err_json = json.dumps({"error": content})
                    yield f"event: error\ndata: {err_json}\n\n"
                    return
                token_json = json.dumps({"content": content})
                yield f"event: {event_type}\ndata: {token_json}\n\n"
        except Exception as exc:
            logger.exception("Error in ask_question_stream generator")
            err_json = json.dumps({"error": str(exc)})
            yield f"event: error\ndata: {err_json}\n\n"
            return

        complete_answer = "".join(full_answer_acc).strip()
        if not complete_answer and thinking_acc:
            complete_answer = "".join(thinking_acc).strip()
            token_json = json.dumps({"content": complete_answer})
            yield f"event: answer\ndata: {token_json}\n\n"

        if not complete_answer:
            complete_answer = "Could not generate an answer from the retrieved sources."
            token_json = json.dumps({"content": complete_answer})
            yield f"event: answer\ndata: {token_json}\n\n"

        elapsed = round(time.time() - start_time, 2)

        chat_obj = ChatMessage.objects.create(
            user_query=user_question,
            assistant_response=complete_answer,
            mode=mode,
            status="grounded",
            sources=sources_meta,
            latency=f"{elapsed}s",
        )

        done_data = json.dumps({"latency": f"{elapsed}s", "msg_id": chat_obj.id})
        yield f"event: done\ndata: {done_data}\n\n"

    response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response

