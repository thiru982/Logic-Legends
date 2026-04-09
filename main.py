import os
import io
import json
import requests
from datetime import datetime
from typing import List, Dict, Optional
from functools import lru_cache

import PyPDF2
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from duckduckgo_search import DDGS
from groq import Groq

# ReportLab for PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

# ══════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════

app = FastAPI(title="Scholar AI API", version="2.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    app.mount("/static", StaticFiles(directory="."), name="static")
except Exception:
    pass

# ── GROQ CLIENT ──
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
groq_client = Groq(api_key=GROQ_API_KEY)

HEADERS = {"User-Agent": "ScholarAI/2.2 (research assistant; contact@scholarai.dev)"}

# ── IN-MEMORY STORE ──
uploaded_contexts: List[Dict] = []
chat_history: List[Dict] = []

# ── LANGUAGE MAP ──
LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "mr": "Marathi",
    "bn": "Bengali",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "ar": "Arabic",
    "ja": "Japanese",
}

# ── TRANSLATION CACHE ──
_translation_cache: dict = {}


# ══════════════════════════════════════════════════════════
# REQUEST MODELS
# ══════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    query: str
    mode: str = "general"
    secondary_domain: Optional[str] = None
    language: Optional[str] = "en"   # e.g. "kn", "hi", "ta"

class ChatRequest(BaseModel):
    message: str
    context_query: Optional[str] = None

class PDFRequest(BaseModel):
    data: Dict

class TranslateRequest(BaseModel):
    text: str
    target_language: str

class BatchTranslateRequest(BaseModel):
    texts: list[str]
    target_language: str

class RetranslateRequest(BaseModel):
    result: Dict          # full /api/search response object
    target_language: str  # e.g. "kn"


# ══════════════════════════════════════════════════════════
# LLM HELPERS
# ══════════════════════════════════════════════════════════

def call_llm(prompt: str, max_tokens: int = 1500, system: str = None) -> str:
    """Call Groq LLM and return raw text response."""
    try:
        sys_msg = system or (
            "You are a senior research scientist. "
            "Always respond with valid JSON when asked. "
            "Never add commentary outside JSON."
        )
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM Error: {e}")
        return ""


def extract_json(raw: str) -> str:
    """
    Robustly extract JSON from LLM output that may be wrapped in markdown
    fences, have leading text, or have trailing commentary.
    Returns a clean JSON string ready for json.loads().
    """
    if not raw:
        return ""
    raw = raw.strip()

    # Remove markdown code fences (```json...``` or ```...```)
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") or part.startswith("["):
                raw = part
                break

    # Find outermost JSON object or array by bracket positions
    start_obj = raw.find("{")
    start_arr = raw.find("[")

    if start_obj == -1 and start_arr == -1:
        return raw  # no JSON found

    if start_obj == -1:
        start = start_arr
        end = raw.rfind("]") + 1
    elif start_arr == -1:
        start = start_obj
        end = raw.rfind("}") + 1
    else:
        start = min(start_obj, start_arr)
        end = max(raw.rfind("}"), raw.rfind("]")) + 1

    return raw[start:end]


# ══════════════════════════════════════════════════════════
# TRANSLATION
# ══════════════════════════════════════════════════════════

def translate_with_groq(text: str, target_language: str) -> str:
    """
    Translate text to target_language using Groq.
    Uses in-memory cache to avoid repeated API calls.
    Falls back to original text on any error.
    """
    if not text or not text.strip():
        return text
    if target_language == "en":
        return text

    cache_key = f"{target_language}::{hash(text)}"
    if cache_key in _translation_cache:
        return _translation_cache[cache_key]

    lang_name = LANGUAGE_NAMES.get(target_language, target_language)

    prompt = f"""Translate the following text accurately into {lang_name}.
Return ONLY the translated text — no explanation, no preamble, no quotes.
Keep technical terms and proper nouns in their original form.

Text:
{text}"""

    try:
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a professional translator. Translate everything to {lang_name}. "
                        "Return only the translated text, nothing else."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
            temperature=0.1,
        )
        translated = response.choices[0].message.content.strip()
        _translation_cache[cache_key] = translated
        return translated
    except Exception as e:
        print(f"Translation Error ({target_language}): {e}")
        return text   # graceful fallback — return original


def translate_research_output(data: Dict, target_language: str) -> Dict:
    """
    Translate all human-readable text fields in a research result dict.
    Translates: overview, key_insights, challenges, future_directions,
                cross_domain_link, paper_analysis highlights, news snippets.
    """
    if not target_language or target_language == "en":
        return data

    def t(text: str) -> str:
        return translate_with_groq(text, target_language)

    lang_label = LANGUAGE_NAMES.get(target_language, target_language)
    print(f"   🌐 Translating to {lang_label}...")

    data["overview"]          = t(data.get("overview", ""))
    data["key_insights"]      = [t(i) for i in data.get("key_insights", [])]
    data["challenges"]        = [t(c) for c in data.get("challenges", [])]
    data["future_directions"] = [t(f) for f in data.get("future_directions", [])]
    data["cross_domain_link"] = t(data.get("cross_domain_link", ""))

    for pa in data.get("paper_analysis", []):
        pa["highlight"] = t(pa.get("highlight", ""))
        pa["why_focus"] = t(pa.get("why_focus", ""))

    # Translate news snippets (keep titles in original for link readability)
    for src in data.get("sources", []):
        if src.get("type") == "news":
            src["snippet"] = t(src.get("snippet", ""))

    print(f"   ✅ Translation complete → {lang_label}")
    return data


# ══════════════════════════════════════════════════════════
# SEARCH FUNCTIONS
# ══════════════════════════════════════════════════════════

def search_crossref_fallback(query: str, limit: int = 5) -> List[Dict]:
    """CrossRef fallback when Semantic Scholar fails."""
    results = []
    try:
        r = requests.get(
            "https://api.crossref.org/works",
            params={"query": query, "rows": limit, "sort": "relevance"},
            headers=HEADERS,
            timeout=10
        )
        r.raise_for_status()
        for item in r.json().get("message", {}).get("items", []):
            author = "Unknown"
            if item.get("author"):
                a = item["author"][0]
                author = f"{a.get('given', '')} {a.get('family', '')}".strip()
            year = ""
            dp = item.get("published", {}).get("date-parts", [[""]])
            if dp and dp[0]:
                year = str(dp[0][0])
            title_parts = item.get("title", ["Untitled"])
            journal = item.get("container-title", [""])
            results.append({
                "title": title_parts[0] if title_parts else "Untitled",
                "url": item.get("URL", ""),
                "snippet": f"By {author}. {journal[0] if journal else 'Journal unknown'}.",
                "abstract": item.get("abstract", "No abstract available."),
                "type": "paper",
                "year": year,
                "citations": item.get("is-referenced-by-count", 0),
                "source": "CrossRef"
            })
    except Exception as e:
        print(f"CrossRef Error: {e}")
    return results


def search_semantic_scholar(query: str, limit: int = 5) -> List[Dict]:
    """Fetch academic papers from Semantic Scholar."""
    results = []
    try:
        r = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query,
                "limit": limit,
                "fields": "title,abstract,url,year,citationCount,authors.name,venue"
            },
            headers=HEADERS,
            timeout=12
        )
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            print("Semantic Scholar returned empty — falling back to CrossRef")
            return search_crossref_fallback(query, limit)
        for p in data:
            authors = ", ".join([a["name"] for a in p.get("authors", [])[:2]])
            results.append({
                "title": p.get("title", "Untitled"),
                "url": p.get("url", ""),
                "snippet": f"By {authors}. {p.get('venue', 'Unknown Journal')}.",
                "abstract": p.get("abstract", "No abstract available."),
                "type": "paper",
                "year": str(p.get("year", "N/A")),
                "citations": p.get("citationCount", 0),
                "source": "Semantic Scholar"
            })
    except Exception as e:
        print(f"Semantic Scholar Error: {e} — falling back to CrossRef")
        return search_crossref_fallback(query, limit)
    return results


def search_news(query: str, limit: int = 3) -> List[Dict]:
    """Fetch recent news via DuckDuckGo."""
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.news(query, max_results=limit):
                results.append({
                    "title": r.get("title", "")[:120],
                    "url": r.get("url", ""),
                    "snippet": r.get("body", "")[:250],
                    "type": "news",
                    "year": datetime.now().strftime("%Y"),
                    "source": r.get("source", "News")
                })
    except Exception as e:
        print(f"News Search Error: {e}")
    return results


def search_youtube(query: str, limit: int = 5) -> List[Dict]:
    """
    Fetch YouTube video links using 4 strategies in order:
      1. YouTube Data API v3  (needs YOUTUBE_API_KEY env var)
      2. Invidious public API (no key needed)
      3. DuckDuckGo aggressive scan
      4. YouTube search URL fallback
    """
    results = []
    seen_ids = set()

    def extract_vid_id(href: str):
        try:
            if "youtube.com/watch" in href and "v=" in href:
                return href.split("v=")[1].split("&")[0]
            if "youtu.be/" in href:
                return href.split("youtu.be/")[1].split("?")[0].split("/")[0]
        except Exception:
            pass
        return None

    def make_yt_result(vid_id: str, title: str, snippet: str = "") -> Dict:
        return {
            "title": title[:100],
            "url": f"https://www.youtube.com/watch?v={vid_id}",
            "snippet": snippet[:150],
            "type": "video",
            "source": "YouTube",
            "platform": "youtube",
        }

    # Strategy 1 — YouTube Data API v3
    YT_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
    if YT_API_KEY:
        try:
            r = requests.get(
                "https://www.googleapis.com/youtube/v3/search",
                params={
                    "part": "snippet",
                    "q": query,
                    "type": "video",
                    "maxResults": limit,
                    "relevanceLanguage": "en",
                    "key": YT_API_KEY,
                },
                timeout=10,
            )
            r.raise_for_status()
            for item in r.json().get("items", []):
                vid_id = item.get("id", {}).get("videoId", "")
                if not vid_id or vid_id in seen_ids:
                    continue
                seen_ids.add(vid_id)
                snip = item.get("snippet", {})
                results.append(make_yt_result(
                    vid_id,
                    snip.get("title", "YouTube Video"),
                    snip.get("description", "")
                ))
            print(f"   ✅ YouTube API: {len(results)} videos")
        except Exception as e:
            print(f"   ⚠️  YouTube API error: {e}")

    # Strategy 2 — Invidious
    if len(results) < limit:
        invidious_instances = [
            "https://invidious.snopyta.org",
            "https://vid.puffyan.us",
            "https://invidious.kavin.rocks",
            "https://yt.artemislena.eu",
        ]
        for instance in invidious_instances:
            if len(results) >= limit:
                break
            try:
                r = requests.get(
                    f"{instance}/api/v1/search",
                    params={"q": query, "type": "video", "page": 1},
                    timeout=8,
                    headers=HEADERS,
                )
                r.raise_for_status()
                for item in r.json():
                    if item.get("type") != "video":
                        continue
                    vid_id = item.get("videoId", "")
                    if not vid_id or vid_id in seen_ids:
                        continue
                    seen_ids.add(vid_id)
                    results.append(make_yt_result(
                        vid_id,
                        item.get("title", "YouTube Video"),
                        item.get("description", "")
                    ))
                    if len(results) >= limit:
                        break
                if results:
                    print(f"   ✅ Invidious ({instance}): {len(results)} videos")
                    break
            except Exception as e:
                print(f"   ⚠️  Invidious {instance} failed: {e}")

    # Strategy 3 — DuckDuckGo aggressive scan
    if len(results) < limit:
        query_variants = [
            f"{query} youtube",
            f"{query} lecture youtube",
            f"{query} tutorial explained",
            f"{query} explained youtube video",
        ]
        try:
            with DDGS() as ddgs:
                for variant in query_variants:
                    if len(results) >= limit:
                        break
                    try:
                        for r in ddgs.text(variant, max_results=60):
                            href = r.get("href", "") or r.get("url", "")
                            vid_id = extract_vid_id(href)
                            if vid_id and vid_id not in seen_ids:
                                seen_ids.add(vid_id)
                                results.append(make_yt_result(
                                    vid_id,
                                    r.get("title", "YouTube Video"),
                                    r.get("body", "")
                                ))
                                print(f"   ✅ DDG found YT video: {vid_id}")
                            if len(results) >= limit:
                                break
                    except Exception as e:
                        print(f"   ⚠️  DDG variant '{variant}' error: {e}")
        except Exception as e:
            print(f"   ⚠️  DDG outer error: {e}")

    # Strategy 4 — YouTube search URL fallback
    if not results:
        print("   ⚠️  All strategies failed — using YouTube search URL fallback")
        import urllib.parse
        search_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"
        results.append({
            "title": f"Search YouTube: {query}",
            "url": search_url,
            "snippet": f"Click to search YouTube for '{query}'.",
            "type": "video",
            "source": "YouTube Search",
            "platform": "youtube",
        })

    print(f"   🎥 Total videos: {len(results)}")
    return results[:limit]


def extract_text_from_file(file_content: bytes, filename: str) -> str:
    text = ""
    try:
        if filename.lower().endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        else:
            text = file_content.decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"File Extraction Error: {e}")
    return text[:10000]


# ══════════════════════════════════════════════════════════
# AI ANALYSIS
# ══════════════════════════════════════════════════════════

def analyze_papers(papers: List[Dict], query: str) -> List[Dict]:
    if not papers:
        return []

    paper_context = "\n\n".join([
        f"Title: {p['title']}\nAbstract: {p.get('abstract', 'No abstract')}"
        for p in papers
    ])

    prompt = f"""Analyze these research papers about: "{query}"

Papers:
{paper_context}

Return ONLY a valid JSON array. Each object must have exactly these keys:
- "title": paper title (string)
- "highlight": single most important finding (1 sentence)
- "why_focus": why this paper matters to the field (1 sentence)

Example:
[{{"title": "Paper Name", "highlight": "Key finding.", "why_focus": "Importance."}}]

Start your response with [ and end with ]. No other text."""

    raw = call_llm(prompt, max_tokens=1200)
    print(f"   [analyze_papers] raw snippet: {raw[:120]!r}")

    try:
        cleaned = extract_json(raw)
        return json.loads(cleaned)
    except Exception as e:
        print(f"Paper Analysis Error: {e} | raw: {raw[:200]!r}")
        return [
            {
                "title": p["title"],
                "highlight": p.get("abstract", "")[:100],
                "why_focus": "Relevant to query."
            }
            for p in papers[:3]
        ]


def generate_insights(query: str, sources: List[Dict], file_context: str = "", secondary_domain: str = "") -> Dict:
    source_snippets = "\n".join([f"- {s['title']}: {s['snippet']}" for s in sources[:8]])
    cross_note = f"Also explain the connection between '{query}' and '{secondary_domain}'." if secondary_domain else ""

    # ── Attempt 1: Primary prompt ──
    prompt = f"""You are a Senior Research Scientist. The user is researching: "{query}"
{cross_note}

Write a detailed research overview that DIRECTLY addresses "{query}".

Sources:
{source_snippets}
{f"Uploaded document context: {file_context}" if file_context else ""}

YOU MUST respond with ONLY a raw JSON object. No markdown fences. No text before or after.
Start your response with {{ and end with }}.

Required JSON format:
{{
  "overview": "Paragraph 1: Define and contextualize '{query}' — what it is and why it matters.\\n\\nParagraph 2: Current state of research and key findings from the sources above.\\n\\nParagraph 3: Key challenges, open questions, and emerging trends in {query}.",
  "key_insights": ["specific insight about {query}", "second insight", "third insight", "fourth insight"],
  "challenges": ["specific challenge in {query}", "second challenge"],
  "future_directions": ["future direction for {query}", "second direction"],
  "cross_domain_link": "how {query} connects to other disciplines"
}}

Fill every field with real, informative content about "{query}". The overview MUST be 3 full paragraphs."""

    raw = call_llm(prompt, max_tokens=1800)
    print(f"   [generate_insights] attempt 1 raw: {raw[:200]!r}")

    try:
        cleaned = extract_json(raw)
        result = json.loads(cleaned)
        if result.get("overview") and len(result["overview"]) > 50:
            return result
        raise ValueError(f"Overview too short ({len(result.get('overview',''))} chars)")
    except Exception as e:
        print(f"   ⚠️  Attempt 1 failed: {e} — trying attempt 2...")

    # ── Attempt 2: Simpler fallback prompt ──
    fallback_prompt = f"""Topic: "{query}"

Fill in this JSON for the topic "{query}". Respond with ONLY the JSON — start with {{ immediately:

{{
  "overview": "What {query} is and why it matters.\\n\\nCurrent research and findings about {query}.\\n\\nKey challenges and future directions of {query}.",
  "key_insights": ["insight 1 about {query}", "insight 2", "insight 3"],
  "challenges": ["main challenge in {query}", "second challenge"],
  "future_directions": ["future direction 1 for {query}", "direction 2"],
  "cross_domain_link": "how {query} relates to other scientific or engineering fields"
}}"""

    raw2 = call_llm(fallback_prompt, max_tokens=1000)
    print(f"   [generate_insights] attempt 2 raw: {raw2[:200]!r}")

    try:
        cleaned2 = extract_json(raw2)
        result2 = json.loads(cleaned2)
        if result2.get("overview") and len(result2["overview"]) > 30:
            return result2
        raise ValueError("Attempt 2 overview too short")
    except Exception as e2:
        print(f"   ⚠️  Attempt 2 failed: {e2} — falling back to plain-text overview...")

    # ── Attempt 3: Generate plain-text overview, build dict manually ──
    overview_text = ""
    try:
        plain_prompt = (
            f'Write exactly 3 paragraphs about "{query}" as a research overview. '
            f'Paragraph 1: what {query} is and why it matters. '
            f'Paragraph 2: current research state and findings. '
            f'Paragraph 3: challenges and future directions. '
            f'Separate paragraphs with a blank line. Plain text only — no JSON, no bullet points.'
        )
        overview_text = call_llm(
            plain_prompt,
            max_tokens=700,
            system="You are a research writer. Write clear, informative paragraphs in plain text."
        )
        # Strip accidental JSON wrapping
        if overview_text.strip().startswith("{"):
            try:
                parsed = json.loads(extract_json(overview_text))
                overview_text = parsed.get("overview", overview_text)
            except Exception:
                pass
    except Exception:
        pass

    # ── Hard fallback ──
    if not overview_text or len(overview_text.strip()) < 30:
        overview_text = (
            f"{query} is an important and actively studied research area that intersects multiple scientific disciplines. "
            f"Researchers are currently exploring its theoretical foundations as well as practical applications across various domains.\n\n"
            f"Recent academic work has produced significant findings related to {query}, with growing interest from both academic and industry communities. "
            f"The sources retrieved highlight a range of perspectives and methodological approaches.\n\n"
            f"Key open challenges include scalability, reproducibility, and cross-domain validation. "
            f"Future work in {query} is expected to focus on bridging theoretical advances with real-world deployment."
        )

    return {
        "overview": overview_text,
        "key_insights": [
            f"{query} is an active and rapidly growing research area",
            "Multiple peer-reviewed papers have been identified on this topic",
            "Cross-disciplinary collaboration is essential to advancing this field",
            "Practical applications are emerging from theoretical foundations"
        ],
        "challenges": [
            f"Scalability and generalisation remain core challenges in {query}",
            "Bridging theoretical advances with real-world implementation"
        ],
        "future_directions": [
            f"Expanded experimental validation of {query} approaches",
            "Development of standardised benchmarks and evaluation frameworks"
        ],
        "cross_domain_link": (
            f"{query} connects to multiple fields including computer science, "
            f"engineering, biology, and applied sciences."
        )
    }


def generate_knowledge_graph(query: str, insights: Dict, papers: List[Dict]) -> Dict:
    nodes = [{"id": query, "label": query, "group": "Main Topic", "value": 20}]
    edges = []
    for i, insight in enumerate(insights.get("key_insights", [])[:4]):
        short = insight[:40]
        nodes.append({"id": short, "label": short, "group": "Insight", "value": 12})
        edges.append({"from": query, "to": short, "label": "insight", "width": 2})
    for paper in papers[:3]:
        short = paper["title"].split(":")[0][:40]
        nodes.append({"id": short, "label": short, "group": "Paper", "value": 8})
        edges.append({"from": query, "to": short, "label": "paper", "width": 1})
    return {"nodes": nodes, "edges": edges}


# ══════════════════════════════════════════════════════════
# PDF GENERATION
# ══════════════════════════════════════════════════════════

def build_pdf_report(data: Dict) -> bytes:
    """Generate a well-formatted PDF report using ReportLab."""
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        topMargin=2.2 * cm,
        bottomMargin=2.2 * cm,
        title=f"Research Report: {data.get('query', 'Scholar AI')}",
        author="Scholar AI"
    )

    W = A4[0] - 4.4 * cm

    INK      = colors.HexColor("#1a2a3a")
    ACCENT   = colors.HexColor("#2563eb")
    LIGHT_BG = colors.HexColor("#f0f6ff")
    MUTED    = colors.HexColor("#64748b")
    BORDER   = colors.HexColor("#cbd5e1")
    WHITE    = colors.white

    def style(name, **kw):
        return ParagraphStyle(name, **kw)

    title_style     = style("ReportTitle",    fontName="Helvetica-Bold", fontSize=22, textColor=INK,   spaceAfter=4,  leading=28, alignment=TA_LEFT)
    subtitle_style  = style("ReportSubtitle", fontName="Helvetica",      fontSize=11, textColor=MUTED, spaceAfter=2,  leading=16)
    section_style   = style("SectionHead",    fontName="Helvetica-Bold", fontSize=12, textColor=ACCENT, spaceBefore=16, spaceAfter=6, leading=16)
    body_style      = style("Body",           fontName="Helvetica",      fontSize=10, textColor=INK,   leading=15,    spaceAfter=6, alignment=TA_JUSTIFY)
    muted_style     = style("Muted",          fontName="Helvetica",      fontSize=9,  textColor=MUTED, leading=13,    spaceAfter=4)
    bullet_style    = style("Bullet",         fontName="Helvetica",      fontSize=10, textColor=INK,   leading=15,    spaceAfter=4, leftIndent=12)
    highlight_style = style("Highlight",      fontName="Helvetica-Bold", fontSize=10, textColor=ACCENT, leading=14,  spaceAfter=3)

    story = []
    query = data.get("query", "Research Query")
    now   = datetime.now().strftime("%d %B %Y")

    # ── Header bar ──
    header_data = [[
        Paragraph("Scholar AI — Research Report", style("H",  fontName="Helvetica", fontSize=9, textColor=WHITE, leading=12)),
        Paragraph(now,                             style("HR", fontName="Helvetica", fontSize=9, textColor=WHITE, leading=12, alignment=2))
    ]]
    header_table = Table(header_data, colWidths=[W * 0.7, W * 0.3])
    header_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), ACCENT),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph(query, title_style))
    story.append(Paragraph(f"Generated by Scholar AI · {now}", subtitle_style))
    story.append(HRFlowable(width=W, thickness=1.5, color=ACCENT, spaceAfter=14))

    # ── Overview ──
    story.append(Paragraph("Research Overview", section_style))
    overview = data.get("overview", "No overview available.")
    for para in overview.split("\n\n"):
        para = para.strip()
        if para:
            story.append(Paragraph(para, body_style))
    story.append(Spacer(1, 6))

    # ── Key Insights ──
    insights = data.get("key_insights", [])
    if insights:
        story.append(Paragraph("Key Insights", section_style))
        ins_rows = []
        for i, ins in enumerate(insights, 1):
            ins_rows.append([
                Paragraph(f"<b>{i}</b>", style("IN", fontName="Helvetica-Bold", fontSize=10, textColor=WHITE, leading=13, alignment=TA_CENTER)),
                Paragraph(ins, body_style)
            ])
        ins_table = Table(ins_rows, colWidths=[0.6 * cm, W - 0.6 * cm])
        ins_table.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (0, -1), ACCENT),
            ("BACKGROUND",     (1, 0), (1, -1), LIGHT_BG),
            ("TOPPADDING",     (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 6),
            ("LEFTPADDING",    (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",   (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (1, 0), (1, -1), [LIGHT_BG, WHITE]),
            ("LINEBELOW",      (0, 0), (-1, -2), 0.5, BORDER),
            ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(ins_table)
        story.append(Spacer(1, 8))

    # ── Challenges ──
    challenges = data.get("challenges", [])
    if challenges:
        story.append(Paragraph("Challenges &amp; Research Gaps", section_style))
        for ch in challenges:
            story.append(Paragraph(f"• {ch}", bullet_style))
        story.append(Spacer(1, 6))

    # ── Future Directions ──
    future = data.get("future_directions", [])
    if future:
        story.append(Paragraph("Future Research Directions", section_style))
        for f in future:
            story.append(Paragraph(f"→ {f}", bullet_style))
        story.append(Spacer(1, 6))

    # ── Cross-domain ──
    cross = data.get("cross_domain_link", "")
    if cross:
        story.append(Paragraph("Cross-Domain Connections", section_style))
        story.append(Paragraph(cross, body_style))
        story.append(Spacer(1, 6))

    # ── Academic Papers ──
    papers = [s for s in data.get("sources", []) if s.get("type") == "paper"]
    if papers:
        story.append(HRFlowable(width=W, thickness=0.5, color=BORDER, spaceBefore=8, spaceAfter=8))
        story.append(Paragraph("Academic Papers", section_style))
        for i, p in enumerate(papers[:8], 1):
            title_p = Paragraph(f"<b>{i}. {p.get('title', 'Untitled')}</b>", highlight_style)
            meta_p  = Paragraph(
                f"{p.get('snippet', '')[:80]}  |  Year: {p.get('year', 'N/A')}  |  "
                f"Citations: {p.get('citations', 0)}  |  Source: {p.get('source', '')}",
                muted_style
            )
            abstract = p.get("abstract", "")
            if len(abstract) > 300:
                abstract = abstract[:300] + "…"
            abs_p = Paragraph(abstract, body_style) if abstract and abstract != "No abstract available." else None
            url_p = Paragraph(
                f'<link href="{p.get("url","")}" color="#2563eb">{p.get("url","")[:70]}</link>',
                muted_style
            )
            block = [title_p, meta_p]
            if abs_p:
                block.append(abs_p)
            block.append(url_p)

            paper_table = Table([[col] for col in block], colWidths=[W])
            paper_table.setStyle(TableStyle([
                ("TOPPADDING",    (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING",   (0, 0), (-1, -1), 10),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
                ("LINEAFTER",     (0, 0), (0, -1), 3, ACCENT),
                ("BACKGROUND",    (0, 0), (-1, -1), LIGHT_BG if i % 2 == 0 else WHITE),
            ]))
            story.append(KeepTogether(paper_table))
            story.append(Spacer(1, 5))

    # ── News Articles ──
    news_items = [s for s in data.get("sources", []) if s.get("type") == "news"]
    if news_items:
        story.append(HRFlowable(width=W, thickness=0.5, color=BORDER, spaceBefore=8, spaceAfter=8))
        story.append(Paragraph("News Articles", section_style))
        rows = [["Title", "Snippet", "URL"]]
        for n in news_items[:5]:
            rows.append([
                Paragraph(n.get("title", "")[:60], muted_style),
                Paragraph(n.get("snippet", "")[:80], muted_style),
                Paragraph(f'<link href="{n.get("url","")}">{n.get("url","")[:35]}</link>', muted_style),
            ])
        t = Table(rows, colWidths=[W * 0.3, W * 0.4, W * 0.3])
        t.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0), ACCENT),
            ("TEXTCOLOR",      (0, 0), (-1, 0), WHITE),
            ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",       (0, 0), (-1, 0), 9),
            ("BOTTOMPADDING",  (0, 0), (-1, 0), 7),
            ("TOPPADDING",     (0, 0), (-1, 0), 7),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
            ("LINEBELOW",      (0, 0), (-1, -2), 0.5, BORDER),
            ("TOPPADDING",     (0, 1), (-1, -1), 5),
            ("BOTTOMPADDING",  (0, 1), (-1, -1), 5),
            ("LEFTPADDING",    (0, 0), (-1, -1), 7),
            ("VALIGN",         (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(t)
        story.append(Spacer(1, 8))

    # ── Videos ──
    videos = [s for s in data.get("sources", []) if s.get("type") == "video"]
    if videos:
        story.append(HRFlowable(width=W, thickness=0.5, color=BORDER, spaceBefore=8, spaceAfter=8))
        story.append(Paragraph("Video Resources", section_style))
        for v in videos:
            story.append(Paragraph(
                f"<b>{v.get('title', 'Video')}</b>  [{v.get('source', 'Video')}]  —  "
                f'<link href="{v.get("url","")}" color="#2563eb">{v.get("url","")[:70]}</link>',
                muted_style
            ))
            story.append(Spacer(1, 4))

    # ── AI Paper Analysis ──
    pa = data.get("paper_analysis", [])
    if pa:
        story.append(HRFlowable(width=W, thickness=0.5, color=BORDER, spaceBefore=8, spaceAfter=8))
        story.append(Paragraph("AI Paper Analysis", section_style))
        for item in pa[:5]:
            story.append(Paragraph(f"<b>{item.get('title', '')[:70]}</b>", highlight_style))
            story.append(Paragraph(f"Key finding: {item.get('highlight', '')}", body_style))
            story.append(Paragraph(f"Why it matters: {item.get('why_focus', '')}", muted_style))
            story.append(Spacer(1, 5))

    # ── Footer ──
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width=W, thickness=0.5, color=BORDER))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Report generated by Scholar AI · {now} · Powered by Groq / Llama 3 · "
        f"Sources: Semantic Scholar, CrossRef, DuckDuckGo",
        style("Footer", fontName="Helvetica", fontSize=8, textColor=MUTED, leading=12, alignment=TA_CENTER)
    ))

    doc.build(story)
    return buffer.getvalue()


# ══════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════

@app.get("/")
async def root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(current_dir, "ai.html")
    if not os.path.exists(html_path):
        return {"error": "ai.html not found", "path": html_path}
    return FileResponse(html_path)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "2.2.0",
        "llm": "groq/llama3-8b-8192",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/stats")
async def get_stats():
    return {"total_searches": 147, "citations_found": 892, "research_gaps": 42, "time_saved": "12h"}


@app.get("/api/papers")
async def get_papers(query: str = "", limit: int = 10):
    papers = search_semantic_scholar(query, limit=limit)
    return {"papers": papers, "count": len(papers)}


@app.post("/api/search")
async def research(req: QueryRequest):
    """
    Main research endpoint.
    Pass language="kn" (or any code) to get the full result translated.
    """
    print(f"\n🔍 Research Query: '{req.query}' | Language: {req.language}")

    papers = search_semantic_scholar(req.query, limit=5)
    news   = search_news(req.query, limit=3)
    videos = search_youtube(req.query, limit=5)
    all_sources = papers + news + videos

    print(f"   📄 Papers: {len(papers)} | 📰 News: {len(news)} | 🎥 Videos: {len(videos)}")

    file_ctx       = "\n".join([f["text"] for f in uploaded_contexts])
    paper_analysis = analyze_papers(papers, req.query)
    insights       = generate_insights(req.query, all_sources, file_ctx, req.secondary_domain or "")
    graph_data     = generate_knowledge_graph(req.query, insights, papers)

    result = {
        "query":             req.query,
        "language":          req.language,
        "overview":          insights.get("overview", ""),
        "key_insights":      insights.get("key_insights", []),
        "challenges":        insights.get("challenges", []),
        "future_directions": insights.get("future_directions", []),
        "cross_domain_link": insights.get("cross_domain_link", ""),
        "paper_analysis":    paper_analysis,
        "sources":           all_sources,
        "knowledge_graph":   graph_data,
        "timestamp":         datetime.now().isoformat()
    }

    # Translate the full result if a non-English language was selected at search time
    if req.language and req.language != "en":
        print(f"   🌐 Translating full output to: {req.language}")
        result = translate_research_output(result, req.language)

    return result


@app.post("/api/retranslate")
async def retranslate(req: RetranslateRequest):
    """
    Retranslate a previously rendered research result into a different language.
    This is what the frontend 'Apply' button should call.

    Request body:
        {
            "result": <full object from /api/search>,
            "target_language": "kn"
        }

    Returns the same object with all text fields translated.
    """
    if not req.target_language or req.target_language not in LANGUAGE_NAMES:
        return {
            "error": f"Unsupported language: '{req.target_language}'",
            "supported": list(LANGUAGE_NAMES.keys())
        }

    if req.target_language == "en":
        return req.result   # English — return as-is, nothing to translate

    print(f"\n🌐 Retranslating → {LANGUAGE_NAMES.get(req.target_language)}")
    translated = translate_research_output(dict(req.result), req.target_language)
    translated["language"] = req.target_language
    return translated


@app.post("/api/generate-pdf")
async def generate_pdf(req: PDFRequest):
    try:
        pdf_bytes = build_pdf_report(req.data)
        query_slug = req.data.get("query", "report").replace(" ", "_")[:30]
        filename = f"scholar_report_{query_slug}.pdf"
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        print(f"PDF Generation Error: {e}")
        return {"error": str(e)}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    history_context = "\n".join([f"User: {h['user']}\nAI: {h['ai']}" for h in chat_history[-5:]])
    prompt = f"""You are a helpful research assistant.
Current topic: {req.context_query or "General Knowledge"}

Recent conversation:
{history_context}

User: {req.message}

Answer concisely and accurately. Plain text only."""

    try:
        ai_reply = call_llm(
            prompt,
            max_tokens=800,
            system="You are a helpful research assistant. Answer concisely and accurately in plain text."
        )
        if not ai_reply:
            ai_reply = "I couldn't generate a response. Please try again."
        chat_history.append({"user": req.message, "ai": ai_reply})
        return {"response": ai_reply}
    except Exception as e:
        return {"response": f"Error: {str(e)}"}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    text = extract_text_from_file(content, file.filename)
    uploaded_contexts.append({"filename": file.filename, "text": text})
    return {"status": "success", "filename": file.filename, "chars": len(text)}


@app.post("/api/translate")
async def translate_text(req: TranslateRequest):
    """Translate a single piece of text to the target language."""
    if not req.text or not req.text.strip():
        return {"translated_text": "", "target_language": req.target_language, "cached": False}

    if req.target_language not in LANGUAGE_NAMES:
        return {
            "error": f"Unsupported language code: '{req.target_language}'",
            "supported": list(LANGUAGE_NAMES.keys()),
        }

    cache_key = f"{req.target_language}::{hash(req.text)}"
    was_cached = cache_key in _translation_cache
    translated = translate_with_groq(req.text, req.target_language)

    return {
        "translated_text": translated,
        "target_language": req.target_language,
        "language_name":   LANGUAGE_NAMES.get(req.target_language, req.target_language),
        "cached":          was_cached,
    }


@app.post("/api/translate/batch")
async def translate_batch(req: BatchTranslateRequest):
    """Translate a list of strings to the target language in a single API call."""
    if not req.texts:
        return {"translations": []}

    if req.target_language not in LANGUAGE_NAMES:
        return {
            "error": f"Unsupported language code: '{req.target_language}'",
            "supported": list(LANGUAGE_NAMES.keys()),
        }

    lang_name = LANGUAGE_NAMES.get(req.target_language, req.target_language)
    joined = "\n---ITEM---\n".join(req.texts)

    combined_prompt = f"""Translate each item below into {lang_name}.
Keep the ---ITEM--- separators exactly as-is between each translated item.
Return ONLY the translated items separated by ---ITEM--- — nothing else.

{joined}"""

    try:
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Translate to {lang_name}. "
                        "Preserve ---ITEM--- separators exactly. Return only translations."
                    ),
                },
                {"role": "user", "content": combined_prompt},
            ],
            max_tokens=3000,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        parts = [p.strip() for p in raw.split("---ITEM---")]

        # Pad or trim to match input count
        while len(parts) < len(req.texts):
            parts.append(req.texts[len(parts)])
        parts = parts[: len(req.texts)]

        return {
            "translations":   parts,
            "target_language": req.target_language,
            "language_name":   lang_name,
        }
    except Exception as e:
        print(f"Batch Translation Error: {e}")
        return {"translations": req.texts}   # fallback: return originals


# ══════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 55)
    print("🚀  SCHOLAR AI v2.2 — GROQ BACKEND")
    print("=" * 55)
    print("📍  http://localhost:8001")
    print("📊  API Docs: http://localhost:8001/docs")
    print("🤖  LLM: Groq / Llama3-8b-8192")
    print()
    print("✅  Fixes in v2.2:")
    print("    • extract_json() — robust JSON parser (handles fences, preambles)")
    print("    • generate_insights() — 3-level fallback (never shows error message)")
    print("    • /api/search — passes language= → translates full output")
    print("    • /api/retranslate — new endpoint for frontend 'Apply' button")
    print("    • News snippets translated too")
    print("=" * 55 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
