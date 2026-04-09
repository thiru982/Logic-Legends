# ── IMPORTS ──
import os
import json
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from duckduckgo_search import DDGS
from datetime import datetime
import PyPDF2
import io

# ── APP SETUP ──
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="."), name="static")

# Store uploaded files context & search history
uploaded_contexts = []
search_history = []

# ── MODELS ──
class QueryRequest(BaseModel):
    query: str
    include_related: bool = True

class SourceItem(BaseModel):
    title: str
    url: str
    snippet: str
    type: str
    source: str

class Node(BaseModel):
    id: str
    label: str
    group: str

class Edge(BaseModel):
    from_node: str
    to_node: str
    label: str

class KnowledgeGraph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

class SearchResponse(BaseModel):
    query: str
    overview: str
    key_facts: List[str]
    innovations: List[str]
    challenges: List[str]
    related_topics: List[str]
    sources: List[SourceItem]
    wiki_summary: str
    wiki_url: str
    knowledge_graph: Optional[KnowledgeGraph] = None
    source_stats: Dict
    file_context: str
    timestamp: str

# ── FILE PROCESSING ──
def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """Extract text from uploaded files (PDF, TXT, MD)"""
    text = ""
    try:
        if filename.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif filename.endswith('.txt') or filename.endswith('.md'):
            text = file_content.decode('utf-8', errors='ignore')
        else:
            text = file_content.decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"❌ Error extracting text from {filename}: {e}")
    return text[:5000]  # Limit to 5000 chars to avoid token limits

# ── SEARCH FUNCTIONS ──
def search_web(query, limit=5):
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=limit):
                results.append({
                    "title": r.get("title", "")[:100], 
                    "url": r.get("href", ""), 
                    "snippet": r.get("body", "")[:200], 
                    "type": "web", 
                    "source": "Web"
                })
    except Exception as e:
        print(f"❌ Web search error: {e}")
    return results

def search_papers(query, limit=5):
    """Search academic papers from Semantic Scholar"""
    results = []
    try:
        r = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search", 
            params={
                "query": query, 
                "limit": limit, 
                "fields": "title,abstract,url,year,citationCount"
            }, 
            timeout=10
        )
        for p in r.json().get("data", []):
            results.append({
                "title": p.get("title", "")[:100], 
                "url": p.get("url", ""), 
                "snippet": (p.get("abstract", "") or "")[:200], 
                "type": "paper", 
                "source": f"Paper ({p.get('year', 'N/A')})"
            })
    except Exception as e:
        print(f"❌ Paper search error: {e}")
    return results

def search_youtube(query, limit=3):
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(f"{query} site:youtube.com", max_results=limit):
                results.append({
                    "title": r.get("title", "")[:100], 
                    "url": r.get("href", ""), 
                    "snippet": r.get("body", "")[:200], 
                    "type": "video", 
                    "source": "YouTube"
                })
    except Exception as e:
        print(f"❌ YouTube search error: {e}")
    return results

def search_related_topics(query, limit=5):
    """Find related research topics for deeper exploration"""
    related = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(f"related to {query} research topics", max_results=limit):
                title = r.get("title", "")[:150]
                if title and query.lower() not in title.lower():
                    related.append(title)
    except Exception as e:
        print(f"❌ Related topics error: {e}")
    
    # Fallback suggestions if search fails
    if not related:
        related = [
            f"{query} Applications",
            f"{query} Future Trends",
            f"{query} Challenges",
            f"{query} Case Studies",
            f"{query} Best Practices"
        ]
    return related[:limit]

def get_wikipedia(query):
    """Get Wikipedia summary for additional context"""
    try:
        enc = requests.utils.quote(query)
        s = requests.get(
            f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={enc}&srlimit=1&format=json&origin=*", 
            timeout=10
        ).json()
        if s.get('query', {}).get('search'):
            title = s['query']['search'][0]['title']
            p = requests.get(
                f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts|pageimages&exintro=true&explaintext=true&pithumbsize=600&titles={requests.utils.quote(title)}&format=json&origin=*", 
                timeout=10
            ).json()
            page = list(p.get('query', {}).get('pages', {}).values())[0]
            return {
                "summary": (page.get('extract', '') or "")[:800], 
                "image": page.get('thumbnail', {}).get('source', ''), 
                "url": f"https://en.wikipedia.org/wiki/{requests.utils.quote(title)}"
            }
    except Exception as e:
        print(f"❌ Wikipedia error: {e}")
    return {"summary": "", "image": "", "url": ""}

# ── AI SYNTHESIS (LLAMA 3) ──
def synthesize(query, context, file_context=""):
    """Generate AI analysis using Llama 3"""
    prompt = f"""You are an expert research assistant. Analyze the topic: {query}

Context from Web Research:
{context}

{"Additional Context from Uploaded Files:\n" + file_context if file_context else ""}

Return a valid JSON object (no markdown, no code blocks) with this structure:
{{
    "overview": "A comprehensive 3-paragraph summary of the topic based on all context.",
    "key_facts": ["Fact 1", "Fact 2", "Fact 3", "Fact 4", "Fact 5"],
    "innovations": ["Innovation 1", "Innovation 2", "Innovation 3"],
    "challenges": ["Challenge 1", "Challenge 2"],
    "related_topics": ["Related Topic 1", "Related Topic 2", "Related Topic 3"]
}}"""
    
    try:
        r = requests.post(
            "http://localhost:11434/api/generate", 
            json={
                "model": "llama3", 
                "prompt": prompt, 
                "stream": False, 
                "format": "json"
            }, 
            timeout=120
        )
        text = r.json().get("response", "").replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"❌ AI Synthesis error: {e}")
        return {
            "overview": f"AI Error: {str(e)}. Ensure Ollama is running with: ollama serve", 
            "key_facts": ["Run: ollama serve", "Run: ollama pull llama3"], 
            "innovations": [], 
            "challenges": [], 
            "related_topics": []
        }

def generate_knowledge_graph(query, context, file_context=""):
    """Generate knowledge graph showing entity relationships"""
    prompt = f"""Analyze the relationships in this research topic: {query}
Context: {context}
{file_context}

Extract key entities and their relationships. Return ONLY valid JSON:
{{
    "nodes": [
        {{"id": "entity1", "label": "Entity Name", "group": "Concept"}},
        {{"id": "entity2", "label": "Person/Org", "group": "Actor"}}
    ],
    "edges": [
        {{"from": "entity1", "to": "entity2", "label": "influences"}}
    ]
}}
Keep it simple (max 10 nodes, 10 edges). Include main topic and related concepts."""

    try:
        r = requests.post(
            "http://localhost:11434/api/generate", 
            json={
                "model": "llama3", 
                "prompt": prompt, 
                "stream": False, 
                "format": "json"
            }, 
            timeout=120
        )
        text = r.json().get("response", "").replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"❌ Knowledge Graph error: {e}")
        return None

# ── ENDPOINTS ──
@app.get("/")
def root():
    """Serve the main HTML page"""
    return FileResponse("index.html")

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file for additional research context"""
    global uploaded_contexts
    content = await file.read()
    text = extract_text_from_file(content, file.filename)
    
    uploaded_contexts.append({
        "filename": file.filename,
        "text": text,
        "uploaded_at": datetime.now().isoformat(),
        "size": len(content)
    })
    
    print(f"✅ Uploaded: {file.filename} ({len(text)} chars)")
    return {
        "status": "success", 
        "filename": file.filename, 
        "chars": len(text),
        "size_kb": round(len(content) / 1024, 2)
    }

@app.get("/uploaded-files")
async def get_uploaded_files():
    """Get list of all uploaded files"""
    return {
        "files": [
            {
                "filename": f["filename"], 
                "chars": len(f["text"]), 
                "uploaded_at": f["uploaded_at"],
                "size_kb": round(f["size"] / 1024, 2)
            } 
            for f in uploaded_contexts
        ]
    }

@app.post("/clear-files")
async def clear_files():
    """Clear all uploaded files from memory"""
    global uploaded_contexts
    count = len(uploaded_contexts)
    uploaded_contexts = []
    print(f"🗑️ Cleared {count} uploaded files")
    return {"status": "cleared", "count": count}

@app.post("/search", response_model=SearchResponse)
async def search(req: QueryRequest):
    """Main search endpoint - combines web search + AI analysis"""
    print(f"\n🔍 Researching: {req.query} (Using Llama 3)...")
    start_time = datetime.now()
    
    # Combine all file contexts
    file_context = "\n\n".join([f"File: {f['filename']}\n{f['text']}" for f in uploaded_contexts])
    if file_context:
        print(f"📁 Using {len(uploaded_contexts)} uploaded file(s) as context")
    
    # Gather external research
    web = search_web(req.query)
    papers = search_papers(req.query)
    youtube = search_youtube(req.query)
    wiki = get_wikipedia(req.query)
    
    # Find related topics
    related = search_related_topics(req.query) if req.include_related else []
    
    # Combine all sources
    all_sources = web + papers + youtube
    context = "\n".join([f"- {s['title']}: {s['snippet']}" for s in all_sources]) or "No sources found"
    
    # AI Synthesis
    print("🤖 Generating AI analysis...")
    ai_data = synthesize(req.query, context, file_context)
    
    # Knowledge Graph
    print("🕸️ Generating knowledge graph...")
    graph_data = generate_knowledge_graph(req.query, context, file_context)
    
    # Calculate timing
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save to history
    search_history.append({
        "query": req.query,
        "timestamp": timestamp,
        "duration_seconds": round(duration, 2),
        "sources_found": len(all_sources),
        "overview_preview": ai_data.get("overview", "")[:200]
    })
    
    print(f"✅ Search completed in {duration:.2f}s | Found {len(all_sources)} sources")
    
    return SearchResponse(
        query=req.query,
        overview=ai_data.get("overview", ""),
        key_facts=ai_data.get("key_facts", []),
        innovations=ai_data.get("innovations", []),
        challenges=ai_data.get("challenges", []),
        related_topics=ai_data.get("related_topics", []) + related,
        sources=all_sources,
        wiki_summary=wiki.get("summary", ""),
        wiki_url=wiki.get("url", ""),
        knowledge_graph=graph_data,
        source_stats={
            "web": len(web), 
            "papers": len(papers), 
            "videos": len(youtube), 
            "total": len(all_sources)
        },
        file_context=file_context[:500] + "..." if len(file_context) > 500 else file_context,
        timestamp=timestamp
    )

@app.get("/search-history")
async def get_history():
    """Get recent search history"""
    return {"history": search_history[-10:]}  # Last 10 searches

@app.post("/clear-history")
async def clear_history():
    """Clear search history"""
    global search_history
    search_history = []
    return {"status": "cleared"}

# ── MAIN ──
if __name__ == "__main__":
    import uvicorn
    
    # Check if Ollama is running
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        print("✅ Ollama is running")
    except:
        print("⚠️  WARNING: Ollama may not be running! Run: ollama serve")
    
    print("\n" + "="*60)
    print("🚀 SCHOLAR AI (LLAMA 3 + KNOWLEDGE GRAPH)")
    print("="*60)
    print("📍 http://localhost:8000")
    print("📁 Upload: PDF, TXT, MD files for additional context")
    print("📄 Export: PDF reports & Knowledge Graph images")
    print("🔗 Related Topics: Auto-discovered during search")
    print("⚠️  Make sure: ollama pull llama3")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
