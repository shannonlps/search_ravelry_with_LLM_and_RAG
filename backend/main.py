"""
Ravelry AI Pattern Search — FastAPI Backend (RAG version)
Supports Anthropic (Claude), OpenAI (GPT-4o), and Google (Gemini).
Uses RAG retriever to resolve entities before calling the LLM.
"""

import os
import sys
import httpx
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Literal
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add rag/ to path so retriever can be imported
sys.path.insert(0, str(Path(__file__).parent.parent / "rag"))
from retriever import retrieve_context

app = FastAPI(title="Ravelry AI Search + RAG", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RAVELRY_BASE = "https://api.ravelry.com"

BASE_SYSTEM_PROMPT = """You are an expert at translating natural language knitting and crochet searches
into Ravelry API query parameters.

## Parameters you can set

### query (string)
Free-text keyword search. Use the main subject only: "sweater", "hat", "socks", "shawl".
Keep it to 1-2 words. Do NOT put difficulty, weight, craft, or sort concepts here.

### craft (string)
Only if clearly stated: "knitting", "crochet", "machine knitting", "loom knitting"

### availability (string)
Only if user says free/no cost: "free", "for-sale", "online", "ravelry"

### weight (string)
Map carefully: thread, cobweb, lace, fingering, sport, dk, worsted, aran, bulky, super-bulky, jumbo

### sort (string)
Map intent: "best" (default), "projects" (most popular/made), "rating" (highest rated),
"favorites" (most saved), "recently-added" (newest), "recently-popular" (trending)

### difficulty (string)
Pipe-separated 1-10 scale: beginner=1|2, intermediate=3|4|5|6, advanced=7|8, expert=9|10

### ratings (string)
Pipe-separated stars: "5" or "4|5" or "3|4|5"

### pc (string)
Pattern category — use exact Ravelry pc values like: sweater, cardigan, pullover, hat,
beanie-toque, socks, shawl-wrap, cowl, mittens, blanket, baby-blanket, etc.

### pa (string)
Pattern attributes joined with +. Use exact values like: top-down, bottom-up, seamless,
cables, lace, fairisle, stranded, v-neck, raglan, toe-up, heel-flap, etc.

### fit (string)
Combine age+fit+ease+gender with +: adult, baby, child, teen, petite, plus, tall,
maternity, fitted, oversized, female, male, unisex, negative-ease, positive-ease

### fiber (string)
Fiber type: wool, merino, alpaca, cotton, silk, cashmere, mohair, linen, acrylic, etc.

### colors (integer)
Number of colors used

### page_size (integer)
Number of results — use whatever the app passes in

## Rules
1. Only set fields clearly implied by the user's query
2. Never guess — omit a field rather than guess
3. Return raw JSON only, no markdown, no explanation
4. If RAG context is provided above, do NOT override those values — only fill in what's missing"""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

LLMProvider = Literal["anthropic", "openai", "gemini"]


class SearchRequest(BaseModel):
    query: str
    ravelry_username: str
    ravelry_password: str
    llm_provider: LLMProvider = "anthropic"
    llm_api_key: str
    page_size: Optional[int] = 5


class ParsedParams(BaseModel):
    query: Optional[str] = None
    craft: Optional[str] = None
    availability: Optional[str] = None
    fit: Optional[str] = None
    weight: Optional[str] = None
    difficulty: Optional[str] = None
    ratings: Optional[str] = None
    pc: Optional[str] = None
    pa: Optional[str] = None
    fiber: Optional[str] = None
    colors: Optional[int] = None
    designer_id: Optional[str] = None
    needle_size_mm: Optional[float] = None
    sort: Optional[str] = "best"
    page_size: int = 5


# ---------------------------------------------------------------------------
# LLM providers
# ---------------------------------------------------------------------------

def parse_with_anthropic(prompt: str, api_key: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=400,
        system=BASE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in message.content if hasattr(block, "text"))


def parse_with_openai(prompt: str, api_key: str) -> str:
    import openai
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=400,
        messages=[
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""


def parse_with_gemini(prompt: str, api_key: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=BASE_SYSTEM_PROMPT,
    )
    response = model.generate_content(prompt)
    return response.text or ""


def call_llm(prompt: str, provider: LLMProvider, api_key: str) -> str:
    try:
        if provider == "anthropic":
            return parse_with_anthropic(prompt, api_key)
        elif provider == "openai":
            return parse_with_openai(prompt, api_key)
        elif provider == "gemini":
            return parse_with_gemini(prompt, api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{provider} API error: {str(e)}")


def merge_rag_and_llm(rag_context: dict, llm_json: dict, page_size: int) -> ParsedParams:
    """Merge RAG-resolved values with LLM-inferred values. RAG always wins."""
    params = ParsedParams(page_size=page_size)

    # Apply LLM params first (lower priority)
    for field in ["query", "craft", "availability", "weight", "difficulty",
                  "ratings", "pc", "pa", "fiber", "colors", "sort"]:
        val = llm_json.get(field)
        if val is not None:
            setattr(params, field, val)

    # Apply RAG params (override LLM where RAG has a confident match)
    if rag_context.get("designer"):
        params.designer_id = rag_context["designer"]["designer_id"]

    if rag_context.get("categories"):
        params.pc = rag_context["categories"][0]["pc"]

    if rag_context.get("attributes"):
        pa_values = "+".join(a["pa"] for a in rag_context["attributes"])
        params.pa = pa_values

    if rag_context.get("fit_params"):
        fit_values = "+".join(f["api_value"] for f in rag_context["fit_params"])
        params.fit = fit_values

    if rag_context.get("fiber"):
        params.fiber = rag_context["fiber"]["api_value"]

    if rag_context.get("needle_size"):
        params.needle_size_mm = rag_context["needle_size"]["mm"]

    for param, value in rag_context.get("parameters", {}).items():
        if param == "sort":
            params.sort = value
        elif param == "availability":
            params.availability = value
        elif param == "difficulty":
            params.difficulty = value
        elif param == "rating":
            params.ratings = value

    return params


# ---------------------------------------------------------------------------
# Ravelry helpers
# ---------------------------------------------------------------------------

def ravelry_client(username: str, password: str) -> httpx.Client:
    return httpx.Client(
        auth=(username, password),
        headers={"Accept": "application/json"},
        timeout=15.0,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/api/search")
def search_patterns(req: SearchRequest):
    # 1. RAG retrieval
    rag_context = retrieve_context(req.query)

    # 2. Build enriched prompt for LLM
    if rag_context["rag_resolved"] and rag_context["prompt_injection"]:
        llm_prompt = f"{rag_context['prompt_injection']}\n\nUser query: {req.query}"
    else:
        llm_prompt = req.query

    # 3. Call LLM for remaining params
    raw = call_llm(llm_prompt, req.llm_provider, req.llm_api_key)
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        llm_json = json.loads(cleaned)
    except Exception:
        llm_json = {}

    # 4. Merge RAG + LLM
    params = merge_rag_and_llm(rag_context, llm_json, req.page_size or 25)

    # 5. Build Ravelry search params
    search_params: dict = {"page_size": params.page_size, "sort": params.sort or "best"}
    if params.query:        search_params["query"] = params.query
    if params.craft:        search_params["craft"] = params.craft
    if params.availability: search_params["availability"] = params.availability
    if params.fit:          search_params["fit"] = params.fit
    if params.weight:       search_params["weight"] = params.weight
    if params.difficulty:   search_params["difficulty"] = params.difficulty
    if params.ratings:      search_params["ratings"] = params.ratings
    if params.pc:           search_params["pc"] = params.pc
    if params.pa:           search_params["pa"] = params.pa
    if params.fiber:        search_params["fiber"] = params.fiber
    if params.colors:       search_params["colors"] = params.colors
    if params.designer_id:  search_params["designer-id"] = params.designer_id

    with ravelry_client(req.ravelry_username, req.ravelry_password) as client:
        search_resp = client.get(f"{RAVELRY_BASE}/patterns/search.json", params=search_params)

    if search_resp.status_code == 401:
        raise HTTPException(status_code=401, detail="Invalid Ravelry credentials.")
    if not search_resp.is_success:
        raise HTTPException(status_code=search_resp.status_code, detail=search_resp.text)

    pattern_stubs = search_resp.json().get("patterns", [])
    if not pattern_stubs:
        return {
            "params": params.model_dump(),
            "rag_context": rag_context,
            "patterns": []
        }

    # 6. Fetch full details via individual pattern endpoint /patterns/{id}.json
    # The bulk patterns.json endpoint is unreliable with basic auth — use individual calls
    # Use a thread pool to fetch concurrently and keep it fast
    import concurrent.futures

    ids = [str(p["id"]) for p in pattern_stubs[:params.page_size]]
    auth = (req.ravelry_username, req.ravelry_password)

    def fetch_one(pattern_id: str):
        try:
            resp = httpx.get(
                f"{RAVELRY_BASE}/patterns/{pattern_id}.json",
                auth=auth,
                headers={"Accept": "application/json"},
                timeout=10.0
            )
            if resp.is_success:
                return resp.json().get("pattern")
        except Exception:
            pass
        return None

    patterns = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_one, pid): pid for pid in ids}
        # Preserve order
        results = {pid: None for pid in ids}
        for future in concurrent.futures.as_completed(futures):
            pid = futures[future]
            result = future.result()
            if result:
                results[pid] = result
        patterns = [results[pid] for pid in ids if results[pid] is not None]


    return {
        "params": params.model_dump(),
        "rag_context": rag_context,
        "patterns": patterns
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Serve built React frontend in production
# ---------------------------------------------------------------------------

frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        return FileResponse(frontend_dist / "index.html")
