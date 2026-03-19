"""
Ravelry RAG Ingest Script
Run once (and resume daily) to build the ChromaDB vector database.

Usage:
    python ingest.py

Configuration via .env or constants below.
"""

import os
import json
import time
import httpx
import chromadb
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DESIGNER_MIN_PATTERNS  = 100
INGEST_RAVELRY_USERNAME = os.environ.get("INGEST_RAVELRY_USERNAME", "")
INGEST_RAVELRY_PASSWORD = os.environ.get("INGEST_RAVELRY_PASSWORD", "")
EMBEDDING_MODEL        = "all-MiniLM-L6-v2"
DAILY_REQUEST_LIMIT    = 90          # conservative buffer under Ravelry's 100/day
REQUESTS_PER_SECOND    = 1           # 1 request per second max

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR      = Path(__file__).parent
STATIC_DIR    = BASE_DIR / "data" / "static"
CHROMA_DIR    = BASE_DIR / "data" / "chroma_db"
PROGRESS_FILE = BASE_DIR / "data" / "progress.json"
SEED_FILE     = BASE_DIR / "data" / "seed_designers.txt"

RAVELRY_BASE  = "https://api.ravelry.com"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {
        "last_updated": None,
        "designers_completed": 0,
        "designers_total": 0,
        "last_processed": None,
        "requests_today": 0,
        "status": "not_started"
    }


def save_progress(progress: dict):
    progress["last_updated"] = datetime.now().isoformat()
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def ravelry_get(client: httpx.Client, path: str, params: dict = None) -> dict:
    """Make a rate-limited GET request to the Ravelry API."""
    time.sleep(1.0 / REQUESTS_PER_SECOND)
    resp = client.get(f"{RAVELRY_BASE}{path}", params=params or {})
    resp.raise_for_status()
    return resp.json()


def make_document(texts: list[str]) -> str:
    """Join a list of text fragments into a single searchable document."""
    return " | ".join(t for t in texts if t)


def parse_seed_designers(seed_file: Path) -> list[dict]:
    """
    Parse seed_designers.txt.
    Format: alternating lines of username then display name.
    If a line has no corresponding pair it's treated as display-name only.
    """
    lines = [l.strip() for l in seed_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    designers = []
    i = 0
    while i < len(lines):
        username = lines[i]
        # Check if next line looks like a display name (i.e. different from username)
        if i + 1 < len(lines):
            display = lines[i + 1]
            # If display name looks like it could be a username (no spaces, all lowercase)
            # and is identical to username, treat as display-name-only
            if display.lower() == username.lower():
                designers.append({"username": username, "display_name": display})
                i += 2
            else:
                designers.append({"username": username, "display_name": display})
                i += 2
        else:
            designers.append({"username": username, "display_name": username})
            i += 1
    return designers


# ---------------------------------------------------------------------------
# Static collection builders
# ---------------------------------------------------------------------------

def build_parameters_collection(collection, model: SentenceTransformer):
    data = json.loads((STATIC_DIR / "parameters.json").read_text())
    docs, ids, metas = [], [], []

    for param_name, param_data in data.items():
        for item in param_data.get("values", []):
            text = make_document([
                item["display"],
                item["api_value"],
                param_name,
                param_data.get("param", ""),
                *item.get("aliases", [])
            ])
            doc_id = f"param_{param_name}_{item['api_value']}".replace(" ", "_").replace("|", "")[:100]
            docs.append(text)
            ids.append(doc_id)
            metas.append({
                "type": "parameter",
                "param_name": param_name,
                "api_param": param_data.get("param", param_name),
                "api_value": item["api_value"],
                "display": item["display"],
                "note": param_data.get("note", "")
            })

    embeddings = model.encode(docs).tolist()
    collection.add(documents=docs, embeddings=embeddings, ids=ids, metadatas=metas)
    return len(docs)


def build_categories_collection(collection, model: SentenceTransformer):
    data = json.loads((STATIC_DIR / "categories.json").read_text())
    docs, ids, metas = [], [], []

    def flatten(items, parent_breadcrumb=""):
        for item in items:
            breadcrumb = f"{parent_breadcrumb} > {item['display']}" if parent_breadcrumb else item["display"]
            text = make_document([item["display"], item["pc"], breadcrumb, *item.get("aliases", [])])
            doc_id = f"cat_{item['pc']}".replace("/", "-")[:100]
            docs.append(text)
            ids.append(doc_id)
            metas.append({
                "type": "category",
                "pc": item["pc"],
                "display": item["display"],
                "breadcrumb": breadcrumb
            })
            if "children" in item:
                flatten(item["children"], breadcrumb)

    flatten(data)
    embeddings = model.encode(docs).tolist()
    collection.add(documents=docs, embeddings=embeddings, ids=ids, metadatas=metas)
    return len(docs)


def build_attributes_collection(collection, model: SentenceTransformer):
    data = json.loads((STATIC_DIR / "attributes.json").read_text())
    docs, ids, metas = [], [], []

    for group_data in data:
        group = group_data["group"]
        for item in group_data["attributes"]:
            text = make_document([item["display"], item["pa"], group, *item.get("aliases", [])])
            group_slug = group.replace(" ", "_").replace("/", "-").replace(".", "")
            doc_id = f"attr_{group_slug}_{item['pa']}".replace("/", "-")[:100]
            docs.append(text)
            ids.append(doc_id)
            metas.append({
                "type": "attribute",
                "pa": item["pa"],
                "display": item["display"],
                "group": group
            })

    embeddings = model.encode(docs).tolist()
    collection.add(documents=docs, embeddings=embeddings, ids=ids, metadatas=metas)
    return len(docs)


def build_fibers_collection(collection, model: SentenceTransformer):
    data = json.loads((STATIC_DIR / "fibers.json").read_text())
    docs, ids, metas = [], [], []

    for item in data:
        text = make_document([item["display"], item["api_value"], *item.get("aliases", [])])
        doc_id = f"fiber_{item['api_value']}"[:100]
        docs.append(text)
        ids.append(doc_id)
        metas.append({
            "type": "fiber",
            "api_value": item["api_value"],
            "display": item["display"]
        })

    embeddings = model.encode(docs).tolist()
    collection.add(documents=docs, embeddings=embeddings, ids=ids, metadatas=metas)
    return len(docs)


def build_needle_sizes_collection(collection, model: SentenceTransformer):
    data = json.loads((STATIC_DIR / "needle_sizes.json").read_text())
    docs, ids, metas = [], [], []

    for item in data:
        text = make_document([
            f"US {item['us']}",
            f"{item['mm']}mm",
            f"US size {item['us']}",
            *item.get("aliases", [])
        ])
        doc_id = f"needle_us{item['us'].replace('.', '_')}"[:100]
        docs.append(text)
        ids.append(doc_id)
        metas.append({
            "type": "needle_size",
            "us": item["us"],
            "mm": item["mm"],
            "display": f"US {item['us']} / {item['mm']}mm"
        })

    embeddings = model.encode(docs).tolist()
    collection.add(documents=docs, embeddings=embeddings, ids=ids, metadatas=metas)
    return len(docs)


# ---------------------------------------------------------------------------
# Designer collection builder
# ---------------------------------------------------------------------------

def build_designers_collection(
    collection,
    model: SentenceTransformer,
    progress: dict,
    seed_designers: list[dict]
) -> tuple[dict, int]:
    """
    Fetch designers from Ravelry API using the seed list.
    Respects daily rate limit and resumes from progress.json.
    Returns updated progress and number of requests made this run.
    """
    if not INGEST_RAVELRY_USERNAME or not INGEST_RAVELRY_PASSWORD:
        print("  ⚠  No ingest Ravelry credentials found — skipping designer fetch.")
        print("     Set INGEST_RAVELRY_USERNAME and INGEST_RAVELRY_PASSWORD in .env")
        return progress, 0

    requests_this_run = 0
    completed = progress.get("designers_completed", 0)
    total = len(seed_designers)
    progress["designers_total"] = total

    remaining = seed_designers[completed:]

    client = httpx.Client(
        auth=(INGEST_RAVELRY_USERNAME, INGEST_RAVELRY_PASSWORD),
        headers={"Accept": "application/json"},
        timeout=15.0
    )

    docs, ids, metas = [], [], []

    # Load any already-embedded designers so we can add incrementally
    existing = collection.get()
    existing_ids = set(existing["ids"]) if existing["ids"] else set()

    try:
        for i, designer in enumerate(remaining):
            actual_index = completed + i

            if requests_this_run >= DAILY_REQUEST_LIMIT - 10:
                print(f"\n  ⏸  Daily limit approaching ({DAILY_REQUEST_LIMIT} requests). Stopping cleanly.")
                print(f"     Progress saved. Run again tomorrow to continue.")
                break

            username = designer["username"]
            display_name = designer["display_name"]

            try:
                # Use pattern search with designer param — works with basic auth
                # /people/search.json requires OAuth, but patterns/search returns designer info
                data = ravelry_get(client, "/patterns/search.json", {
                    "designer": username,
                    "page_size": 1,
                    "sort": "best"
                })
                requests_this_run += 1

                patterns = data.get("patterns", [])
                matched = None

                if patterns:
                    p = patterns[0]
                    designer_info = p.get("designer", {})
                    matched = {
                        "id": designer_info.get("id"),
                        "username": username,
                        "display_name": designer_info.get("name", display_name)
                    }

                if matched and matched.get("id"):
                    designer_id = matched.get("id")
                    matched_username = matched.get("username", username)
                    matched_display = matched.get("display_name") or display_name

                    doc_id = f"designer_{designer_id}"
                    if doc_id not in existing_ids:
                        text = make_document([
                            matched_display,
                            matched_username,
                            display_name,
                            username,
                            f"designer {matched_display}",
                            f"patterns by {matched_display}"
                        ])
                        docs.append(text)
                        ids.append(doc_id)
                        metas.append({
                            "type": "designer",
                            "designer_id": str(designer_id),
                            "username": matched_username,
                            "display_name": matched_display,
                            "seed_display_name": display_name
                        })
                    print(f"  [{actual_index + 1}/{total}] {display_name} → id:{designer_id} ✓")
                else:
                    print(f"  [{actual_index + 1}/{total}] {display_name} → not found, skipping")

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    print(f"\n  🛑 Rate limit hit (429). Saving progress and stopping.")
                    break
                else:
                    print(f"  [{actual_index + 1}/{total}] {display_name} → HTTP error {e.response.status_code}, skipping")

            except Exception as e:
                print(f"  [{actual_index + 1}/{total}] {display_name} → error: {e}, skipping")

            progress["designers_completed"] = actual_index + 1
            progress["last_processed"] = display_name
            progress["requests_today"] = requests_this_run

    finally:
        client.close()

    # Embed and add new designers in one batch
    if docs:
        embeddings = model.encode(docs).tolist()
        collection.add(documents=docs, embeddings=embeddings, ids=ids, metadatas=metas)

    if progress["designers_completed"] >= total:
        progress["status"] = "complete"
    else:
        progress["status"] = "in_progress"

    return progress, requests_this_run


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 50)
    print("  Ravelry RAG Ingest")
    print("=" * 50)
    print()

    # Load model
    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("  ✓ Model loaded\n")

    # Init ChromaDB — always rebuild static, append designers
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # ---------------------------------------------------------------------------
    # Static collections — always rebuilt from scratch
    # ---------------------------------------------------------------------------
    print("Building static collections (no API calls)...")

    for name in ["parameters", "categories", "attributes", "fibers", "needle_sizes", "designers"]:
        try:
            chroma.delete_collection(name)
        except Exception:
            pass

    params_col    = chroma.create_collection("parameters")
    cats_col      = chroma.create_collection("categories")
    attrs_col     = chroma.create_collection("attributes")
    fibers_col    = chroma.create_collection("fibers")
    needles_col   = chroma.create_collection("needle_sizes")
    designers_col = chroma.create_collection("designers")

    n = build_parameters_collection(params_col, model)
    print(f"  ✓ parameters     ({n} documents)")

    n = build_categories_collection(cats_col, model)
    print(f"  ✓ categories     ({n} documents)")

    n = build_attributes_collection(attrs_col, model)
    print(f"  ✓ attributes     ({n} documents)")

    n = build_fibers_collection(fibers_col, model)
    print(f"  ✓ fibers         ({n} documents)")

    n = build_needle_sizes_collection(needles_col, model)
    print(f"  ✓ needle_sizes   ({n} documents)")

    print()

    # ---------------------------------------------------------------------------
    # Designers — resumable, rate-limited
    # ---------------------------------------------------------------------------
    progress = load_progress()
    seed_designers = parse_seed_designers(SEED_FILE)
    total = len(seed_designers)
    completed = progress.get("designers_completed", 0)

    print(f"Fetching designers from seed list...")
    if completed > 0 and completed < total:
        print(f"  Resuming from: {progress.get('last_processed', '?')} ({completed}/{total} completed)")
    elif completed >= total:
        print(f"  ✓ All {total} designers already fetched.")
        progress["status"] = "complete"
        save_progress(progress)
        print()
        print("Done. Database is complete.")
        return

    progress, requests_made = build_designers_collection(
        designers_col, model, progress, seed_designers
    )

    save_progress(progress)

    print()
    print(f"Done. {requests_made} API requests used this run.")
    print(f"Designer progress: {progress['designers_completed']}/{total}")
    if progress["status"] == "complete":
        print("✓ Ingest complete! Database is ready.")
    else:
        print(f"Run again tomorrow to continue ({total - progress['designers_completed']} designers remaining).")


if __name__ == "__main__":
    main()
