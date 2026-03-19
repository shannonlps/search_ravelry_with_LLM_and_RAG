"""
Ravelry RAG Retriever
Imported by backend/main.py. Resolves entities and parameters from ChromaDB
before passing anything to the LLM.
"""

import re
from pathlib import Path
from functools import lru_cache
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DESIGNER_THRESHOLD  = 0.85
CATEGORY_THRESHOLD  = 0.80
ATTRIBUTE_THRESHOLD = 0.75
FIBER_THRESHOLD     = 0.80
NEEDLE_THRESHOLD    = 0.80

CHROMA_DIR = Path(__file__).parent / "data" / "chroma_db"

# ---------------------------------------------------------------------------
# Rule-based parameter extraction (no vector search needed)
# ---------------------------------------------------------------------------

SORT_RULES = {
    "most popular": "projects", "most made": "projects", "most projects": "projects",
    "high project count": "projects", "popular": "projects", "most knitted": "projects",
    "most crocheted": "projects", "most used": "projects", "most worked": "projects",
    "highest rated": "rating", "top rated": "rating", "best rated": "rating",
    "most loved": "rating", "best reviews": "rating", "most stars": "rating",
    "most favorited": "favorites", "most saved": "favorites", "most queued": "favorites",
    "newest": "recently-added", "latest": "recently-added", "most recent": "recently-added",
    "just added": "recently-added", "new patterns": "recently-added",
    "trending": "recently-popular", "hot right now": "recently-popular",
    "currently popular": "recently-popular",
}

AVAILABILITY_RULES = {
    "free": "free", "no cost": "free", "free download": "free",
    "free pattern": "free", "for free": "free",
    "paid": "for-sale", "purchase": "for-sale", "buy": "for-sale", "for sale": "for-sale",
}

DIFFICULTY_RULES = {
    "beginner": "1|2", "easy": "1|2", "simple": "1|2", "quick knit": "1|2",
    "no experience": "1|2", "first project": "1|2", "beginner friendly": "1|2",
    "intermediate": "3|4|5|6", "medium": "3|4|5|6", "moderate": "3|4|5|6",
    "advanced": "7|8", "difficult": "7|8", "challenging": "7|8", "complex": "7|8",
    "expert": "9|10", "very difficult": "9|10", "master": "9|10",
}

RATING_RULES = {
    "5 stars": "5", "five stars": "5", "perfect": "5",
    "4 stars": "4|5", "four stars": "4|5", "highly rated": "4|5",
    "3 stars": "3|4|5", "three stars": "3|4|5",
}


def extract_rule_based_params(query: str) -> dict:
    """Extract parameters using simple rule matching."""
    q = query.lower()
    params = {}

    for phrase, value in SORT_RULES.items():
        if phrase in q:
            params["sort"] = value
            break

    for phrase, value in AVAILABILITY_RULES.items():
        if phrase in q:
            params["availability"] = value
            break

    for phrase, value in DIFFICULTY_RULES.items():
        if phrase in q:
            params["difficulty"] = value
            break

    for phrase, value in RATING_RULES.items():
        if phrase in q:
            params["rating"] = value
            break

    return params


# ---------------------------------------------------------------------------
# Designer name detection heuristic
# ---------------------------------------------------------------------------

def looks_like_designer_name(query: str) -> Optional[str]:
    """
    Heuristic to detect a possible designer name in the query.
    Looks for capitalized multi-word phrases not at the start of a sentence.
    Returns the suspected name or None.
    """
    # Remove common query words that are capitalized but not names
    stop_words = {
        "free", "easy", "beginner", "intermediate", "advanced", "pattern",
        "patterns", "knitting", "crochet", "sweater", "hat", "sock", "shawl",
        "most", "popular", "top", "rated", "best", "adult", "baby", "child",
        "show", "find", "get", "give", "me", "some", "any", "the", "and", "or",
        "with", "for", "by", "in", "of", "a", "an", "new", "old", "latest",
        "worsted", "bulky", "lace", "dk", "fingering", "aran", "sport"
    }

    # Look for sequences of capitalized words (2+ words, each starting uppercase)
    # that don't match known stop words
    words = query.split()
    candidates = []
    i = 0
    while i < len(words):
        word = re.sub(r'[^\w]', '', words[i])
        if word and word[0].isupper() and word.lower() not in stop_words and len(word) > 1:
            # Start of a potential name
            name_parts = [words[i]]
            j = i + 1
            while j < len(words):
                next_word = re.sub(r'[^\w]', '', words[j])
                if next_word and next_word[0].isupper() and next_word.lower() not in stop_words:
                    name_parts.append(words[j])
                    j += 1
                else:
                    break
            if len(name_parts) >= 1:
                candidate = " ".join(name_parts).strip(".,?!")
                # Only consider as name if it's more than one word OR
                # it's a single uncommon capitalized word not at start of sentence
                if len(name_parts) >= 2 or (i > 0 and len(name_parts) == 1):
                    candidates.append(candidate)
            i = j
        else:
            i += 1

    # Return the longest candidate (most specific)
    if candidates:
        return max(candidates, key=len)
    return None


# ---------------------------------------------------------------------------
# ChromaDB client (lazy-loaded)
# ---------------------------------------------------------------------------

_chroma_client = None
_collections = {}


def get_collection(name: str):
    global _chroma_client, _collections
    if _chroma_client is None:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if name not in _collections:
        try:
            _collections[name] = _chroma_client.get_collection(name)
        except Exception:
            return None
    return _collections[name]


def vector_search(collection_name: str, query: str, n_results: int = 3) -> list[dict]:
    """Search a ChromaDB collection and return results with distances."""
    col = get_collection(collection_name)
    if col is None:
        return []
    try:
        results = col.query(query_texts=[query], n_results=n_results)
        items = []
        for i, doc_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            # Convert cosine distance to similarity score (0-1)
            confidence = max(0.0, 1.0 - distance)
            items.append({
                "id": doc_id,
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "confidence": round(confidence, 3)
            })
        return items
    except Exception as e:
        return []


# ---------------------------------------------------------------------------
# Main retrieval function
# ---------------------------------------------------------------------------

def retrieve_context(user_query: str) -> dict:
    """
    Main entry point. Called before the LLM with the raw user query.
    Returns resolved entities and a pre-formatted prompt injection string.
    """
    result = {
        "designer": None,
        "categories": [],
        "attributes": [],
        "fit_params": [],
        "fiber": None,
        "needle_size": None,
        "parameters": {},
        "prompt_injection": "",
        "rag_resolved": False
    }

    db_available = CHROMA_DIR.exists()

    # 1. Rule-based parameter extraction (always runs, no DB needed)
    rule_params = extract_rule_based_params(user_query)
    if rule_params:
        result["parameters"] = rule_params
        result["rag_resolved"] = True

    if not db_available:
        result["prompt_injection"] = _build_prompt_injection(result)
        return result

    # 2. Designer detection
    suspected_name = looks_like_designer_name(user_query)
    if suspected_name:
        hits = vector_search("designers", suspected_name, n_results=1)
        if hits and hits[0]["confidence"] >= DESIGNER_THRESHOLD:
            meta = hits[0]["metadata"]
            result["designer"] = {
                "name": suspected_name,
                "display_name": meta.get("display_name", ""),
                "username": meta.get("username", ""),
                "designer_id": meta.get("designer_id", ""),
                "confidence": hits[0]["confidence"]
            }
            result["rag_resolved"] = True

    # 3. Category detection
    cat_hits = vector_search("categories", user_query, n_results=2)
    for hit in cat_hits:
        if hit["confidence"] >= CATEGORY_THRESHOLD:
            result["categories"].append({
                "display": hit["metadata"].get("display", ""),
                "pc": hit["metadata"].get("pc", ""),
                "confidence": hit["confidence"]
            })
            result["rag_resolved"] = True

    # 4. Attribute detection
    attr_hits = vector_search("attributes", user_query, n_results=3)
    for hit in attr_hits:
        if hit["confidence"] >= ATTRIBUTE_THRESHOLD:
            result["attributes"].append({
                "display": hit["metadata"].get("display", ""),
                "pa": hit["metadata"].get("pa", ""),
                "group": hit["metadata"].get("group", ""),
                "confidence": hit["confidence"]
            })
            result["rag_resolved"] = True

    # 5. Fiber detection
    fiber_hits = vector_search("fibers", user_query, n_results=1)
    if fiber_hits and fiber_hits[0]["confidence"] >= FIBER_THRESHOLD:
        meta = fiber_hits[0]["metadata"]
        result["fiber"] = {
            "display": meta.get("display", ""),
            "api_value": meta.get("api_value", ""),
            "confidence": fiber_hits[0]["confidence"]
        }
        result["rag_resolved"] = True

    # 6. Needle size detection
    if any(term in user_query.lower() for term in ["needle", "mm", "us size", "us 0", "us 1", "us 2", "us 3", "us 4", "us 5", "us 6", "us 7", "us 8", "us 9", "us 10", "us 11", "us 13", "us 15"]):
        needle_hits = vector_search("needle_sizes", user_query, n_results=1)
        if needle_hits and needle_hits[0]["confidence"] >= NEEDLE_THRESHOLD:
            meta = needle_hits[0]["metadata"]
            result["needle_size"] = {
                "display": meta.get("display", ""),
                "us": meta.get("us", ""),
                "mm": meta.get("mm", ""),
                "confidence": needle_hits[0]["confidence"]
            }
            result["rag_resolved"] = True

    # 7. Fit/age/gender from parameters collection
    fit_hits = vector_search("parameters", user_query, n_results=5)
    for hit in fit_hits:
        meta = hit["metadata"]
        if meta.get("param_name") in ("fit", "age", "gender", "ease") and hit["confidence"] >= CATEGORY_THRESHOLD:
            result["fit_params"].append({
                "display": meta.get("display", ""),
                "api_value": meta.get("api_value", ""),
                "param_name": meta.get("param_name", ""),
                "confidence": hit["confidence"]
            })
            result["rag_resolved"] = True

    # 8. Build prompt injection string
    result["prompt_injection"] = _build_prompt_injection(result)

    return result


def _build_prompt_injection(result: dict) -> str:
    """Format RAG results into a string for injection into the LLM prompt."""
    if not result["rag_resolved"]:
        return ""

    lines = ["RAG context — already resolved, do not override these values:"]

    if result["designer"]:
        d = result["designer"]
        lines.append(f'- Designer: "{d["display_name"]}" → designer_id: {d["designer_id"]} (confidence: {d["confidence"]})')

    for cat in result["categories"]:
        lines.append(f'- Category: pc = "{cat["pc"]}" ({cat["display"]}) (confidence: {cat["confidence"]})')

    if result["attributes"]:
        pa_values = "+".join(a["pa"] for a in result["attributes"])
        attr_displays = ", ".join(a["display"] for a in result["attributes"])
        lines.append(f'- Attributes: pa = "{pa_values}" ({attr_displays})')

    if result["fit_params"]:
        fit_values = "+".join(f["api_value"] for f in result["fit_params"])
        fit_displays = ", ".join(f["display"] for f in result["fit_params"])
        lines.append(f'- Fit: fit = "{fit_values}" ({fit_displays})')

    if result["fiber"]:
        f = result["fiber"]
        lines.append(f'- Fiber: fiber = "{f["api_value"]}" ({f["display"]})')

    if result["needle_size"]:
        n = result["needle_size"]
        lines.append(f'- Needle size: needle-size = "{n["mm"]}mm" (US {n["us"]})')

    for param, value in result["parameters"].items():
        lines.append(f'- {param.capitalize()}: {param} = "{value}"')

    lines.append("")
    lines.append("Your job: resolve any remaining parameters not listed above.")
    lines.append("Do NOT override or second-guess the RAG-resolved values above.")
    lines.append("Return JSON for unresolved parameters only.")

    return "\n".join(lines)
