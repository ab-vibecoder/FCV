import os
import re
import json
import pickle
import logging
import faiss
import numpy as np                          # ← added for isinstance check
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import openai

# -------------------------------------------------------------------------
# Configuration Constants
# -------------------------------------------------------------------------
BASE_FOLDER   = "download_fcv/Structure Extraction Sample"
INDEX_PATH    = os.path.join(BASE_FOLDER, "faiss_index.bin")
META_PATH     = os.path.join(BASE_FOLDER, "metadata_mapping.pkl")
EMBED_MODEL   = "paraphrase-mpnet-base-v2"
CHAT_MODEL    = "gpt-4.1-nano-2025-04-14"
LOG_FILE      = os.path.join(BASE_FOLDER, "fcv_query.log")


# ----------------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------------
import os
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="w")
    ]
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Text cleanup patterns
# ----------------------------------------------------------------------------
REP_PATTERNS = [r"\d+", r"[^\w\s]", r"\s+"]

def clean_text(text: str) -> str:
    # ← handle non-str (e.g. numpy.ndarray) by coercing to python str
    if not isinstance(text, str):
        try:
            # for 0-dim ndarray
            text = text.item()
        except Exception:
            text = str(text)
    # now safe to lower()
    text = text.lower()
    for pat in REP_PATTERNS:
        text = re.sub(pat, " ", text)
    return " ".join(tok for tok in text.split() if len(tok) > 2)

# -------------------------------------------------------------------------
# Load FAISS index & metadata
# -------------------------------------------------------------------------
def load_index_and_metadata():
    logger.info(f"Loading FAISS index from {INDEX_PATH}…")
    index = faiss.read_index(INDEX_PATH)
    logger.info(f"Index loaded: {index.ntotal} vectors")
    logger.info(f"Loading metadata from {META_PATH}…")
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    logger.info(f"Metadata entries = {len(metadata)}")
    return index, metadata

# -------------------------------------------------------------------------
# Embed query using SentenceTransformer
# -------------------------------------------------------------------------
def embed_query(text: str) -> np.ndarray:
    model = SentenceTransformer(EMBED_MODEL)
    vecs = model.encode([text], convert_to_numpy=True)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs.astype("float32")

# -------------------------------------------------------------------------
# Retrieve & heading-filtered snippets
# -------------------------------------------------------------------------
def retrieve_sections(
    index,
    metadata,
    question: str,
    query_emb: np.ndarray,
    k: int = 1000,
    max_search: int = 800000,
    filter_type: str | None = None,
    section_label: str | None = None,
    doc_type: str | None = None,
    use_heading_filter: bool = True
) -> list[dict]:
    """
    1) Pull top `max_search` hits from FAISS
    2) If use_heading_filter, only keep hits whose heading tokens overlap with question tokens
    3) Return up to `k` snippets, each with:
         - 'score' = raw FAISS inner‐product (∈[-1,1])
         - 'heading_match_score' = fraction of query‐tokens found in heading (∈[0,1])
    """
    # 1) tokenize the question
    qtokens = set(clean_text(question).split())

    # 2) FAISS search
    raw_dists, raw_idxs = index.search(query_emb, max_search)

    hits = []
    seen = set()
    for raw_score, idx in zip(raw_dists[0], raw_idxs[0]):
        entry = metadata[idx]

        # apply your existing metadata filters
        if section_label and str(entry.get("section_label","")).lower() != section_label.lower():
            continue
        if doc_type    and str(entry.get("doc_type","")).lower()      != doc_type.lower():
            continue
        if filter_type and str(entry.get("type","")).lower()         != filter_type.lower():
            continue

        heading = entry.get("heading","") or ""
        content = entry.get("content","").strip()
        if len(content.split()) < 10:
            continue

        # heading‐based screening
        if use_heading_filter:
            htokens = set(clean_text(heading).split())
            if not (qtokens & htokens):
                continue

        # dedupe by content
        if content in seen:
            continue
        seen.add(content)

        # compute heading match score
        htokens = set(clean_text(heading).split())
        heading_match_score = (len(qtokens & htokens) / len(qtokens)) if qtokens else 0.0

        hits.append({
            "project_id":         entry.get("project_id","").split("_",1)[0],
            "heading":            heading,
            "raw_snippet":        content,
            "score":              float(raw_score),        # raw inner‐product
            "heading_match_score": heading_match_score,    # new field
            "country":            entry.get("Country",""),
            "region":             entry.get("Region",""),
            "fiscal_year":        entry.get("Project Fiscal Year",""),
            "commitment_amount":  entry.get("Commitment Amount",""),
            "document_type":      entry.get("document_type", ""),
            "project_title":      entry.get("document_name","")
        })

        if len(hits) >= k:
            break

    logger.info(f"Retrieved {len(hits)} snippets (k={k}, max_search={max_search})")
    return hits

# -------------------------------------------------------------------------
# Function schema for project-wise aggregation
# -------------------------------------------------------------------------
AGGREGATION_FUNCTION = {
    "name": "aggregate_project_info",
    "description": "For each project, produce a JSON list of {project_id, country, project_title, summary}.",
    "parameters": {
        "type": "object",
        "properties": {
            "projects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "project_id":    {"type": "string"},
                        "country":       {"type": "string"},
                        "project_title": {"type": "string"},
                        "summary":       {"type": "string"}
                    },
                    "required": ["project_id","summary"]
                }
            }
        },
        "required": ["projects"]
    }
}

# -------------------------------------------------------------------------
# Per-project aggregation via ChatGPT function call
# -------------------------------------------------------------------------
def extract_and_aggregate(
    question: str,
    index,
    metadata,
    k: int = 1000,
    prefiltered_hits: list[dict] | None = None
):
    """
    If you pass `prefiltered_hits` (your dashboard’s filtered list), we
    use it verbatim.  Otherwise we do a FAISS search up to `k`.
    """
    hits = (
        prefiltered_hits
        if prefiltered_hits is not None
        else retrieve_sections(
            index, metadata,
            question, embed_query(question),
            k=k
        )
    )

    if not hits:
        return [], None, None

    first_score      = hits[0]["score"]
    thousandth_score = hits[min(k-1, len(hits)-1)]["score"]

    # group by project
    by_proj = defaultdict(list)
    for h in hits:
        by_proj[h["project_id"]].append(h["raw_snippet"])

    # build function schema prompt, now including country & title
    prompt_parts = []
    for pid, snippets in by_proj.items():
        # find metadata from any hit in this project
        entry = next(h for h in hits if h["project_id"] == pid)
        country      = entry.get("country","")
        project_title= entry.get("project_title","")
        joined       = " ".join(f"\"{s}\"" for s in snippets)
        prompt_parts.append(
            f"Project {pid} (Country: {country}, Title: \"{project_title}\"): {joined}"
        )

    prompt_text = (
        "\n".join(prompt_parts)
        + f"\n\nPlease answer the question or topic:\n\"{question}\"\n"
        + "For each project above, explain in detail in 100 to 200 words."
        + "\nRespond only with JSON via the aggregate_project_info function."
    )

    resp = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role":"system","content":"You are an expert evaluation assistant. You **must** respond by calling the function aggregate_project_info."},
            {"role":"user",  "content":prompt_text}
        ],
        functions    =[AGGREGATION_FUNCTION],
        function_call={"name":AGGREGATION_FUNCTION["name"]},
        max_completion_tokens=10000
    )

    fc   = resp.choices[0].message.function_call
    data = json.loads(fc.arguments) if fc else {}
    return data.get("projects", []), first_score, thousandth_score

# -------------------------------------------------------------------------
# Portfolio-level summarization via a single ChatGPT call
# -------------------------------------------------------------------------
def summarise_portfolio(
    question: str,
    index,
    metadata,
    k: int = 1000,
    prefiltered_hits: list[dict] | None = None
):
    # unchanged...
    hits = (
        prefiltered_hits
        if prefiltered_hits is not None
        else retrieve_sections(
            index, metadata,
            question, embed_query(question),
            k=k
        )
    )
    if not hits:
        return "No relevant snippets found.", None, None

    first_score      = hits[0]["score"]
    thousandth_score = hits[min(k-1, len(hits)-1)]["score"]

    # pick one best per project (unchanged)…
    best_per_proj = {}
    for h in hits:
        pid = h["project_id"]
        if pid not in best_per_proj or h["score"] > best_per_proj[pid]["score"]:
            best_per_proj[pid] = h
    snippets = list(best_per_proj.values())

    # build short prompt (unchanged)…
    by_proj, meta_proj = defaultdict(list), {}
    for h in snippets:
        pid = h["project_id"]
        by_proj[pid].append(h["raw_snippet"])
        if pid not in meta_proj:
            meta_proj[pid] = {
                "country":           h["country"],
                "commitment_amount": h["commitment_amount"]
            }

    prompt_parts = []
    for pid, snips in by_proj.items():
        m = meta_proj[pid]
        txt = snips[0]
        if len(txt) > 120:
            txt = txt[:120].rsplit(" ",1)[0] + "…"
        prompt_parts.append(f"{pid}|{m['country']}|{m['commitment_amount']}: {txt}")

    prompt_text = (
        "You are an expert portfolio analyst.\n\n"
        "Here is one representative snippet per project:\n"
        + "\n".join(prompt_parts)
        + f"\n\nQuestion: {question}\n"
        "Write a well-structured and detailed portfolio-level summary of the above."
    )

    resp = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"user","content":prompt_text}],
        max_completion_tokens=4000
    )

    return resp.choices[0].message.content.strip(), first_score, thousandth_score

# -------------------------------------------------------------------------
# AI-driven cluster summaries
# -------------------------------------------------------------------------
def ai_cluster_summaries(
    all_hits: list[dict],
    cluster_map: dict[str,int],
    max_snippets: int = 1000
) -> dict[int,str]:
    # unchanged…
    by_cluster: dict[int, list[dict]] = defaultdict(list)
    for h in all_hits:
        pid = h["project_id"]
        cl  = cluster_map.get(pid)
        if cl is not None:
            by_cluster[cl].append(h)

    summaries: dict[int,str] = {}
    for cl, hits in by_cluster.items():
        hits_sorted = sorted(hits, key=lambda x: x.get("score", 0), reverse=True)
        examples    = [h["raw_snippet"] for h in hits_sorted[:max_snippets]]

        prompt_text = (
            f"These are representative snippets for cluster {cl}:\n"
            + "\n".join(f"- {snippet}" for snippet in examples)
            + "\n\nIn around 100 words and a few bullet points "
            + "briefly summarize key similarities (in bold) of all projects in this cluster."
        )

        resp = openai.chat.completions.create(
            model                 = CHAT_MODEL,
            messages              = [{"role":"user","content":prompt_text}],
            max_completion_tokens = 1000
        )
        summaries[cl] = resp.choices[0].message.content.strip()

    return summaries
