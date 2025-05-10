import os
import json
import pickle
import faiss
import numpy as np
import logging
import re
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import openai
from collections import defaultdict

# ─── Configuration ───────────────────────────────────────────────────────
BASE_FOLDER    = "download_fcv/Structure Extraction Sample"
INDEX_PATH     = os.path.join(BASE_FOLDER, "faiss_index.bin")
META_PATH      = os.path.join(BASE_FOLDER, "metadata_mapping.pkl")
EMBED_MODEL    = "paraphrase-mpnet-base-v2"
CHAT_MODEL     = "gpt-4.1-mini-2025-04-14"
LOG_FILE       = os.path.join(BASE_FOLDER, "old_query.log")

# ─── Logging ─────────────────────────────────────────────────────────────
import os
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, mode="w")]
)
logger = logging.getLogger(__name__)

# ─── Stopwords & Cleaning ─────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
STOPWORDS    = set(stopwords.words("english"))
REP_PATTERNS = [r"\d+", r"[^\w\s]", r"\s+"]

def clean_text(text: str) -> str:
    text = text.lower()
    for pat in REP_PATTERNS:
        text = re.sub(pat, " ", text)
    return " ".join(tok for tok in text.split() if tok not in STOPWORDS and len(tok) > 2)

# ─── Load Index & Metadata ───────────────────────────────────────────────
def load_index_and_metadata():
    logger.info(f"Loading FAISS index from {INDEX_PATH}…")
    index = faiss.read_index(INDEX_PATH)
    logger.info(f"Index loaded: {index.ntotal} vectors")
    logger.info(f"Loading metadata from {META_PATH}…")
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    logger.info(f"Metadata entries = {len(metadata)}")
    return index, metadata

# ─── Embed Query ─────────────────────────────────────────────────────────
def embed_query(text: str) -> np.ndarray:
    model = SentenceTransformer(EMBED_MODEL)
    vec   = model.encode([text], convert_to_numpy=True)
    vec   = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    return vec.astype("float32")

# ─── Retrieve Snippets ───────────────────────────────────────────────────
def retrieve_sections(index, metadata, query_emb, k=1000, filter_type=None):
    dist, idxs = index.search(query_emb, k)
    hits, seen = [], set()
    for score, idx in zip(dist[0], idxs[0]):
        entry = metadata[idx]
        raw   = entry.get("content","").strip()
        if len(raw.split()) < 10:
            continue
        if not clean_text(raw):
            continue
        if raw in seen:
            continue
        seen.add(raw)

        hits.append({
            "project_id":         entry.get("project_id",""),
            "country":            entry.get("Country",""),
            "region":             entry.get("Region",""),
            "fiscal_year":        entry.get("Project Fiscal Year",""),
            "total_project_cost": entry.get("Total Project Cost",""),
            "commitment_amount":  entry.get("Commitment Amount", ""),
            "project_title":      entry.get("Project Title", ""),
            "raw_snippet":        raw,
            "score":              float(score)
        })
    logger.info(f"Retrieved {len(hits)} snippets (k={k})")
    return hits

# ─── Function Schema ─────────────────────────────────────────────────────
AGGREGATION_FUNCTION = {
    "name": "aggregate_project_info",
    "description": "For each project, produce JSON with project_id and a summary.",
    "parameters": {
        "type": "object",
        "properties": {
            "projects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "string"},
                        "summary":    {"type": "string"}
                    },
                    "required": ["project_id","summary"]
                }
            }
        },
        "required": ["projects"]
    }
}

# ─── Project-wise Aggregation ────────────────────────────────────────────
def extract_and_aggregate(
    question,
    index, metadata,
    k=1000,
    prefiltered_hits: list[dict] | None = None
):
    """
    If prefiltered_hits is provided, use it directly.
    Otherwise, call retrieve_sections().
    """
    hits = prefiltered_hits if prefiltered_hits is not None else \
           retrieve_sections(index, metadata, embed_query(question), k)

    if not hits:
        return [], None, None

    first_s     = hits[0]["score"]
    thousandth_s = hits[min(999, len(hits)-1)]["score"]

    by_proj = defaultdict(list)
    for h in hits:
        by_proj[h["project_id"]].append(h["raw_snippet"])

    prompt_parts = []
    for pid, snippets in by_proj.items():
        joined = " ".join(f"\"{s}\"" for s in snippets)
        prompt_parts.append(f"Project {pid}: {joined}")

    prompt_text = (
        "\n".join(prompt_parts)
        + f"\n\nPlease answer the question:\n\"{question}\"\n"
        + "For each project above, write one coherent, detailed paragraph explaining how it relates to the question."
        + "\nRespond with JSON via the aggregate_project_info function."
    )

    resp = openai.chat.completions.create(
        model        = CHAT_MODEL,
        messages     = [{"role":"user","content":prompt_text}],
        functions    = [AGGREGATION_FUNCTION],
        function_call={"name":AGGREGATION_FUNCTION["name"]},
        max_completion_tokens=20000
    )

    call = resp.choices[0].message.function_call
    data = json.loads(call.arguments) if call else {}
    projs = data.get("projects", [])
    return projs, first_s, thousandth_s

# ─── Portfolio-level Aggregation ─────────────────────────────────────────
def summarise_portfolio(
    question,
    index, metadata,
    k=1000,
    prefiltered_hits: list[dict] | None = None
):
    hits = prefiltered_hits if prefiltered_hits is not None else \
           retrieve_sections(index, metadata, embed_query(question), k)

    if not hits:
        return "No snippets found.", None, None

    first_s     = hits[0]["score"]
    thousandth_s = hits[min(999, len(hits)-1)]["score"]

    by_proj, meta_proj = defaultdict(list), {}
    for h in hits:
        pid = h["project_id"]
        by_proj[pid].append(h["raw_snippet"])
        if pid not in meta_proj:
            meta_proj[pid] = {
                "country": h["country"],
                "cost":    h["total_project_cost"],
                "commitment_amount": h["commitment_amount"]
            }

    prompt_parts = []
    for pid, snippets in by_proj.items():
        m     = meta_proj[pid]
        label = f"Project {pid} (Country: {m['country']}, Cost: {m['cost']})"
        joined = " ".join(f"\"{s}\"" for s in snippets)
        prompt_parts.append(f"{label}: {joined}")

    prompt_text = (
        "\n".join(prompt_parts)
        + f"\n\nPlease answer the question:\n\"{question}\"\n"
        + "Write a concise portfolio-level summary."
        #+ "highlighting similarities and differences across all projects."
    )

    resp = openai.chat.completions.create(
        model              = CHAT_MODEL,
        messages           = [{"role":"user","content":prompt_text}],
        max_completion_tokens=20000
    )
    return resp.choices[0].message.content.strip(), first_s, thousandth_s

# ── in fcv_query.py, near the bottom ──────────────────────────────────────

# ai_cluster_summaries.py

import openai
from collections import defaultdict
from fcv_query import CHAT_MODEL  # wherever you define CHAT_MODEL

def ai_cluster_summaries(
    all_hits: list[dict],
    cluster_map: dict[str,int],
    max_snippets: int = 1000
) -> dict[int,str]:
    """
    Group all_hits by cluster_map[project_id], then for each cluster
    take the top-scoring max_snippets raw_snippet examples and ask ChatGPT
    for a theme. Returns {cluster_id: summary_text}.
    """
    # 1) bucket by cluster
    by_cluster: dict[int, list[dict]] = defaultdict(list)
    for h in all_hits:
        pid = h["project_id"]
        cl  = cluster_map.get(pid)
        if cl is not None:
            by_cluster[cl].append(h)

    summaries: dict[int,str] = {}
    # 2) for each cluster, sort by score, pick top N
    for cl, hits in by_cluster.items():
        hits_sorted = sorted(hits, key=lambda x: x.get("score", 0), reverse=True)
        examples = [h["raw_snippet"] for h in hits_sorted[:max_snippets]]

        # 3) build prompt
        prompt_text = (
            f"These are representative snippets for cluster {cl}:\n"
            + "\n".join(f"- {snippet}" for snippet in examples)
            + "\n\nIn around 200 words and a few bullet points, briefly summarize key similarities of all projects in this cluster."
            + "\nAlso give an approximate estimate of number of projects in this cluster."
        )

        # 4) GPT call
        resp = openai.chat.completions.create(
            model                 = CHAT_MODEL,
            messages              = [{"role":"user","content":prompt_text}],
            max_completion_tokens = 20000
        )

        # 5) save summary
        summaries[cl] = resp.choices[0].message.content.strip()

    return summaries
