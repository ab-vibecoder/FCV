# dashboard_full.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import defaultdict

from fcv_query import (
    load_index_and_metadata,
    embed_query, retrieve_sections,
    extract_and_aggregate, summarise_portfolio,
    ai_cluster_summaries
)

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
st.set_page_config("FCV Dashboard", layout="wide")
st.title("🌐 Large-Index FCV Dashboard")

@st.cache_resource
def get_resources():
    return load_index_and_metadata()

idx, meta = get_resources()

# ────────────────────────────────────────────────────────────────────────────
#  Sidebar: summary controls in a form
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar.form("summary_controls"):
    query        = st.text_input("🔍 Enter your query")
    use_heading  = st.checkbox("Enable heading-based screening", value=True)
    k            = st.slider("Snippets to retrieve (k)", 10000, 800000, 200000, 10000)
    mode         = st.radio("Answer granularity", ["Project-wise","Entire portfolio"])
    sel_pid      = st.multiselect("Project ID",   sorted({m["project_id"] for m in meta}))
    sel_cou      = st.multiselect("Country",      sorted({m["Country"]    for m in meta}))
    sel_reg      = st.multiselect("Region",       sorted({m["Region"]     for m in meta}))
    sel_year     = st.multiselect("Fiscal Year",  sorted({m["Project Fiscal Year"] for m in meta}))
    sel_doctype = st.multiselect("Document Type", sorted({m["document_type"] for m in meta}))
    run_summary = st.form_submit_button("Run Summary")

# ────────────────────────────────────────────────────────────────────────────
#  Sidebar: clustering controls (outside the form)
# ────────────────────────────────────────────────────────────────────────────
num_clusters   = st.sidebar.slider("Number of clusters", 2, 20, 6)
run_clustering = st.sidebar.button("Run Clustering Analysis")

# ────────────────────────────────────────────────────────────────────────────
#  SUMMARY SECTION
# ────────────────────────────────────────────────────────────────────────────
if run_summary:
    # 1) Retrieve top-k (with or without heading filter)
    all_hits = retrieve_sections(
        idx, meta,
        question=query,
        query_emb=embed_query(query),
        k=k,
        filter_type=None,
        use_heading_filter=use_heading
    )

    # 2) Apply facet filters first
    filtered_hits = all_hits
    if sel_pid:  filtered_hits = [h for h in filtered_hits if h["project_id"] in sel_pid]
    if sel_cou:  filtered_hits = [h for h in filtered_hits if h["country"]    in sel_cou]
    if sel_reg:  filtered_hits = [h for h in filtered_hits if h["region"]     in sel_reg]
    if sel_year: filtered_hits = [h for h in filtered_hits if h["fiscal_year"] in sel_year]
    if sel_doctype: filtered_hits = [h for h in filtered_hits if h["document_type"] in sel_doctype]

    # 3) Sort by combined score and cap to 5 per project
    filtered_hits_sorted = sorted(
        filtered_hits,
        key=lambda h: (h["score"] + h.get("heading_match_score",0)) / 2,
        reverse=True
    )
    limited_hits = []
    counts = defaultdict(int)
    for h in filtered_hits_sorted:
        pid = h["project_id"]
        if counts[pid] < 5:
            limited_hits.append(h)
            counts[pid] += 1

    # 4) If still empty, warn and bail out
    if not limited_hits:
        st.warning("No snippets survived your filters. Please broaden your query or selections.")
        st.stop()

    # 5) Build DataFrame & metrics
    df = pd.DataFrame(limited_hits)
    st.markdown(f"**{len(df)} snippets** after filtering and capping")
    total_proj  = len({m['project_id'] for m in meta})
    num_matched = df['project_id'].nunique()
    st.metric("Projects matched", f"{num_matched} / {total_proj}")

    # 6) Coverage vs. Confidence
    st.subheader("📊 Coverage vs. Confidence")
    stats = (
        df.groupby("project_id")
          .agg(
             snippet_count=("raw_snippet","count"),
             avg_score    =("score","mean")
          )
          .reset_index()
    )
    fig_stats = px.scatter(
        stats, x="snippet_count", y="avg_score",
        hover_name="project_id",
        title="Per-Project Snippet Count vs. Avg. Score",
        labels={"snippet_count":"# Snippets","avg_score":"Avg. Score"}
    )
    st.plotly_chart(fig_stats, use_container_width=True)

    # 7) Table & Score distribution
    st.dataframe(df[[
        'project_id','country','region','fiscal_year', "document_type",
        'commitment_amount','heading','raw_snippet','score', 'heading_match_score'
    ]])
    st.subheader("Score Distribution")
    fig, ax = plt.subplots()
    ax.hist(df['score'], bins=20)
    st.pyplot(fig)

    # 8) Word Cloud
    st.subheader("☁️ Word Cloud")
    wc_mode = st.radio("Generate for …", ["Overall portfolio","Single project"])
    if wc_mode=="Single project":
        pid  = st.selectbox("Project", sorted(df['project_id'].unique()))
        text = " ".join(df[df['project_id']==pid]['raw_snippet'])
    else:
        text = " ".join(df['raw_snippet'])
    if text:
        st.image(
            WordCloud(width=800, height=300, background_color="white")
            .generate(text).to_array()
        )
    else:
        st.info("No text to display.")

    # 9) SHOW exactly which snippets go to ChatGPT
    if mode=="Entire portfolio":
        by_proj     = defaultdict(list)
        for h in limited_hits:
            by_proj[h["project_id"]].append(h)
        one_per_proj = [
            max(v, key=lambda x: (x["score"] + x.get("heading_match_score",0)) / 2)
            for v in by_proj.values()
        ]

        st.subheader("🗂️ Snippets sent for Portfolio Summary")
        st.metric("Unique projects in prompt", len(one_per_proj))
        st.dataframe(pd.DataFrame(one_per_proj)[[
            "project_id","country","commitment_amount", "document_type", "heading","raw_snippet","score","heading_match_score"
        ]])

        # 10a) Call summariser
        summary, first_s, thousandth_s = summarise_portfolio(
            query, idx, meta, k,
            prefiltered_hits=one_per_proj
        )

    else:
        st.subheader("🗂️ Snippets sent for Project-wise Summary")
        st.metric("Unique projects in prompt", df['project_id'].nunique())
        st.dataframe(df[[
            "project_id","country","commitment_amount", "document_type", "heading","raw_snippet","score","heading_match_score"
        ]])

        # 10b) Call project-wise
        projects, first_s, thousandth_s = extract_and_aggregate(
            query, idx, meta, k,
            prefiltered_hits=limited_hits
        )

    # ────────────────────────────────────────────────────────────────────────
    #  OUTPUT SUMMARY with meta_map for title lookup
    # ────────────────────────────────────────────────────────────────────────
    # build a quick map from limited_hits to country & title
    meta_map = {h["project_id"]: (h["country"], h["project_title"]) for h in limited_hits}

    st.subheader("📝 Answers")
    st.write(f"First score: **{first_s:.3f}**, 1000th: **{thousandth_s:.3f}**")
    if mode=="Entire portfolio":
        st.markdown("### Portfolio-wide Summary")
        st.markdown(summary)
    else:
        for p in projects:
            pid     = p["project_id"]
            country, title = meta_map.get(pid, ("",""))
            hdr     = f"{pid} — {country}"
            with st.expander(hdr):
                st.markdown(f"**Summary**: {p['summary']}")

# ────────────────────────────────────────────────────────────────────────────
#  CLUSTERING SECTION (runs only when you click its button)
# ────────────────────────────────────────────────────────────────────────────
if run_clustering:
    st.subheader("🔸 Project Clustering (3D)")
    all_proj_ids = sorted({e["project_id"] for e in meta})
    proj_to_idxs = defaultdict(list)
    for i, entry in enumerate(meta):
        proj_to_idxs[entry["project_id"]].append(i)

    emb_list = []
    for pid in all_proj_ids:
        vecs = np.vstack([idx.reconstruct(i) for i in proj_to_idxs[pid]])
        avg  = vecs.mean(axis=0)
        norm = np.linalg.norm(avg)
        emb_list.append(avg/(norm if norm else 1.0))
    emb_matrix = np.vstack(emb_list)

    km      = KMeans(n_clusters=num_clusters, random_state=0).fit(emb_matrix)
    labels  = km.labels_
    coords3 = PCA(n_components=3, random_state=0).fit_transform(emb_matrix)

    df_proj = pd.DataFrame({
        "project_id": all_proj_ids,
        "cluster":    labels,
        "PC1":        coords3[:,0],
        "PC2":        coords3[:,1],
        "PC3":        coords3[:,2],
    })

    # AI-driven cluster summaries (over your limited_hits set)
    cluster_map       = dict(zip(all_proj_ids, labels))
    cluster_summaries = ai_cluster_summaries(limited_hits, cluster_map, max_snippets=200)

    st.subheader("🔹 Cluster Themes")
    for cl, theme in sorted(cluster_summaries.items()):
        st.markdown(f"**Cluster {cl}:** {theme}")

    df_proj["cluster"] = df_proj["cluster"].astype(str)
    st.metric("Projects in this scatter plot", df_proj.shape[0])

    fig3d = px.scatter_3d(
        df_proj,
        x="PC1", y="PC2", z="PC3",
        color="cluster",
        hover_name="project_id",
        color_discrete_sequence=px.colors.qualitative.Bold,
        opacity=0.7,
        width=800, height=600,
        title="3D PCA of Projects (by cluster)"
    )
    fig3d.update_traces(marker=dict(size=4))
    st.plotly_chart(fig3d, use_container_width=True)

# streamlit run dashboard_full.py
