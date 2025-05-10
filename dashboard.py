import os, pandas as pd, matplotlib.pyplot as plt
import streamlit as st
from wordcloud import WordCloud

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

from fcv_se_query2 import (
    load_index_and_metadata,
    embed_query, retrieve_sections,
    extract_and_aggregate,
    summarise_portfolio # now returns portfolio summary when called from dashboard
)

st.set_page_config("FCV Query Dashboard", layout="wide")
st.title("🌐 FCV Query-Answer Dashboard")

@st.cache_resource
def get_resources():
    return load_index_and_metadata()

idx, meta = get_resources()

query = st.text_input("🔍 Enter your query")
k     = st.sidebar.slider("Snippets to retrieve (k)", 10, 10000, 1000)
mode  = st.sidebar.radio("Answer granularity", ["Project-wise", "Entire portfolio"])
if not query:
    st.stop()

with st.spinner("Searching…"):
    hits = retrieve_sections(idx, meta, embed_query(query), k)

if not hits:
    st.warning("No snippets matched.")
    st.stop()

df = pd.DataFrame(hits)

st.sidebar.markdown("### 📑 Facet filters")
sel_doc   = st.sidebar.multiselect("doc_type", sorted(df.doc_type.unique()))
sel_sec   = st.sidebar.multiselect("section_label", sorted(df.section_label.dropna().unique()))
sel_proj  = st.sidebar.multiselect("project_id", sorted(df.project_id.unique()))
min_score = st.sidebar.slider("Min score", 0.0, 1.0, 0.0, 0.01)

mask = (
    (df.score >= min_score) &
    (df.doc_type.isin(sel_doc)      if sel_doc  else True) &
    (df.section_label.isin(sel_sec) if sel_sec else True) &
    (df.project_id.isin(sel_proj)   if sel_proj else True)
)
df = df[mask]
st.markdown(f"**{len(df)} snippets** after filtering")

st.dataframe(df[['project_id','doc_type','section_label','raw_snippet','score']])

c1, c2 = st.columns(2)
with c1:
    st.subheader("Matches per Project")
    st.bar_chart(df.project_id.value_counts())
with c2:
    st.subheader("Score Distribution")
    fig, ax = plt.subplots()
    ax.hist(df.score, bins=20)
    st.pyplot(fig)

st.subheader("☁️ Word Cloud")
wc_mode = st.radio("Generate for …", ["Overall portfolio", "Single project"])
if wc_mode == "Single project":
    proj_choice = st.selectbox("Choose project", sorted(df.project_id.unique()))
    text_src    = " ".join(df[df.project_id==proj_choice].raw_snippet)
else:
    text_src    = " ".join(df.raw_snippet)

if text_src.strip():
    st.image(WordCloud(width=800, height=300, background_color="white")
             .generate(text_src).to_array())
else:
    st.info("No text available for word cloud.")

if mode == "Entire portfolio":
    with st.spinner("Generating portfolio summary…"):
        summary, first_s, hundredth_s = summarise_portfolio(query, idx, meta, k=k)
    st.write(
        f"First snippet score: **{first_s or 'N/A'}**   |   "
        f"100th: **{hundredth_s or 'N/A'}**"
    )
    st.markdown("### Portfolio-wide Summary")
    st.markdown(summary)

# Different answer format
# st.subheader("📝 Answers")
# if mode == "Entire portfolio":
#    summary, first_s, hundredth_s = extract_and_aggregate(query, idx, meta, k=k)
#    st.write(f"First snippet score: **{first_s or 'N/A'}**   |   100th: **{hundredth_s or 'N/A'}**")
#    st.markdown("### 📋 Portfolio-wide Summary")
#    st.write(summary)
else:
    projects, first_s, hundredth_s = extract_and_aggregate(query, idx, meta, k=k)
    st.write(f"First snippet score: **{first_s or 'N/A'}**   |   100th: **{hundredth_s or 'N/A'}**")
    for proj in projects:
        header = proj["project_id"]
        if proj.get("Project Title"): header += f" — {proj['Project Title']}"
        if proj.get("Country"):       header += f" ({proj['Country']})"
        with st.expander(header):
            st.markdown(f"**Summary**: {proj['summary']}")
            # show up to 10 raw snippets for this project
            # for s in df[df['project_id']==pid]['raw_snippet'].head(10):
            #    st.write("•", s)
            # To run in local terminal: streamlit run dashboard.py