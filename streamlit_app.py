import json
import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
api = httpx.Client(base_url=API_URL, timeout=300)

st.set_page_config(page_title="Production RAG", layout="wide")
st.title("Ask your PDFs")

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Add PDFs", type="pdf", accept_multiple_files=True)
    category = st.text_input("Category (optional metadata)")
    if uploads and st.button("Ingest"):
        for f in uploads:
            meta = json.dumps({"category": category} if category else {})
            with st.spinner(f"Indexing {f.name}..."):
                r = api.post("/ingest", files={"file": (f.name, f.getvalue(), "application/pdf")},
                             data={"metadata": meta})
            if r.is_success:
                d = r.json()
                st.success(f"{f.name}: already indexed" if d.get("skipped")
                           else f"{f.name}: {d['num_chunks']} chunks")
            else:
                st.error(f"{f.name}: {r.text}")

    try:
        docs = api.get("/documents").json()
    except httpx.HTTPError:
        st.error(f"API not reachable at {API_URL}")
        st.stop()
    for d in docs:
        st.caption(f"{d['source']} ({d['num_pages']} pages, {d['num_chunks']} chunks)")

    st.header("Filters")
    selected = st.multiselect("Only search these documents", [d["source"] for d in docs])
    max_page = max([d["num_pages"] for d in docs] + [2])
    pages = st.slider("Page range", 1, max_page, (1, max_page))
    top_k = st.slider("Sources per answer", 1, 10, 5)

question = st.text_input("Question")
if question:
    flt = {"sources": selected or None, "page_gte": pages[0], "page_lte": pages[1]}
    if category:
        flt["metadata"] = {"category": category}
    with st.spinner("Searching..."):
        r = api.post("/query", json={"question": question, "top_k": top_k, "filter": flt})
    if not r.is_success:
        st.error(r.text)
        st.stop()
    a = r.json()

    st.markdown(a["answer"])
    if a["refused"]:
        st.info("No source passed the relevance threshold, or the answer could not be grounded in one.")
    for c in a["citations"]:
        with st.expander(f"[{c['number']}] {c['source']}, page {c['page']}"):
            st.write(c["snippet"])

    t = a["timings_ms"]
    st.caption(f"{t.get('total', 0):.0f} ms total, retrieval {t.get('retrieval_total', 0):.0f} ms"
               f"{' (cached)' if a['cache_hit'] else ''}, generation {t.get('generation', 0):.0f} ms")
    with st.expander("Retrieved chunks and scores"):
        for s in a["sources"]:
            score = s["rerank_score"]
            st.markdown(f"**{s['source']} p.{s['page']}** rerank={score:.3f}" if score is not None
                        else f"**{s['source']} p.{s['page']}** score={s['score']:.3f}")
            st.text(s["text"][:600])
