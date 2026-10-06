<<<<<<< HEAD
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
=======
import asyncio
from pathlib import Path
import time

import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests

load_dotenv()

st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)


def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path


async def send_rag_ingest_event(pdf_path: Path) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )


st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    with st.spinner("Uploading and triggering ingestion..."):
        path = save_uploaded_pdf(uploaded)
        # Kick off the event and block until the send completes
        asyncio.run(send_rag_ingest_event(path))
        # Small pause for user feedback continuity
        time.sleep(0.3)
    st.success(f"Triggered ingestion for: {path.name}")
    st.caption("You can upload another PDF if you like.")

st.divider()
st.title("Ask a question about your PDFs")


async def send_rag_query_event(question: str, top_k: int) -> None:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )

    return result[0]


def _inngest_api_base() -> str:
    # Local dev server default; configurable via env
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")
        time.sleep(poll_interval_s)


with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("How many chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Sending event and generating answer..."):
            # Fire-and-forget event to Inngest for observability/workflow
            event_id = asyncio.run(send_rag_query_event(question.strip(), int(top_k)))
            # Poll the local Inngest API for the run's output
            output = wait_for_run_output(event_id)
            answer = output.get("answer", "")
            sources = output.get("sources", [])

        st.subheader("Answer")
        st.write(answer or "(No answer)")
        if sources:
            st.caption("Sources")
            for s in sources:
                st.write(f"- {s}")
>>>>>>> 887dabe57a6aba3b9c98461ffc930e31cf02fe74
