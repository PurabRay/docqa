"""DocQA web UI: a Documents page and a Chat page. No business logic here.

Run: uv run streamlit run ui/streamlit_app.py
All state lives in st.session_state; all HTTP goes through ui/api_client.py.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from typing import Any

import httpx
import streamlit as st
from api_client import ApiClient, ApiError  # streamlit puts ui/ on sys.path
from streamlit_pdf_viewer import pdf_viewer

from docqa.settings import load_settings

SETTINGS = load_settings(env_file=None)
PRIVACY_WARNING = (
    "Answers come from free-tier hosted models, which may use prompts to improve their "
    "products. Don't upload confidential files (use local mode for private documents)."
)
NOTICES = {
    "low_relevance": "I couldn't find this in your documents. Try rephrasing.",
    "insufficient_context": "The documents don't seem to answer this.",
}


IN_PROCESS = "inprocess"  # DOCQA_API_URL=inprocess: run the API inside this process


@st.cache_resource
def in_process_http() -> httpx.Client:
    """Streamlit Community Cloud: one API app per server process, called without a network hop."""
    from fastapi.testclient import TestClient

    from docqa.api.app import create_app

    client = TestClient(create_app())
    client.__enter__()  # runs the lifespan (Mongo client, worker) for the process lifetime
    return client


def api() -> ApiClient:
    """One client per browser session, with its own session id."""
    if "api" not in st.session_state:
        url = SETTINGS.ui_api_url()
        http = (
            in_process_http()
            if url == IN_PROCESS
            else httpx.Client(base_url=url, timeout=SETTINGS.ui.request_timeout_s)
        )
        st.session_state.api = ApiClient(http, session_id=uuid.uuid4().hex)
        st.session_state.messages = []
    client: ApiClient = st.session_state.api
    return client


def documents_page() -> None:
    """Upload (up to ui.max_uploads), live status table, delete."""
    st.header("Documents")
    docs = api().list_documents()
    room = SETTINGS.ui.max_uploads - len(docs)
    files = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True, disabled=room <= 0
    )
    if room <= 0:
        st.info(
            f"This session already has {SETTINGS.ui.max_uploads} documents; "
            "delete one to add another."
        )
    if files and st.button("Upload"):
        for file in files[:room]:
            try:
                api().upload(file.name, file.getvalue())
            except ApiError as err:
                st.error(f"{file.name}: {err.message}")
        if len(files) > room:
            st.warning(
                f"Only the first {room} files were uploaded (limit {SETTINGS.ui.max_uploads})."
            )
    for doc in docs:
        cols = st.columns([4, 2, 1, 1])
        cols[0].write(doc["filename"])
        cols[1].write(doc["status"] if not doc["error"] else f"failed: {doc['error']}")
        cols[2].write(f"{doc['page_count']} pages")
        if cols[3].button("Delete", key=f"del-{doc['doc_id']}"):
            api().delete(doc["doc_id"])
            st.rerun()
    if any(d["status"] not in ("ready", "failed") for d in docs):
        time.sleep(SETTINGS.ui.poll_interval_s)  # poll until every document is ready or failed
        st.rerun()


def chat_page() -> None:
    """Pick ready documents, ask, read the streamed answer, open citations, give feedback."""
    st.header("Chat")
    ready = {d["doc_id"]: d["filename"] for d in api().list_documents() if d["status"] == "ready"}
    if not ready:
        st.info("Upload a document and wait until it is ready.")
        return
    chosen = st.multiselect("Documents", list(ready), default=list(ready), format_func=ready.get)
    for index, message in enumerate(st.session_state.messages):
        show_message(index, message, ready)
    question = st.chat_input("Ask about your documents")
    if question and chosen:
        st.session_state.messages.append({"role": "user", "text": question})
        st.session_state.messages.append(ask(question, chosen))
        st.rerun()
    show_citation_viewer()


def ask(question: str, doc_ids: list[str]) -> dict[str, Any]:
    """Stream one answer into the page and return it as a chat message."""
    message: dict[str, Any] = {"role": "assistant", "text": "", "citations": [], "notice": None}

    def tokens() -> Iterator[str]:
        for event, data in api().query(question, doc_ids):
            if event == "meta":
                message["trace_id"] = data["trace_id"]
            elif event == "token":
                yield data["text"]
            elif event == "citations":
                message["citations"] = data["citations"]
            elif event == "abstain":
                message["notice"] = NOTICES[data["reason"]]
            elif event == "degraded":
                message["notice"] = (
                    "No language model is available right now; here are the top passages."
                )
                message["passages"] = [p["chunk"]["text"] for p in data["passages"]]
            elif event == "error":
                message["notice"] = data["message"]

    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            message["text"] = st.write_stream(tokens()) or ""
        except ApiError as err:
            message["notice"] = err.message
    return message


def show_message(index: int, message: dict[str, Any], docs: dict[str, str]) -> None:
    """One chat bubble with citation chips and feedback."""
    with st.chat_message(message["role"]):
        st.write(message["text"])
        if message.get("notice"):
            st.warning(message["notice"])
        for passage in message.get("passages", []):
            st.caption(passage)
        names = {name: doc_id for doc_id, name in docs.items()}
        for n, citation in enumerate(message.get("citations", [])):
            label = f"[{citation['doc_name']}, p. {citation['page']}]"
            if st.button(label, key=f"cite-{index}-{n}", help=citation["quote"]):
                st.session_state.open_citation = (
                    names.get(citation["doc_name"]),
                    citation["page"],
                    citation["quote"],
                )
        if message["role"] == "assistant" and message.get("trace_id"):
            feedback_controls(index, message["trace_id"])


def feedback_controls(index: int, trace_id: str) -> None:
    """Thumbs up/down with an optional comment."""
    rating = st.feedback("thumbs", key=f"fb-{index}")
    comment = st.text_input("Comment (optional)", key=f"fbc-{index}")
    if rating is not None and st.button("Send feedback", key=f"fbs-{index}"):
        api().feedback(trace_id, 1 if rating == 1 else -1, comment or None)
        st.success("Thanks for the feedback.")


def show_citation_viewer() -> None:
    """The cited page of the PDF, with the quote highlighted."""
    citation = st.session_state.get("open_citation")
    if not citation or not citation[0]:
        return
    doc_id, page, quote = citation
    found = api().highlights(doc_id, page, quote)
    boxes = [
        {
            "page": page,
            "x": r["x"],
            "y": r["y"],
            "width": r["width"],
            "height": r["height"],
            "color": "red",
        }
        for r in found["rects"]
    ]
    st.subheader(f"Page {page}")
    if not boxes:
        st.caption("The quote could not be located on this page; showing the page.")
    pdf_viewer(api().file(doc_id), annotations=boxes, pages_to_render=[page], scroll_to_page=page)


def main() -> None:
    """Page layout."""
    st.set_page_config(page_title="DocQA", layout="wide")
    st.warning(PRIVACY_WARNING)
    page = st.sidebar.radio("Page", ["Documents", "Chat"])
    try:
        documents_page() if page == "Documents" else chat_page()
    except (ApiError, httpx.HTTPError) as err:
        st.error(f"The API is not reachable or returned an error: {err}")


main()
