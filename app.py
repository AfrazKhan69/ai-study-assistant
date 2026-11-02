import os
import streamlit as st

from rag_index import build_index, list_documents, delete_document, safe_name
from rag_answer import answer, NOT_FOUND

st.set_page_config(page_title="AI Study Assistant", page_icon="📚")
st.title("📚 AI Study Assistant")
st.caption("Runs fully on your laptop. Ask questions about your PDFs.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_message(m):
    """Draw one chat message, plus its sources if it has any."""
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            if m.get("searched"):
                st.caption(f"Searched for: {m['searched']}")
            with st.expander("Sources (retrieved chunks)"):
                for s in m["sources"]:
                    st.markdown(
                        f"**{s['doc']} · page {s['page']}** · similarity {s['score']:.2f}"
                    )
                    st.text(s["text"])
                    st.divider()


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Add documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs", type="pdf", accept_multiple_files=True
    )

    if uploaded_files and st.button("Process PDFs"):
        existing = list_documents()
        os.makedirs("uploads", exist_ok=True)

        for up in uploaded_files:
            name = safe_name(up.name)
            if name in existing:
                st.info(f"{up.name} is already processed, skipped.")
                continue

            pdf_path = os.path.join("uploads", name + ".pdf")
            with open(pdf_path, "wb") as f:
                f.write(up.getbuffer())

            bar = st.progress(0.0, text=f"Embedding {up.name}...")
            try:
                build_index(
                    pdf_path,
                    name,
                    progress=lambda frac, bar=bar, n=up.name: bar.progress(
                        frac, text=f"Embedding {n}..."
                    ),
                )
                st.success(f"Added {up.name}")
            except ValueError as e:
                st.error(f"{up.name}: {e}")
            bar.empty()

    st.header("Your library")
    docs = list_documents()
    if docs:
        # the key changes when the library changes, so new documents get selected
        selected = st.multiselect(
            "Search in", docs, default=docs, key="sel_" + "|".join(docs)
        )
        to_delete = st.selectbox("Remove a document", ["-"] + docs)
        if to_delete != "-" and st.button(f"Delete {to_delete}"):
            delete_document(to_delete)
            st.rerun()
    else:
        selected = []

    st.header("Settings")
    min_score = st.slider(
        "Minimum similarity", 0.0, 1.0, 0.55, 0.01,
        help="Chunks scoring below this are ignored. Higher = stricter.",
    )

# ---------- Stop if nothing is ready ----------
if not docs:
    st.info("Upload PDFs in the sidebar and click **Process PDFs** to start.")
    st.stop()
if not selected:
    st.warning("Select at least one document in the sidebar.")
    st.stop()

# ---------- Show earlier messages ----------
for m in st.session_state.messages:
    show_message(m)

# ---------- Handle a new question ----------
question = st.chat_input("Ask about your notes")
if question:
    history = list(st.session_state.messages)  # history BEFORE the new question
    user_msg = {"role": "user", "content": question}
    st.session_state.messages.append(user_msg)
    show_message(user_msg)

    with st.spinner("Thinking..."):
        reply, sources, standalone = answer(
            question, selected, history, min_score=min_score
        )

    found = NOT_FOUND.lower() not in reply.lower()
    assistant_msg = {
        "role": "assistant",
        "content": reply,
        "sources": sources if found else [],
        "searched": standalone if found else "",
    }
    st.session_state.messages.append(assistant_msg)
    show_message(assistant_msg)