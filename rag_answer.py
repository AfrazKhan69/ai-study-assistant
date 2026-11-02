import re
import ollama

from rag_index import search, list_documents

CHAT_MODEL = "qwen3:1.7b"
NOT_FOUND = "I couldn't find this in the notes."


def clean(text):
    """Remove any <think>...</think> block the model may add."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def rewrite_question(question, history):
    """Turn a follow-up like 'explain that more simply' into a standalone question."""
    if not history:
        return question

    convo = ""
    for m in history[-4:]:  # only the last 4 messages
        convo += f"{m['role'].upper()}: {m['content'][:300]}\n"

    prompt = f"""Rewrite the LAST QUESTION so it can be understood on its own, using the conversation for context.
If it is already clear on its own, return it unchanged.
Return ONLY the rewritten question, nothing else.

CONVERSATION:
{convo}
LAST QUESTION: {question}

/no_think"""

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "num_ctx": 2048},
    )
    rewritten = clean(response["message"]["content"])
    return rewritten or question


def build_prompt(question, results):
    context = ""
    for r in results:
        context += f"[{r['doc']}, page {r['page']}]\n{r['text']}\n\n"

    return f"""You are a study assistant. Answer the question using ONLY the notes below.
If the notes do not contain the answer, say "{NOT_FOUND}"
Keep the answer short and clear.

NOTES:
{context}
QUESTION: {question}

/no_think"""


def answer(question, doc_names, history=None, top_k=4, min_score=0.0):
    standalone = rewrite_question(question, history or [])
    results = search(standalone, doc_names, top_k=top_k)

    # drop weak chunks
    results = [r for r in results if r["score"] >= min_score]
    if not results:
        return NOT_FOUND, [], standalone  # skip the model entirely

    prompt = build_prompt(standalone, results)
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2, "num_ctx": 4096},
    )
    return clean(response["message"]["content"]), results, standalone


if __name__ == "__main__":
    docs = list_documents()
    print("Searching in:", docs)
    history = []
    while True:
        q = input("\nAsk a question (or type 'quit'): ")
        if q.strip().lower() == "quit":
            break
        reply, sources, standalone = answer(q, docs, history, min_score=0.55)
        print("\n(searched for:", standalone + ")")
        print("\nANSWER:\n" + reply)
        print("\nSOURCES:", [(r["doc"], r["page"], round(r["score"], 2)) for r in sources])
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": reply})