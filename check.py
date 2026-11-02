from rag_index import search, list_documents

docs = list_documents()
print("Documents:", docs)

questions = [
    "what is hadoop",
    "what is YARN",
    "Who won the 2011 cricket World Cup?",
]

for q in questions:
    results = search(q, docs, top_k=4)
    print(f"\n{q}")
    print("  scores:", [round(r["score"], 2) for r in results])
    print("  from:  ", [(r["doc"], r["page"]) for r in results])