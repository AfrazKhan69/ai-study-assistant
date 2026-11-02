from pypdf import PdfReader


def read_pdf(path):
    """Return a list of {page, text} for every page in the PDF."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"page": i + 1, "text": text})
    return pages


def chunk_text(text, chunk_size=400, overlap=100):
    """Split text into pieces of ~chunk_size characters that overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def build_chunks(path):
    """Read the PDF and return a list of {page, text} chunks."""
    all_chunks = []
    for page in read_pdf(path):
        for piece in chunk_text(page["text"]):
            if piece.strip():
                all_chunks.append({"page": page["page"], "text": piece})
    return all_chunks


if __name__ == "__main__":
    chunks = build_chunks("sample.pdf")
    print("Total chunks:", len(chunks))
    print("\nChunk 1 (page", chunks[0]["page"], "):")
    print(chunks[0]["text"])
