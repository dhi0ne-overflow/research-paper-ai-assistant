def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # clean chunk
        chunk = chunk.strip()
        if len(chunk) > 100:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks