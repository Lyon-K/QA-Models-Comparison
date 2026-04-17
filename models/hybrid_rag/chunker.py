from typing import List, Dict


def chunk_text_by_words(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping chunks by words.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def build_chunks(documents: List[Dict], chunk_size: int, overlap: int) -> List[Dict]:
    """
    Convert full documents into chunk-level records.
    """
    chunk_records = []

    for doc in documents:
        chunks = chunk_text_by_words(doc["text"], chunk_size, overlap)

        for chunk_idx, chunk_text in enumerate(chunks):
            chunk_records.append(
                {
                    "chunk_id": f'{doc["doc_id"]}_{chunk_idx}',
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "source": doc["source"],
                    "chunk_index": chunk_idx,
                    "text": chunk_text,
                }
            )

    return chunk_records