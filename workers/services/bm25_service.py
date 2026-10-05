"""
BM25 keyword vectors, computed by Qdrant itself (built-in "qdrant/bm25" model, Qdrant 1.15.2+).

We send the text; Qdrant tokenizes, drops stopwords, stems ("dragons" -> "dragon") and turns words into
stable sparse vectors. The collection's IDF modifier adds the IDF part at search time. Nothing to keep in
sync on our side: search just sends the query text with the same model and options (`query_document`).

(FastEmbed would do this client-side, but it can't be installed next to Marker: pillow version conflict.)
"""

from typing import List

from qdrant_client import models

from workers.config import Config

BM25_MODEL = "qdrant/bm25"


def bm25_options() -> dict:
    return {
        "k": Config.BM25_K,
        "b": Config.BM25_B,
        "avg_len": Config.BM25_AVG_LEN,
        "language": Config.BM25_LANGUAGE,
    }


class BM25Service:

    def document(self, text: str) -> models.Document:
        """What goes in a point's "sparse" vector slot; Qdrant turns it into the BM25 vector on upsert."""
        return models.Document(text=text, model=BM25_MODEL, options=bm25_options())

    def documents(self, texts: List[str]) -> List[models.Document]:
        return [self.document(text) for text in texts]

    def query_document(self, query: str) -> models.Document:
        """For search: query_points(query=..., using="sparse"). Must use the same model + options."""
        return models.Document(text=query, model=BM25_MODEL, options=bm25_options())
