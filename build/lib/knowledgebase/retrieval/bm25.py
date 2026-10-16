"""Okapi BM25 as a sparse matrix — scoring a query is one sparse column-sum.

Exact names (repo names, library names, error strings) dominate personal-recall
queries; embeddings blur them, BM25 nails them.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import sparse

from knowledgebase.text import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.vocab: dict[str, int] = {}
        self.matrix: sparse.csc_matrix | None = None
        self.n_docs = 0

    def fit(self, texts: list[str]) -> "BM25Index":
        rows, cols, tfs = [], [], []
        lengths = np.zeros(len(texts), dtype=np.float32)
        for doc_id, text in enumerate(texts):
            counts = Counter(tokenize(text))
            lengths[doc_id] = sum(counts.values())
            for term, tf in counts.items():
                col = self.vocab.setdefault(term, len(self.vocab))
                rows.append(doc_id)
                cols.append(col)
                tfs.append(tf)
        self.n_docs = len(texts)
        if not self.n_docs:
            self.matrix = sparse.csc_matrix((0, 0), dtype=np.float32)
            return self
        tf = sparse.csr_matrix((np.asarray(tfs, dtype=np.float32), (rows, cols)), shape=(self.n_docs, len(self.vocab)))
        df = np.bincount(np.asarray(cols), minlength=len(self.vocab)).astype(np.float32)
        idf = np.log1p((self.n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        avgdl = float(lengths.mean()) or 1.0
        norm = self.k1 * (1 - self.b + self.b * lengths / avgdl)
        tf = tf.tocoo()
        data = tf.data * (self.k1 + 1) / (tf.data + norm[tf.row]) * idf[tf.col]
        self.matrix = sparse.csc_matrix((data.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)
        return self

    def scores(self, query: str) -> np.ndarray:
        if self.matrix is None or not self.n_docs:
            return np.zeros(self.n_docs, dtype=np.float32)
        ids = [self.vocab[t] for t in tokenize(query) if t in self.vocab]
        if not ids:
            return np.zeros(self.n_docs, dtype=np.float32)
        return np.asarray(self.matrix[:, ids].sum(axis=1)).ravel()
