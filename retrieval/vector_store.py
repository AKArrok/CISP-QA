"""本地向量库：numpy 矩阵余弦检索，npz + json 持久化。

语料规模（约 1.3k 块）下全量内存余弦 < 10ms，无需 FAISS/Chroma。
"""
from __future__ import annotations

import json
import os

import numpy as np

import config


class VectorStore:
    def __init__(self, vectors: np.ndarray, metadata: list[dict]):
        # 归一化后用点积 = 余弦
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        self._matrix = vectors / norms
        self.metadata = metadata

    @classmethod
    def load(cls) -> "VectorStore":
        npz = np.load(config.VECTORS_PATH)
        with open(config.VECTORS_PATH + ".meta.json", encoding="utf-8") as fp:
            metadata = json.load(fp)
        return cls(npz["vectors"], metadata)

    @classmethod
    def empty(cls) -> "VectorStore":
        return cls(np.zeros((0, 1), dtype=np.float32), [])

    def save(self) -> None:
        os.makedirs(config.DATA_DIR, exist_ok=True)
        np.savez_compressed(config.VECTORS_PATH, vectors=self._matrix.astype(np.float32))
        with open(config.VECTORS_PATH + ".meta.json", "w", encoding="utf-8") as fp:
            json.dump(self.metadata, fp, ensure_ascii=False)

    def search(self, query_vec: list[float], top_k: int) -> list[dict]:
        if self._matrix.shape[0] == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        scores = self._matrix @ q
        top_idx = np.argsort(-scores)[:top_k]
        return [
            {**self.metadata[i], "score": float(scores[i])}
            for i in top_idx
        ]
