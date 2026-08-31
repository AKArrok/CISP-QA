"""BM25 稀疏检索（jieba 分词），从 kb_chunks.json 内存构建，无需外部索引。"""
from __future__ import annotations

import math
import re
from collections import Counter

import jieba

jieba.setLogLevel(60)  # 静音建词日志

_STOPWORDS = {"的", "了", "和", "是", "在", "与", "及", "或", "一个", "以下", "以下哪一项", "下列",
              "哪些", "哪一种", "哪项", "对于", "关于", "通过", "进行", "可以", "能够", "主要"}


def tokenize(text: str) -> list[str]:
    words = jieba.lcut(text.lower())
    return [w for w in words if w.strip() and w not in _STOPWORDS and not re.fullmatch(r"[\W\d]", w)]


class BM25Index:
    def __init__(self, docs_tokens: list[list[str]], metadata: list[dict]):
        self.metadata = metadata
        self.k1, self.b = 1.5, 0.75
        self.doc_lens = [len(d) for d in docs_tokens]
        self.avgdl = sum(self.doc_lens) / max(len(docs_tokens), 1)
        self.tfs = [Counter(d) for d in docs_tokens]
        df: Counter = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(docs_tokens)
        # BM25+ 风格平滑 IDF
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    @classmethod
    def build(cls, chunks: list[dict]) -> "BM25Index":
        return cls([
            tokenize(c.get("embedding_text", c["text"])) for c in chunks
        ], chunks)

    def search(self, query: str, top_k: int) -> list[dict]:
        q_tokens = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tfs):
            if not tf:
                scores.append(0.0)
                continue
            score = 0.0
            dl = self.doc_lens[i]
            for t in q_tokens:
                if t in tf:
                    score += self.idf.get(t, 0.0) * tf[t] * (self.k1 + 1) / (
                        tf[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                    )
            scores.append(score)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [
            {**self.metadata[i], "score": scores[i]}
            for i in ranked if scores[i] > 0
        ]
