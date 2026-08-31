"""构建索引：知识块的检索文本 → embedding → vectors.npz。

按稳定 Chunk ID 复用未变化向量；删除、重排或新增 Chunk 时保持索引与 JSON 对齐。
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def main() -> None:
    import numpy as np
    from llms import get_embeddings
    from retrieval.vector_store import VectorStore

    with open(config.KB_CHUNKS_PATH, encoding="utf-8") as fp:
        chunks = json.load(fp)

    old_vectors, old_meta = None, []
    if os.path.exists(config.VECTORS_PATH):
        npz = np.load(config.VECTORS_PATH)
        old_vectors, old_meta = npz["vectors"], json.load(open(config.VECTORS_PATH + ".meta.json", encoding="utf-8"))
        print(f"已有索引: {len(old_meta)} 块，断点续跑")

    emb = get_embeddings()
    from data_ingest.manifest import read_manifest
    previous_model = read_manifest().get("embedding_model")
    # 仅当清单记录的模型与当前完全一致时才复用旧向量；清单缺失（旧索引）无法确认
    # 模型身份，也强制全量重建，避免换了后端后静默混用不同模型的向量。
    reuse_old = previous_model == emb.model_identity
    old_by_id = {}
    if reuse_old and old_vectors is not None:
        old_by_id = {meta["id"]: old_vectors[i] for i, meta in enumerate(old_meta)}
    elif old_meta:
        print(f"Embedding 模型变化: {previous_model} → {emb.model_identity}，全量重建")

    todo = [c for c in chunks if c["id"] not in old_by_id]
    print(f"待向量化: {len(todo)} / {len(chunks)} 块")
    new_by_id = {}
    if todo:
        new_vectors = emb.embed_documents([
            c.get("embedding_text", c["text"]) for c in todo
        ])
        new_by_id = {
            chunk["id"]: np.asarray(vector, dtype=np.float32)
            for chunk, vector in zip(todo, new_vectors, strict=True)
        }

    vectors = np.asarray([
        old_by_id[c["id"]] if c["id"] in old_by_id else new_by_id[c["id"]]
        for c in chunks
    ], dtype=np.float32)
    metadata = [{"id": c["id"]} for c in chunks]

    store = VectorStore(vectors, metadata)
    store.save()
    from data_ingest.manifest import update_manifest
    update_manifest(
        vector_count=len(metadata),
        embedding_model=emb.model_identity,
    )
    print(f"索引已保存: {len(metadata)} 块 × {vectors.shape[1]} 维 → {config.VECTORS_PATH}")


if __name__ == "__main__":
    main()
