"""构建索引：kb_chunks.json → embedding（Ark doubao）→ vectors.npz。

支持断点续跑：已有 vectors.npz 时仅对新增 chunk 调 embedding。
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

    done_ids = {m["id"] for m in old_meta}
    todo = [c for c in chunks if c["id"] not in done_ids]
    print(f"待向量化: {len(todo)} / {len(chunks)} 块")
    if todo:
        emb = get_embeddings()
        new_vectors = emb.embed_documents([c["text"] for c in todo])
        if old_vectors is not None and old_vectors.size:
            import numpy as _np
            vectors = _np.vstack([old_vectors, _np.asarray(new_vectors, dtype=old_vectors.dtype)])
        else:
            vectors = np.asarray(new_vectors, dtype=np.float32)
        metadata = old_meta + [{"id": c["id"]} for c in todo]
    else:
        vectors, metadata = old_vectors, old_meta

    store = VectorStore(vectors, metadata)
    store.save()
    from data_ingest.manifest import update_manifest
    model_identity = None
    if todo:
        model_identity = emb.model_identity  # 仅真实调用 embedding 时更新模型身份
    update_manifest(
        vector_count=len(metadata),
        **({"embedding_model": model_identity} if model_identity else {}),
    )
    print(f"索引已保存: {len(metadata)} 块 × {vectors.shape[1]} 维 → {config.VECTORS_PATH}")


if __name__ == "__main__":
    main()
