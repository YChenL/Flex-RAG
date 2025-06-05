"""
Merge every *.json file under /data/huali_data/  (each is a list of instances)
into one big JSON list and save to /data/huali_mm/huali_corpus.json
"""
import json
from tqdm import tqdm
from pathlib import Path
from itertools import islice
from collections import Counter, defaultdict
from langchain.docstore.document import Document



def load_serialized_docs(path: Path):
    with open(path, encoding="utf-8") as f:
        raw_items = json.load(f)         # list[dict]

    # 重新构造成 Document
    docs = [
        Document(page_content=item["page_content"],
                 metadata=item["metadata"])
        for item in raw_items
    ]
    return docs



def preview_docs_by_type(docs, n_preview=5):
    """按 metadata['type'] 分组打印前 n_preview 个 Document"""
    buckets = defaultdict(list)
    for d in docs:
        buckets[d.metadata["type"]].append(d)
    
    for t in ["text", "image", "table", "equation"]:
        lst = buckets.get(t, [])
        print(f"\n=== {t.upper()}  (共 {len(lst)} 条) ===")
        for i, d in enumerate(islice(lst, n_preview), 1):
            print(f"{i}.", d)


# def union_docs(docs):
#     unique_docs, seen = [], set()
#     for d in docs:
#         key = (d.metadata["book_idx"], d.metadata["page_idx"], d.page_content.strip())
#         if key not in seen:
#             unique_docs.append(d)
#             seen.add(key)
#     docs = unique_docs
#     print(f"After dedup: {len(docs)}")


