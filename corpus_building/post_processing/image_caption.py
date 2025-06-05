import os, json
from tqdm import tqdm
from typing import List
from langchain.docstore.document import Document
from concurrent.futures import ThreadPoolExecutor, as_completed
from .utils import _process_image_inst


def load_corpus_image(KB_PATH, IMAGE_ROOT, parallel_image_workers: int = 16) -> List[Document]:
    docs: List[Document] = []
    with open(KB_PATH, encoding="utf-8") as f:
        all_insts = json.load(f)

    # 1) 先把非 image 类型的都处理好
    image_insts = []
    for inst in all_insts:
        t = inst.get("type")
        if t == "text":
            docs.append(Document(
                page_content=inst["text"],
                metadata={
                    "type":     "text",
                    "book_idx": inst.get("book_idx", -1),
                    "page_idx": inst.get("page_idx", -1),
                    **{k: inst.get(k) for k in ("text_level",) if inst.get(k) is not None}
                }
            ))
        elif t == "equation":
            docs.append(Document(
                page_content=inst["text"],
                metadata={
                    "type":     "equation",
                    "book_idx": inst.get("book_idx", -1),
                    "page_idx": inst.get("page_idx", -1),
                    **{k: inst.get(k) for k in ("text_format",) if inst.get(k) is not None}
                }
            ))
        elif t == "table":
            try:
                img_path = os.path.join(IMAGE_ROOT, inst.get("img_path",""))
                cap_list = inst.get("table_caption") or []
                body_list = [inst.get("table_body")] or []
                docs.append(Document(
                    page_content=" ".join(cap_list + body_list),
                    metadata={
                        "type":          "table",
                        "book_idx":      inst.get("book_idx", -1),
                        "page_idx":      inst.get("page_idx", -1),
                        "img_path":      img_path,
                        "table_caption": cap_list,
                        "table_body":    body_list,
                        **{k: inst.get(k) for k in ("table_footnote",) if inst.get(k) is not None}
                    }
                ))
            except:
                print(inst)
                
        elif t == "image":
            image_insts.append(inst)
        # 其它类型若有，可继续 elif

    # 2) 并行处理所有 image_inst
    with ThreadPoolExecutor(max_workers=parallel_image_workers) as ex:
        # 用 as_completed 可以加进度条
        futures = { ex.submit(_process_image_inst, inst, IMAGE_ROOT): inst for inst in image_insts }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Captioning images"):
            try:
                docs.append(fut.result())
            except Exception as e:
                inst = futures[fut]
                print(f"[Error] image inst {inst.get('img_path')} caption failed: {e}")

    print(f"Loaded {len(docs)} documents")
    return docs
