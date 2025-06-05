import os, json
from tqdm import tqdm
from typing import List
from .prompts import CAPTION_PROMPT
from dashscope import MultiModalConversation
from langchain.docstore.document import Document
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_corpus_trivial(KB_PATH, parallel_image_workers: int = 16) -> List[Document]:
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
   
    return docs

