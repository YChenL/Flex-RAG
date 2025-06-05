import os, json
from tqdm import tqdm
from typing import List
from langchain.docstore.document import Document
from concurrent.futures import ThreadPoolExecutor, as_completed
from .utils import _process_image_inst, _process_table_inst


def load_corpus_mm(KB_PATH, IMAGE_ROOT, parallel_image_workers: int = 16) -> List[Document]:
    """
    读取 KB_PATH → 解析四类 inst → 并行调用 Qwen-VL
      · image  : 调用 _process_image_inst → caption+description
      · table  : 调用 _process_table_inst → caption+LLM description
      · text   : 直接写入
      · equation: 直接写入
    返回统一的 docs 列表
    """
    docs: List[Document] = []
    # ------- 读取原始 JSON -------
    with open(KB_PATH, encoding="utf-8") as f:
        all_insts = json.load(f)

    # ------- 按类型分桶 -------
    image_insts, table_insts = [], []
    for inst in all_insts:
        t = inst.get("type")
        if t == "text":
            docs.append(
                Document(
                    page_content=inst["text"],
                    metadata={
                        "type":     "text",
                        "book_idx": inst.get("book_idx", -1),
                        "page_idx": inst.get("page_idx", -1),
                        **{k: inst.get(k) for k in ("text_level",) if inst.get(k) is not None}
                    },
                )
            )
        elif t == "equation":
            docs.append(
                Document(
                    page_content=inst["text"],
                    metadata={
                        "type":     "equation",
                        "book_idx": inst.get("book_idx", -1),
                        "page_idx": inst.get("page_idx", -1),
                        **{k: inst.get(k) for k in ("text_format",) if inst.get(k) is not None}
                    },
                )
            )
        elif t == "image":
            image_insts.append(inst)
        elif t == "table":
            table_insts.append(inst)
        # 其它类型可继续 elif

    # ------- 并行处理 image & table -------
    media_total = len(image_insts) + len(table_insts)
    if media_total:
        with ThreadPoolExecutor(max_workers=parallel_image_workers) as ex:
            fut2inst = {}

            # 提交任务
            for inst in image_insts:
                fut2inst[ex.submit(_process_image_inst, inst, IMAGE_ROOT)] = ("image", inst)
            for inst in table_insts:
                fut2inst[ex.submit(_process_table_inst, inst, IMAGE_ROOT)] = ("table", inst)

            # 收集结果
            for fut in tqdm(
                as_completed(fut2inst),
                total=media_total,
                desc="Captioning media (image+table)",
            ):
                typ, inst = fut2inst[fut]
                try:
                    docs.append(fut.result())
                except Exception as e:
                    print(f"[Error] {typ} inst {inst.get('img_path')} caption failed: {e}")

    print(f"Loaded {len(docs)} documents")
    return docs
