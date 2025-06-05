"""
Merge every *.json file under /data/huali_data/  (each is a list of instances)
into one big JSON list and save to /data/huali_mm/huali_corpus.json
"""
import os, json
from tqdm import tqdm
from pathlib import Path
from itertools import islice
from collections import Counter, defaultdict
from langchain.docstore.document import Document

from .prompts import CAPTION_PROMPT
from dashscope import MultiModalConversation


# 初始化captioner
qwen_key = os.getenv("DASHSCOPE_API_KEY")
if not qwen_key:
    raise RuntimeError("请先确保 DASHSCOPE_API_KEY 已正确设置并激活了 mmrag 环境")

def img_cap(image_path):
    messages = [{"role": "system",
                "content": [{"text": "You are a helpful assistant for image captioning. Think step by step."}]},
                {'role':'user',
                'content': [{'image': image_path},   
                            {'text': CAPTION_PROMPT}
                           ]}]
    response = MultiModalConversation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
        api_key=qwen_key,
        model='qwen2.5-vl-7b-instruct', #'qwen2-vl-2b-instruct' free
        messages=messages,
        vl_high_resolution_images=False)

    return response["output"]["choices"][0]["message"].content[0]["text"]



def _process_image_inst(inst, IMAGE_ROOT):
    """
    辅助函数：给一个 inst 调用 img_cap 并返回一个 Document
    """
    img_path = os.path.join(IMAGE_ROOT, inst["img_path"])
    cap_list = inst.get("img_caption") or []
    descrip_list = [img_cap(img_path).strip()]
    return Document(
        page_content=" ".join(cap_list + descrip_list),
        metadata={
            "type":        "image",
            "book_idx":    inst.get("book_idx", -1),
            "page_idx":    inst.get("page_idx", -1),
            "img_path":    img_path,
            "img_caption": cap_list,
            "img_descrip": descrip_list,
            **{k: inst.get(k) for k in ("img_footnote",) if inst.get(k) is not None}
        }
    )



def _process_table_inst(inst, IMAGE_ROOT):
    """
    辅助函数：给一个 inst 调用 img_cap 并返回一个 Document
    """
    img_path = os.path.join(IMAGE_ROOT, inst["img_path"])
    cap_list = inst.get("table_caption") or []
    descrip_list = [img_cap(img_path).strip()]
    return Document(
        page_content=" ".join(cap_list + descrip_list),
        metadata={
            "type":        "table",
            "book_idx":    inst.get("book_idx", -1),
            "page_idx":    inst.get("page_idx", -1),
            "img_path":    img_path,
            "table_caption": cap_list,
            "table_descrip": descrip_list,
            **{b: inst.get(b) for b in ("table_body",) if inst.get(b) is not None},
            **{k: inst.get(k) for k in ("table_footnote",) if inst.get(k) is not None}
        }
    )



def analyze_kb_types(kb_path):
    with open(kb_path, encoding="utf-8") as f:
        all_insts = json.load(f)

    type_counter = Counter()
    missing_img_path_idxs = []
    samples = defaultdict(list)

    for idx, inst in enumerate(tqdm(all_insts, desc="Analyzing KB instances")):
        t = inst.get("type")
        type_counter[t] += 1

        if t == "image" and not inst.get("img_path"):
            missing_img_path_idxs.append(idx)

        if len(samples[t]) < 5:
            samples[t].append(inst)

    print(f"\n一共发现 {len(type_counter)} 种不同的 type：")
    for t, cnt in type_counter.items():
        print(f"  - {t!r}: {cnt} 条")
    if missing_img_path_idxs:
        print(f"\n注意：有 {len(missing_img_path_idxs)} 条 type='image' 的记录缺少 'img_path'，示例索引：{missing_img_path_idxs[:10]}{'...' if len(missing_img_path_idxs)>10 else ''}")
    else:
        print("\n所有 type='image' 的记录均包含 'img_path'。")

    print("\n每种 type 的前 5 个实例（完整内容）：")
    for t, inst_list in samples.items():
        print(f"\n--- Type = {t!r} （共 {type_counter[t]} 条） ---")
        for i, inst in enumerate(inst_list, start=1):
            print(f"{i}. {inst}")



def save_docs(docs, OUTPUT_DOCS= Path("/data/huali_mm/docs.json")):
    serializable = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in docs
    ]
    with open(OUTPUT_DOCS, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(docs)} docs to {OUTPUT_DOCS}")

