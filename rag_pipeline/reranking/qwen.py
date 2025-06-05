import os, tiktoken
from tqdm import tqdm
from typing import List, Tuple
from dotenv import load_dotenv
from dashscope import MultiModalConversation
from langchain.docstore.document import Document
from concurrent.futures import ThreadPoolExecutor, as_completed
from .utils import classify_block


load_dotenv()
ENC = tiktoken.get_encoding("o200k_base")
qwen_key = os.getenv("DASHSCOPE_API_KEY")
if not qwen_key:
    raise RuntimeError("请先确保 DASHSCOPE_API_KEY 已正确设置并激活了 mmrag 环境")

# ---------- Qwen 调用：单块评分 ----------
def qwen_score_block(query: str, block: Document) -> float:
    """给父块打相关分（0~1）。若失败返回 0."""
    t   = block.metadata["type"]
    txt = block.page_content[:2000]      # ⬅️ 如过长截断，确保单块 < 4k tok
    if t in {"image", "table"}:
        # 图或表：把图片+描述 / 表格描述发进去
        img_path = block.metadata.get("img_path")
        user_content = []
        if img_path and os.path.exists(img_path) and t == "image":
            user_content.append({"image": img_path})
        user_content.append({"text":
            f"Query: {query}\n\nContext ({t}):\n{txt}\n\n"
            "请给出其与 Query 的相关性分数，0~1 间小数。仅回复分数字面值。"})
    else:
        # 文本块
        user_content = [{"text":
            f"Query: {query}\n\nContext (text):\n{txt}\n\n"
            "请给出其与 Query 的相关性分数，0~1 间小数。仅回复分数字面值。"}]

    messages = [
        {"role": "system",
         "content": [{"text": "You are a helpful assistant for relevance scoring."}]},
        {"role": "user", "content": user_content}
    ]

    try:
        resp = MultiModalConversation.call(
            api_key = qwen_key,
            model   = "qwen2.5-vl-7b-instruct",
            messages= messages,
            vl_high_resolution_images=False
        )
        score_txt = resp["output"]["choices"][0]["message"].content[0]["text"]
        return float(score_txt.strip())
    except Exception as e:
        print("评分失败:", e)
        return 0.0


# ───────────平坦 chunks 的重排 ───────────
def rerank_chunks_with_llm(
    query: str,
    chunks: List[Document],
    n_text: int,         # ← 需多少文本块
    n_media: int,        # ← 需多少多媒体块
    batch: int = 8,      # 并行线程数
) -> Tuple[List[Document], List[Document]]:
    """
    对同质 chunks 做 LLM 相关性重排。
    - 文本块 & 多媒体块分别打分、排序、各取前 k。
    - 返回 (top_text_chunks, top_media_chunks)
    """
    # 1) 分类
    text_chunks  = [c for c in chunks if classify_block(c) == "text"]
    media_chunks = [c for c in chunks if classify_block(c) == "media"]

    # 2) 并行逐块评分
    def score_and_pack(block):
        s = qwen_score_block(query, block)
        return (s, block)

    scored_text, scored_media = [], []

    with ThreadPoolExecutor(max_workers=batch) as ex:
        futures = {
            ex.submit(score_and_pack, blk): blk for blk in text_chunks + media_chunks
        }
        for fut in tqdm(as_completed(futures),
                        total=len(futures),
                        desc="Qwen Scoring"):
            score, blk = fut.result()
            (scored_text if blk in text_chunks else scored_media).append((score, blk))

    # 3) 排序 & 截断
    scored_text.sort(key=lambda x: x[0], reverse=True)
    scored_media.sort(key=lambda x: x[0], reverse=True)

    top_text  = [blk for _, blk in scored_text[:n_text]]
    top_media = [blk for _, blk in scored_media[:n_media]]

    # 4) 打印统计（可选）
    print(
        f"最终选出 {len(top_text)} 段文本 + {len(top_media)} 个多媒体块 "
        f"(文本/多媒体请求阈值 = {n_text}/{n_media})"
    )
    return top_text, top_media


# ---------- rerank ----------
def rerank_parents_with_llm(
    query: str,
    parents: List[Document],
    n_text: int,  # TOP_TEXT
    n_media: int, # TOP_MEDIA
    batch: int,   # BATCH
) -> Tuple[List[Document], List[Document]]:
    """返回 (top_text_parents, top_media_parents)"""
    # 1) 分类
    text_blocks  = [p for p in parents if p.metadata["type"] in {"parent","text"}]
    media_blocks = [p for p in parents if p.metadata["type"] in {"image","table"}]

    # 2) 并行逐块评分
    scored_text  = []
    scored_media = []

    def score_and_pack(block):
        s = qwen_score_block(query, block)
        return (s, block)

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {
            ex.submit(score_and_pack, blk): ("text" if blk in text_blocks else "media")
            for blk in text_blocks + media_blocks
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Qwen Scoring"):
            typ = futures[fut]
            score, blk = fut.result()
            (scored_text if typ=="text" else scored_media).append((score, blk))

    # 3) 排序 & 截断
    scored_text.sort(key=lambda x: x[0], reverse=True)
    scored_media.sort(key=lambda x: x[0], reverse=True)

    top_text  = [blk for _, blk in scored_text[:n_text]]
    top_media = [blk for _, blk in scored_media[:n_media]]

    return top_text, top_media
