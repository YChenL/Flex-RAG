import os, torch, pathlib
from tqdm import tqdm
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers.cross_encoder import CrossEncoder
from langchain.docstore.document import Document
from .utils import classify_block  


# 本地模型路径
LOCAL_MODEL_DIR = pathlib.Path(
    "/data/yfli/code/Flex-RAG/models/MiniLM-L-6-v2"
).expanduser()
if not LOCAL_MODEL_DIR.exists():
    raise FileNotFoundError(f"模型目录不存在: {LOCAL_MODEL_DIR}")

# 初始化Cross-Encoder
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cross_encoder = CrossEncoder(str(LOCAL_MODEL_DIR), device=DEVICE)


# 单块评分（文本/媒体统一，只用 page_content）
MAX_LEN = 512        # 适当截断，防止超长度

def ce_score_block(query: str, block: Document) -> float:
    """Cross-Encoder 给 chunk 打 0~1 分；失败返回 0."""
    try:
        ctx = block.page_content[:4096]              # 4k 字符硬截断
        pair = (query, ctx[:MAX_LEN])
        score = cross_encoder.predict([pair])[0]     # 原始范围 ~ (-inf, +inf)
        # 映射到 0~1 Sigmoid；若模型本身就是 [0,1] 可省略
        return float(torch.sigmoid(torch.tensor(score)).item())
    except Exception as e:
        print("Cross-Encoder 评分失败:", e)
        return 0.0


# 平坦 chunks 的重排：文本 / 媒体分组，各取前 k
def rerank_chunks_ce(
    query: str,
    chunks: List[Document],
    n_text: int,
    n_media: int,
    batch: int = 8,
) -> Tuple[List[Document], List[Document]]:
    text_chunks  = [c for c in chunks if classify_block(c) == "text"]
    media_chunks = [c for c in chunks if classify_block(c) == "media"]

    def score_block(b): return (ce_score_block(query, b), b)

    scored_text, scored_media = [], []
    with ThreadPoolExecutor(max_workers=batch) as ex:
        futures = {ex.submit(score_block, b): b for b in text_chunks + media_chunks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="CE Scoring"):
            score, blk = fut.result()
            (scored_text if blk in text_chunks else scored_media).append((score, blk))

    scored_text.sort(key=lambda x: x[0], reverse=True)
    scored_media.sort(key=lambda x: x[0], reverse=True)

    top_text  = [blk for _, blk in scored_text[:n_text]]
    top_media = [blk for _, blk in scored_media[:n_media]]

    print(f"Cross-Encoder：选出 {len(top_text)} 文本 + {len(top_media)} 媒体")
    return top_text, top_media



# 分层切分重排：沿用逻辑更换评分器
def rerank_parents_ce(
    query: str,
    parents: List[Document],
    n_text: int,
    n_media: int,
    batch: int = 8,
) -> Tuple[List[Document], List[Document]]:
    text_blocks  = [p for p in parents if p.metadata["type"] in {"parent", "text"}]
    media_blocks = [p for p in parents if p.metadata["type"] in {"image", "table"}]

    def score_block(b): return (ce_score_block(query, b), b)

    scored_text, scored_media = [], []
    with ThreadPoolExecutor(max_workers=batch) as ex:
        futures = {ex.submit(score_block, b): ("text" if b in text_blocks else "media")
                   for b in text_blocks + media_blocks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="CE Scoring"):
            typ = futures[fut]
            score, blk = fut.result()
            (scored_text if typ == "text" else scored_media).append((score, blk))

    scored_text.sort(key=lambda x: x[0], reverse=True)
    scored_media.sort(key=lambda x: x[0], reverse=True)

    top_text  = [blk for _, blk in scored_text[:n_text]]
    top_media = [blk for _, blk in scored_media[:n_media]]

    return top_text, top_media
