import os, pathlib, torch
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from sentence_transformers.cross_encoder import CrossEncoder
from langchain.docstore.document import Document

from .utils import classify_block   # 需保证现有 utils 中存在


class CrossEncoder_Reranker():
    """
    通用 Cross-Encoder 重排器
    --------------------------------------------------------------
    • 支持把 **本地模型目录** 作为参数传入，兼容 e5 / MiniLM / Qwen3-Reranker 等。
    • 自动判断模型输出是否已在 0-1 区间；若不是则套 Sigmoid。
    • 提供平坦 chunks 与分层 parents 两种重排 API。
    """

    def __init__(
        self,
        model_dir: str | os.PathLike,
        device: str | None = None,
        max_len: int = 512,
        prob_already: bool | None = None,
        trust_remote_code: bool = True,
    ):
        """
        Parameters
        ----------
        model_dir        : 本地模型文件夹路径（CrossEncoder / SBERT 结构）
        device           : "cuda" | "cpu" | None(自动)
        max_len          : 每对 (query, context) 的截断长度
        prob_already     : 显式告诉模型输出是否已在 0~1；若 None 自动判断
        trust_remote_code: 对 Qwen3 等含自定义 pooling 的模型需保持 True
        """
        self.model_dir = str(pathlib.Path(model_dir).expanduser())
        if not pathlib.Path(self.model_dir).exists():
            raise FileNotFoundError(f"模型目录不存在: {self.model_dir}")

        self.device   = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len  = max_len

        self.ce = CrossEncoder(
            self.model_dir,
            device=self.device,
            trust_remote_code=trust_remote_code,
            max_length=max_len
        )

        if prob_already is None:
            # Qwen3-Reranker 系列官方说明输出 range [0,1]
            prob_already = "qwen" in self.model_dir.lower()
        self.prob_already = prob_already


    # 单块评分
    def _score_block(self, query: str, block: Document) -> float:
        """对单个 Document 计算相关性分数"""
        ctx = block.page_content[:4096]               # 粗截断
        pair = (query, ctx[: self.max_len])

        try:
            score = self.ce.predict([pair])[0]        # float
            if not self.prob_already:
                score = torch.sigmoid(torch.tensor(score)).item()
            return float(score)
        except Exception as e:
            print("Cross-Encoder 评分失败:", e)
            return 0.0

    
    # 平坦 chunks 重排
    def rerank_chunks(
        self,
        query: str,
        chunks: List[Document],
        n_text: int,
        n_media: int,
        batch: int = 8,
    ) -> Tuple[List[Document], List[Document]]:
        text_chunks  = [c for c in chunks if classify_block(c) == "text"]
        media_chunks = [c for c in chunks if classify_block(c) == "media"]

        def score_and_pack(b):
            return (self._score_block(query, b), b)

        scored_text, scored_media = [], []
        with ThreadPoolExecutor(max_workers=batch) as ex:
            futures = {ex.submit(score_and_pack, b): b for b in text_chunks + media_chunks}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="CE Scoring"):
                score, blk = fut.result()
                if blk in text_chunks:
                    scored_text.append((score, blk))
                else:
                    scored_media.append((score, blk))

        scored_text.sort(key=lambda x: x[0], reverse=True)
        scored_media.sort(key=lambda x: x[0], reverse=True)

        top_text  = [blk for _, blk in scored_text[:n_text]]
        top_media = [blk for _, blk in scored_media[:n_media]]

        print(
            f"[{pathlib.Path(self.model_dir).name}] 选出 "
            f"{len(top_text)} 文本 + {len(top_media)} 多媒体"
        )
        return top_text, top_media


    # 分层 parents 重排
    def rerank_parents(
        self,
        query: str,
        parents: List[Document],
        n_text: int,
        n_media: int,
        batch: int = 8,
    ) -> Tuple[List[Document], List[Document]]:
        text_blocks  = [p for p in parents if p.metadata.get("type") in {"parent", "text"}]
        media_blocks = [p for p in parents if p.metadata.get("type") in {"image", "table"}]

        def score_and_pack(b):
            return (self._score_block(query, b), b)

        scored_text, scored_media = [], []
        with ThreadPoolExecutor(max_workers=batch) as ex:
            futures = {
                ex.submit(score_and_pack, b): ("text" if b in text_blocks else "media")
                for b in text_blocks + media_blocks
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="CE Scoring"):
                typ = futures[fut]
                score, blk = fut.result()
                (scored_text if typ == "text" else scored_media).append((score, blk))

        scored_text.sort(key=lambda x: x[0], reverse=True)
        scored_media.sort(key=lambda x: x[0], reverse=True)

        top_text  = [blk for _, blk in scored_text[:n_text]]
        top_media = [blk for _, blk in scored_media[:n_media]]
        return top_text, top_media



# # e5 / MiniLM-L-6-v2 交叉编码器
# e5_reranker = CrossEncoderReranker(
#     model_dir="Flex-RAG/models/MiniLM-L-6-v2",
#     prob_already=False         # e5 logits，需要 sigmoid
# )

# # Qwen3-Reranker-8B
# qwen_reranker = CrossEncoderReranker(
#     model_dir="Flex-RAG/models/Qwen3-Reranker-8B"
#     # prob_already=None => 自动检测为 True
# )

# top_text, top_media = qwen_reranker.rerank_chunks(
#     query="Explain convolution theorem",
#     chunks=flat_chunks,
#     n_text=5,
#     n_media=2
# )
