import os
import tiktoken
from dotenv import load_dotenv
from tqdm import tqdm
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from dashscope import MultiModalConversation
from langchain.docstore.document import Document
from .utils import classify_block  # 假设 classify_block 已在 utils 中定义


class Qwenvl_Reranker():
    """
    使用 Qwen-VL 系列模型做多模态的相关性重排。
    
    初始化时会加载环境变量、检查 API Key，并准备 tokenizer 编码器。然后可调用
    rerank_chunks() 或 rerank_parents() 来对平坦 chunks 或分层 parents 进行重排。
    """
    
    def __init__(self, model_name: str = "qwen2.5-vl-7b-instruct"):
        # 加载 .env 并读取 DASHSCOPE_API_KEY
        load_dotenv()
        self.ENC = tiktoken.get_encoding("o200k_base")
        self.qwen_key = os.getenv("DASHSCOPE_API_KEY")
        if not self.qwen_key:
            raise RuntimeError("请先确保 DASHSCOPE_API_KEY 已正确设置并激活了 mmrag 环境")

        self.llm = model_name

    
    # 单块评分：给一个 chunk/块 返回 0~1 浮点分数
    def _score_block(self, query: str, block: Document) -> float:
        t = block.metadata.get("type", "text")
        txt = block.page_content[:2000]  # 截断到 2000 字符，保证单块 token < 4k
        
        if t in {"image", "table"}:
            img_path = block.metadata.get("img_path")
            user_content = []
            if img_path and os.path.exists(img_path) and t == "image":
                user_content.append({"image": img_path})
            user_content.append({
                "text": (
                    f"Query: {query}\n\n"
                    f"Context ({t}):\n{txt}\n\n"
                    "请给出其与 Query 的相关性分数，0~1 间小数。仅回复分数字面值。"
                )
            })
        else:
            user_content = [{
                "text": (
                    f"Query: {query}\n\n"
                    "Context (text):\n"
                    f"{txt}\n\n"
                    "请给出其与 Query 的相关性分数，0~1 间小数。仅回复分数字面值。"
                )
            }]
        
        messages = [
            {
                "role": "system",
                "content": [{"text": "You are a helpful assistant for relevance scoring."}]
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        
        try:
            resp = MultiModalConversation.call(
                api_key=self.qwen_key,
                model=self.llm,
                messages=messages,
                vl_high_resolution_images=False
            )
            score_txt = resp["output"]["choices"][0]["message"].content[0]["text"]
            return float(score_txt.strip())
        except Exception as e:
            print("评分失败:", e)
            return 0.0
    

    # 对“平坦 chunks”做重排：文本 / 媒体分别打分、排序、各取前 n
    def rerank_chunks(
        self,
        query: str,
        chunks: List[Document],
        n_text: int,
        n_media: int,
        batch: int = 5 # batch <= 6
    ) -> Tuple[List[Document], List[Document]]:
        """
        对同质 chunks（无 parent_id）做 LLM 相关性重排。
        - 文本块 & 多媒体块分别打分、排序、各取前 k。
        - 返回 (top_text_chunks, top_media_chunks)
        """
        text_chunks = [c for c in chunks if classify_block(c) == "text"]
        media_chunks = [c for c in chunks if classify_block(c) == "media"]

        def score_and_pack(b):
            return (self._score_block(query, b), b)

        scored_text, scored_media = [], []
        with ThreadPoolExecutor(max_workers=batch) as ex:
            futures = {
                ex.submit(score_and_pack, blk): blk
                for blk in text_chunks + media_chunks
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Qwen Scoring"):
                score, blk = fut.result()
                if blk in text_chunks:
                    scored_text.append((score, blk))
                else:
                    scored_media.append((score, blk))

        scored_text.sort(key=lambda x: x[0], reverse=True)
        scored_media.sort(key=lambda x: x[0], reverse=True)

        top_text = [blk for _, blk in scored_text[:n_text]]
        top_media = [blk for _, blk in scored_media[:n_media]]

        print(
            f"最终选出 {len(top_text)} 段文本 + {len(top_media)} 个多媒体块 "
            f"(文本/多媒体请求阈值 = {n_text}/{n_media})"
        )
        return top_text, top_media


    # 对“分层 parents”做重排：先分类再打分
    def rerank_parents(
        self,
        query: str,
        parents: List[Document],
        n_text: int,
        n_media: int,
        batch: int = 5 # batch <= 6
    ) -> Tuple[List[Document], List[Document]]:
        """
        对层次式父块列表做 LLM 相关性重排：
        - text_blocks & media_blocks 分组后各自打分、排序、取前 k
        - 返回 (top_text_parents, top_media_parents)
        """
        text_blocks = [p for p in parents if p.metadata.get("type") in {"parent", "text"}]
        media_blocks = [p for p in parents if p.metadata.get("type") in {"image", "table"}]

        def score_and_pack(b):
            return (self._score_block(query, b), b)

        scored_text, scored_media = [], []
        with ThreadPoolExecutor(max_workers=batch) as ex:
            futures = {
                ex.submit(score_and_pack, blk): ("text" if blk in text_blocks else "media")
                for blk in text_blocks + media_blocks
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Qwen Scoring"):
                typ = futures[fut]
                score, blk = fut.result()
                if typ == "text":
                    scored_text.append((score, blk))
                else:
                    scored_media.append((score, blk))

        scored_text.sort(key=lambda x: x[0], reverse=True)
        scored_media.sort(key=lambda x: x[0], reverse=True)

        top_text = [blk for _, blk in scored_text[:n_text]]
        top_media = [blk for _, blk in scored_media[:n_media]]
        return top_text, top_media
