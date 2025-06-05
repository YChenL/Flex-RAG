from pathlib import Path
from typing import List, Tuple
from collections import Counter, defaultdict
from rank_bm25 import BM25Okapi
from langchain.docstore.document import Document



class Sparse_Retriever():
    """
    仅使用 BM25 的稀疏检索器
    -------------------------------------------------
    configs 需要包含：
        BM25_PICK   — 层次式检索时 child top-k
        TOP_PARENT  — 层次式检索时 parent top-k
        k_child     — 文本过滤后 child top-k
        k_parent    — 文本过滤后 parent top-k
        CHUNK_PICK  — 平坦检索时 chunk top-k  (可选；缺省用 BM25_PICK)
    """
    # 初始化：构建 BM25 语料
    def __init__(self, children: List[Document], parents: List[Document], configs: dict):
        self.children = children
        self.parents  = parents
        self.configs  = configs

        corpus_tokens = [c.page_content.split() for c in children]
        self.bm25 = BM25Okapi(corpus_tokens)

    
    # ------------------------------------------------------------------
    def bm25_retrieve_parents(self, query: str) -> Tuple[List[Document], List[Document]]:
        """不区分内容类型的 BM25 → top-k child → parent 映射"""
        scores   = self.bm25.get_scores(query.split())
        bm25_k   = self.configs["BM25_PICK"]
        top_idx  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:bm25_k]
        child_hits = [self.children[i] for i in top_idx]

        # 映射到 parent
        parent_ids, seen = [], set()
        for ch in child_hits:
            pid = ch.metadata["parent_id"]
            if pid not in seen:
                parent_ids.append(pid)
                seen.add(pid)
            if len(parent_ids) >= self.configs["TOP_PARENT"]:
                break
        parent_hits = [self.parents[i] for i in parent_ids]
        return child_hits, parent_hits

    
    def bm25_retrieve_text_parents(self, query: str) -> Tuple[List[Document], List[Document]]:
        """仅保留文本类 parent（排除图像 / 表格 parent）"""
        scores     = self.bm25.get_scores(query.split())
        idx_sorted = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        child_hits, parent_ids = [], []
        for i in idx_sorted:
            if len(child_hits) >= self.configs["k_child"]:
                break
            pid = self.children[i].metadata["parent_id"]
            if self.parents[pid].metadata["type"] != "parent":       # 只要文本 parent
                continue
            child_hits.append(self.children[i])
            parent_ids.append(pid)
            if len(set(parent_ids)) >= self.configs["k_parent"]:
                break

        parent_hits = [self.parents[i] for i in set(parent_ids)]
        return child_hits, parent_hits

    
    # ------------------------------------------------------------------
    def sparse_retrieve_parents(self, query: str) -> List[Document]:
        child_hits, parent_hits = self.bm25_retrieve_text_parents(query)

        cnt = Counter(p.metadata["type"] for p in parent_hits)
        print(
            f"BM25 检索到 {len(child_hits)} 个子块，映射到 {len(parent_hits)} 个父块，"
            f"{cnt.get('parent',0)+cnt.get('text',0)} 段文本，"
            f"{cnt.get('image',0)} 张图像，"
            f"{cnt.get('table',0)} 个表格"
        )
        return parent_hits

    
    # ------------------------------------------------------------------
    def bm25_retrieve_chunks(self, query: str, k: int | None = None) -> List[Document]:
        """
        对“平坦切分”的 chunks 直接做 BM25：
            - 不映射父块
            - k 默认为 configs['CHUNK_PICK'] 或 BM25_PICK
        """
        k = k or self.configs.get("CHUNK_PICK", self.configs["BM25_PICK"])
        scores   = self.bm25.get_scores(query.split())
        top_idx  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        chunk_hits = [self.children[i] for i in top_idx]

        cnt = Counter(ch.metadata.get("type", "text") for ch in chunk_hits)
        print(
            f"BM25 检索到 {len(chunk_hits)} 个 chunk："
            f"{cnt.get('text',0)} 段文本，"
            f"{cnt.get('image',0)} 张图像，"
            f"{cnt.get('table',0)} 个表格"
        )
        return chunk_hits

    
    # 公式关联 / 预览
    def related_equs(self, top_text_parents: List[Document], docs: List[Document]):
        pages_of_text = {
            (p.metadata["book_idx"], p.metadata["page_idx"])
            for p in top_text_parents
        }
        eq_by_loc = defaultdict(list)
        for d in docs:
            if d.metadata["type"] == "equation":
                loc = (d.metadata["book_idx"], d.metadata["page_idx"])
                eq_by_loc[loc].append(d)

        top_equations, seen = [], set()
        for loc in pages_of_text:
            for eq in eq_by_loc.get(loc, []):
                if id(eq) not in seen:
                    top_equations.append(eq)
                    seen.add(id(eq))

        print(f"\n关联到同页公式 {len(top_equations)} 条")
        self.preview_equations(top_equations)
        return top_equations

    
    def preview_equations(self, eq_list: List[Document], n: int = 3):
        for i, d in enumerate(eq_list[:n], 1):
            snippet = (d.page_content[:100] + "…") if len(d.page_content) > 100 else d.page_content
            print(f"{i}. (equation, p{d.metadata['page_idx']})  {snippet}")













