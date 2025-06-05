from pathlib import Path
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from collections import Counter, defaultdict
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings



class Dense_Retriever():
    """仅使用稠密向量检索的版本"""
    def __init__(self, children: List[Document], parents: List[Document], configs: dict):
        self.children = children
        self.parents  = parents
        self.configs  = configs

    
        embeddings = HuggingFaceEmbeddings(
            model_name   = configs["DENSE_MODEL"],
            model_kwargs = {"local_files_only": True}
        )

        index_dir = Path(configs["INDEX_PATH"])
        if index_dir.exists():
            vectordb = FAISS.load_local(str(index_dir), embeddings)
        else:
            vectordb = FAISS.from_documents(children, embeddings)
            vectordb.save_local(str(index_dir))

        self.dense_retriever = vectordb.as_retriever(
            search_kwargs={"k": configs["DENSE_PICK"]}
        )

    # ---------------------------------------------------------------------
    def dense_retrieve_chunks(self, query: str, k: int | None = None) -> List[Document]:
        """
        仅对平坦 chunk 进行稠密检索：
        - 不依赖 parent_id，也不做父块映射
        - 默认 top-k = configs['CHUNK_PICK'] (若有)；否则复用 DENSE_PICK
        """
        k = k or self.configs.get("CHUNK_PICK", self.configs["DENSE_PICK"])
        chunk_hits = self.vectordb.similarity_search(query, k=k)
        
        stats = Counter(d.metadata.get("type", "text") for d in chunk_hits)
        print(
            f"稠密检索到 {len(chunk_hits)} 个 chunk："
            f"{stats.get('text',0)} 段文本，"
            f"{stats.get('image',0)} 张图像，"
            f"{stats.get('table',0)} 个表格"
        )
        return chunk_hits
    
    # ---------------------------------------------------------------------
    def dense_retrieve_parents(self, query: str) -> List[Document]:
        """稠密检索 → 子块 → 映射父块 → 去重后返回"""
        dense_child_hits = self.dense_retriever.get_relevant_documents(query)
        parent_ids       = {ch.metadata["parent_id"] for ch in dense_child_hits}
        parent_hits      = [self.parents[i] for i in parent_ids]

        # 结果统计（可选）
        stats = Counter(p.metadata["type"] for p in parent_hits)
        print(
            f"稠密检索到 {len(dense_child_hits)} 个子块，映射到 {len(parent_hits)} 个父块，"
            f"包含 {stats.get('parent',0)+stats.get('text',0)} 段文本，"
            f"{stats.get('image',0)} 张图像，"
            f"{stats.get('table',0)} 个表格"
        )
        return parent_hits

    # ---------------------------------------------------------------------
    def related_equs(self, top_text_parents: List[Document], docs: List[Document]):
        """根据文本父块所在页，附加同页公式。逻辑与旧版一致。"""
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














