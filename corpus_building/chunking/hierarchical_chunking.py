import json
import tiktoken
from collections import defaultdict
from typing import List, Dict, Tuple
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Hier_TextSplitter:
    def count_tokens(self, text: str, encoding_name: str = "o200k_base") -> int:
        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))

    
    def split_docs(
        self,
        docs: List[Document],
        chunk_size: int = 300,
        chunk_overlap: int = 50
    ) -> Tuple[List[Document], List[Document]]:
        """
        返回 (parents, children)
          parents  : 供 LLM 使用的父块（page 级文本 + image/table 原文）
          children : 供稠密/稀疏检索的子块（text/equation 滑窗 + image/table 自身）
        """
        page_text: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    
        parents:  List[Document] = []
        children: List[Document] = []
        chunk_id = 0

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name   = "gpt-4o",
            chunk_size   = chunk_size,
            chunk_overlap= chunk_overlap
        )

        for d in docs:
            dtype = d.metadata["type"]

            if dtype in {"text", "equation"}:
                key = (d.metadata["book_idx"], d.metadata["page_idx"])
                page_text[key].append(d.page_content.strip())

            elif dtype in {"image", "table"}:
                p_idx = len(parents)                
                parents.append(d)                   

                children.append(
                    Document(
                        page_content=d.page_content,
                        metadata={
                            **d.metadata,           
                            "type":      "child",
                            "parent_id": p_idx,
                            "chunk_id":  chunk_id,
                            "length_tokens": self.count_tokens(d.page_content)
                        }
                    )
                )
                chunk_id += 1

        for (book, page), pieces in page_text.items():
            parent_text = " ".join(pieces).strip()

            p_idx = len(parents)
            parents.append(
                Document(
                    page_content=parent_text,
                    metadata={"type": "parent", "book_idx": book, "page_idx": page}
                )
            )

            for ch in splitter.split_text(parent_text):
                children.append(
                    Document(
                        page_content=ch,
                        metadata={
                            "type":      "child",
                            "book_idx":  book,
                            "page_idx":  page,
                            "parent_id": p_idx,
                            "chunk_id":  chunk_id,
                            "length_tokens": self.count_tokens(ch)
                        }
                    )
                )
                chunk_id += 1

        return parents, children