import tiktoken
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextSplitter:
    # ---------- 1. 计数工具 ----------
    def count_tokens(self, text: str, encoding_name: str = "o200k_base") -> int:
        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))

    # ---------- 2. 纯粹 chunking ----------
    def split_docs(
        self,
        docs: List[Document],
        chunk_size: int = 300,
        chunk_overlap: int = 50
    ) -> List[Document]:
        """
        普通文本切分：
        - 不区分 parent / child
        - 直接按 `chunk_size` & `chunk_overlap` 切块
        - 返回统一的 List[Document]
        """
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        chunks: List[Document] = []
        chunk_id = 0

        for doc in docs:
            # 逐块切分当前文档
            for chunk in splitter.split_text(doc.page_content):
                chunks.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            **doc.metadata,            # 保留原元数据
                            "chunk_id":       chunk_id,
                            "length_tokens":  self.count_tokens(chunk),
                        }
                    )
                )
                chunk_id += 1

        return chunks
