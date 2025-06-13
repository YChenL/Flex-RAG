import textwrap
from typing import List
from langchain.docstore.document import Document



def block_fmt(doc: Document, idx: int) -> str:
    tp  = doc.metadata["type"]
    pg  = doc.metadata["page_idx"]
    b   = doc.metadata["book_idx"]
    head = f"[{idx:02d}] ({tp.upper()} | book={b}, page={pg})"
    body = doc.page_content.strip()
    return f"{head}\n{body}"


def build_retrieval_prompt(
    query: str,
    top_text_parents: List[Document],
    top_media_parents: List[Document]
) -> str:
    """
    根据检索出的文本、图表和公式父块，生成最终发给 LLM 的 Prompt 字符串。
    """
    # A) 文本父块
    text_section = "\n\n".join(
        block_fmt(d, i + 1) 
        for i, d in enumerate(top_text_parents)
    )
    # B) 图表父块
    media_section = "\n\n".join(
        block_fmt(d, i + 1) 
        for i, d in enumerate(top_media_parents)
    )

    # 组装最终 prompt
    prompt = textwrap.dedent(f"""
        ## Query
        {query}

        ## Retrieved Context - TEXT (Top {len(top_text_parents)})
        {text_section}

        ## Retrieved Context - IMAGE/TABLE (Top {len(top_media_parents)})
        {media_section}

        ## Instruction
        你是专业的技术写作助手，请仔细阅读 **Retrieved Context**，并基于这些检索到的信息回答 **Query**。
        - 如果问题中的某些点你非常确信可以回答，但是未在检索结果出现，可以尝试采用你掌握的知识进行补充;
        - 回答过程中请充分结合检索到的公式进行专业的说明;
        - 回答用简体中文，结构清晰，必要时分条列出。
    """).strip()

    return prompt
