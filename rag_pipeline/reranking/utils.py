from langchain.docstore.document import Document


def classify_block(block: Document) -> str:
    """
    返回 'text' 或 'media'。
    - 缺失 'type' 元数据时默认按 'text' 处理。
    """
    t = block.metadata.get("type", "text")
    return "media" if t in {"image", "table"} else "text"
