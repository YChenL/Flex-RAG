import os, json5, re, textwrap
from typing import List, Tuple
from .prompts import REWRITE_PROMPT
from dashscope import MultiModalConversation
from langchain.docstore.document import Document


# 初始化captioner
qwen_key = os.getenv("DASHSCOPE_API_KEY")
if not qwen_key:
    raise RuntimeError("请先确保 DASHSCOPE_API_KEY 已正确设置并激活了 mmrag 环境")
    

def build_media_inputs(media_docs: List[Document],
                       max_n: int = 5) -> Tuple[List[str], str]:
    """
    根据 top_media_parents 生成:
      media_paths : 最多 max_n 个文件路径 (image截图/table截图)
      media_block : 用于 prompt 的文字描述
    """
    media_paths  = []
    block_lines  = []

    for idx, doc in enumerate(media_docs[:max_n], start=1):
        tag      = f"<MEDIA_{idx}>"
        typ      = doc.metadata["type"].capitalize()      # Image / Table
        img_path = doc.metadata["img_path"]

        # 取 caption
        if typ == "Image":
            cap_raw = doc.metadata.get("img_caption", "")
        else:  # Table
            cap_raw = doc.metadata.get("table_caption", "")
        if isinstance(cap_raw, list):
            caption = " ".join(cap_raw)
        else:
            caption = cap_raw

        # 组装
        media_paths.append(img_path)
        block_lines.append(f"{tag}  ({typ})  caption: {caption}")

    media_block = "\n".join(block_lines)
    return media_paths, media_block



def safe_json_load(txt: str):
    # 把 ```json … ``` 外围去掉
    m = re.search(r"```json\s*(.*?)\s*```", txt, re.S | re.I)
    core = m.group(1) if m else txt

    # 用 json5 解析
    return json5.loads(core)



def rewrite_with_mllm(answer_text: str,
                       media_docs: List[Document],
                       max_media: int = 5) -> dict:
    """
    调用 Qwen-VL 7B 让其润色 AnswerText 并决定 Media 插入点。
    返回 {"enhanced_paragraphs":[...], "unused_media":[...]} (dict)
    """
    media_paths, media_block = build_media_inputs(media_docs, max_n=5)

    # ---------- 1. 组织 media_paths 与 media_block ----------
    media_paths, media_lines = [], []
    for idx, doc in enumerate(media_docs[:max_media], 1):
        tag  = f"<MEDIA_{idx}>"
        path = doc.metadata["img_path"]
        typ  = doc.metadata["type"].capitalize()   # Image / Table
        cap_raw = (doc.metadata.get("img_caption") or
                   doc.metadata.get("table_caption") or "")
        cap = " ".join(cap_raw) if isinstance(cap_raw, list) else cap_raw

        media_paths.append(path)
        media_lines.append(f"{tag}  ({typ})  caption: {cap}")

    media_block = "\n".join(media_lines)

    # ---------- 2. 构造 messages ----------
    user_content = (
        [{"text": REWRITE_PROMPT}] +                   # 提示
        [{"image": p} for p in media_paths] +       # 多张图(表)一次传
        [{"text": textwrap.dedent(f"""
            ## AnswerText
            {answer_text}

            ## Media List
            {media_block}
            """)}]
    )

    messages = [
        {"role": "system",
         "content": [{"text": "You are a careful technical writer."}]},
        {"role": "user",
         "content": user_content}
    ]

    # ---------- 3. 调用 Dashscope MultiModalConversation ----------
    resp = MultiModalConversation.call(
        api_key = qwen_key,
        model   = "qwen2.5-vl-72b-instruct",
        messages= messages,
        vl_high_resolution_images = False
    )

    # ---------- 4. 解析返回 JSON ----------
    raw_txt = resp["output"]["choices"][0]["message"].content[0]["text"]
    result  = safe_json_load(raw_txt)
    return result



def render_mm_results(result, top_media_parents, response):
    # 把 media tag 映射到路径 & caption
    tag2info = {}
    for idx, doc in enumerate(top_media_parents, 1):
        tag = f"<MEDIA_{idx}>"
        path = doc.metadata["img_path"]
        caption = (doc.metadata.get("img_caption") or
                   doc.metadata.get("table_caption") or "")
        caption = " ".join(caption) if isinstance(caption, list) else caption
        tag2info[tag] = (path, caption)

    # 显示思考过程
    display(Markdown("## 🤔 思考过程\n\n" +
                     response.choices[0].message.reasoning_content))

    # 显示润色后回答并插入图表
    display(Markdown("## 💡 回答\n"))
    for para in result["enhanced_paragraphs"]:
        # 逐段输出，替换占位符
        for tag, (img_path, cap) in tag2info.items():
            if tag in para:
                # 段落文本去掉 tag 占位
                para_text = para.replace(tag, "").strip()
                if para_text:
                    display(Markdown(para_text))
                display(Image(img_path))
                display(Markdown(f"*{cap}*"))
                break
        else:
            # 普通段落
            display(Markdown(para))
