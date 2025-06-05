REWRITE_PROMPT = """
你将得到：
1. 原始 **AnswerText**（来自 DeepSeek Reasoner）。
2. 最多 5 个 **Media**（Image 或 Table）的描述，每个带唯一占位名 <MEDIA_i>。
3. 每个 Media 的 **Caption** （图/表标题）。

目标：  
- 对 AnswerText 进行润色，使行文连贯、信息丰富，但不得改变事实。  
- 判断哪些段落需要插入哪些 Media；请在相应段落 **行尾** 插入 `<MEDIA_i>` 占位符，可多张图表插同段，也可某些图表不使用。  
- 输出 **JSON**，字段：  
  ```json
  {
    "enhanced_paragraphs": ["第一段", "第二段", ...],
    "unused_media": ["<MEDIA_3>", ...]      // 若全部用完给 []
  }
注意：
不要改动 <MEDIA_i> 占位符格式。
不要生成除 JSON 之外的任何额外内容。
请务必输出 纯 JSON，不要加 ```json 包裹，也不要出现未转义的反斜杠，例如 \( \] 等。
"""
