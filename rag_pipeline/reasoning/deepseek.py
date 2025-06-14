import os
from openai import OpenAI
from typing import Iterator, Tuple
from .utils import build_retrieval_prompt



def DeepSeek(query: str,
             top_text_parents: str,
             top_media_parents: str,
             model: str = "deepseek-reasoner", # R1, "deepseek-chat" v3,
             max_tokens: int = 64 * 1024,
             temperature: float = 0.7):

    # build llm input
    prompt_input = build_retrieval_prompt(query, top_text_parents, top_media_parents)
    
    # llm reasoning
    print("------------llm reasoning--------------")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_key:
        raise RuntimeError("请先确保 DEEPSEEK_API_KEY 已正确设置并激活了 mmrag 环境")

    client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt_input},
        ],
        max_tokens  = max_tokens,     
        temperature = temperature,
        stream=False
    )
    
    return response


def DeepSeek_Stream(query: str,
                    top_text_parents: str,
                    top_media_parents: str,
                    model: str = "deepseek-reasoner", # R1, "deepseek-chat" v3,    
                    max_tokens: int = 64 * 1024,
                    temperature: float = 0.7) -> Tuple[Iterator[str], "callable"]:

    """
    返回 (token_iterator, done)：
          token_iterator — 逐 token 字符串，可直接 for 循环 / yield 给前端
          done()         — 调用后得到完整回复文本（已自动累计）
    """
  
    # build llm input
    prompt_input = build_retrieval_prompt(query, top_text_parents, top_media_parents)

    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_key:
        raise RuntimeError("DEEPSEEK_API_KEY 未设置")

    client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")

    # -------------------------------------------
    stream_resp = client.chat.completions.create(
        model       = model,
        messages    = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": prompt_input},
        ],
        max_tokens  = max_tokens,
        temperature = temperature,
        stream      = True,
    )

    # ------------------
    reasoning_parts, answer_parts = [], []
    def _iter_tokens() -> Iterator[str]:
        """
        1) 推理阶段先输出 <think>…</think> 包裹的思考 token
        2) 后续正文直接输出
        """
        thinking_started = False         # 是否已输出 <think>
        thinking_ended   = False         # 是否已输出 </think>

        for chunk in stream_resp:
            delta = chunk.choices[0].delta

            # ---------- 思考阶段 ----------
            if getattr(delta, "reasoning_content", None):
                if not thinking_started:
                    thinking_started = True
                    yield "<think>"                     # 开始标签
                tok = delta.reasoning_content
                reasoning_parts.append(tok)
                yield tok
                continue

            # ---------- 正文阶段 ----------
            if getattr(delta, "content", None):
                if thinking_started and not thinking_ended:
                    thinking_ended = True
                    yield "</think>"                    # 结束标签
                tok = delta.content
                answer_parts.append(tok)
                yield tok

    
    def _done() -> Tuple[str, str]:
        return "".join(reasoning_parts), "".join(answer_parts)

    
    return _iter_tokens(), _done




# token_iter, done = deepseek_r1_stream("Explain Fourier transform.")

# for tok in token_iter:            # 实时打印 / 推送
#     print(tok, end="", flush=True)

# print("\n\n--- 完整结果 ---")
# print(done())

