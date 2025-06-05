import os
from openai import OpenAI
from typing import Iterator, Tuple


def deepseek(prompt_input: str,
             model: str = "deepseek-reasoner", # R1, "deepseek-chat" v3,
             max_tokens: int = 64 * 1024,
             temperature: float = 0.7):
    
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


def deepseek_stream(prompt_input: str,
                    model: str = "deepseek-reasoner", # R1, "deepseek-chat" v3,    
                    max_tokens: int = 64 * 1024,
                    temperature: float = 0.7) -> Tuple[Iterator[str], "callable"]:
    
    """
    返回 (token_iterator, done)：
          token_iterator — 逐 token 字符串，可直接 for 循环 / yield 给前端
          done()         — 调用后得到完整回复文本（已自动累计）
    """
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
    full_text_parts = []

    def _iter_tokens() -> Iterator[str]:
        for chunk in stream_resp:                     # ChatCompletionChunk
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                token = delta.content
                full_text_parts.append(token)
                yield token                           
                

    def _done() -> str:
        """运行完后再获取完整字符串"""
        return "".join(full_text_parts)

    return _iter_tokens(), _done




# token_iter, done = deepseek_r1_stream("Explain Fourier transform.")

# for tok in token_iter:            # 实时打印 / 推送
#     print(tok, end="", flush=True)

# print("\n\n--- 完整结果 ---")
# print(done())

