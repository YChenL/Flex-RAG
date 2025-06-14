import os
import time
from rag_pipeline.retrieval import Hybrid_Retriever, load_serialized_docs
from rag_pipeline.reranking import Qwenvl_Reranker
from rag_pipeline.reasoning import DeepSeek_Stream
from post_processing import rewrite_with_mllm, render_mm_results


# loading Corpus
mm_parents  = load_serialized_docs("/data/huali_mm/chunks/mm_parents.json")
mm_children = load_serialized_docs("/data/huali_mm/chunks/mm_children.json")

# Initializaing Retriever
qwen_configs={
"DENSE_MODEL": "./models/Qwen3-Embedding-0.6B",
"INDEX_PATH" : "./indexes/qwen/mm_hier_index",
"DENSE_PICK" : 200,         # 稠密召回数
"BATCH"      : 6,           # 嵌入模型batch大小 显存大约占用为21g
"BM25_PICK"  : 200,         # 稀疏召回数
"TOP_PARENT" : 200,
"k_child"    : 200,
"k_parent"   : 20
}
mm_hier_hret = Hybrid_Retriever((mm_children, mm_parents), qwen_configs)
llm_reranker = Qwenvl_Reranker()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_text",     type=int,  default=20)
    parser.add_argument("--top_media",    type=int,  default=5)
    parser.add_argument("--rerank_batch", type=int,  default=8)
    parser.add_argument("--rewrite",      type=bool, default=False)    
    args = parser.parse_args()

    query = input()
    # Retrieval
    results = mm_hier_hret.hybrid_retrieve_parents(query)
    # Reranking
    top_text_parents, top_media_parents = llm_reranker.rerank_parents(
    query  =query,
    parents=results,
    n_text =args.top_text,
    n_media=args.top_media,
    batch  =args.rerank_batch
    )
    # Reasoning
    token_iter, response = DeepSeek_Stream(query,
                                       top_text_parents,
                                       top_media_parents)
    # Printing Results
    for tok in token_iter:           
        print(tok, end="", flush=True)

    if args.rewrite:
        text = response()
        start = time.perf_counter()
        print("------------rewriting results--------------")
        result = rewrite_with_mllm(
            answer_text    = text[1],               # DeepSeek 回答文本
            media_docs     = top_media_parents      # 5 条图/表父块
        )
        # # rendering
        render_mm_results(result, top_media_parents, text[0])