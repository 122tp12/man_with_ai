import os
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
from sentence_transformers.util import normalize_embeddings, semantic_search
from datasets import load_dataset

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="./model_local",
    device="cuda",
)

ds = load_dataset("tmskss/linux-man-pages-tldr-summarized")

corpus_embeddings = model.encode(ds["train"]["Text"], convert_to_tensor=True)



corpus_embeddings = corpus_embeddings.to("cuda")
corpus_embeddings = normalize_embeddings(corpus_embeddings)

queries = [
    input("Enter your query: ")
]

query_embeddings = model.encode(queries, convert_to_tensor=True)
query_embeddings = query_embeddings.to("cuda")
query_embeddings = normalize_embeddings(query_embeddings)

hits = semantic_search(query_embeddings, corpus_embeddings)

for i in range(10):
    print(str(hits[0][i]["score"]) + ":" + ds["train"]["Summary"][hits[0][i]["corpus_id"]])