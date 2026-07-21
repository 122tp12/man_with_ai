from sentence_transformers import SentenceTransformer
from sentence_transformers.util import dot_score, normalize_embeddings, semantic_search

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")

corpus = [
    "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
    "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
    "Neural networks are computing systems vaguely inspired by the biological neural networks that constitute animal brains.",
    "Mars rovers are robotic vehicles designed to travel on the surface of Mars to collect data and perform experiments.",
    "The James Webb Space Telescope is the largest optical telescope in space, designed to conduct infrared astronomy.",
    "SpaceX's Starship is designed to be a fully reusable transportation system capable of carrying humans to Mars and beyond.",
    "Global warming is the long-term heating of Earth's climate system observed since the pre-industrial period due to human activities.",
    "Renewable energy sources include solar, wind, hydro, and geothermal power that naturally replenish over time.",
    "Carbon capture technologies aim to collect CO2 emissions before they enter the atmosphere and store them underground.",
]

queries = [
    "How do artificial neural networks work?",
    "What technology is used for modern space exploration?",
    "How can we address climate change challenges?",
]

corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

query_embeddings = model.encode(queries, convert_to_tensor=True)


corpus_embeddings = corpus_embeddings.to("cuda")
corpus_embeddings = normalize_embeddings(corpus_embeddings)

query_embeddings = query_embeddings.to("cuda")
query_embeddings = normalize_embeddings(query_embeddings)
hits = semantic_search(query_embeddings, corpus_embeddings, score_function=dot_score)

print(hits)