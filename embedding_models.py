import numpy as np
from langchain_openai.embeddings import OpenAIEmbeddings


embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
text = "This is the text you are going to embed"
query_result = embeddings.embed_query(text)
len(query_result)

positive_reviews = [
    "A cinematic masterpiece that captivates from start to finish.",
    "Outstanding storytelling with breathtaking performances!",
    "Exquisitely crafted and emotionally resonant.",
    "An enthralling experience that stays with you.",
    "Exceptional film! A triumph in every aspect."
]

negative_reviews = [
    "A tedious experience, lacks creativity and depth.",
    "The story drags and never quite lands.",
    "Lackluster performances and dull script.",
    "Disappointingly shallow plot and weak characterization.",
    "Fails to deliver on its promise. A missed opportunity."
]

positive_embeds, negative_embeds = [], []

for review in positive_reviews:
    embedding = embeddings.embed_query(review)
    positive_embeds.append(embedding)

for review in negative_reviews:
    embedding = embeddings.embed_query(review)
    negative_embeds.append(embedding)

max_sim = 0
similarity_dict = {}

for i, embed in enumerate(positive_embeds):
    sim = np.dot(positive_embeds[0], embed)
    similarity_dict[f"p{i}"] = sim
    if max_sim < sim:
        max_sim = sim

for i, embed in enumerate(negative_embeds):
    sim = np.dot(negative_embeds[0], embed)
    similarity_dict[f"p{i}"] = sim
    if max_sim < sim:
        max_sim = sim

for similarity in similarity_dict:
    similarity_dict[similarity] = 100 * (similarity_dict[similarity] / max_sim)

similarity_dict
