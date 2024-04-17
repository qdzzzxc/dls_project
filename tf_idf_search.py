import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if os.path.exists(os.path.join(".", "all_rows.pqt")):
    all_rows = pd.read_parquet("all_rows.pqt")
else:
    all_rows = pd.DataFrame(
        columns=["id", "link", "name", "rate", "description", "reviews"]
    )

    for name in os.listdir():
        if name.endswith(".csv") and name.startswith("rows"):
            temp_df = pd.read_csv(name)
            all_rows = pd.concat([all_rows, temp_df])

    all_rows.to_parquet("all_rows.pqt", index=False)


vectorizer = TfidfVectorizer()

tfidf_vectors = vectorizer.fit_transform(all_rows.name)


def cosine_search(new_sentence):
    new_tfidf_vector = vectorizer.transform([new_sentence])

    new_tfidf_vector_array = new_tfidf_vector.toarray()

    euclidean_dist = cosine_similarity(
        new_tfidf_vector_array.reshape(1, -1), tfidf_vectors
    )

    return all_rows.iloc[np.argmax(euclidean_dist)]
