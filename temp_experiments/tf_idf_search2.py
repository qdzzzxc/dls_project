import os

import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sqlalchemy import text, create_engine

import aiopg

# from config import *
DATA_TYPE = 'PANDAS'

if DATA_TYPE == "PANDAS":

    all_rows = pd.DataFrame(
        columns=["id", "link", "name", "rate", "description", "reviews"]
    )

    for name in os.listdir():
        if name.endswith(".csv") and name.startswith("rows"):
            temp_df = pd.read_csv(name)
            all_rows = pd.concat([all_rows, temp_df])

    def data_getter(x):
        logging.info('pandas request')
        return all_rows.iloc[x]
elif DATA_TYPE == "SQL":
    engine = create_engine(DSN, echo=False)

    def data_getter(x):
        logging.info('sql request')
        with engine.connect() as conn:
            ret = conn.execute(
                text(
                    f"select id, link, name, rate, description, reviews from all_rows where index = {x}"
                )
            ).fetchall()
            return pd.Series(
                ret[0], index=["id", "link", "name", "rate", "description", "reviews"]
            )


vectorizer_path = os.path.join("data", "tf-idf", "vectorizer.pkl")

vectors_path = os.path.join("data", "tf-idf", "tfidf_vectors.npy")
logging.info("Vectorizer created")
vectorizer = TfidfVectorizer()

tfidf_vectors = vectorizer.fit_transform(all_rows.name)

joblib.dump(vectorizer, vectorizer_path)
np.save(vectors_path, tfidf_vectors)


def cosine_search(new_sentence):
    new_tfidf_vector = vectorizer.transform([new_sentence])

    new_tfidf_vector_array = new_tfidf_vector.toarray()

    euclidean_dist = cosine_similarity(
        new_tfidf_vector_array.reshape(1, -1), tfidf_vectors
    )

    return data_getter(np.argmax(euclidean_dist))

def find_top_ten_books(new_sentence):
    new_tfidf_vector = vectorizer.transform([new_sentence])

    new_tfidf_vector_array = new_tfidf_vector.toarray()

    euclidean_dist = cosine_similarity(
        new_tfidf_vector_array.reshape(1, -1), tfidf_vectors
    )

    return euclidean_dist, data_getter
    return [data_getter(x) for x in np.argsort(euclidean_dist)[:10]]
