import pickle
import pandas as pd
import bertopic
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer


def draw_piechart(counts, fname: str = None, inpct: float = 0.7, pctcut: int = 2):
    fig = plt.figure(figsize=(6,6))
    plt.pie(counts, labels=counts.index, autopct=lambda pct: ('%1.1f%%' % pct) if pct > pctcut else '', colors=mpl.colormaps.get_cmap("tab20").colors, 
            startangle=90, pctdistance=0.8, explode=[0.05]*len(counts), rotatelabels=True, labeldistance=1.)
    centre_circle = plt.Circle((0,0),inpct,fc='white')
    fig.gca().add_artist(centre_circle)
    plt.axis("equal")
    plt.tight_layout()
    # plt.figtext(0.5, 0.5, fname.replace("_", " ").title(), ha='center', va='center', fontsize=18)
    if fname is not None:
        plt.savefig(f"output/{fname}.pdf")


def get_counts(data):
    items = []
    for item in data:
        items.extend(item)
    return pd.Series(items).value_counts()


def model_topics(corpus, dates, path_to_data="output", min_cluster_size=3):
    rerun = True

    embeddings_path = os.path.join(path_to_data, "embeddings.p")
    if not os.path.exists(embeddings_path):
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = sentence_model.encode(corpus, show_progress_bar=True)
        pickle.dump(embeddings, open(embeddings_path, "wb"))
    else:
        embeddings = pickle.load(open(embeddings_path, "rb"))

    topic_model_path = os.path.join(path_to_data, "topic_model.p")
    if not os.path.exists(topic_model_path) or rerun:
        ctfidf_model = bertopic.vectorizers.ClassTfidfTransformer(reduce_frequent_words=True)
        vectorizer_model = CountVectorizer(stop_words="english")
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

        topic_model = bertopic.BERTopic(
            hdbscan_model=hdbscan_model, 
            ctfidf_model=ctfidf_model, 
            vectorizer_model=vectorizer_model,
            verbose=True)
        topic_model.fit(corpus, embeddings)
        topic_model.save(topic_model_path)
    else:
        topic_model = bertopic.BERTopic.load(topic_model_path)

    return topic_model, embeddings
