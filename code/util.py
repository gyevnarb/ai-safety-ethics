import pickle
import pandas as pd
import bertopic
import os
import  matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.cluster import hierarchy as sch
from collections import Counter
from hdbscan import HDBSCAN
from umap import UMAP
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


def model_topics(corpus, dates, min_cluster_size=3):
    rerun = True

    if not os.path.exists("output/embeddings.p"):
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = sentence_model.encode(corpus, show_progress_bar=True)
        pickle.dump(embeddings, open("output/embeddings.p", "wb"))
    else:
        embeddings = pickle.load(open("output/embeddings.p", "rb"))

    if not os.path.exists("output/topic_model.p") or rerun:
        ctfidf_model = bertopic.vectorizers.ClassTfidfTransformer(reduce_frequent_words=True)
        vectorizer_model = CountVectorizer(stop_words="english")
        hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

        topic_model = bertopic.BERTopic(
            hdbscan_model=hdbscan_model, 
            ctfidf_model=ctfidf_model, 
            vectorizer_model=vectorizer_model,
            verbose=True)
        topic_model.fit(corpus, embeddings)
        topic_model.save("output/topic_model.p")
    else:
        topic_model = bertopic.BERTopic.load("output/topic_model.p")

    print(topic_model.get_document_info(corpus))
    hierarchical_topics = topic_model.hierarchical_topics(corpus)
    fig = topic_model.visualize_topics()
    fig.show()

    linkage_function = lambda x: sch.linkage(x, 'single', optimal_ordering=True)
    hierarchical_topics = topic_model.hierarchical_topics(corpus, linkage_function=linkage_function)
    fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
    fig.show()

    fig = topic_model.visualize_heatmap()
    fig.show()

    # Run the visualization with the original embeddings
    fig = topic_model.visualize_documents(corpus, embeddings=embeddings)
    fig.show()

    reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    fig = topic_model.visualize_documents(corpus, reduced_embeddings=reduced_embeddings)
    fig.show()

    fig = topic_model.visualize_barchart()
    fig.show()

    topics_over_time = topic_model.topics_over_time(corpus, dates)
    fig = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10)
    fig.show()