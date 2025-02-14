import sys
import json
import os
import nltk
import pickle
import pandas as pd
import wordcloud as wc
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import plotly.io as pio
import numpy as np
import networkx as nx

from sklearn.feature_extraction.text import TfidfVectorizer
from maps import risk_map, methods_map
from collections import Counter


def load_data():
    ann = json.load(open("data/structured_search.json", "r", encoding="utf-8"))
    ann.extend(json.load(open("data/snowballing.json", "r", encoding="utf-8")))

    anns = []
    for an in ann:
        # print(an)
        result = {
                "title": an["Title"],
                "year": int(an["Year"]),
                "bibtex": an["BibtexKey"],
                "keywords": list(map(lambda x: x.lower(), an["Keywords"])),
                "akeywords": list(map(str.lower, an["AuthorKeywords"])),
                "risks": list(map(lambda x: risk_map(str.lower(x)), an["RiskTypes"])),
                "stages": list(map(str.lower, an["Stage"])),
                "method": [methods_map(an["Method"].lower())],
                "affiliation": list(map(str.lower, an["AuthorAffiliation"])),
                "affiliation_type": list(map(str.lower, an["AuthorAffiliationType"])),
                "countries": list(map(lambda x: x.split(", ")[1], an["AuthorAffiliation"])),
                "citations": int(an["Citations"])
            }
        result["allkws"] = list(set(result["keywords"] + result["akeywords"]))
        result["method"] = list(map(methods_map, result["method"]))
        for key in ["framework", "algorithm"]:
            result[key] = str(key in result["keywords"])
        result["type"] =  "applied" if "applied" in result["keywords"] else "theoretical"
        anns.append(result)

    refs = json.load(open("data/Found.json", "r", encoding="utf-8"))
    refs.extend(json.load(open("data/Q4A.json", "r", encoding="utf-8")))
    selected = refs.copy()
    refs.extend(json.load(open("data/Q4B.json", "r", encoding="utf-8")))
    refs.extend(json.load(open("data/Q4C.json", "r", encoding="utf-8")))

    # Prepare data
    corpus, dates,  = [], []
    for item in refs:
        corpus.append(item["title"] + "; " + item["abstract"])
        dates.append(int(item["issued"]["date-parts"][0][0]))

    return pd.DataFrame.from_records(anns), refs, corpus, dates


def main():
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    plt.rc('font', size=11)

    if not os.path.exists("output"):
        os.mkdir("output")

    if os.path.exists("output/data.p"):
        annotated, references, corpus, dates = pickle.load(open("output/data.p", "rb"))
    else:
        annotated, references, corpus, dates = load_data()
        pickle.dump((annotated, references, corpus, dates), open("output/data.p", "wb"))

    # Citation ordering
    df = []
    for key, item in annotated.iterrows():
        for risk in item["risks"]:
            df.append({
                "title": item["title"],
                "risk": risk,
                "citations": item["citations"],
                "method": item["method"][0],
                "bibtex": item["bibtex"]
            })
    df = pd.DataFrame.from_records(df)
    most_cited = df.sort_values(by=["citations"], ascending=False).groupby("risk").head(5).sort_values(by=["risk", "citations"], ascending=False)
    most_cited.to_csv("output/most_cited.csv", sep="\t")

    # Co-occurence matrix
    vocabulary = []
    for kws in annotated["allkws"]:
        vocabulary.append(dict(zip(kws, [1] * len(kws))))
    df = pd.DataFrame.from_records(vocabulary)
    df[pd.isna(df)] = 0
    cooc = df.T.dot(df)
    np.fill_diagonal(cooc.values, 0)


    # # Select only specific keywords for plotting
    # g = nx.Graph(cooc)
    # keys = ["x risks", "existential risk", "artificial general intelligence (agi)", "aixi", "friendly ai"]  # "fairness", "safety", "ethics", "ai ethics",

    # h = g.copy()
    # to_drop = [
    #     n for n, nbrs in h.adj.items() if n not in keys and not any([nbr in keys for nbr in nbrs])
    # ] + ["theoretical", "framework", "applied", "algorithm"]
    # h.remove_nodes_from(to_drop)
    # labels = {n: n if " " not in n else n.replace(" ", "\n").replace("\nand", " and") for n in h.nodes()}
    # widths = np.array(list(nx.get_edge_attributes(h, 'weight').values()))
    # options = {
    #     'node_color': ["green" if n in keys else "red" for n in h.nodes()],
    #     'node_size': 1000,
    #     'width': widths / widths.max() * 0.75,
    #     'font_size': 6
    # }
    # fig = plt.figure(figsize=(20, 12))
    # pos = nx.kamada_kawai_layout(h)
    # nx.draw_networkx(h, pos, with_labels=False, **options)
    # nx.draw_networkx_labels(h, pos, labels)
    # plt.axis("off")

    return 1


if __name__ == "__main__":
    sys.exit(main())
