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
from maps import *
from util import draw_piechart, get_counts, model_topics
from collections import Counter


def load_data():
    ann = json.load(open("data/structured_search.json", "r"))
    ann.extend(json.load(open("data/snowballing.json", "r")))

    anns = []
    for an in ann:
        # print(an)
        if an["Title"].lower() in responsible_titles:
            continue

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
    # selected = refs.copy()
    # refs.extend(json.load(open("data/Q4B.json", "r", encoding="utf-8")))
    # refs.extend(json.load(open("data/Q4C.json", "r", encoding="utf-8")))

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

    # Create topic models and plot different figures
    model_topics(corpus, dates, min_cluster_size=2)


    # Citation ordering
    # df = []
    # for key, item in annotated.iterrows():
    #     for risk in item["risks"]:
    #         df.append({
    #             "title": item["title"],
    #             "risk": risk,
    #             "citations": item["citations"],
    #             "method": item["method"][0],
    #             "bibtex": item["bibtex"]
    #         })
    # df = pd.DataFrame.from_records(df)
    # most_cited = df.sort_values(by=["citations"], ascending=False).groupby("risk").head(5).sort_values(by=["risk", "citations"], ascending=False)
    # most_cited.to_csv("output/most_cited.csv", sep="\t")


    # Plot wordclouds
    # stemmer = nltk.PorterStemmer()
    # analyzer = TfidfVectorizer().build_analyzer()
    # def stemmed_words(doc):
    #     return (stemmer.stem(w.lower()) for w in analyzer(doc) if w not in nltk.corpus.stopwords.words("english"))
    # tfidf_vectorizer = TfidfVectorizer(analyzer=stemmed_words)
    # vecs = tfidf_vectorizer.fit_transform(corpus)
    # tfidf_scores = pd.DataFrame(vecs.todense().tolist(), columns=tfidf_vectorizer.get_feature_names_out()).T.sum(axis=1)
    # print(tfidf_scores.sort_values(ascending=False).head(20))

    # wordcloud = wc.WordCloud(width=1000, height=1000, background_color="white", colormap="tab10", random_state=42).generate_from_frequencies(tfidf_scores)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis("off")
    # plt.tight_layout()
    # plt.savefig("output/wc-tfidf.pdf")
    # plt.show()


    # Overall publication counts
    # plt.rc('font', size=13)
    # sns.set_style("whitegrid")
    # plt.figure(figsize=(6, 3.5))
    # years, counts = list(zip(*list(sorted(Counter(all_dates).items()))))
    # plt.plot(years, counts)
    # annotated["year"].value_counts().sort_index().plot.line(x="count", linewidth=3)
    # plt.xlabel("Year")
    # plt.ylabel("Publication Count")
    # plt.xlim(1998, 2023)
    # plt.legend(["All", "Selected"])
    # plt.tight_layout()
    # plt.savefig("output/pub-years.pdf")
    # plt.show()
    # plt.rc('font', size=11)


    # # Co-occurence matrix 
    # vocabulary = []
    # for kws in annotated["allkws"]:
    #     vocabulary.append(dict(zip(kws, [1] * len(kws))))
    # df = pd.DataFrame.from_records(vocabulary)
    # df[pd.isna(df)] = 0
    # cooc = df.T.dot(df)
    # np.fill_diagonal(cooc.values, 0)

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

    # left, bottom, width, height = [0.76, 0.76, 0.25, 0.25]
    # ax2 = fig.add_axes([left, bottom, width, height])
    # widths = np.array(list(nx.get_edge_attributes(g, 'weight').values()))
    # options = {
    #     'node_color': ["green" if n in keys else "red" for n in g.nodes()],
    #     'node_size': 20,
    #     'width': 0.05,
    #     'font_size': 6
    # }
    # pos = nx.kamada_kawai_layout(g)
    # nx.draw_networkx(g, pos, ax=ax2, with_labels=False, **options)
    # plt.axis("off")
    # plt.tight_layout()
    # plt.savefig("output/ann_kw_graph.png", dpi=300)
    # plt.show()


    # Plot risk stats
    # counts = get_counts(annotated["risks"])
    # draw_piechart(counts, "risks", inpct=0.6, pctcut=3)
    # plt.show()
    

    # # Plot method types
    # counts = get_counts(annotated["method"])
    # counts["analysis\nframework"] += counts["dataset"]
    # counts = counts.drop("dataset")
    # draw_piechart(counts.rename({"simulated\nagents": "agent\nsimulation"}), "methods")
    # plt.show()

    # counts = get_counts(annotated["keywords"])
    # methods = ["supervised learning", "unsupervised learning", "semi-supervised learning", "self-supervised learning", "reinforcement learning", "machine learning"]
    # draw_piechart(counts[methods].rename({ix: ix.replace(" ", "\n") for ix in methods}), fname="areas")
    # plt.show()

    # counts = get_counts(annotated["countries"])
    # counts["Other"] = len(counts[counts < 4])
    # counts = counts[counts > 3]
    # draw_piechart(counts, "countries")
    # plt.show()


    # # Plot institutions
    # plt.rc('font', size=14)
    # counts = get_counts(annotated["affiliation_type"].apply(lambda x: [i.replace(" ", "\n") for i in x]))
    # sns.set_style("whitegrid")
    # plt.figure(figsize=(6, 3.5))
    # counts.plot.barh(color=mpl.colormaps.get_cmap("tab10").colors, width=0.9)
    # # draw_piechart(counts, "affiliation_type")
    # plt.tight_layout()
    # plt.savefig("output/affiliation_type.pdf")
    # plt.show()
    # plt.rc('font', size=11)

    # counts = get_counts(annotated["affiliation"])
    # df = []
    # for key, count in counts.items():
    #     for idx, countries in annotated["countries"].items():
    #         if key in annotated.loc[idx, "affiliation"]:
    #             country = countries[annotated.loc[idx, "affiliation"].index(key)]
    #             afftype = annotated.loc[idx, "affiliation_type"][0] if len(annotated.loc[idx, "affiliation_type"]) == 1 else "UNK"
    #             break
    #     df.append({
    #         "Country": country,
    #         "Type": afftype,
    #         "Institution": key,
    #         "Count": count
    #     })
    # df = pd.DataFrame.from_records(df)
    # df["Institution"] = df["Institution"].str.title()
    # df[df["Count"] > 2].groupby(["Country", "Type", "Institution"]).max().to_latex("output/institutions.txt")


    # # Plot method type stats
    # params = plt.rcParams.copy()
    # sns.set_style("whitegrid")
    # plt.rc("font", size=18)
    # a = annotated.groupby(["framework", "algorithm"])["type"].value_counts()
    # a = a.reset_index()
    # g = sns.catplot(a, x="algorithm", y="count", col="type", kind="bar", legend="auto", errorbar=None)
    # g.set_xlabels("")
    # g.set_ylabels("Counts")
    # g.set_xticklabels(["Without Algorithm", "With Algorithm"])
    # # g.legend.set_title("Proposes\nFramework?")
    # # g.legend.set_bbox_to_anchor((0.7, 0.5))
    # g.axes[0, 0].set_title("Theoretical/No Evaluation")
    # g.axes[0, 1].set_title("Applied/Evaluated")
    # plt.tight_layout()
    # plt.savefig("output/framework_algo.pdf")
    # plt.show()
    # plt.rcParams = params

    return 1


if __name__ == "__main__":
    sys.exit(main())
