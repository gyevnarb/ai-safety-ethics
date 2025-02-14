import sys
import json
import os
import re
import pickle

import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib as mpl
from graph_tool import Graph

from maps import risk_map, methods_map


def load_data():
    ann = json.load(open("data/structured_search.json", "r"))
    ann.extend(json.load(open("data/snowballing.json", "r")))

    anns = []
    for an in ann:
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
    refs.extend(json.load(open("data/Q4B.json", "r", encoding="utf-8")))
    refs.extend(json.load(open("data/Q4C.json", "r", encoding="utf-8")))

    # Prepare data
    corpus, dates,  = [], []
    for item in refs:
        corpus.append(item["title"] + "; " + item["abstract"])
        dates.append(int(item["issued"]["date-parts"][0][0]))

    return pd.DataFrame.from_records(anns), refs, corpus, dates


def main():
    # mpl.rcParams['pdf.fonttype'] = 42
    # mpl.rcParams['ps.fonttype'] = 42
    # plt.rc('font', size=12)

    if not os.path.exists("output"):
        os.mkdir("output")

    if os.path.exists("output/data.p"):
        annotated, references, corpus, dates = pickle.load(open("output/data.p", "rb"))
    else:
        annotated, references, corpus, dates = load_data()
        pickle.dump((annotated, references, corpus, dates), open("output/data.p", "wb"))


    # Co-occurence matrix
    vocabulary = []
    for kws in annotated["allkws"]:
        vocabulary.append(dict(zip(kws, [1] * len(kws))))
    df = pd.DataFrame.from_records(vocabulary)
    df[pd.isna(df)] = 0
    cooc = df.T.dot(df)
    np.fill_diagonal(cooc.values, 0)


    # Select only specific keywords for plotting
    g = Graph(cooc)
    # keys = ["x risks", "existential risk", "artificial general intelligence (agi)", "aixi", "friendly ai"]  # "fairness", "safety", "ethics", "ai ethics",

    h = g.copy()
    to_drop = [
        n for n, nbrs in h.adj.items() if n not in keys and not any([nbr in keys for nbr in nbrs])
    ] + ["theoretical", "framework", "applied", "algorithm"]
    h.remove_nodes_from(to_drop)
    labels = {n: n if " " not in n else n.replace(" ", "\n").replace("\nand", " and") for n in h.nodes()}
    widths = np.array(list(nx.get_edge_attributes(h, 'weight').values()))
    options = {
        'node_color': ["green" if n in keys else "red" for n in h.nodes()],
        'node_size': 1000,
        'width': widths / widths.max() * 0.75,
        'font_size': 6
    }
    fig = plt.figure(figsize=(20, 12))
    pos = nx.kamada_kawai_layout(h)
    nx.draw_networkx(h, pos, with_labels=False, **options)
    nx.draw_networkx_labels(h, pos, labels)
    plt.axis("off")

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

    return 1


if __name__ == "__main__":
    os.chdir("../")
    sys.exit(main())
