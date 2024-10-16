import json, os
import pandas as pd

from maps import risk_map, methods_map, responsible_titles


def load_data(cwd: str = "."):
    ann = json.load(open(os.path.join(cwd, "data/annotations/structured_search.json"), "r"))
    ann.extend(json.load(open(os.path.join(cwd, "data/annotations/snowballing.json"), "r")))

    anns = []
    for an in ann:
        # print(an)
        if an["Title"].lower() in responsible_titles:
            continue

        result = {
                "title": an["Title"],
                "year": int(an["Year"]),
                "bibtex": an["BibtexKey"],
                "keywords": set(map(lambda x: x.lower(), an["Keywords"])),
                "akeywords": set(map(str.lower, an["AuthorKeywords"])),
                "risks": list(set(map(lambda x: risk_map(str.lower(x)), an["RiskTypes"]))),
                "stages": list(map(str.lower, an["Stage"])),
                "method": [methods_map(an["Method"].lower())]
            }
        result["allkws"] = list(set(result["keywords"].union(result["akeywords"])))
        result["method"] = list(map(methods_map, result["method"]))
        for key in ["framework", "algorithm"]:
            result[key] = str(key in result["keywords"])
        result["type"] =  "applied" if "applied" in result["keywords"] else "theoretical"
        anns.append(result)

    refs = json.load(open(os.path.join(cwd, "data/export/Found.json"), "r", encoding="utf-8"))
    refs.extend(json.load(open(os.path.join(cwd, "data/export/Q4A.json"), "r", encoding="utf-8")))
    refs.extend(json.load(open(os.path.join(cwd, "data/export/Q4B.json"), "r", encoding="utf-8")))
    refs.extend(json.load(open(os.path.join(cwd, "data/export/Q4C.json"), "r", encoding="utf-8")))

    # Prepare data
    corpus, dates,  = [], []
    for item in refs:
        corpus.append(item["title"] + "; " + item["abstract"])
        dates.append(int(item["issued"]["date-parts"][0][0]))
    
    return pd.DataFrame.from_records(anns), refs, corpus, dates


def get_counts(data):
    items = []
    for item in data:
        items.extend(item)
    return pd.Series(items).value_counts()