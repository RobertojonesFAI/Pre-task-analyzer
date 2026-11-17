#!/usr/bin/env python3
"""Glider models and classes for every result: one name per model, and its class (CCC, EN-D,
EN-C, EN-B, EN-A).

Pilots and scoring sites write the same glider many ways ("Ozone Enzo 3", "OZONE ENZO3",
"Enzo 3 MS"). Names are reduced to a key (lower case, no brand, no size, no spaces), then each
key gets a class:

    declared   the class most results with that key declare (results.glider_cert)
    curated    otherwise, a short list of well-known competition families (below)
    unknown    otherwise

Writes DB/result_gliders.parquet (result_id, glider_key, glider_class, class_source) and
DB/gliders.parquet (one row per key: example name, results, class, how it was found).

    python3 scripts/gliders.py /content/drive/MyDrive/IGCS/database
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

BRANDS = ["ozone", "gin", "gingliders", "advance", "niviuk", "flow", "flowparagliders", "axis", "macpara", "mac para",
          "skywalk", "triple seven", "777", "nova", "swing", "phi", "bgd", "dudek", "up", "u-turn", "uturn", "sol",
          "airdesign", "air design", "supair", "aircross", "icaro", "skyman", "ozone paragliders", "gradient",
          "porcher", "windtech", "independence", "firebird", "apco", "freex", "paratech", "papillon"]
SIZES = {"xxs", "xs", "s", "sm", "ms", "m", "ml", "l", "xl", "xxl", "xxxs"}
# Well-known competition families: used only when no result declares a class for that model
CURATED = {
    "enzo": "CCC", "boomerang": "CCC", "venus": "CCC", "icepeakxone": "CCC", "xone": "CCC", "xcracer": "CCC",
    "zeno": "EN-D", "mantra": "EN-D", "omegaxalps": "EN-D", "peak": "EN-D",
    "delta": "EN-C", "sigma": "EN-C",
    "rush": "EN-B", "mentor": "EN-B",
}
CLASSES = ["CCC", "EN-D", "EN-C", "EN-B", "EN-A"]


def key(name):
    """'OZONE Enzo 3 MS' -> 'enzo3'."""
    if not isinstance(name, str) or not name.strip():
        return ""
    s = name.lower().replace("x-alps", "xalps").replace("x alps", "xalps").replace("x-one", "xone")
    for b in sorted(BRANDS, key=len, reverse=True):
        s = re.sub(rf"(^|[\s\-_/]){re.escape(b)}(?=$|[\s\-_/])", " ", s)
    s = re.sub(r"(\d)\s+\d{2}(?:[.,]\d+)?\s*(?:m2|m²|sqm)?\s*$", r"\1", s.strip())   # area after the model number
    s = re.sub(r"[()\[\],.;:/_\-+]", " ", s)
    words = s.split()
    while words and (words[-1] in SIZES or re.fullmatch(r"\d{2}(\.\d)?m?2?", words[-1]) and len(words) > 2):
        words.pop()   # a trailing size ("MS") or area ("22.5") after the model number
    return "".join(words)


def family(k):
    """'enzo3' -> 'enzo', 'omegaxalps3' -> 'omegaxalps'."""
    return re.sub(r"\d+[a-z]?$", "", k)


def cert(c):
    if not isinstance(c, str):
        return None
    c = c.upper().replace(" ", "").replace("-", "")
    if c in ("OPEN", "COMPETITION", "COMP"):
        return "CCC"
    c = c.replace("EN", "EN-") if c.startswith("EN") else c
    for k in CLASSES:
        if k.replace("-", "") == c.replace("-", ""):
            return k
    if c in ("A", "B", "C", "D"):
        return f"EN-{c}"
    return None


def classify(results):
    r = results[["result_id", "glider", "glider_cert"]].copy()
    r["glider_key"] = r["glider"].map(key)
    r["declared"] = r["glider_cert"].map(cert)
    votes = r.dropna(subset=["declared"]).query("glider_key != ''")
    major = votes.groupby("glider_key")["declared"].agg(lambda x: x.value_counts().index[0])
    declared = r["glider_key"].map(major)
    curated = r["glider_key"].map(lambda k: CURATED.get(family(k)) or CURATED.get(k))
    r["glider_class"] = declared.where(declared.notna(), curated)
    r["class_source"] = np.where(declared.notna(), "declared", np.where(curated.notna(), "curated", "unknown"))
    return r[["result_id", "glider_key", "glider_class", "class_source"]], r


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db", type=Path)
    args = parser.parse_args()
    results = pd.read_parquet(args.db / "results.parquet", columns=["result_id", "glider", "glider_cert", "status"])
    out, r = classify(results)
    out.to_parquet(args.db / "result_gliders.parquet", index=False)
    g = r.groupby("glider_key").agg(example=("glider", lambda x: x.value_counts().index[0] if x.notna().any() else ""),
                                    results=("result_id", "size"), glider_class=("glider_class", "first"),
                                    class_source=("class_source", "first")).reset_index()
    g = g[g["glider_key"] != ""].sort_values("results", ascending=False)
    g.to_parquet(args.db / "gliders.parquet", index=False)
    named = r["glider_key"] != ""
    print(f"results: {len(r):,}; with a glider name {named.mean():.0%}; {g.shape[0]:,} distinct models after "
          f"normalising {r.loc[named, 'glider'].nunique():,} spellings")
    print("class found:", r["class_source"].value_counts(normalize=True).map("{:.1%}".format).to_dict())
    print("classes:", r["glider_class"].value_counts(normalize=True).map("{:.1%}".format).to_dict())
    print("\nMost flown models:\n" + g.head(20)[["example", "glider_key", "results", "glider_class", "class_source"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
