"""Small helpers shared by the parsers."""

import html
import io
import json
import re
import unicodedata
import warnings

import pandas as pd


def norm_name(name):
    """Lower-case ASCII name with single spaces: 'José  Pérez-Díaz' -> 'jose perez diaz'."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    s = unicodedata.normalize("NFKD", html.unescape(str(name))).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", " ", s)
    return " ".join(s.split())


def strip_tags(text):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", str(text))).split())


def to_float(value):
    """First number in a value ('112.93 km', '1,234', '<del>31.8</del>') or None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"-?\d+(?:\.\d+)?", strip_tags(value).replace(",", ""))
    return float(m.group(0)) if m else None


def to_int(value):
    f = to_float(value)
    return int(f) if f is not None else None


def hms(value):
    """'3:17:06' -> 11826 seconds; None when there is no time."""
    s = strip_tags(value)
    m = re.search(r"(-?)(\d+):(\d{2})(?::(\d{2}))?", s)
    if not m:
        return None
    sign = -1 if m.group(1) else 1
    return sign * (int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4) or 0))


def read_tables(page):
    """All HTML tables of a page as DataFrames (empty list when there are none)."""
    if not page:
        return []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.read_html(io.StringIO(page))
        except ValueError:  # no tables
            return []
        except Exception:  # one malformed table spoils the whole page: read them one by one
            tables = []
            for chunk in re.findall(r"<table\b.*?</table>", page, flags=re.S | re.I):
                try:
                    tables += pd.read_html(io.StringIO(chunk))
                except Exception:
                    continue
            return tables


def js_value(text, marker):
    """Parse the JSON object or array that follows `marker` in a page (e.g. 'window.ATDATA.pilots =')."""
    i = text.find(marker)
    if i < 0:
        return None
    start = i + len(marker)
    while start < len(text) and text[start] not in "{[":
        start += 1
    if start >= len(text):
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
        return value
    except json.JSONDecodeError:
        return None


def pair_table(tables):
    """Merge every two-column table (label, value) of a page into one dict."""
    out = {}
    for t in tables:
        if t.shape[1] != 2:
            continue
        cols = [str(c) for c in t.columns]
        if cols == ["param", "value"] or cols == ["0", "1"] or all(c.isdigit() for c in cols):
            for k, v in zip(t.iloc[:, 0], t.iloc[:, 1]):
                if isinstance(k, str):
                    out.setdefault(k.strip(), v)
    return out
