"""
Step 2 - Data cleaning for the Movie Review Prediction project.

Takes the raw scrape (indian_movies_reviews.csv) and produces a lean, ASCII-only
dataset (movies_cleaned.csv) ready for modelling:

  1. movie_title reduced to letters, numbers and a small set of punctuation
  2. review_text dropped - it was 96.9% of the file and review_url already
     points at the same content
  3. slug dropped - it is embedded in review_url, so nothing is lost

The raw CSV is left untouched, so cleaning rules can change without re-scraping.

Run:  python data_cleaning.py
"""

import os
import re

import pandas as pd

IN_CSV = "indian_movies_reviews.csv"
OUT_CSV = "movies_cleaned.csv"
DROP_COLUMNS = ["review_text", "slug"]

# Punctuation that is allowed to survive in a title. The full stop is kept so
# decimal-style titles stay intact ("Raman Raghav 2.0", "Shaadi No.1").
ALLOWED_PUNCTUATION = " '+,-_:/@."

# Unicode punctuation mapped to its ASCII counterpart FIRST, so that characters
# like the en dash land inside the allowed set instead of being stripped out.
# This is what turns "Toilet <en dash> Ek Prem Katha" into "Toilet - Ek Prem
# Katha" rather than "Toilet Ek Prem Katha".
UNICODE_MAP = {
    "–": "-",   # en dash
    "—": "-",   # em dash
    "‒": "-",   # figure dash
    "’": "'",   # right single quote (It's)
    "‘": "'",   # left single quote
    "ʼ": "'",   # modifier letter apostrophe
    "“": "",    # left double quote
    "”": "",    # right double quote
    "…": " ",   # ellipsis
}

# Anything not a letter, digit or allowed punctuation.
OFF_LIST = re.compile(r"[^0-9A-Za-z '+,\-_:/@.]")


def clean_title(text):
    """Reduce a title to letters, numbers, spaces and allowed punctuation."""
    text = str(text)
    for unicode_char, ascii_char in UNICODE_MAP.items():
        text = text.replace(unicode_char, ascii_char)
    text = text.replace("&", " and ")            # before anything is stripped
    text = OFF_LIST.sub(" ", text)               # everything off-list -> space
    text = re.sub(r"\s+", " ", text).strip()     # collapse runs of whitespace
    return text.strip(ALLOWED_PUNCTUATION)       # no dangling punctuation


def verify(df, original):
    """Fail loudly on anything that would poison the modelling step."""
    assert len(df) == len(original), "row count changed"
    assert not df["movie_title"].str.contains(OFF_LIST, regex=True).any(), \
        "a title still holds a disallowed character"
    for column in df.columns:
        values = df[column].dropna().astype(str)
        assert not values.str.contains(r"[^\x00-\x7F]", regex=True).any(), \
            "non-ASCII survived in column %s" % column
    assert (df["movie_title"].str.len() > 0).all(), "a title cleaned to empty"
    assert not df["movie_title"].str.contains("  ").any(), "doubled space"
    assert (df["movie_title"] == df["movie_title"].str.strip()).all(), \
        "leading or trailing whitespace"
    for column in DROP_COLUMNS:
        assert column not in df.columns, "%s was not dropped" % column
    assert df[["movie_title", "lead_actor", "sentiment"]].notna().all().all(), \
        "null in a mandatory column"
    expected = (df["rating"] >= 3.5).map({True: "positive", False: "negative"})
    assert (df["sentiment"] == expected).all(), "sentiment no longer matches rating"
    assert not df.duplicated(subset=["movie_title", "year"]).any(), \
        "(movie_title, year) is not unique"


def main():
    df = pd.read_csv(IN_CSV)
    original = df.copy()
    before_size = os.path.getsize(IN_CSV)

    # Movie title only in text and numbers, no unparsable characters
    df["movie_title"] = df["movie_title"].map(clean_title)

    # Removal of review text column
    # Removal of slug column
    df = df.drop(columns=DROP_COLUMNS)

    verify(df, original)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    after_size = os.path.getsize(OUT_CSV)

    changed = original["movie_title"] != df["movie_title"]
    print("Cleaned titles: %d of %d changed\n" % (changed.sum(), len(df)))
    for old, new in zip(original.loc[changed, "movie_title"],
                        df.loc[changed, "movie_title"]):
        print("  %-42.42s -> %s" % (old, new))

    print("\n" + "=" * 62)
    print("  rows            %d -> %d" % (len(original), len(df)))
    print("  columns         %d -> %d  (dropped %s)"
          % (original.shape[1], df.shape[1], ", ".join(DROP_COLUMNS)))
    print("  file size       %.1f MB -> %.0f KB"
          % (before_size / 1e6, after_size / 1e3))
    print("  non-ASCII cells %d -> 0"
          % original.apply(
              lambda c: c.dropna().astype(str)
              .str.contains(r"[^\x00-\x7F]", regex=True).sum()).sum())
    print("  wrote %s" % OUT_CSV)
    print("=" * 62)


if __name__ == "__main__":
    main()
