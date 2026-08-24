"""
Step 1 - Data collection for the Movie Review Prediction project.

Scrapes Indian (Hindi) movies from Bollywood Hungama and writes a CSV with, at
minimum, the movie title, its lead actor, and whether the critic review was
positive or negative.

Source : https://www.bollywoodhungama.com  (public pages; robots.txt allows
         crawling and sets no Crawl-delay)
Usage  : personal / educational. The critic ratings and review text remain the
         copyright of Bollywood Hungama and are not redistributed here.

How it works (two passes, so the dataset favours actors with several films):
  Pass A - fetch every film's critic-review page. Gives the rating plus the
           film's actors. Films that survive to Pass B needed this page anyway,
           so only the discards are wasted work.
  Pass B - fetch the cast page for the selected films. This is the ONLY place
           the cast is in true billing order, so it is where lead_actor comes
           from. It also carries Language, Genre and Music Director.

Runs with 5 workers and finishes in roughly 40 minutes. TIME_BUDGET_MIN is a
hard ceiling: if the crawl runs late it stops fetching and builds the CSV from
whatever it has, rather than overrunning.

Run:  python scrape_movies.py
Resumable - press Ctrl-C any time and re-run to continue where it stopped.
"""

import html
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from bs4 import BeautifulSoup

# --------------------------------------------------------------------------
# Config - everything you might want to change lives here
# --------------------------------------------------------------------------
BASE = "https://www.bollywoodhungama.com"
SITEMAP_INDEX = BASE + "/sitemap_index.xml"

TARGET_ROWS = 700           # rows we want in the final CSV
MIN_FILMS_PER_ACTOR = 3     # drop leads with fewer films than this, so the
                            # actor feature has something to generalise from
PASS_A_POOL = None          # films whose review page we fetch; None = all
PASS_B_POOL = 950           # films whose cast page we fetch (~85% survive)
POSITIVE_THRESHOLD = 3.5    # rating >= this is "positive" (scale is 0-5)
LANGUAGE = "Hindi"          # keep only films in this language

MAX_WORKERS = 5             # concurrent requests; a browser opens more
TIME_BUDGET_MIN = 60        # hard ceiling for the whole run
TIMEOUT = 30
MAX_RETRIES = 3
RANDOM_SEED = 42

# Set to a small number (e.g. 20) to smoke-test the pipeline in ~1 min.
# None = full run.
SMOKE_TEST = None

OUT_CSV = "indian_movies_reviews.csv"
CANDIDATES_FILE = "scrape_candidates.json"
PROGRESS_REVIEWS = "scrape_progress_reviews.jsonl"
PROGRESS_CAST = "scrape_progress_cast.jsonl"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Slug suffixes Bollywood Hungama uses for non-Hindi releases. Dropping these
# early saves requests; the Language field in Pass B is the real filter.
FOREIGN_SUFFIXES = (
    "-english", "-tamil", "-telugu", "-malayalam", "-kannada",
    "-marathi", "-punjabi", "-bengali", "-korean", "-japanese",
)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})
# Give the connection pool room for every worker, else they queue on each other.
_adapter = requests.adapters.HTTPAdapter(
    pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS
)
SESSION.mount("https://", _adapter)
SESSION.mount("http://", _adapter)

_write_lock = threading.Lock()

REVIEW_SLUG_RE = re.compile(
    r"^https://www\.bollywoodhungama\.com/movie/([^/]+)/critic-review/"
)
JSONLD_RE = re.compile(
    r'<script[^>]*application/ld\+json[^>]*>(.*?)</script>', re.S
)
LOC_RE = re.compile(r"<loc>([^<]+)</loc>")


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------
def get(url):
    """Fetch a URL and return its text, or None if it permanently fails.

    Returning None rather than raising means one bad film never kills a run
    that is already half an hour deep.
    """
    for attempt in range(MAX_RETRIES):
        try:
            resp = SESSION.get(url, timeout=TIMEOUT)
            if resp.status_code == 200:
                resp.encoding = "utf-8"
                return resp.text
            if resp.status_code == 404:
                return None
            if resp.status_code not in (429, 500, 502, 503, 504):
                return None
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)  # 1s, 2s, 4s
    return None


def read_jsonl(path):
    """Load a progress file into {slug: record}. Missing file -> {}."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a half-written last line after Ctrl-C
            out[rec["slug"]] = rec
    return out


def append_jsonl(path, record):
    with _write_lock:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_pool(todo, worker, path, label, deadline):
    """Run `worker` over `todo` with MAX_WORKERS threads.

    Each result is appended to `path` as it lands, so progress survives a
    Ctrl-C. Stops starting new work once `deadline` passes.
    """
    results = {}
    if not todo:
        return results

    stop = threading.Event()

    def guarded(item):
        if stop.is_set():
            return None
        try:
            return worker(item)
        except Exception:
            return None  # never let one film kill the whole pass

    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(guarded, item) for item in todo]
        try:
            for future in as_completed(futures):
                record = future.result()
                done += 1
                if record is not None:
                    append_jsonl(path, record)
                    results[record["slug"]] = record
                if time.time() > deadline and not stop.is_set():
                    stop.set()
                    print("\n  time budget reached - wrapping up this pass")
                left = max(0, int(deadline - time.time()) // 60)
                print("  %s [%d/%d] kept %d  (%dm left)   "
                      % (label, done, len(todo), len(results), left), end="\r")
        except KeyboardInterrupt:
            stop.set()
            print("\n  interrupted - progress saved, re-run to continue")
            raise
    print()
    return results


# --------------------------------------------------------------------------
# Discovery - which films exist
# --------------------------------------------------------------------------
def discover_slugs():
    """Return every movie slug that has a critic review, via the sitemaps."""
    if os.path.exists(CANDIDATES_FILE):
        with open(CANDIDATES_FILE, encoding="utf-8") as fh:
            slugs = json.load(fh)
        print("  reusing %d candidates from %s" % (len(slugs), CANDIDATES_FILE))
        return slugs

    index = get(SITEMAP_INDEX)
    if index is None:
        sys.exit("Could not fetch the sitemap index - check your connection.")

    review_maps = [u for u in LOC_RE.findall(index) if "review-sitemap" in u]
    print("  %d review sitemaps to scan" % len(review_maps))

    def fetch_map(url):
        xml = get(url)
        found = set()
        if xml:
            for loc in LOC_RE.findall(xml):
                if "music-critic-review" in loc:
                    continue  # song reviews, not film reviews
                m = REVIEW_SLUG_RE.match(loc)
                if m:
                    found.add(m.group(1))
        return found

    slugs = set()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for i, found in enumerate(pool.map(fetch_map, review_maps), 1):
            slugs |= found
            print("  [%d/%d] %d unique films so far   "
                  % (i, len(review_maps), len(slugs)), end="\r")

    slugs = sorted(s for s in slugs if not s.endswith(FOREIGN_SUFFIXES))
    print("\n  %d candidate films after dropping non-Hindi slugs" % len(slugs))

    with open(CANDIDATES_FILE, "w", encoding="utf-8") as fh:
        json.dump(slugs, fh)
    return slugs


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def _jsonld_blocks(html):
    for raw in JSONLD_RE.findall(html):
        try:
            yield json.loads(raw.strip())
        except json.JSONDecodeError:
            continue


def _clean_text(value):
    """Titles come out of the JSON-LD HTML-escaped (Toilet &#8211; Ek Prem
    Katha). Decode them so the CSV holds real characters."""
    if isinstance(value, str):
        return html.unescape(value).strip()
    return value


def _names(value):
    """schema.org person field -> list of names."""
    if isinstance(value, list):
        return [d.get("name") for d in value
                if isinstance(d, dict) and d.get("name")]
    if isinstance(value, dict) and value.get("name"):
        return [value["name"]]
    return []


def parse_review(html):
    """Pull the schema.org/Review block off a critic-review page.

    Note: the JSON-LD 'actor' list is NOT in billing order (verified - one film
    lists the supporting actress before the lead), so it is only used here to
    measure how often an actor appears. The real lead comes from the cast page.
    """
    review = None
    for block in _jsonld_blocks(html):
        if isinstance(block, dict) and block.get("@type") == "Review":
            review = block
            break
    if review is None:
        return None

    rating = (review.get("reviewRating") or {}).get("ratingValue")
    try:
        rating = float(rating)  # site emits both "2" and "2.0"
    except (TypeError, ValueError):
        return None

    item = review.get("itemReviewed") or {}
    date = review.get("datePublished") or item.get("datePublished") or ""
    year = re.search(r"\b(1[89]\d{2}|20\d{2})\b", date)

    soup = BeautifulSoup(html, "lxml")
    article = soup.select_one("article")
    text = None
    if article:
        text = re.sub(r"\s+", " ", article.get_text(" ", strip=True))
        text = re.sub(r"^\d+\s*", "", text).strip() or None

    return {
        "title": review.get("name"),
        "rating": rating,
        "release_date": date,
        "year": int(year.group(1)) if year else None,
        "critic": (review.get("author") or {}).get("name"),
        "director": ", ".join(_names(item.get("director"))) or None,
        "jsonld_actors": _names(item.get("actor")),
        "review_text": text,
    }


def parse_cast(html):
    """Pull billing-ordered cast + crew details off a /cast/ page.

    Structure is ul.crew-list > li, where h4.entry-title.name is the label and
    whatever remains in the li is the value.
    """
    soup = BeautifulSoup(html, "lxml")

    fields = {}
    for li in soup.select(".crew-wrapper li"):
        label_el = li.select_one("h4.entry-title.name")
        if label_el is None:
            continue
        label = label_el.get_text(" ", strip=True).rstrip(":").strip()
        label_el.extract()  # drop the label so only the value text is left
        values = [v.strip() for v in li.get_text("~", strip=True).split("~")
                  if v.strip()]
        if values:
            fields[label] = values

    def clean(val):
        if not val or val.lower().startswith("not available"):
            return None
        return val

    def first(label):
        vals = fields.get(label)
        return clean(vals[0]) if vals else None

    def joined(label):
        vals = fields.get(label)
        return clean(", ".join(vals)) if vals else None

    # "Govinda ... Ganga" -> "Govinda";  "Ramdas Jadhav ..." -> "Ramdas Jadhav"
    cast = []
    for entry in fields.get("Star Cast", []):
        name = entry.split("...")[0].strip()
        if name:
            cast.append(name)

    if not cast:
        return None

    return {
        "cast": cast,
        "language": first("Language"),
        "genre": joined("Genre"),
        "music_director": joined("Music Director"),
        "cast_release_date": " ".join(fields.get("Release Date", [])) or None,
    }


# --------------------------------------------------------------------------
# The two passes
# --------------------------------------------------------------------------
def pass_a(slugs, deadline):
    """Fetch review pages. Returns {slug: record}."""
    done = read_jsonl(PROGRESS_REVIEWS)
    todo = [s for s in slugs if s not in done]
    print("Pass A - review pages: %d already done, %d to fetch"
          % (len(done), len(todo)))

    def work(slug):
        url = "%s/movie/%s/critic-review/" % (BASE, slug)
        html = get(url)
        record = parse_review(html) if html else None
        if record and record.get("title"):
            record["slug"] = slug
            record["review_url"] = url
            return record
        # Remember the failure so a re-run does not retry it forever.
        return {"slug": slug, "rating": None}

    done.update(run_pool(todo, work, PROGRESS_REVIEWS, "review", deadline))
    print("  %d review pages usable"
          % sum(1 for r in done.values() if r.get("rating") is not None))
    return done


def select_for_pass_b(reviews):
    """Pick which films get a cast-page fetch, favouring recurring actors.

    The JSON-LD actor ORDER is unreliable, but actor MEMBERSHIP is fine - which
    is all we need to spot the actors who keep showing up.
    """
    usable = [r for r in reviews.values() if r.get("rating") is not None]

    freq = Counter()
    for rec in usable:
        for name in rec.get("jsonld_actors", []):
            freq[name] += 1

    def score(rec):
        actors = rec.get("jsonld_actors") or []
        return max((freq[a] for a in actors), default=0)

    usable.sort(key=lambda r: (-score(r), r["slug"]))
    limit = PASS_B_POOL or len(usable)
    return usable[:limit]


def pass_b(selected, deadline):
    """Fetch cast pages for the selected films. Returns {slug: record}."""
    done = read_jsonl(PROGRESS_CAST)
    todo = [r["slug"] for r in selected if r["slug"] not in done]
    print("Pass B - cast pages: %d already done, %d to fetch"
          % (len(done), len(todo)))

    def work(slug):
        html = get("%s/movie/%s/cast/" % (BASE, slug))
        cast = parse_cast(html) if html else None
        row = {"slug": slug}
        if cast:
            row.update(cast)
        return row

    done.update(run_pool(todo, work, PROGRESS_CAST, "cast  ", deadline))
    print("  %d cast pages usable"
          % sum(1 for r in done.values() if r.get("cast")))
    return done


# --------------------------------------------------------------------------
# Build the dataset
# --------------------------------------------------------------------------
def build_dataset(reviews, casts):
    rows = []
    for slug, cast in casts.items():
        review = reviews.get(slug)
        if not review or review.get("rating") is None or not cast.get("cast"):
            continue
        names = [_clean_text(n) for n in cast["cast"]]
        rows.append({
            "movie_title": _clean_text(review["title"]),
            "lead_actor": names[0],
            "rating": review["rating"],
            "year": review["year"],
            "release_date": review["release_date"] or cast.get("cast_release_date"),
            "language": cast.get("language"),
            "genre": _clean_text(cast.get("genre")),
            "director": _clean_text(review.get("director")),
            "actor_2": names[1] if len(names) > 1 else None,
            "actor_3": names[2] if len(names) > 2 else None,
            "music_director": cast.get("music_director"),
            "critic": review.get("critic"),
            "review_text": review.get("review_text"),
            "review_url": review.get("review_url"),
            "slug": slug,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("No rows survived - nothing was scraped successfully.")
    print("\nBuilding dataset from %d scraped films" % len(df))

    df = df[df["language"] == LANGUAGE]
    print("  %d after keeping only %s films" % (len(df), LANGUAGE))

    df = df.dropna(subset=["movie_title", "lead_actor", "rating"])

    # A few cast pages echo the film name as the lead (e.g. the film "Sheen"
    # lists "Sheen"). Those are parse artifacts, not real actors.
    df = df[df["lead_actor"].str.strip().str.lower()
            != df["movie_title"].str.strip().str.lower()]
    print("  %d after dropping missing / artifact leads" % len(df))

    # Slugs carry numeric disambiguators that are NOT sequels (yodha-2 is the
    # film "Yodha"), so de-duplicate on the real title, never the slug.
    df = df.drop_duplicates(subset=["slug"])
    df = df.drop_duplicates(subset=["movie_title", "year"])
    print("  %d after de-duplication" % len(df))

    df["sentiment"] = (df["rating"] >= POSITIVE_THRESHOLD).map(
        {True: "positive", False: "negative"}
    )

    # Keep only leads with enough films for the actor signal to generalise.
    # Most of the archive is a long tail of one-film actors.
    counts = df["lead_actor"].value_counts()
    df = df[df["lead_actor"].map(counts) >= MIN_FILMS_PER_ACTOR]
    print("  %d after dropping leads with < %d films (%d actors left)"
          % (len(df), MIN_FILMS_PER_ACTOR, df["lead_actor"].nunique()))

    # Trim the surplus off the most over-represented actors, so we lose films
    # from Akshay Kumar's 74 rather than dropping whole actors.
    df["_lead_films"] = df["lead_actor"].map(df["lead_actor"].value_counts())
    if len(df) > TARGET_ROWS:
        df = df.sort_values(["_lead_films", "lead_actor", "year"],
                            ascending=[False, True, True])
        df = df.iloc[len(df) - TARGET_ROWS:]
        print("  %d after trimming to the target row count" % len(df))
    else:
        print("  NOTE: %d rows collected (wanted %d). Just re-run - the"
              % (len(df), TARGET_ROWS))
        print("        progress files mean it resumes where it stopped.")
    df = df.drop(columns="_lead_films")

    columns = [
        "movie_title", "lead_actor", "sentiment",
        "rating", "year", "release_date", "language", "genre",
        "director", "actor_2", "actor_3", "music_director", "critic",
        "review_text", "review_url", "slug",
    ]
    return df[columns].reset_index(drop=True)


def verify_and_report(df):
    """Fail loudly on anything that would poison the modelling step."""
    mandatory = ["movie_title", "lead_actor", "sentiment"]
    assert df[mandatory].notna().all().all(), "null in a mandatory column"
    assert (df["language"] == LANGUAGE).all(), "non-%s film slipped through" % LANGUAGE
    assert df["rating"].between(0, 5).all(), "rating outside 0-5"
    assert not df["slug"].duplicated().any(), "duplicate slug"
    expected = (df["rating"] >= POSITIVE_THRESHOLD).map(
        {True: "positive", False: "negative"}
    )
    assert (df["sentiment"] == expected).all(), "sentiment does not match rating"

    print("\n" + "=" * 62)
    print("DATASET SUMMARY  (%d rows -> %s)" % (len(df), OUT_CSV))
    print("=" * 62)

    counts = df["sentiment"].value_counts()
    for label in ("positive", "negative"):
        n = int(counts.get(label, 0))
        print("  %-9s %4d  (%.1f%%)" % (label, n, 100 * n / len(df)))
    print("  baseline accuracy if you always guess the majority class: %.1f%%"
          % (100 * counts.max() / len(df)))

    leads = df["lead_actor"].value_counts()
    print("\n  distinct lead actors  : %d" % len(leads))
    print("  films per lead actor  : mean %.1f, max %d"
          % (leads.mean(), leads.max()))
    print("  leads with only 1 film: %d" % int((leads == 1).sum()))
    print("  year range            : %s - %s"
          % (int(df["year"].min()), int(df["year"].max())))

    print("\n  rating histogram:")
    for rating, n in sorted(df["rating"].value_counts().items()):
        print("    %.1f  %s %d" % (rating, "#" * int(40 * n / len(df)), n))

    print("\n  top 15 lead actors by film count:")
    for name, n in leads.head(15).items():
        pos = int((df[df["lead_actor"] == name]["sentiment"] == "positive").sum())
        print("    %-28.28s %3d films  %d positive" % (name, n, pos))
    print("=" * 62)


# --------------------------------------------------------------------------
def main():
    random.seed(RANDOM_SEED)
    started = time.time()
    # Reserve ~60% of the budget for Pass A and the rest for Pass B, so a slow
    # Pass A can never starve Pass B and leave us with zero usable rows.
    budget = TIME_BUDGET_MIN * 60
    deadline_a = started + budget * 0.60
    deadline_b = started + budget * 0.92

    print("Discovering films from the sitemaps...")
    slugs = discover_slugs()

    random.shuffle(slugs)
    if SMOKE_TEST:
        slugs = slugs[:SMOKE_TEST]
        print("SMOKE TEST: only %d films. Set SMOKE_TEST = None for a full run.\n"
              % SMOKE_TEST)
    elif PASS_A_POOL:
        slugs = slugs[:PASS_A_POOL]

    print("Budget %d min, %d workers.\n" % (TIME_BUDGET_MIN, MAX_WORKERS))

    reviews = pass_a(slugs, deadline_a)
    selected = select_for_pass_b(reviews)
    casts = pass_b(selected, deadline_b)

    df = build_dataset(reviews, casts)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    verify_and_report(df)

    mins, secs = divmod(int(time.time() - started), 60)
    print("\nDone in %dm %ds. Wrote %s" % (mins, secs, OUT_CSV))
    print("Progress files kept for resuming; delete them to re-scrape clean:")
    print("  %s, %s, %s" % (CANDIDATES_FILE, PROGRESS_REVIEWS, PROGRESS_CAST))


if __name__ == "__main__":
    main()
