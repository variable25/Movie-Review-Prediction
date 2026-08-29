"""
Step 6 - Serving: turn a lead actor's name into a probability.

This file has no user interface in it. app.py imports it, and so could an API, a
notebook or a test - which is the point. All it does is load the frozen model and
the actor snapshot, hand the ten columns over in the right order, and return the
answer together with the evidence behind it.

Nothing here recomputes a career average. The snapshot was built by the same
build_as_of() that built the training data (see feature_engineering.py), so the
numbers the model is asked to score at serve time were produced by the identical
code that produced the numbers it learned on.

    import predict
    predict.for_actor("Akshay Kumar")["probability"]   -> 0.46

Run:  python predict.py        (self-check: prints a few actors and exits non-zero
                                if anything is inconsistent)
"""

import json
from pathlib import Path

import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL_PKL = HERE / "model.pkl"
SNAPSHOT_CSV = HERE / "actor_snapshot.csv"
CARD_JSON = HERE / "model_card.json"
CLEANED_CSV = HERE / "movies_cleaned.csv"

DEBUT_ACTOR = "__debut__"       # matches feature_engineering.DEBUT_ACTOR
DEBUT_LABEL = "An actor with no track record"
POSITIVE_THRESHOLD = 3.5        # a "positive" review, per data_cleaning.py
THRESHOLD = 0.5                 # the decision cut, per Model-Selection.py

_LOADED = None


def load():
    """Read the model and its data once, then hand back the same objects.

    Reading a 386 KB model off disk per request would be a silly way to spend a
    web server's time, so it happens once per process. Streamlit caches on top of
    this as well; the module-level guard means it is cheap either way.
    """
    global _LOADED
    if _LOADED is not None:
        return _LOADED

    for path in (MODEL_PKL, SNAPSHOT_CSV, CARD_JSON, CLEANED_CSV):
        if not path.exists():
            raise FileNotFoundError(
                "%s is missing. Run:  python feature_engineering.py  then  "
                "python train_model.py" % path.name)

    model = joblib.load(MODEL_PKL)
    snapshot = pd.read_csv(SNAPSHOT_CSV, index_col="lead_actor")
    card = json.loads(CARD_JSON.read_text(encoding="utf-8"))
    features = card["features"]

    films = pd.read_csv(CLEANED_CSV, usecols=["movie_title", "lead_actor",
                                              "release_date", "rating"])
    films["release_date"] = pd.to_datetime(films["release_date"], format="mixed")
    films = films.sort_values("release_date")

    # The model is positional about its columns, so this is worth being loud about.
    missing = [c for c in features if c not in snapshot.columns]
    if missing:
        raise ValueError("actor_snapshot.csv is missing %s - regenerate it with "
                         "feature_engineering.py" % missing)

    _LOADED = {
        "model": model,
        "snapshot": snapshot,
        "card": card,
        "features": features,
        "films": films,
        # What an average film does, so a bare percentage can be read against
        # something. 33.9% of all 700 films cleared 3.5.
        "base_rate": float((films["rating"] >= POSITIVE_THRESHOLD).mean()),
    }
    return _LOADED


def list_actors():
    """The lead actors the model has a record for, alphabetically."""
    snapshot = load()["snapshot"]
    return sorted(name for name in snapshot.index if name != DEBUT_ACTOR)


def history(actor):
    """Every film this actor led, oldest first. Empty frame for a newcomer."""
    films = load()["films"]
    return films[films["lead_actor"] == actor][
        ["release_date", "movie_title", "rating"]].reset_index(drop=True)


def for_actor(actor):
    """The prediction for this actor's next film, and the evidence behind it.

    Pass DEBUT_ACTOR (or use for_debut) for a lead with no record: the snapshot
    carries a purpose-built row for that case, priced off the era average exactly
    as a real debut in the training data was.
    """
    data = load()
    snapshot, features = data["snapshot"], data["features"]

    if actor not in snapshot.index:
        raise KeyError("no record for %r - call list_actors() for the %d known names"
                       % (actor, len(list_actors())))

    row = snapshot.loc[[actor]]
    probability = float(data["model"].predict_proba(row[features])[:, 1][0])
    base_rate = data["base_rate"]
    n_films = int(row["prior_n_films"].iloc[0])

    return {
        "actor": DEBUT_LABEL if actor == DEBUT_ACTOR else actor,
        "is_debut": actor == DEBUT_ACTOR,
        "probability": probability,
        "verdict": "positive" if probability >= THRESHOLD else "negative",
        "base_rate": base_rate,
        "lift": probability - base_rate,

        # The evidence. A percentage on its own is a party trick; these are the
        # numbers it was computed from, so a reader can disagree with it.
        "films_to_date": n_films,
        "career_mean": None if n_films == 0 else float(row["prior_mean"].iloc[0]),
        "adjusted_mean": float(row["prior_adj_mean"].iloc[0]),
        "positive_rate": float(row["prior_positive_rate"].iloc[0]),
        "last3_mean": float(row["prior_last3_mean"].iloc[0]),
        "last5_mean": float(row["prior_last5_mean"].iloc[0]),
        "momentum": float(row["prior_momentum"].iloc[0]),
        "consistency": float(row["prior_std"].iloc[0]),
        "vs_era": float(row["prior_mean_vs_era"].iloc[0]),
        "era_mean": float(row["era_prior_mean"].iloc[0]),
        "tier": str(row["prior_tier"].iloc[0]),
        "as_of": str(row["as_of_date"].iloc[0]),
    }


def for_debut():
    """The newcomer case, spelled out so callers do not need the sentinel name."""
    return for_actor(DEBUT_ACTOR)


def card():
    """The model card - scores, limitations, what it was trained on."""
    return load()["card"]


def _self_check():
    """Run by CI. Fails loudly rather than returning something plausible-but-wrong."""
    data = load()
    actors = list_actors()
    assert len(actors) == 47, "expected 47 lead actors, found %d" % len(actors)
    assert DEBUT_ACTOR not in actors, "the sentinel leaked into the public list"

    results = [for_actor(name) for name in actors]
    for result in results:
        assert 0.0 <= result["probability"] <= 1.0, \
            "%s: probability out of range" % result["actor"]
        assert result["films_to_date"] >= 3, \
            "%s: fewer films than the scraper's floor" % result["actor"]
        assert len(history(result["actor"])) == result["films_to_date"], \
            "%s: history and film count disagree" % result["actor"]

    debut = for_debut()
    assert debut["is_debut"] and debut["films_to_date"] == 0, \
        "the debut row has picked up a history"
    assert debut["career_mean"] is None, "a newcomer cannot have a career average"

    # Shrinkage: a 4-film record must be pulled toward the middle harder than a
    # 27-film one. If this ever stops holding, the snapshot is not what we think.
    small, large = for_actor("Ranveer Singh"), for_actor("Akshay Kumar")
    assert abs(small["adjusted_mean"] - small["career_mean"]) > \
           abs(large["adjusted_mean"] - large["career_mean"]), \
        "shrinkage is not pulling small samples toward the average"

    strong = for_actor("Akshay Kumar")["probability"]
    weak = for_actor("Bobby Deol")["probability"]
    assert strong > weak, "the model ranks a 3.32 average below a 1.70 average"

    print("predict.py self-check passed")
    print("  %d actors, probabilities %.0f%% to %.0f%%, base rate %.0f%%"
          % (len(actors),
             100 * min(r["probability"] for r in results),
             100 * max(r["probability"] for r in results),
             100 * data["base_rate"]))
    for name in ("Akshay Kumar", "Ranveer Singh", "Bobby Deol"):
        r = for_actor(name)
        print("  %-16s %5.1f%%  (%d films, career mean %.2f, adjusted %.2f)"
              % (name, 100 * r["probability"], r["films_to_date"],
                 r["career_mean"], r["adjusted_mean"]))
    r = for_debut()
    print("  %-16s %5.1f%%  (no record - priced off the era average %.2f)"
          % ("newcomer", 100 * r["probability"], r["era_mean"]))


if __name__ == "__main__":
    _self_check()
