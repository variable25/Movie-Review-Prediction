"""
Step 3 - Feature engineering for the Movie Review Prediction project.

Turns movies_cleaned.csv into movies_features.csv, one row per movie. Each lead
actor's track record is summarised from their EARLIER films only, so no movie
helps predict itself. In the output, every numeric column except `is_positive`
is a feature - the modelling step needs no column list and no manifest.

    df = pd.read_csv("movies_features.csv", index_col="movie_id")
    train = df[df.split == "train"]
    X, y = train.select_dtypes("number").drop(columns=["is_positive"]), train.is_positive

Run:  python feature_engineering.py
"""

import numpy as np
import pandas as pd

IN_CSV = "movies_cleaned.csv"
OUT_CSV = "movies_features.csv"
SNAPSHOT_CSV = "actor_snapshot.csv"  # one row per actor, as of today - fed to the app
ID_COL = "movie_id"         # written as the CSV index, so it is never a column
TARGET_COL = "is_positive"  # the one numeric column that is not a feature
DEBUT_ACTOR = "__debut__"   # stands for any lead with no record; never a real name

# Readable columns kept beside the features. All TEXT, so select_dtypes("number")
# skips them and a model handed one raises instead of silently training on it.
LABEL_COLS = ["movie_title", "lead_actor", "release_date", "critic",
              "director", "prior_tier", "split"]

POSITIVE_THRESHOLD = 3.5    # a "positive" review, per data_cleaning.py:78-79
M = 15                      # shrinkage weight; EDA.ipynb settles on 15 (band 12-30)
MIN_PROVEN = 15             # films before an actor's mean counts as established
ERA_FLOOR = 20              # films needed before an as-of era mean is trusted
TEST_FRACTION = 0.20        # last 20% of the timeline is held out

SCALE_MIDPOINT = 3.0        # midpoint of the 1-5 rating scale, known a priori
FALLBACK_GAP_DAYS = 365.0   # stand-in gap before any real gap has been observed
FALLBACK_SD = 0.9           # stand-in spread before any within-actor sd exists
MIN_GENRE_ROWS = 10         # a genre token below this is folded into genre_other
TIERS = ["unproven", "promising", "proven_weak", "proven_positive"]

# Never features: `rating` is the target on its raw scale, `is_positive` is the
# target itself. Keys are carried for traceability, not for the model.
TARGET_COLS = ["rating", "is_positive"]
KEY_COLS = ["movie_title", "lead_actor", "release_date", "year",
            "critic", "director"]

# EDA.ipynb put the honest ceiling of actor features near AUC 0.69. Anything well
# above that has not got smarter, it has seen the answer.
AUC_TRIPWIRE = 0.75
AUC_DRIFT_TOLERANCE = 0.02

# Built but not modelled. `prior_tier` stays as a readable text label (the tier_*
# flags carry the same thing numerically); the rest are dropped from the file.
DIAGNOSTIC_COLS = ["prior_tier", "era_prior_n", "prior_career_years"]
DROP_FROM_OUTPUT = ["era_prior_n", "rating", "year", "prior_career_years"]

# A feature must take values in the test period that it also took in training,
# or the model is extrapolating. `year` and `prior_career_years` failed this.
MIN_RANGE_OVERLAP = 0.80


def load():
    """Read the cleaned CSV and put it in strict release order."""
    df = pd.read_csv(IN_CSV)
    df["release_date"] = pd.to_datetime(df["release_date"], format="mixed")
    assert df["release_date"].notna().all(), "a release_date failed to parse"

    # Ties on the same date break by title, so the ordering - and every feature
    # built from it - is identical run to run.
    df = df.sort_values(["release_date", "movie_title"], kind="mergesort")
    return df.reset_index(drop=True)


def build_as_of(df):
    """Actor-history features, computed from earlier films only.

    Walks the dataset in release order, reading history before writing to it, so
    a movie is scored on what was known the instant before it appeared.
    """
    hist = {}          # lead_actor -> list of (date, rating, era_residual)
    all_ratings = []   # every rating seen so far, in order
    all_gaps = []      # every actor's film-to-film gap seen so far, in days
    rows = {}

    # Grouping by date means same-day releases are scored together and only then
    # pushed into history, so two films out on one Friday cannot inform each other.
    for _, same_day in df.groupby("release_date", sort=True):
        pending = []

        # The era: a running mean of every film released before today, pulled
        # toward the scale midpoint so the first few rows cannot swing wildly.
        n_era = len(all_ratings)
        era_sum = float(np.sum(all_ratings)) if n_era else 0.0
        era_mean = (era_sum + ERA_FLOOR * SCALE_MIDPOINT) / (n_era + ERA_FLOOR)
        era_sd = float(np.std(all_ratings, ddof=1)) if n_era > 1 else 0.0
        era_pos_rate = (float(np.mean(np.asarray(all_ratings) >= POSITIVE_THRESHOLD))
                        if n_era else 0.0)
        gap_fill = float(np.median(all_gaps)) if all_gaps else FALLBACK_GAP_DAYS

        # Shrinkage guesses an ACTOR's true mean, and that actor was drawn from
        # the population of actors - so the target is one vote per actor.
        actor_means = [np.mean([h[1] for h in v]) for v in hist.values() if v]
        actor_pop_mean = float(np.mean(actor_means)) if actor_means else era_mean

        # Typical spread WITHIN a career, used when an actor has too few films to
        # measure their own. The era-wide spread would be the spread ACROSS actors.
        within = [float(np.std([h[1] for h in v], ddof=1))
                  for v in hist.values() if len(v) > 1]
        sd_fill = float(np.median(within)) if within else FALLBACK_SD

        for position, movie in zip(same_day.index, same_day.itertuples(index=False)):
            past = hist.get(movie.lead_actor, [])
            n = len(past)

            if n:
                ratings = np.array([h[1] for h in past], dtype=float)
                residuals = np.array([h[2] for h in past], dtype=float)
                prior_mean = float(ratings.mean())
                prior_std = float(ratings.std(ddof=1)) if n > 1 else sd_fill
                prior_pos_rate = float((ratings >= POSITIVE_THRESHOLD).mean())
                last3 = float(ratings[-3:].mean())
                last5 = float(ratings[-5:].mean())
                # How far above or below their era this actor scored, restated on
                # today's scale - the strongest single feature EDA.ipynb found.
                prior_vs_era = float(residuals.mean()) + era_mean
                career_years = (movie.release_date - past[0][0]).days / 365.25
                gap_days = float((movie.release_date - past[-1][0]).days)
            else:
                # A debut has no record, so the era stands in for every one of
                # these. `has_history` flags the row so a model can tell.
                prior_mean = era_mean
                prior_std = sd_fill
                prior_pos_rate = era_pos_rate
                last3 = last5 = era_mean
                prior_vs_era = era_mean
                career_years = 0.0
                gap_days = gap_fill

            # Credit every actor with M imaginary average films: few real films and
            # the estimate is dragged to average, many and their own record stands.
            prior_adj_mean = (n * prior_mean + M * actor_pop_mean) / (n + M)

            # The readable tier, judged on the career SO FAR rather than the
            # finished career, which is why a debut is correctly 'unproven'.
            if n >= MIN_PROVEN:
                tier = "proven_positive" if prior_mean > 3 else "proven_weak"
            else:
                tier = "promising" if (n and prior_mean > 3) else "unproven"

            rows[position] = {
                "prior_n_films": n,
                "has_history": int(n > 0),
                "prior_mean": prior_mean,
                "prior_adj_mean": prior_adj_mean,
                "prior_std": prior_std,
                "prior_positive_rate": prior_pos_rate,
                "prior_last3_mean": last3,
                "prior_last5_mean": last5,
                "prior_momentum": last3 - prior_mean if n else 0.0,
                "prior_mean_vs_era": prior_vs_era,
                "prior_tier": tier,
                "prior_career_years": career_years,
                "days_since_prev_film": gap_days,
                "era_prior_mean": era_mean,
                "era_prior_n": n_era,
                **{"tier_" + name: int(tier == name) for name in TIERS},
            }
            pending.append((movie, era_mean, gap_days, n))

        # Only now, with the whole day scored, do today's films become history.
        for movie, era_mean, gap_days, n in pending:
            if n:
                all_gaps.append(gap_days)
            hist.setdefault(movie.lead_actor, []).append(
                (movie.release_date, movie.rating, movie.rating - era_mean))
            all_ratings.append(movie.rating)

    return pd.DataFrame.from_dict(rows, orient="index").reindex(df.index)


def as_of_today(df, as_of=None):
    """One row per lead actor: their record as it stands for a film released today.

    The modelling rows above answer "what was known before THIS film". The live app
    needs the next question along - "what is known now, for a film nobody has made
    yet" - and that row does not exist in the dataset.

    Rather than recompute career averages here (two functions meant to agree is how
    they end up disagreeing), append one placeholder film per actor, all dated after
    every real release, and push the whole thing back through build_as_of. It groups
    by date and reads history before writing to it, so those rows are scored against
    the complete 700-film history and cannot see each other. Then read them off.

    A "__debut__" actor is included so the app can price an unknown newcomer using
    the same cold-start path a real debut takes.

    Only `days_since_prev_film` and `prior_career_years` depend on `as_of`, and
    neither is modelled - so the ten features the app actually uses are the same
    whichever day this is regenerated.
    """
    if as_of is None:
        as_of = pd.Timestamp.today().normalize()
    as_of = pd.Timestamp(as_of)
    assert as_of > df["release_date"].max(), \
        "as_of must be after every real release, or the placeholders join history early"

    actors = sorted(df["lead_actor"].unique()) + [DEBUT_ACTOR]
    placeholder = pd.DataFrame({
        "lead_actor": actors,
        "release_date": as_of,
        # Never read for these rows: build_as_of writes a film's rating into history
        # only after the whole day is scored, and nothing is scored after today.
        "rating": SCALE_MIDPOINT,
    })

    combined = pd.concat([df[["lead_actor", "release_date", "rating"]], placeholder],
                         ignore_index=True)
    snapshot = build_as_of(combined).tail(len(actors)).copy()
    snapshot.insert(0, "lead_actor", actors)

    # Films to date and career average, straight from the source, so the app can show
    # the evidence next to the prediction without re-deriving it.
    history = df.groupby("lead_actor")["rating"]
    snapshot["films_to_date"] = snapshot["lead_actor"].map(history.size()).fillna(0)
    snapshot["career_mean"] = snapshot["lead_actor"].map(history.mean())
    snapshot["as_of_date"] = as_of.strftime("%Y-%m-%d")

    debut = snapshot[snapshot.lead_actor == DEBUT_ACTOR]
    assert len(debut) == 1 and debut["prior_n_films"].iloc[0] == 0, \
        "the debut placeholder picked up a history"
    assert (snapshot[snapshot.lead_actor != DEBUT_ACTOR]["prior_n_films"]
            == df["lead_actor"].value_counts().reindex(actors[:-1]).to_numpy()).all(), \
        "an actor's snapshot does not rest on all of their films"

    return snapshot.set_index("lead_actor")


def build_context(df):
    """Release timing and genre - features that never touch `rating`."""
    tokens = df["genre"].fillna("").str.split(",")
    tokens = tokens.map(lambda parts: sorted({p.strip() for p in parts if p.strip()}))

    # Treating each comma-joined combination as a category gives 157 levels; split
    # apart there are ~22 real tokens, which multi-hot encodes cheaply.
    seen = pd.Series([t for row in tokens for t in row]).value_counts()

    # A token on a handful of films out of 700 cannot support a column of its own,
    # so the rare ones pool into a single flag rather than overfitting three rows.
    vocabulary = sorted(seen[seen >= MIN_GENRE_ROWS].index)
    rare = sorted(seen[seen < MIN_GENRE_ROWS].index)

    out = pd.DataFrame(index=df.index)
    out["release_month"] = df["release_date"].dt.month
    out["n_genres"] = tokens.map(len)
    for token in vocabulary:
        column = "genre_" + token.lower().replace(" ", "_").replace("-", "_")
        out[column] = tokens.map(lambda row, t=token: int(t in row))
    out["genre_other"] = tokens.map(lambda row, r=set(rare): int(bool(r & set(row))))
    return out, vocabulary, rare


def add_split(df):
    """Hold out the last slice of the timeline, never a random slice.

    Ratings drifted heavily over 2000-2026, so a random split would let the model
    train on 2024 and test on 2004 - something it never gets to do in production.
    """
    cutoff = df["release_date"].quantile(1 - TEST_FRACTION)
    df["split"] = np.where(df["release_date"] <= cutoff, "train", "test")
    return cutoff


def auc(scores, labels):
    """Rank-based AUC. No sklearn in this .venv, and ranks handle ties properly."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    n_pos, n_neg = labels.sum(), len(labels) - labels.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank().to_numpy()
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def verify(out, source, feature_cols):
    """Fail loudly on anything that would poison the modelling step."""
    # Nothing lost or duplicated on the way through.
    assert len(out) == len(source), "row count changed"
    assert not out.duplicated(subset=["movie_title", "year"]).any(), \
        "(movie_title, year) is not unique"
    assert (out.groupby("lead_actor").size().sort_index()
            .equals(source.groupby("lead_actor").size().sort_index())), \
        "an actor gained or lost movies"

    # The target must not be reachable from the features. `sentiment` is exactly
    # (rating >= 3.5), so it is the target restated and must not survive.
    for column in TARGET_COLS:
        assert column not in feature_cols, "%s is exposed as a feature" % column
    assert "sentiment" not in out.columns, "sentiment must not survive"

    # Every feature must be numeric - a text column would skip the correlation
    # check, the AUC tripwire and the range guard below, which all read numbers.
    numeric = out[feature_cols].select_dtypes("number")
    assert list(numeric.columns) == list(feature_cols), (
        "not model-ready: %s are not numeric"
        % sorted(set(feature_cols) - set(numeric.columns)))
    worst = numeric.corrwith(out["rating"]).abs().max()
    assert worst < 0.99, "a feature is a near-perfect copy of rating (r=%.4f)" % worst
    assert out[feature_cols].notna().all().all(), "null in a feature column"

    # Causality, checked the slow obvious way: recompute prior_mean from earlier
    # films directly. If a movie ever informed its own feature, these disagree.
    sample = out.sample(n=min(40, len(out)), random_state=0)
    for movie in sample.itertuples(index=False):
        earlier = out[(out.lead_actor == movie.lead_actor)
                      & (out.release_date < movie.release_date)]
        assert len(earlier) == movie.prior_n_films, \
            "%s: history size disagrees with prior_n_films" % movie.movie_title
        if movie.prior_n_films:
            assert abs(earlier.rating.mean() - movie.prior_mean) < 1e-9, \
                "%s: prior_mean used movies it should not see" % movie.movie_title

    # Cold starts are flagged, not hidden. A debut is every film on an actor's
    # earliest release date - a few actors open with a same-day pair.
    debuts = out.prior_n_films == 0
    first_release = out.groupby("lead_actor")["release_date"].transform("min")
    assert (debuts == (out.release_date == first_release)).all(), (
        "a debut is not on the actor's first release date, or vice versa")
    assert set(out.loc[debuts, "lead_actor"]) == set(out.lead_actor), (
        "an actor has no debut row")
    assert (out.loc[debuts, "has_history"] == 0).all(), "a debut claims history"
    assert (out.loc[~debuts, "has_history"] == 1).all(), "a non-debut denies history"

    # The leakage tripwire: a single feature scoring far above the honest ceiling
    # has not got smarter, it has seen the answer.
    scored = {}
    for column in numeric.columns:
        value = auc(numeric[column], out["is_positive"])
        scored[column] = max(value, 1 - value)      # direction-agnostic
    hottest, hot_value = max(scored.items(), key=lambda kv: kv[1])
    assert hot_value < AUC_TRIPWIRE, \
        "%s scores AUC %.3f, above the %.2f tripwire - that is a leak, not a win" \
        % (hottest, hot_value, AUC_TRIPWIRE)

    assert out.loc[out.split == "train", "release_date"].max() \
        <= out.loc[out.split == "test", "release_date"].min(), \
        "train and test overlap in time"

    # No feature may drift out of the range it was learned on. A feature that
    # climbs with time scores well in cross-validation and pays nothing at test.
    train, test = out[out.split == "train"], out[out.split == "test"]
    for column in numeric.columns:
        if out[column].nunique() <= 2:
            continue          # a 0/1 flag has no range it can drift out of
        low, high = train[column].min(), train[column].max()
        inside = float(test[column].between(low, high).mean())
        assert inside >= MIN_RANGE_OVERLAP, (
            "%s: only %.1f%% of test rows fall inside its training range "
            "[%.4g, %.4g] - it drifts with time, so it will look strong in "
            "cross-validation and pay nothing back on test; add it to "
            "DIAGNOSTIC_COLS and DROP_FROM_OUTPUT"
            % (column, 100 * inside, low, high))
    return scored


def verify_output(out, feature_cols):
    """Prove the written file IS the contract, not merely described by it.

    The modelling step relies on "every numeric column except is_positive is a
    feature", so that is asserted here about the exact frame being written.
    """
    assert out.index.name == ID_COL, "not indexed by " + ID_COL
    assert out.index.is_unique, ID_COL + " is not unique"

    numeric = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    assert set(numeric) == set(feature_cols) | {TARGET_COL}, (
        "the one rule is broken - these numeric columns are not features: %s"
        % sorted(set(numeric) - set(feature_cols) - {TARGET_COL}))

    # Everything readable must stay text, so select_dtypes("number") skips it.
    for column in LABEL_COLS:
        assert column in out.columns, "%s went missing" % column
        assert not pd.api.types.is_numeric_dtype(out[column]), (
            "%s is numeric, so a model would pick it up as a feature" % column)

    for column in DROP_FROM_OUTPUT:
        assert column not in out.columns, (
            "%s must not be written - it is numeric and not a feature" % column)

    assert out[feature_cols + [TARGET_COL]].notna().all().all(), (
        "null in a modelled column")


# Figures EDA.ipynb measured for the same features, so any drift in this script
# shows up next to the numbers it is meant to reproduce.
PUBLISHED_AUC = {
    "prior_mean": 0.633,
    "prior_adj_mean": 0.628,
    "prior_mean_vs_era": 0.688,
    "prior_last5_mean": 0.669,
    "prior_last3_mean": 0.668,
}


def main():
    # Every guarantee above is an assert, and `python -O` deletes asserts. This is
    # a real branch so the script refuses to run rather than skip its own checks.
    if not __debug__:
        raise SystemExit("refusing to run under -O: the leak checks in "
                         "verify() are assertions and -O deletes them")

    source = load()

    # `sentiment` is exactly (rating >= 3.5), so it adds nothing to the target and
    # is not written out. The assert holds that identity to account.
    target = pd.DataFrame(index=source.index)
    target["rating"] = source["rating"]
    target["is_positive"] = (source["rating"] >= POSITIVE_THRESHOLD).astype(int)
    assert (target["is_positive"]
            == (source["sentiment"] == "positive").astype(int)).all(), \
        "sentiment no longer matches the 3.5 threshold"

    as_of = build_as_of(source)
    context, vocabulary, rare_genres = build_context(source)

    out = pd.concat([source[KEY_COLS], target, as_of, context], axis=1)
    cutoff = add_split(out)

    # `critic` is not a feature - it is only known once the review exists, and it
    # is confounded with the era anyway. `year` is dropped per MIN_RANGE_OVERLAP.
    feature_cols = [c for c in list(as_of.columns) + list(context.columns)
                    if c not in DIAGNOSTIC_COLS]
    scored = verify(out, source, feature_cols)

    # Labels first so the file is scannable by eye, then the target, then the
    # features. release_date is written as text so it stays a label, not a number.
    out.index.name = ID_COL
    out["release_date"] = out["release_date"].dt.strftime("%Y-%m-%d")

    out = out[LABEL_COLS + [TARGET_COL] + feature_cols]
    verify_output(out, feature_cols)
    out.to_csv(OUT_CSV, index=True, encoding="utf-8")

    # The serving side of the same features. Written here rather than in the app so
    # there is exactly one piece of code that knows how a prior is computed.
    # Actor-history columns only. The genre and release-month features are properties
    # of a specific film, not of an actor, so a "next film" row cannot carry them -
    # and Model-Selection.py leaves them out of the model anyway.
    snapshot = as_of_today(source)
    missing = [c for c in as_of.columns if c not in snapshot.columns]
    assert not missing, "the snapshot cannot serve these actor features: %s" % missing
    snapshot.to_csv(SNAPSHOT_CSV, encoding="utf-8")

    # ---------------------------------------------------------------- report
    n_train = (out.split == "train").sum()
    print("Wrote %s: %d rows x %d columns"
          % (OUT_CSV, len(out), out.shape[1]))
    print("  %d numeric features + %s, plus %d text labels the model skips"
          % (len(feature_cols), TARGET_COL, len(LABEL_COLS)))
    print("  X = df.select_dtypes(\"number\").drop(columns=[\"%s\"])"
          % TARGET_COL)
    print("  genres multi-hot   %d tokens: %s"
          % (len(vocabulary), ", ".join(vocabulary[:8]) + ", ..."))
    print("  genre_other        %d rare tokens pooled (<%d movies each): %s"
          % (len(rare_genres), MIN_GENRE_ROWS, ", ".join(rare_genres)))
    print("  cold-start rows    %d of %d across %d actors (flagged, not dropped)"
          % ((out.prior_n_films == 0).sum(), len(out), out.lead_actor.nunique()))
    print("  time split         train %d (to %s) | test %d (from %s)"
          % (n_train, cutoff.date(), len(out) - n_train,
             out.loc[out.split == "test", "release_date"].min()))
    print("  %-18s %d actors + %s, as of %s (same build_as_of, no second copy)"
          % (SNAPSHOT_CSV, len(snapshot) - 1, DEBUT_ACTOR,
             snapshot["as_of_date"].iloc[0]))

    print("\n" + "=" * 62)
    print("  Single-feature AUC vs is_positive, against EDA.ipynb")
    print("=" * 62)
    # EDA.ipynb measured on a 531-row subset and this runs on all 700, so a small
    # gap is expected; a large one means the recomputation changed the feature.
    for column, value in sorted(scored.items(), key=lambda kv: -kv[1])[:10]:
        published = PUBLISHED_AUC.get(column)
        note = ""
        if published is not None:
            drift = value - published
            flag = ("  << DRIFT %+.3f" % drift
                    if abs(drift) > AUC_DRIFT_TOLERANCE else "")
            note = "  (EDA: %.3f%s)" % (published, flag)
        print("  %-22s %.3f%s" % (column, value, note))
    print("  tripwire at %.2f - anything above it is a leak, not a result"
          % AUC_TRIPWIRE)

    positive_rate = out.is_positive.mean()
    print("\n" + "=" * 62)
    print("  Carry into modelling")
    print("=" * 62)
    print("  - Baseline to beat: predict positive for everything -> F1 %.3f"
          % (2 * positive_rate / (1 + positive_rate)))
    print("    (%.1f%% of all 700 rows are positive)" % (100 * positive_rate))
    print("  - Split and cross-validate by TIME, never at random")
    print("  - Tune M (currently %d) inside the 12-30 band" % M)
    print("  - rating is not normal within an actor: no z-scores,")
    print("    prefer rank-based or ordinal treatments")
    print("  - prior_std is unreliable where an actor's ratings are lopsided;")
    print("    below 2 films it is a stand-in, so gate it on prior_n_films")
    print("  - Every numeric column except %s is a feature - no list needed"
          % TARGET_COL)
    print("=" * 62)


if __name__ == "__main__":
    main()
