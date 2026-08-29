"""
Step 5 - Freeze the chosen model for production.

Model-Selection.py compares four candidates and argues for one. This script does
not re-open that argument. It takes the winner - the random forest - re-derives
its hyperparameters the same way (grid search on the training years, scored on
time-ordered folds), retrains it on every film available, and writes two files:

  model.pkl        the trained model, ready to load in milliseconds
  model_card.json  what this model is, what it scored, and what it cannot do

Why freeze it at all: Model-Selection.py spends a minute grid-searching on every
run. A web page cannot do that per visitor. So the work happens once, here, and
the app only ever loads the result.

Why retrain on all 700 rows rather than the 561 training rows: the measurement is
already done and recorded. Once a model has been chosen honestly, you give the
shipped copy every row you have. The scores in the model card come from the
139-film test set and describe the METHOD, not this exact artifact - which is why
the card says so in as many words.

Run:  python train_model.py

Runs from any working directory - paths resolve against this file, not the shell.
"""

import json
import platform
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

TARGET = "is_positive"
SEED = 0                 # same seed as Model-Selection.py, so the fit is identical
CV_SPLITS = 5
THRESHOLD = 0.5          # Model-Selection.py argues for a fixed, interpretable cut

HERE = Path(__file__).resolve().parent
DATA_CSV = HERE / "movies_features.csv"
SNAPSHOT_CSV = HERE / "actor_snapshot.csv"
MODEL_PKL = HERE / "model.pkl"
CARD_JSON = HERE / "model_card.json"

# The ten columns, in this order. Order matters: a fitted sklearn model is a
# positional thing, so predict.py must hand the columns over exactly like this.
# Kept in step with ACTOR_RECORD in Model-Selection.py, and asserted below.
ACTOR_RECORD = [
    "prior_adj_mean",
    "prior_mean",
    "prior_mean_vs_era",
    "prior_positive_rate",
    "prior_last3_mean",
    "prior_last5_mean",
    "prior_momentum",
    "prior_std",
    "prior_n_films",
    "has_history",
]

# The winner, and the grid it won on. Re-searched here rather than hardcoded, so a
# change to the data cannot leave a stale hyperparameter baked into the artifact.
GRID = {
    "n_estimators": [400],
    "max_depth": [2, 3, 5],
    "min_samples_leaf": [5, 10, 20],
}

# What Model-Selection.py measured on the held-out test set. Repeated in the card
# so the number a visitor sees is the number that was actually earned, and checked
# against a fresh run below so the two files cannot quietly disagree.
MEASURED = {
    "selected_on_validation_auc": 0.620,
    "test_auc": 0.704,
    "test_accuracy": 0.669,
    "test_precision": 0.552,
    "test_recall": 0.327,
    "test_f1": 0.410,
    "test_films": 139,
    "base_rate": 0.353,
    "always_negative_accuracy": 0.647,
    "actor_only_oracle_accuracy": 0.703,
    "eda_realistic_ceiling_auc": 0.69,
    "top25pct_hit_rate": 0.56,
}
VALIDATION_TOLERANCE = 0.02   # 561 rows and 5 folds: small wobble is not a fault


def main():
    if not __debug__:
        raise SystemExit("refusing to run under -O: the checks below are assertions "
                         "and -O deletes them")

    assert DATA_CSV.exists(), \
        "%s not found - run feature_engineering.py first" % DATA_CSV.name

    df = pd.read_csv(DATA_CSV, index_col="movie_id")
    X, y = df[ACTOR_RECORD], df[TARGET]

    # The same readiness checks Model-Selection.py makes, because this script can be
    # run on its own and must not trust that the other one ran first.
    assert set(y.unique()) == {0, 1}, "target is not a clean 0/1 label"
    assert not X.isna().any().any(), "a feature column contains nulls"
    assert X.select_dtypes(exclude="number").empty, "a non-numeric column reached X"
    assert "rating" not in df.columns, "the raw target survived into the file"
    assert X.corrwith(y).abs().max() < 0.9, "a feature correlates too well - leakage"

    train = df[df.split == "train"]
    assert train.release_date.max() < df[df.split == "test"].release_date.min(), \
        "train and test overlap in time - the split is not a time split"

    print("=" * 70)
    print("  TRAINING THE SHIPPING MODEL")
    print("=" * 70)
    print("  %d films, %d features, %d lead actors"
          % (len(df), len(ACTOR_RECORD), df.lead_actor.nunique()))

    # ---- 1. Re-derive the hyperparameters, exactly as they were chosen ---------
    # Searched on the TRAINING years only. Letting the test films into this step
    # would tune the model on the data it is later judged by.
    search = GridSearchCV(
        RandomForestClassifier(random_state=SEED, n_jobs=-1),
        GRID, scoring="roc_auc", cv=TimeSeriesSplit(n_splits=CV_SPLITS), n_jobs=-1)
    search.fit(X.loc[train.index], y.loc[train.index])

    drift = search.best_score_ - MEASURED["selected_on_validation_auc"]
    print("\n  grid search on %d training films -> validation AUC %.3f"
          % (len(train), search.best_score_))
    print("  best settings      %s" % search.best_params_)
    assert abs(drift) <= VALIDATION_TOLERANCE, (
        "validation AUC moved %+.3f from the %.3f Model-Selection.py recorded - the "
        "data or the features changed, so re-run Model-Selection.py and update "
        "MEASURED before shipping" % (drift, MEASURED["selected_on_validation_auc"]))

    # ---- 2. Retrain on everything, and ship that -----------------------------
    model = RandomForestClassifier(random_state=SEED, n_jobs=-1,
                                   **search.best_params_)
    model.fit(X, y)
    print("\n  retrained on all %d films (the shipped copy gets every row)" % len(df))

    # ---- 3. Save, then prove the saved copy is the same model ----------------
    joblib.dump(model, MODEL_PKL)
    reloaded = joblib.load(MODEL_PKL)
    before, after = model.predict_proba(X)[:, 1], reloaded.predict_proba(X)[:, 1]
    assert np.allclose(before, after), \
        "the model changed on the way to disk - do not ship this"
    assert list(reloaded.feature_names_in_) == ACTOR_RECORD, \
        "the saved model expects different columns than predict.py will send"
    print("  wrote %s (%.0f KB), reloaded, predictions identical"
          % (MODEL_PKL.name, MODEL_PKL.stat().st_size / 1e3))

    # ---- 4. Prove it can actually serve the app ------------------------------
    # The snapshot is what predict.py feeds the model. If the two disagree about
    # columns, better to find out here than on a live page.
    if SNAPSHOT_CSV.exists():
        snapshot = pd.read_csv(SNAPSHOT_CSV, index_col="lead_actor")
        missing = [c for c in ACTOR_RECORD if c not in snapshot.columns]
        assert not missing, "actor_snapshot.csv is missing %s" % missing
        served = reloaded.predict_proba(snapshot[ACTOR_RECORD])[:, 1]
        assert ((served >= 0) & (served <= 1)).all(), "a served probability is not a probability"
        print("  serves all %d actors in %s: %.0f%% to %.0f%%"
              % (len(snapshot), SNAPSHOT_CSV.name, 100 * served.min(), 100 * served.max()))
    else:
        print("  NOTE: %s not found - run feature_engineering.py to generate it"
              % SNAPSHOT_CSV.name)

    # ---- 5. The model card ---------------------------------------------------
    importance = dict(zip(ACTOR_RECORD, model.feature_importances_.round(4).tolist()))
    card = {
        "model": "RandomForestClassifier",
        "hyperparameters": search.best_params_,
        "trained_on": date.today().isoformat(),
        "trained_rows": int(len(df)),
        "lead_actors": int(df.lead_actor.nunique()),
        "date_range": [df.release_date.min()[:10], df.release_date.max()[:10]],
        "features": ACTOR_RECORD,
        "feature_importance": dict(sorted(importance.items(),
                                          key=lambda kv: -kv[1])),
        "decision_threshold": THRESHOLD,
        "target": "critic rating >= 3.5 out of 5",
        "scores": MEASURED,
        "scores_note": (
            "Measured by Model-Selection.py on 139 films from 2020-12-31 onward, "
            "held out by time and never used for training or tuning. The shipped "
            "model is retrained on all 700 films, so these numbers describe the "
            "method rather than this exact file."),
        "limitations": [
            "Sees one thing about a film: who is starring in it. Not the script, "
            "the director, the budget or the marketing.",
            "The hard ceiling for any lead-actor-only model on this data is 70.3% "
            "accuracy. This model reaches 66.9%, so most of the remaining error is "
            "not fixable by a better algorithm.",
            "Hindi-language films reviewed by Bollywood Hungama, 2000-2026. Says "
            "nothing about other industries, languages or critics.",
            "Actors with few films are pulled toward the average on purpose, so a "
            "small-sample star will look more ordinary than their raw record.",
            "Best used to rank a list, not to judge one film: the top 25% by "
            "predicted probability are 56% positive against a 35% base rate.",
        ],
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    CARD_JSON.write_text(json.dumps(card, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("  MODEL CARD  (wrote %s)" % CARD_JSON.name)
    print("=" * 70)
    print("  what it learned from, most useful column first:")
    for name, value in list(card["feature_importance"].items())[:5]:
        print("    %-22s %.3f" % (name, value))
    print("  test AUC %.3f | accuracy %.1f%% vs %.1f%% for always-negative"
          % (MEASURED["test_auc"], 100 * MEASURED["test_accuracy"],
             100 * MEASURED["always_negative_accuracy"]))
    print("  precision %.0f%% against a %.0f%% base rate"
          % (100 * MEASURED["test_precision"], 100 * MEASURED["base_rate"]))
    print("  ceiling for ANY actor-only model: %.1f%% accuracy"
          % (100 * MEASURED["actor_only_oracle_accuracy"]))
    print("=" * 70)
    print("\n  Ready to serve. Next:  streamlit run app.py")


if __name__ == "__main__":
    main()
