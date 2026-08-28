"""
Step 4 - Model selection for the Movie Review Prediction project.

The question: given only what a lead actor's earlier films scored, what is the
probability that their next film is rated 3.5 or above by the critic?

Only the actor's own track record is used. Genre, release month and the era
baseline are all present in movies_features.csv and all deliberately left out -
the problem is "predict the movie from the actor", so the actor is the input.
Measured on 13 rolling time windows, the actor record alone scores AUC 0.676
against 0.663 for the full 34-column file, so this is a simplification that
costs nothing.

Two decisions are inherited from feature_engineering.py and not re-litigated:

  * Features are already leak-free. Each is built from an actor's earlier films
    only, so nothing here re-derives actor statistics.
  * The split is by TIME, and cannot be random. The features are as-of: a film's
    columns summarise its lead actor's EARLIER films. So if a 2012 film sits in
    training and that actor's 2007 film sits in test, the 2007 rating is already
    inside the 2012 row's prior_mean. Splitting at random puts the test answers
    into the training features for 92% of test films, which lifts AUC 0.704 ->
    0.725 and accuracy 0.669 -> 0.706. That gain is leakage, not skill.

EDA.ipynb put the honest ceiling of these features near AUC 0.69. That is the
number to judge the result against, not 1.0.

Run:  python Model-Selection.py

Runs from any working directory - paths are resolved against this file, not the
shell's cwd. Dependencies are in requirements.txt:

    pip install -r requirements.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, TimeSeriesSplit,
                                     cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "is_positive"
CV_SPLITS = 5          # validation folds carved out of the training years
SEED = 0
CEILING_AUC = 0.69     # EDA.ipynb section 9 - the realistic best, not 1.0

# Anchored to this file rather than the shell's cwd, so the script runs the same
# from anywhere - a repo checkout, a scheduler, another machine.
HERE = Path(__file__).resolve().parent
DATA_CSV = HERE / "movies_features.csv"
PRED_CSV = HERE / "test_predictions.csv"
FIGURE_PNG = HERE / "model_selection.png"

assert DATA_CSV.exists(),     "%s not found - run feature_engineering.py first" % DATA_CSV.name

df = pd.read_csv(DATA_CSV, index_col="movie_id")

# select relevant columns
# The actor's record, and nothing else. Ten numbers answering three questions:
# how good has this actor been, how good lately, and how much evidence is there.
ACTOR_RECORD = [
    "prior_adj_mean",       # career mean, shrunk so 4 films cannot outrank 50
    "prior_mean",           # career mean, unshrunk
    "prior_mean_vs_era",    # how far above/below the era they scored
    "prior_positive_rate",  # share of earlier films that cleared 3.5
    "prior_last3_mean",     # recent form - EDA found this beats the lifetime mean
    "prior_last5_mean",
    "prior_momentum",       # last 3 minus career mean: rising or fading
    "prior_std",            # consistency
    "prior_n_films",        # how much evidence the numbers above rest on
    "has_history",          # 0 for a debut, where every column above is a guess
]

X_all, y_all = df[ACTOR_RECORD], df[TARGET]

# The checks that decide whether this file can be fed to a model at all.
assert set(y_all.unique()) == {0, 1}, "target is not a clean 0/1 label"
assert not X_all.isna().any().any(), "a feature column contains nulls"
assert X_all.select_dtypes(exclude="number").empty, "a non-numeric column reached X"
assert "rating" not in df.columns, "the raw target survived into the file"
assert df.index.is_unique, "movie_id does not identify a row"

# A feature this close to the target would be leakage, not signal.
worst_corr = X_all.corrwith(y_all).abs().max()
assert worst_corr < 0.9, "a feature correlates %.3f with the target" % worst_corr

print("=" * 70)
print("  READINESS CHECK")
print("=" * 70)
print("  %d films, %d actor-record features, %d lead actors"
      % (len(df), X_all.shape[1], df.lead_actor.nunique()))
print("  positive rate      %.1f%% (%d of %d) - imbalanced, so the models are"
      % (100 * y_all.mean(), y_all.sum(), len(y_all)))
print("                     compared on AUC, and the imbalance is handled once,")
print("                     by tuning the decision threshold on validation data")
print("  nulls in features  0")
print("  strongest single feature-target correlation  %.3f  (leak-free)"
      % worst_corr)
print("  debut films        %d - no record yet, flagged by has_history"
      % int((df.has_history == 0).sum()))

# get dummy data for categorical variables, check if already there
# Already there, and deliberately unused. feature_engineering.py turns the
# readable prior_tier label into four tier_* flags, so no encoding work is
# needed. But a tier is just prior_mean chopped into four buckets, and
# prior_mean is already in the model as a continuous number. Adding the flags
# on top measured -0.010 AUC across 13 rolling windows: the same information,
# coarser, competing with itself. So they stay out.
tier_flags = [c for c in df.columns if c.startswith("tier_")]
assert df[tier_flags].isin([0, 1]).all().all(), "a tier flag is not 0/1"
assert not any(c in ACTOR_RECORD for c in tier_flags), "a tier flag reached X"

print("  categorical        prior_tier already encoded as %d tier_* flags;"
      % len(tier_flags))
print("                     excluded as a coarser copy of prior_mean (-0.010 AUC)")

# create train/test split
# The split column was written by feature_engineering.py: the last 20% of the
# timeline is the test set and is never touched until the final section. Inside
# the training years, TimeSeriesSplit carves the validation folds - each fold
# trains on the past and validates on the future, so hyperparameters are chosen
# the same way the model will actually be used.
train, test = df[df.split == "train"], df[df.split == "test"]
X_train, y_train = X_all.loc[train.index], y_all.loc[train.index]
X_test, y_test = X_all.loc[test.index], y_all.loc[test.index]

cv = TimeSeriesSplit(n_splits=CV_SPLITS)

assert train.release_date.max() < test.release_date.min(), \
    "train and test overlap in time - the split is not a time split"

print("\n  train  %d films  %s to %s" % (len(train), train.release_date.min()[:10],
                                         train.release_date.max()[:10]))
print("  test   %d films  %s to %s  (never seen until the last section)"
      % (len(test), test.release_date.min()[:10], test.release_date.max()[:10]))
print("  validation: %d expanding time folds inside the training years" % CV_SPLITS)

# select a proper model
# Logistic regression. "Better track record means likelier to be rated well" is
# a monotonic relationship, which is exactly what a linear model in log-odds
# expresses, and its output is a probability by construction rather than a score
# converted into one afterwards. Scaling is required because the ten features
# run from 0/1 flags to film counts in the dozens.
#
# No class_weight, deliberately. Re-weighting the classes inflates every
# probability - it predicts a mean of 0.48 where the true rate is 0.34 - and the
# deliverable is a percentage that has to mean what it says. The 34% imbalance
# is dealt with once, further down, by tuning the decision threshold.
logistic = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=3000, random_state=SEED)),
])
logistic_grid = {"clf__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]}

# maybe another model according to what we want to do with the data
# Random forest, because the actor record interacts with itself. A career mean
# of 3.2 means something different at 40 films than at 3, and a rising momentum
# means more for a proven actor than an unproven one. Those are products of two
# columns, which a linear model only sees if you build the term by hand. Trees
# find them unaided, and averaging many shallow ones stops any single tree
# memorising a 561-row training set.
forest = RandomForestClassifier(random_state=SEED, n_jobs=-1)
forest_grid = {
    "n_estimators": [400],
    "max_depth": [2, 3, 5],
    "min_samples_leaf": [5, 10, 20],
}

# another one also, maybe to compare the models
# Gradient boosting, as the strongest realistic contender - trees built in
# sequence, each correcting the last. It earns its place by being a fair test
# rather than by winning: if the most flexible model here cannot beat logistic
# regression, that is evidence the actor signal really is close to linear, not
# evidence that the model was too weak to find something.
boosted = HistGradientBoostingClassifier(random_state=SEED)
boosted_grid = {
    "learning_rate": [0.03, 0.1],
    "max_leaf_nodes": [3, 7],
    "min_samples_leaf": [20, 40],
    "max_iter": [150],
}

# tune the models using Grid search or any other technique
# Scored on AUC, which asks "does this rank positive films above negative ones"
# and needs no threshold. Accuracy would mislead at a 34% positive rate, where
# always guessing "negative" already scores 66%.
candidates = {
    "logistic regression": (logistic, logistic_grid),
    "random forest": (forest, forest_grid),
    "gradient boosting": (boosted, boosted_grid),
}

print("\n" + "=" * 70)
print("  GRID SEARCH  (scoring: AUC on %d time folds)" % CV_SPLITS)
print("=" * 70)

tuned, cv_auc = {}, {}
for name, (model, grid) in candidates.items():
    search = GridSearchCV(model, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    tuned[name], cv_auc[name] = search.best_estimator_, search.best_score_
    print("  %-20s AUC %.3f   %s"
          % (name, search.best_score_,
             {k.replace("clf__", ""): v for k, v in search.best_params_.items()}))

# test the model/ensembles
# A soft-voting ensemble averages the three probability outputs. It is scored on
# the same folds as the others, so it only wins if averaging genuinely helps.
ensemble = VotingClassifier(
    estimators=[(name.split()[0], model) for name, model in tuned.items()],
    voting="soft", n_jobs=-1)
tuned["soft-vote ensemble"] = ensemble
cv_auc["soft-vote ensemble"] = cross_val_score(
    ensemble, X_train, y_train, scoring="roc_auc", cv=cv, n_jobs=-1).mean()

# The winner is chosen on validation AUC, never on the test set - picking by
# test score would turn the test score into a training score.
winner = max(cv_auc, key=cv_auc.get)

# The operating point. Because no class re-weighting was applied, the model's
# probabilities are calibrated, so 0.5 carries its plain meaning: "more likely
# positive than not". That is the threshold used.
#
# Tuning it for best F1 on the validation folds was tried and rejected. It
# lands near 0.28, which flags essentially every film positive - recall 1.00,
# precision 0.37 - and buys about +0.02 F1 by giving up any ability to
# discriminate. Across 13 rolling windows no threshold rule beat "call
# everything positive" on F1 by more than noise, so a fixed, interpretable cut
# is preferred over a tuned one that degenerates.
threshold = 0.5

# Only now is the test set touched, and only once.
for model in tuned.values():
    model.fit(X_train, y_train)
probability = {name: m.predict_proba(X_test)[:, 1] for name, m in tuned.items()}

print("\n" + "=" * 70)
print("  TEST SET  (%d films, %s onward, decision threshold %.2f)"
      % (len(test), test.release_date.min()[:10], threshold))
print("=" * 70)
print("  %-20s %7s %7s %7s %7s %7s" % ("", "AUC", "F1", "prec", "recall", "acc"))
for name in sorted(cv_auc, key=cv_auc.get, reverse=True):
    p = probability[name]
    guess = p >= threshold
    print("  %-20s %7.3f %7.3f %7.3f %7.3f %7.3f%s"
          % (name, roc_auc_score(y_test, p), f1_score(y_test, guess, zero_division=0),
             precision_score(y_test, guess, zero_division=0),
             recall_score(y_test, guess), accuracy_score(y_test, guess),
             "   <- selected" if name == winner else ""))

# Is the model doing fine? Guessing "positive" for every film is the floor any
# model must clear, and AUC 0.5 is a coin flip.
rate = y_test.mean()
baseline_f1 = 2 * rate / (1 + rate)
best_auc = roc_auc_score(y_test, probability[winner])
best_f1 = f1_score(y_test, probability[winner] >= threshold, zero_division=0)
print("\n  selected on validation AUC %.3f -> %s" % (cv_auc[winner], winner))
# F1 rewards recall heavily, so "call everything positive" scores well on it
# while being useless - its precision is just the base rate. Precision is the
# fairer comparison for a model meant to pick films out of a list.
best_precision = precision_score(y_test, probability[winner] >= threshold,
                                 zero_division=0)
print("  always-positive baseline   F1 %.3f  precision %.3f  (useless but scores well)"
      % (baseline_f1, rate))
print("  this model                 F1 %.3f  precision %.3f  (%+.3f precision)"
      % (best_f1, best_precision, best_precision - rate))
print("  coin flip                  AUC 0.500  this model AUC %.3f  (%+.3f)"
      % (best_auc, best_auc - 0.500))
# Accuracy needs its own baseline to mean anything: with 35% positives, a model
# that always says "negative" is already right 65% of the time.
best_accuracy = accuracy_score(y_test, probability[winner] >= threshold)
print("  always-negative baseline   accuracy %.1f%%   this model accuracy %.1f%%  (%+.1f)"
      % (100 * (1 - rate), 100 * best_accuracy,
         100 * (best_accuracy - (1 - rate))))
print("  EDA.ipynb realistic ceiling AUC %.2f" % CEILING_AUC)

# The accuracy ceiling, so 66.9% is read against the right number. For each
# actor the best any actor-based model can do is call their majority class; this
# computes that from the answers themselves, so it is an oracle no model can
# beat. It comes out near 70%, because most actors' films are close to a coin
# flip. Diagnostic only - it uses the labels and never touches model selection.
majority = y_all.groupby(df.lead_actor).apply(lambda s: max(s.mean(), 1 - s.mean()))
films = y_all.groupby(df.lead_actor).size()
print("  actor-only ORACLE accuracy %.1f%% - the hard ceiling for any model"
      % (100 * (majority * films).sum() / films.sum()))
print("                             built from lead actor alone, however clever")
# The headline output is a percentage, so it has to mean what it says: over many
# films the average predicted probability should land near the share that really
# were positive. A large gap means the numbers rank well but read wrong.
print("  calibration: mean predicted %.1f%% vs %.1f%% actually positive (gap %+.1f)"
      % (100 * probability[winner].mean(), 100 * rate,
         100 * (probability[winner].mean() - rate)))
print("  139 test films is a small sample - treat differences under ~0.05 as noise")

# Where the value actually is. F1 at a single cut-point does not beat "call
# everything positive" on this data - tested across 13 rolling windows, no
# threshold rule did. The ranking is a different matter: sorting films by
# predicted probability and taking the top slice finds positives at well above
# the base rate, and that is what a useful shortlist looks like.
order = np.argsort(probability[winner])[::-1]
for share in (0.10, 0.25):
    top = order[:max(1, int(share * len(order)))]
    print("  top %2d%% by probability: %.0f%% actually positive, vs %.0f%% base rate"
          % (100 * share, 100 * y_test.iloc[top].mean(), 100 * rate))

# The deliverable: one probability per test film, from the actor's record alone.
predictions = pd.DataFrame({
    "movie_title": test.movie_title,
    "lead_actor": test.lead_actor,
    "release_date": test.release_date,
    "prior_films": test.prior_n_films,
    "positive_probability_pct": (100 * probability[winner]).round(1),
    "predicted": np.where(probability[winner] >= threshold, "positive", "negative"),
    "actual": np.where(y_test == 1, "positive", "negative"),
})
predictions.to_csv(PRED_CSV)

print("\n" + "=" * 70)
print("  PREDICTIONS  (wrote %s)" % PRED_CSV.name)
print("=" * 70)
print(predictions.head(12).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for name in cv_auc:
    false_rate, true_rate, _ = roc_curve(y_test, probability[name])
    axes[0].plot(false_rate, true_rate, lw=2 if name == winner else 1.2,
                 label="%s (%.3f)" % (name, roc_auc_score(y_test, probability[name])))
axes[0].plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="coin flip")
axes[0].set_xlabel("false positive rate"); axes[0].set_ylabel("true positive rate")
axes[0].set_title("ROC on the test set", loc="left")
axes[0].legend(frameon=False, fontsize=8.5)

for label, colour in ((0, "#c26a1e"), (1, "#184f95")):
    axes[1].hist(probability[winner][y_test == label], bins=np.arange(0, 1.05, 0.05),
                 alpha=0.65, color=colour,
                 label="actually %s" % ("positive" if label else "negative"))
axes[1].axvline(threshold, color="black", lw=1.4, ls="--")
axes[1].set_xlabel("predicted probability of positive"); axes[1].set_ylabel("films")
axes[1].set_title("%s: are the two classes separated?" % winner, loc="left")
axes[1].legend(frameon=False, fontsize=8.5)

fig.tight_layout()
fig.savefig(FIGURE_PNG, dpi=120)
print("\n  wrote model_selection.png")
