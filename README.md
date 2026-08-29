# 🎬 Movie Review Predictor

**Given a Bollywood lead actor's track record, what are the odds their next film gets a good review?**

> **Live app:** _(add the Streamlit link here once deployed)_

Pick an actor, get a percentage. The model sees exactly one thing about a film —
who is starring in it — and nothing else. No script, no director, no budget, no
marketing.

That constraint is the point of the project. The interesting question is not "can
this predict films" but **"how far does a single signal actually get you, and how
would you know when you have hit the wall?"**

---

## The honest numbers

Measured on **139 films from 2020 onward** that the model never trained on, held
out by date rather than at random.

| | This model | The baseline it has to beat |
|---|---|---|
| **AUC** (ranking ability) | **0.704** | 0.500 — a coin flip |
| **Accuracy** | **66.9%** | 64.7% — always guess "negative" |
| **Precision** | **55.2%** | 35.3% — the base rate |
| **Top 25% hit rate** | **56% positive** | 35% positive |

Accuracy barely moves, and that is not hidden here — with only 35% of films rated
positive, "always say no" is already right two thirds of the time. **The gain is in
precision.** When this model says positive it is right 55% of the time against a 35%
base rate, and sorting a slate of films by predicted probability puts genuinely
better films at the top. It is a shortlist tool, not an oracle.

### The ceiling — the number that matters most

**70.3%.** That is the best accuracy *any* lead-actor-only model could ever reach on
this data, however clever. It is computed from the answers themselves: for each
actor, take the best possible call you could make about their films, and add it up.

This model reaches 66.9% against that 70.3% ceiling. The remaining error is mostly
not an algorithm problem — it is that most actors' films are close to a coin flip,
and the missing information is in the script, not in the cast list.

Knowing where the ceiling is turned out to be more valuable than any modelling trick.

---

## How it works

```
scrape_movies.py       1  ~700 Hindi films from Bollywood Hungama, two passes,
                          resumable, 60-minute hard budget
data_cleaning.py       2  ASCII-only titles, drop the review text (97% of the file)
feature_engineering.py 3  build the features — and the actor snapshot the app serves
Model-Selection.py     4  compare four models honestly, pick one
train_model.py         5  freeze the winner to model.pkl + model_card.json
predict.py             6  actor name in, probability + evidence out
app.py                 7  the page people actually see
```

Every step ends by checking its own work with `assert`s. A step that cannot prove
its output is sane refuses to write it.

### The two decisions that shaped everything

**1. No feature may see the future.** Every number describing an actor is built from
their *earlier* films only. The pipeline walks the dataset in release order, reading
history before writing to it — so two films released on the same Friday cannot inform
each other, and no film helps predict itself.

**2. Split by time, never at random.** The test set is the most recent 20% of the
timeline. A random split would let the model train on 2024 and test on 2004, and
because the features summarise an actor's earlier films, that leaks the test answers
straight into the training data. It lifts AUC from 0.704 to 0.725 — a gain that is
pure leakage, and rejected on purpose.

There is a tripwire in `feature_engineering.py` that fails the build if any single
feature scores above AUC 0.75, on the reasoning that nothing here is that good and
anything that looks it has seen the answer.

### Serving without drift

The classic way to break a deployed model is to compute a feature slightly
differently in the live app than in training. This project avoids it structurally:
the app's numbers are produced by **the same `build_as_of()` function** that produced
the training data. `as_of_today()` appends one placeholder film per actor, dated after
every real release, and pushes it through that function untouched — so a "next film"
row is scored exactly as a real one would be. There is no second copy of the logic to
drift.

---

## Running it yourself

No virtual environment — this installs into your global Python on purpose, so anyone
who clones it is one command away from running it.

```bash
pip install -r requirements.txt
```

Then, to go from raw data to a live page:

```bash
python data_cleaning.py        # step 2
python feature_engineering.py  # step 3  — writes the features and the actor snapshot
python Model-Selection.py      # step 4  — optional: the model comparison and figure
python train_model.py          # step 5  — writes model.pkl and model_card.json
streamlit run app.py           # step 7  — opens the page in your browser
```

Step 1 (`scrape_movies.py`) only needs running if you want to rebuild the dataset
from scratch — it takes about 40 minutes and the result is already committed. It is
resumable: press Ctrl-C and re-run to continue.

Requires Python 3.11 or newer.

---

## What's in the repo

| File | |
|---|---|
| `EDA.ipynb` | The exploratory analysis — per-actor records, shrinkage, whether ratings are normal within an actor, and the era drift hiding inside actor averages. This is where the ceiling came from. |
| `indian_movies_reviews.csv` | The raw scrape, left untouched so cleaning rules can change without re-scraping |
| `movies_cleaned.csv` | 700 films after cleaning |
| `movies_features.csv` | 700 films × 42 columns, leak-free |
| `actor_snapshot.csv` | One row per actor: their record as it stands today. What the app serves. |
| `model.pkl` | The frozen model, committed so the live app runs exactly what was tested |
| `model_card.json` | What the model is, what it scored, and what it cannot do |
| `test_predictions.csv` | Every held-out film with its predicted probability and what actually happened |
| `model_selection.png` | ROC curves and the probability split between the two classes |

---

## Deployment

Hosted on **Streamlit Community Cloud**, which redeploys automatically on every push
to `main`.

A GitHub Action (`.github/workflows/checks.yml`) runs on every push: it rebuilds the
features from scratch, confirms they reproduce the committed file exactly, retrains
the model, checks the saved copy predicts identically after reloading, and loads the
page headlessly on every branch of its logic. All of it is the project's own existing
`assert`s — the Action just means they run whether or not anyone remembers to.

It deliberately skips the scraper. No automated job should hammer someone else's
website.

---

## What this deliberately doesn't do

- **Streaming or online learning.** Both need a continuous flow of new data. A
  handful of films arrive per month. Wrong tool.
- **Docker.** Genuinely standard, genuinely unnecessary when the host installs from
  `requirements.txt` in a few seconds.
- **Scheduled retraining and drift monitoring.** The right next step — re-scrape
  monthly, retrain, and track whether accuracy is sliding as tastes and critics
  change. Its own project, not a footnote to this one.

---

## Data and licence

Film data scraped from [Bollywood Hungama](https://www.bollywoodhungama.com) —
public pages, `robots.txt` permits crawling and sets no crawl delay. Critic ratings
and review text remain their copyright; **no review text is redistributed here**, only
the numeric rating and the URL it came from. Personal and educational use.
