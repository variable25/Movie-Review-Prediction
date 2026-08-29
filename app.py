"""
Step 7 - The web page.

Everything here is presentation. The thinking lives in predict.py, which this file
imports and does not second-guess. Keeping the split means the model can be tested,
called from a notebook, or put behind an API later without touching a line of this.

Run locally:  streamlit run app.py
Deployed at:  Streamlit Community Cloud, redeployed on every push to main.
"""

import altair as alt
import pandas as pd
import streamlit as st

import predict

st.set_page_config(page_title="Will the critics like it?",
                   page_icon="🎬", layout="centered")

REPO = "https://github.com/variable25/Movie-Review-Prediction"

# Loading the model and the CSVs takes a moment. Doing it per click would make the
# page feel broken, so Streamlit is told to do it once and keep the result.
@st.cache_resource
def _loaded():
    predict.load()
    return predict.card()


def bar(label, value, low, high, helper=None):
    """A 0-1 style meter for a number that lives on a known scale."""
    st.caption(label if helper is None else "%s  ·  %s" % (label, helper))
    st.progress(max(0.0, min(1.0, (value - low) / (high - low))))


# ---------------------------------------------------------------- the question
st.title("🎬 Will the critics like it?")
st.markdown(
    "Pick a Bollywood lead actor. This predicts the chance that **their next film "
    "is rated 3.5 or higher out of 5** by a Bollywood Hungama critic — using "
    "nothing but the ratings their earlier films received."
)

try:
    card = _loaded()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

base_rate = predict.load()["base_rate"]
actors = predict.list_actors()

# ---------------------------------------------------------------- the control
choice = st.selectbox(
    "Lead actor",
    [predict.DEBUT_LABEL] + actors,
    index=1 + actors.index("Akshay Kumar") if "Akshay Kumar" in actors else 0,
    help="47 leads with at least three films each. Or price a newcomer.",
)

result = (predict.for_debut() if choice == predict.DEBUT_LABEL
          else predict.for_actor(choice))
probability = result["probability"]

# ---------------------------------------------------------------- the answer
st.divider()
left, right = st.columns([1, 1.6])

with left:
    # A percentage on its own is unreadable - 55% is only meaningful next to the
    # 34% an average film manages. Ties are shown as a tie, not as "-0 pts".
    points = 100 * result["lift"]
    if abs(points) < 0.5:
        delta, delta_colour = "same as an average film", "off"
    else:
        delta, delta_colour = "%+.0f pts vs average film" % points, "normal"
    st.metric("Chance of a positive review",
              "%.0f%%" % (100 * probability),
              delta=delta, delta_color=delta_colour)

with right:
    if result["is_debut"]:
        st.info(
            "**No track record.** With nothing to go on, the model falls back to "
            "what an average film of this era scored (%.2f out of 5). This is the "
            "same path a real debut took in the training data — it is a baseline, "
            "not a judgement about anyone."
            % result["era_mean"])
    elif result["lift"] > 0.10:
        st.success(
            "**Better than an average film.** An average film clears 3.5 about "
            "%.0f%% of the time. This actor's record puts their next one well above "
            "that." % (100 * base_rate))
    elif result["lift"] > 0:
        st.info(
            "**Slightly better than average**, but not by much — an average film "
            "clears 3.5 about %.0f%% of the time." % (100 * base_rate))
    else:
        st.warning(
            "**Below average.** An average film clears 3.5 about %.0f%% of the "
            "time; this actor's record points lower." % (100 * base_rate))

st.caption(
    "The cut-off for calling it is 50%%, so this one is predicted **%s**. "
    "Read the percentage, not the label — 51%% and 49%% are the same claim."
    % result["verdict"]
)

# ---------------------------------------------------------------- the evidence
if not result["is_debut"]:
    st.divider()
    st.subheader("Where that number comes from")

    a, b, c = st.columns(3)
    a.metric("Films so far", result["films_to_date"])
    b.metric("Career average", "%.2f / 5" % result["career_mean"])
    c.metric("Hit rate", "%.0f%%" % (100 * result["positive_rate"]),
             help="Share of their films that were rated 3.5 or higher.")

    d, e, f = st.columns(3)
    d.metric("Last 3 films", "%.2f / 5" % result["last3_mean"])
    e.metric("Momentum", "%+.2f" % result["momentum"],
             help="Recent form minus career average. Positive means they are on "
                  "a better run than usual.")
    f.metric("Consistency", "±%.2f" % result["consistency"],
             help="How much their ratings swing. Smaller means more predictable.")

    st.write("")
    bar("Adjusted career average — **%.2f / 5**" % result["adjusted_mean"],
        result["adjusted_mean"], 1.0, 5.0,
        helper="pulled toward the average because they have %d films"
               % result["films_to_date"])
    st.caption(
        "An actor with four good films has not proved as much as one with forty. "
        "So the model deliberately drags small records toward the middle: "
        "%s's raw %.2f becomes %.2f once that is accounted for."
        % (result["actor"], result["career_mean"], result["adjusted_mean"])
    )

    # ------------------------------------------------------------ the history
    films = predict.history(result["actor"])
    if len(films):
        st.write("")
        st.markdown("**Every film they have led, and what the critic gave it**")

        dots = alt.Chart(films).mark_circle(size=110, opacity=0.85).encode(
            x=alt.X("release_date:T", title=None),
            y=alt.Y("rating:Q", title="critic rating",
                    scale=alt.Scale(domain=[0.5, 5.2])),
            color=alt.condition(alt.datum.rating >= predict.POSITIVE_THRESHOLD,
                                alt.value("#2e7d5b"), alt.value("#b4552d")),
            tooltip=[alt.Tooltip("movie_title:N", title="film"),
                     alt.Tooltip("rating:Q", title="rating"),
                     alt.Tooltip("release_date:T", title="released")],
        )
        cutoff = alt.Chart(pd.DataFrame({"y": [predict.POSITIVE_THRESHOLD]})).mark_rule(
            strokeDash=[5, 4], color="#888").encode(y="y:Q")

        st.altair_chart((dots + cutoff).properties(height=260),
                        width="stretch")
        st.caption("Green cleared 3.5, orange did not. The dashed line is the "
                   "threshold everything in this project is built around.")

# ---------------------------------------------------------------- the honesty
st.divider()
st.subheader("How much should you trust this?")

scores = card["scores"]
st.markdown(
    """
Not blindly, and the honest numbers are worth more than the prediction:

- On **%d films it had never seen**, it gets the ranking right about **%.0f%%** of
  the time (AUC %.3f, where 50%% is a coin flip).
- Its accuracy is **%.1f%%**. That sounds unimpressive until you notice that always
  guessing "negative" already scores **%.1f%%** — the useful gain is in *precision*:
  when it says positive, it is right **%.0f%%** of the time against a **%.0f%%** base rate.
- **The ceiling for any model like this is %.1f%% accuracy.** That is not a guess —
  it is calculated from the answers themselves, by asking how well you could do if
  you knew each actor's best possible call in advance. This model reaches %.1f%%. It
  is close to the wall, and no cleverer algorithm gets much past it.
"""
    % (scores["test_films"], 100 * scores["test_auc"], scores["test_auc"],
       100 * scores["test_accuracy"], 100 * scores["always_negative_accuracy"],
       100 * scores["test_precision"], 100 * scores["base_rate"],
       100 * scores["actor_only_oracle_accuracy"], 100 * scores["test_accuracy"])
)

with st.expander("What this model cannot see"):
    for limitation in card["limitations"]:
        st.markdown("- %s" % limitation)
    st.markdown(
        "\nThe short version: it knows **one thing** about a film — who is starring "
        "in it. Not the script, the director, the budget or the marketing. That it "
        "gets as far as it does on that alone is the interesting part; that it "
        "cannot get further is not a bug."
    )

with st.expander("What it is actually good for"):
    st.markdown(
        """
Ranking, not verdicts. Sort a slate of upcoming films by this probability and take
the top quarter: **%.0f%% of those turn out positive, against a %.0f%% base rate.**
That is a shortlist worth having. Asking it to call one specific film is asking the
wrong question of it.
        """ % (100 * scores["top25pct_hit_rate"], 100 * scores["base_rate"])
    )

with st.expander("How it was built"):
    st.markdown(
        """
1. **Scraped** ~700 Hindi films from Bollywood Hungama — title, lead actor, critic
   rating, release date.
2. **Cleaned** the titles and dropped the review text.
3. **Built the features** the hard way: every number describing an actor is computed
   from their *earlier* films only, walking the dataset in release order. A film
   never helps predict itself.
4. **Compared four models** — logistic regression, random forest, gradient boosting,
   and an average of all three — splitting the data by *time*, not at random,
   because a random split lets the model train on 2024 and test on 2004.
5. **Froze the winner** (%s) to a file, which is what this page loads.

The full working, including the exploratory analysis that set the ceiling above,
is in the repository.
        """ % card["model"]
    )

st.divider()
st.caption(
    "Model: %s trained on %d films (%s–%s), %d lead actors. Snapshot as of %s.  \n"
    "Ratings © Bollywood Hungama; this is a personal, educational project and "
    "redistributes no review text.  \n[Source and full write-up on GitHub](%s)"
    % (card["model"], card["trained_rows"], card["date_range"][0][:4],
       card["date_range"][1][:4], card["lead_actors"], result["as_of"], REPO)
)
