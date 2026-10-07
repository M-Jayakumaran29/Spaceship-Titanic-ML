# Spaceship Titanic – Learning-First ML

My machine learning project, on Kaggle's [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) competition: predict which passengers were transported to another dimension.

The goal is to understand *why* each step works, not just to climb the leaderboard. Every decision here came from a hypothesis I tested against the data first. The project started as my practical project for Anthropic's AI Fluency course and is still ongoing.

## What I found

- **CryoSleep is the strongest signal:** 82% of sleepers were transported vs 33% of awake passengers.
- **Interactions matter:** among cryosleepers, Martians on deck E were transported only 46% of the time, vs 91% for Martian sleepers overall.
- **Some features are tangled together:** decks A, B, C and T hold only Europans, and deck G holds only Earthlings, so deck G and Earth effects can't be separated.
- **Group size matters:** solo travellers were transported 45% of the time vs 54–64% for people in groups.
- **Not all spending is the same:** among awake passengers, Spa spenders were transported less (28% vs 40% for non-spenders), while FoodCourt spenders were transported slightly more (35% vs 31%).

## Handling missing values

About 24% of passengers have at least one missing field. Instead of filling everything with averages, I found rules in the data and verified each one before using it:

1. CryoSleep = True → all spending is 0
2. Spent anything → not in CryoSleep
3. Age > 12 and spent nothing → CryoSleep (96% accurate)
4. Age ≤ 12 → all spending is 0
5. Deck A/B/C/T → Europa, deck G → Earth
6. HomePlanet = a groupmate's HomePlanet
7. Cabin = a groupmate's Cabin

Whatever the rules couldn't fill gets a generic value (mean, median, most common value, or "Unknown"), computed from training data only.

## Validation

Test-set passengers never share a travel group with training passengers, so I used a **group-aware split** (`GroupShuffleSplit`) that keeps each group entirely on one side. That makes the validation score a more honest estimate of real test performance.

One split isn't enough on its own, though. Repeating the split with 5 different random seeds moved the same model's validation accuracy between about 78.5% and 80.4%. So any difference between models smaller than about a point on a single split is likely just noise.

## Current status

| Model | Train accuracy | Validation accuracy |
|---|---|---|
| Baseline (always predict True) | – | 51.8% |
| Logistic regression (label encoding, scaled, C = 1) | 78.9% | 78.8% (78.5–80.4% across 5 splits) |

Train and validation scores are nearly equal, so the model isn't overfitting. It may be too simple to capture the interactions above.

**Regularization:** C values from 1 to 100 gave identical results, so at the default the model is effectively unregularized. A very strong penalty (C = 0.001) underfit (76.6%). C = 0.1 looked about 2 passengers better on one split, but C = 1 won on 4 of 5 other splits, so I kept the default.

**Next:** try tree-based models with cross-validation, test one-hot vs label encoding, test whether adding VIP back helps, and test splitting spending into two groups.

## Files

- `eda.py` – exploratory analysis, organized by question, with the result and decision noted under each check
- `clean.py` – cleaning pipeline: rule-based fills, generic fills, feature creation and encoding
- `train.py` – loads and cleans data, makes the group-aware split, trains and evaluates the model, plus regularization and split-variance experiments

## How to run

1. Download `train.csv` and `test.csv` from the [competition's data page](https://www.kaggle.com/competitions/spaceship-titanic/data) and put them in the same folder as the scripts. (Data files aren't included in this repo, per Kaggle's rules.)
2. Install the libraries: `pip install pandas scikit-learn`
3. Run `python train.py` (or `python eda.py` to see the exploration)

## AI Use and Diligence Statement

**AI system used:** Claude (Anthropic), via claude.ai, inside a dedicated Project. I used it as my AI collaborator for Anthropic's AI Fluency course, which this project was built for.

**How AI contributed:**
- Explained concepts (overfitting, confounding, interactions, leakage, validation, logistic regression, log loss, gradients, regularization) and pandas/scikit-learn syntax.
- Suggested tests for my hypotheses and challenged my conclusions.
- Wrote parts of the code directly: several fill rules in `clean.py` (the groupmate fills and the strict CryoSleep rule), `get_fill_values`, the group-aware split, the model pipeline, and the experiment loops in `train.py`. It also reorganized `eda.py` into documented sections. I wrote and debugged the remaining code with AI review.
- Drafted this README, the repository description, and this statement, which I reviewed and edited.

**What I did:** Formed every hypothesis, interpreted every result, and made all project decisions: the 7 fill rules, fill strategies, encoding, the GroupSize feature, the group-aware validation split, the model choice, and keeping C = 1.

**How I verified AI contributions:**
- Every fill rule was tested against the data before use.
- AI claims were checked by running code. One AI inference (about deck E sleepers) was partly wrong and was corrected this way. Another AI guess, about why two results were identical, was disproved by my own output. An AI-suggested fix for a pandas warning didn't work at first and was corrected after I ran it.
- The cleaning pipeline was checked with before/after missing-value counts and run on test.csv. Encoding was checked for unmapped values.
- I confirmed the model's average predicted probability on training data matches the actual transported rate (0.49983 vs 0.49986), as the theory of how logistic regression learns predicts.
- AI-written result comments in `eda.py` were checked against real output.

**Privacy and data:** The Kaggle dataset is synthetic, with no real personal data. Competition data files are excluded from this repo. I shared code, outputs and screenshots with the AI, and no credentials or personal data.

**Context:** This is a learning project, intended for AI Fluency course review and as a portfolio piece.

**Responsibility:** I take full responsibility for this project's code, analysis and conclusions, including any errors.