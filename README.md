# Netflix Titles — Machine Learning Workflow

A machine learning workflow on the Netflix Titles dataset (8,807 titles), covering data cleaning, feature engineering, classification (Movie vs. TV Show), and regression (predicting release year) — using XGBoost, LightGBM, and a Neural Network (MLP) for both tasks.



## Dataset

`netflix_titles.csv` — 8,807 Netflix titles with metadata: type, title, director, cast, country, date added, release year, rating, duration, genres, and description.

## Workflow

| Order | Notebook | What it does |
|---|---|---|
| 1 | `1_Cleaning_Stage.ipynb` | Fixes a real shifted-column bug (3 rows had their duration value stuck in the `rating` column), imputes missing rating, drops unrecoverable missing duration rows |
| 2 | `2_Feature_Engineering_Stage.ipynb` | Parses date/duration, converts high-missing text fields (director/cast/country) into usable numeric/categorical signals, derives genre and text-length features |
| 3 | `3_Classification_Analysis_Stage.ipynb` | Predicts `type` (Movie vs. TV Show) using XGBoost, LightGBM, and a Neural Network (MLP) |
| 4 | `4_Regression_Analysis_Stage.ipynb` | Predicts `release_year` using the same three model types |

Notebooks 3 and 4 both branch off `netflix_engineered.csv` (stage 2's output) and can run in either order.

## A real data quality bug worth knowing

Three rows (all Louis C.K. specials) had their `duration` value sitting in the `rating` column instead — e.g. `rating == '74 min'` with `duration` empty. This is a genuine shifted-column error in the source data, not something to guess past. It was caught by scanning `rating` for values containing "min" and moving them into the correct column.

## An important leakage finding — read this before presenting the classification results

Two features were **deliberately excluded** from the classification task:

- **`Duration_Unit` / `Duration_Value`** — Movies are always measured in minutes, TV Shows always in seasons. Including this doesn't teach a model to predict type; it hands it the answer directly.
- **`Primary_Genre`** — Netflix's own genre taxonomy labels differently by type (e.g. "Crime TV Shows" and "Kids' TV" only ever appear on TV Shows; "Action & Adventure" only on Movies). This is a subtler form of the same leakage.

For comparison, including those two features pushed accuracy to **~99%** — which is the kind of suspiciously perfect number that should make you go looking for leakage rather than celebrate. Excluding them gives a fair, still-strong result:

### Classification (Movie vs. TV Show) — leakage-free feature set

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| XGBoost | 0.960 | 0.945 | 0.923 | 0.934 |
| LightGBM | 0.961 | 0.948 | 0.923 | 0.936 |
| Neural Network (MLP) | 0.944 | 0.916 | 0.899 | 0.908 |

LightGBM edged out XGBoost slightly; both tree-based models outperformed the neural network, which is typical for tabular data of this size — MLPs generally need much more data to compete with gradient-boosted trees on structured/tabular problems.

### Regression (predicting release year)

| Model | MSE | RMSE | R² |
|---|---|---|---|
| XGBoost Regressor | 61.46 | 7.84 | 0.326 |
| LightGBM Regressor | 55.80 | 7.47 | 0.388 |
| Neural Network (MLP) Regressor | 80.78 | 8.99 | 0.114 |

LightGBM again led. An R² around 0.35–0.39 means the model explains roughly a third of the variance in release year — reasonable given the main signal is `Year_Added` (titles added to Netflix soon after release vs. older catalog additions), while most of a title's actual release year isn't predictable from its metadata alone. Worth stating plainly rather than overselling.

## Tech Stack

- Python, pandas, numpy
- XGBoost, LightGBM
- scikit-learn (`MLPClassifier`/`MLPRegressor` for the "Neural Network" models — chosen over TensorFlow/Keras to avoid the dependency conflicts you ran into previously, while still giving genuine neural network results)
- scikit-learn preprocessing (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`)

## Setup

```bash
pip install pandas numpy scikit-learn xgboost lightgbm
```

Run notebooks 1 → 2 in order, then 3 and 4 in either order.

## Files

- `netflix_titles.csv` — raw dataset
- `netflix_preprocessed.csv`, `netflix_engineered.csv` — intermediate outputs
- `classification_results.csv`, `regression_results.csv` — model comparison metrics

## What to do before this becomes a portfolio piece

1. **Re-type every cell yourself** and add comments in your own words.
2. **Sit with the leakage finding** until you can explain it unprompted — "why did I exclude genre and duration from the classifier?" is exactly the kind of question an interviewer would ask.
3. **Try including the leaky features yourself** and see the 99% accuracy firsthand — that contrast is more convincing once you've reproduced it, not just read about it.
4. **Consider extra feature ideas**: sentiment or keyword analysis of `description`, cast/director "star power" (how many other titles they appear in), or genre combinations instead of just the primary genre.
