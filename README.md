# ML Playground

Pick an algorithm, pick a dataset, tune the hyperparameters — see the decision boundary update live. Compare two models head-to-head with statistical significance tests, or benchmark every classifier at once with the same protocol used in the [neural-trees](https://github.com/cgrtml/neural-trees) paper.

**[▶ Live Demo](https://ml-playground-fdzxns9xf9xrwi98gxy4n8.streamlit.app)**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ml-playground-fdzxns9xf9xrwi98gxy4n8.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Languages](https://img.shields.io/badge/UI-6%20languages-success)

---

## Why this exists

I kept explaining machine-learning concepts to people and the best I could do was wave my hands at a static diagram. Now I can open this, drag a slider, and show exactly what happens when you crank up the SVM's `C` parameter, drop the tree depth to 1, or ask whether XGBoost is *statistically* better than a Random Forest on a given dataset.

No install needed — just open the link.

## What's inside

### Three modes

| Mode | What it does |
|---|---|
| 🔬 **Single Model** | Decision boundary, confusion matrix, and 5-fold CV for one classifier on one dataset. |
| ⚔️ **Compare Two Models** | Same plots side-by-side for two classifiers on the same train/test split, plus paired *t*-test, McNemar, and Alpaydın's 5×2cv *F*-test on top. |
| 🏆 **Benchmark All Models** | Trains every available classifier with sensible defaults and renders a leaderboard, a CV-accuracy bar chart, a decision-boundary grid, and a confusion-matrix grid. |

### 13 classifiers

**Classical (sklearn):** Decision Tree · Random Forest · Gradient Boosting · SVM (RBF) · *k*-NN · Logistic Regression · MLP · Naive Bayes

**Strong tabular baselines:** XGBoost · LightGBM · CatBoost · TabNet

**Differentiable trees:** Soft Decision Tree (from [neural-trees](https://github.com/cgrtml/neural-trees))

### 7 datasets

Moons · Circles · Blobs (2 / 3 classes) · Iris · Wine · Breast Cancer

### Statistical significance tests

| Test | When to use |
|---|---|
| Paired *t*-test (5×2cv) | Aligned cross-validation scores — quick read on mean difference. |
| Alpaydın 5×2cv *F*-test | Better Type-I error control than the paired *t*-test; recommended in the paper. |
| McNemar (corrected) | Disagreement on a single held-out test set — appropriate when refitting is expensive. |

### Educational content

- Every dataset has an "About this dataset" panel explaining what it is and what it tests.
- Every model has an "About this model" panel with a 2-3 sentence plain-language description.
- Every mode has an "About this mode" panel.
- A 4-step landing tutorial appears before any plot is drawn.

### 6 languages

🇬🇧 English · 🇹🇷 Türkçe · 🇪🇸 Español · 🇮🇹 Italiano · 🇸🇦 العربية (RTL) · 🇷🇺 Русский

Switch from the dropdown at the top of the sidebar.

## Visual conventions

- **Circles** = training points
- **Diamonds** = held-out test points
- **Colored regions** = the model's class predictions on a fine grid

If the boundary looks perfect on the training circles but messy around the test diamonds, you're looking at overfitting in real time.

## Run locally

```bash
git clone https://github.com/cgrtml/ml-playground.git
cd ml-playground
pip install -r requirements.txt
streamlit run app.py
```

Optional: `torch`, `pytorch-tabnet`, and `neural-trees` are listed in `requirements.txt` but are auto-skipped if your environment can't install them — the rest of the playground keeps working without TabNet and the Soft Decision Tree.

## Architecture

```
ml-playground/
├── app.py              # Streamlit UI: 3 modes, 13 classifiers, plots, sig tests
├── translations.py     # 6-language i18n dict (135 keys × 6 langs)
├── requirements.txt    # streamlit, plotly, sklearn, xgboost, lightgbm,
│                       # catboost, mlxtend, torch (CPU), pytorch-tabnet,
│                       # neural-trees
└── README.md
```

Every visible string is routed through `t(key, lang, **fmt)`; English is the fallback for missing keys; Arabic flips the page direction to RTL.

## Related work

- **[neural-trees](https://github.com/cgrtml/neural-trees)** — sklearn-compatible PyTorch implementations of Soft Decision Trees, Hierarchical Mixture-of-Experts, Omnivariate Decision Trees, and Alpaydın's 5×2cv *F*-test. The paper that this playground is the interactive companion to.
- **mlxtend** — the source of the paired *t*, McNemar, and combined 5×2cv *F*-test implementations used here.
- **Alpaydın, *Introduction to Machine Learning* (MIT Press, 2020)** — the textbook reference for the statistical tests and the soft decision tree formulation.

## Citation

If this playground or the underlying [neural-trees](https://github.com/cgrtml/neural-trees) library is useful for your work, please cite:

```bibtex
@software{temel_neuraltrees,
  author = {Temel, Cagri},
  title  = {neural-trees: Differentiable Decision Trees for Interpretable Machine Learning},
  year   = {2026},
  url    = {https://github.com/cgrtml/neural-trees}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Author

[Cagri Temel](https://github.com/cgrtml) · IEEE Senior Member · Hezarfen LLC, Seattle, WA
