# ML Playground

Pick an algorithm, pick a dataset, tune the hyperparameters — see the decision boundary update live.

**[Live Demo](https://ml-playground-fdzxns9xf9xrwi98gxy4n8.streamlit.app)**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ml-playground-fdzxns9xf9xrwi98gxy4n8.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

I built this because I kept explaining ML concepts to people and the best I could do was
wave my hands at a static diagram. Now I can just open this, drag a slider, and show
exactly what happens when you crank up the SVM's C parameter or drop the tree depth to 1.

No install needed — just open the link.

## What's inside

8 classifiers: Decision Tree, Random Forest, Gradient Boosting, SVM, KNN, Logistic Regression, MLP, Naive Bayes

7 datasets: Moons, Circles, Blobs, Iris, Wine, Breast Cancer

- Decision boundary plot with train/test points shown separately
- Compare mode — two models side by side on identical data
- 5-fold CV so you're not fooled by a lucky test split
- Confusion matrix

## Run locally

```bash
git clone https://github.com/cgrtml/ml-playground.git
cd ml-playground
pip install -r requirements.txt
streamlit run app.py
```

## Related

[neural-trees](https://github.com/cgrtml/neural-trees) — sklearn-compatible implementations of Soft Decision Trees, HMoE and classifier comparison tests from Alpaydın's papers.

## License

MIT
