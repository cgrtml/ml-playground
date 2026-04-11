# ML Playground

> Pick an algorithm, pick a dataset, tune the hyperparameters — see the decision boundary update live.

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

I built this because I kept explaining ML concepts to people and the best I could do was
wave my hands at a static diagram. Now I can just open this, drag a slider, and show
exactly what happens when you crank up the SVM's C parameter or drop the tree depth to 1.

No code required on the user side — just run it and play.

## What's inside

| | |
|---|---|
| **8 classifiers** | Decision Tree, Random Forest, Gradient Boosting, SVM, KNN, Logistic Regression, MLP, Naive Bayes |
| **7 datasets** | Moons, Circles, Blobs (2 & 3 class), Iris, Wine, Breast Cancer |
| **Decision boundary** | Interactive Plotly plot with train/test points shown separately |
| **Compare mode** | Two models side by side on the exact same data split |
| **Cross-validation** | 5-fold CV so you're not fooled by a lucky test split |
| **Confusion matrix** | Where exactly does each model make mistakes |
| **Hyperparameter sliders** | Every key parameter exposed — change and re-run instantly |

## Run it

```bash
git clone https://github.com/cgrtml/ml-playground.git
cd ml-playground
pip install -r requirements.txt
streamlit run app.py
```

Opens at [http://localhost:8501](http://localhost:8501).

## How to use

1. **Sidebar → Dataset** — pick synthetic (moons/circles/blobs) or real (iris/wine/cancer)
2. **Sidebar → Mode** — single model or compare two
3. **Sidebar → Algorithm** — choose and tune
4. **Hit ▶ Run**

The train points (circles) and test points (diamonds) are shown separately on the boundary
plot so you can immediately spot overfitting visually.

## Roadmap

- [ ] Upload your own CSV
- [ ] Regression mode
- [ ] Animated training for neural nets (epoch by epoch)
- [ ] Feature importance panel for tree-based models

If you want to add something, open an issue first so we don't duplicate work.

## Related

[neural-trees](https://github.com/cgrtml/neural-trees) — research-grade implementations
of differentiable tree algorithms (Soft Decision Trees, HMoE, etc.) from Alpaydın's papers.

## License

MIT — see [LICENSE](LICENSE).
