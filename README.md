# ML Playground

> **Visualize any classifier on any dataset — interactively.**
> Pick an algorithm, tune hyperparameters, and watch the decision boundary update live.

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is this?

ML Playground is a **Streamlit web app** for exploring machine learning classifiers visually. No code required — just pick your settings and hit Run.

## Features

| Feature | Details |
|---------|---------|
| **8 algorithms** | Decision Tree, Random Forest, Gradient Boosting, SVM, KNN, Logistic Regression, MLP, Naive Bayes |
| **7 datasets** | Moons, Circles, Blobs, Iris, Wine, Breast Cancer |
| **Decision boundary** | Interactive Plotly plot, train/test split shown separately |
| **Compare mode** | Run two models side by side on identical data |
| **Cross-validation** | 5-fold CV scores with mean ± std |
| **Confusion matrix** | Color-coded heatmap for test set |
| **Hyperparameter sliders** | Tune in real time |

## Quickstart

```bash
git clone https://github.com/cgrtml/ml-playground.git
cd ml-playground
pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Screenshots

> Single model mode — Decision Tree on Moons dataset

```
[ Decision Boundary Plot ]  [ Confusion Matrix ]
                            [ 5-Fold CV Scores  ]
```

> Compare mode — SVM vs KNN side by side

```
[ Model A boundary ]  [ Model B boundary ]
[ Model A CM       ]  [ Model B CM       ]
[ Head-to-Head bar chart                 ]
```

## Usage

1. **Sidebar → Dataset**: Choose a synthetic or real-world dataset
2. **Sidebar → Mode**: Single model or compare two models
3. **Sidebar → Algorithm**: Pick classifier and tune hyperparameters
4. **Click ▶ Run**

## Inspiration

This project is inspired by the intuitions built in:

> Alpaydın, E. (2020). *Introduction to Machine Learning* (4th ed.). MIT Press.

The visual intuition behind decision boundaries, bias-variance tradeoff, and model comparison directly maps to concepts explained in that textbook. See also [neural-trees](https://github.com/cgrtml/neural-trees) for research-grade implementations of algorithms from Alpaydın's papers.

## Contributing

Open to PRs! Ideas welcome:
- [ ] Regression mode
- [ ] Upload your own CSV
- [ ] Animated training (epoch-by-epoch for neural nets)
- [ ] Feature importance plots

## License

MIT
