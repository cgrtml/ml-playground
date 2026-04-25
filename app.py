"""
ML Playground — Interactive Machine Learning Visualizer
========================================================
Explore any classifier on any dataset with live decision boundary plots,
confusion matrices, head-to-head comparisons, multi-model benchmarks,
and statistical significance tests.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from sklearn.datasets import (
    make_moons, make_circles, make_classification,
    load_iris, load_wine, load_breast_cancer,
)
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

# Strong tabular baselines — required for the paper's benchmark
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# TabNet and neural-trees both depend on torch — optional on Streamlit Cloud
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch  # noqa: F401
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False

try:
    from neural_trees import SoftDecisionTree
    NEURAL_TREES_AVAILABLE = True
except ImportError:
    NEURAL_TREES_AVAILABLE = False

# Statistical tests — paired t, McNemar, 5x2cv F-test
try:
    from mlxtend.evaluate import (
        paired_ttest_5x2cv,
        mcnemar_table,
        mcnemar,
        combined_ftest_5x2cv,
    )
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False

import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Playground",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #888;
        font-size: 1rem;
        margin-top: 0;
    }
    .stMetric {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

DATASETS = {
    "🌙 Moons": "moons",
    "⭕ Circles": "circles",
    "🔵 Blobs (2 classes)": "blobs2",
    "🔵 Blobs (3 classes)": "blobs3",
    "🌸 Iris": "iris",
    "🍷 Wine": "wine",
    "🔬 Breast Cancer": "cancer",
}

# Classical sklearn zoo — always available
CLASSIFIERS = {
    "Decision Tree": "dt",
    "Random Forest": "rf",
    "Gradient Boosting": "gb",
    "SVM (RBF)": "svm",
    "K-Nearest Neighbors": "knn",
    "Logistic Regression": "lr",
    "Neural Network (MLP)": "mlp",
    "Naive Bayes": "nb",
}

# Strong tabular baselines — added if installed
if XGBOOST_AVAILABLE:
    CLASSIFIERS["⚡ XGBoost"] = "xgb"
if LIGHTGBM_AVAILABLE:
    CLASSIFIERS["⚡ LightGBM"] = "lgbm"
if CATBOOST_AVAILABLE:
    CLASSIFIERS["⚡ CatBoost"] = "cat"
if TABNET_AVAILABLE:
    CLASSIFIERS["🧠 TabNet"] = "tabnet"
if NEURAL_TREES_AVAILABLE:
    CLASSIFIERS["🌳 Soft Decision Tree (neural-trees)"] = "sdt"


def load_dataset(name: str, n_samples: int = 500, noise: float = 0.2, random_state: int = 42):
    """Return X, y, feature_names. Tabular datasets are PCA-reduced to 2D for visualization."""
    feature_names = ("Feature 1", "Feature 2")

    if name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif name == "circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
    elif name == "blobs2":
        X, y = make_classification(
            n_samples=n_samples, n_features=2, n_redundant=0,
            n_informative=2, n_clusters_per_class=1, random_state=random_state,
        )
    elif name == "blobs3":
        X, y = make_classification(
            n_samples=n_samples, n_features=2, n_redundant=0,
            n_informative=2, n_clusters_per_class=1, n_classes=3, random_state=random_state,
        )
    elif name == "iris":
        data = load_iris()
        X, y = data.data[:, :2], data.target
        feature_names = tuple(
            n.replace(" (cm)", "").title() for n in data.feature_names[:2]
        )
    elif name == "wine":
        data = load_wine()
        X, y = data.data[:, :2], data.target
        feature_names = tuple(
            n.replace("_", " ").title() for n in data.feature_names[:2]
        )
    elif name == "cancer":
        data = load_breast_cancer()
        pca = PCA(n_components=2, random_state=random_state)
        X = pca.fit_transform(data.data)
        y = data.target
        feature_names = ("PC 1", "PC 2")
    else:
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

    return X, y, feature_names


def build_classifier(name: str, params: dict):
    if name == "dt":
        return DecisionTreeClassifier(
            max_depth=params.get("max_depth", 5),
            min_samples_split=params.get("min_samples_split", 2),
            random_state=42,
        )
    elif name == "rf":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 5),
            random_state=42,
        )
    elif name == "gb":
        return GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 3),
            random_state=42,
        )
    elif name == "svm":
        return SVC(
            C=params.get("C", 1.0),
            gamma=params.get("gamma", "scale"),
            probability=True,
            random_state=42,
        )
    elif name == "knn":
        return KNeighborsClassifier(n_neighbors=params.get("n_neighbors", 5))
    elif name == "lr":
        return LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=1000,
            random_state=42,
        )
    elif name == "mlp":
        hidden = params.get("hidden_layer_sizes", 64)
        return MLPClassifier(
            hidden_layer_sizes=(hidden, hidden // 2),
            max_iter=500,
            random_state=42,
        )
    elif name == "nb":
        return GaussianNB()
    elif name == "xgb" and XGBOOST_AVAILABLE:
        return XGBClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.1),
            random_state=42,
            eval_metric="mlogloss",
            verbosity=0,
            n_jobs=1,
        )
    elif name == "lgbm" and LIGHTGBM_AVAILABLE:
        return LGBMClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.1),
            random_state=42,
            n_jobs=1,
            num_threads=1,
            force_row_wise=True,
            verbose=-1,
        )
    elif name == "cat" and CATBOOST_AVAILABLE:
        return CatBoostClassifier(
            iterations=params.get("n_estimators", 200),
            depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.1),
            random_state=42,
            verbose=0,
            allow_writing_files=False,
        )
    elif name == "tabnet" and TABNET_AVAILABLE:
        return _TabNetSklearn(
            n_d=params.get("n_d", 8),
            n_a=params.get("n_a", 8),
            n_steps=params.get("n_steps", 3),
            max_epochs=params.get("max_epochs", 30),
        )
    elif name == "sdt" and NEURAL_TREES_AVAILABLE:
        return SoftDecisionTree(
            depth=params.get("depth", 4),
            max_epochs=params.get("max_epochs", 30),
            penalty_coef=params.get("penalty_coef", 1e-3),
        )
    raise ValueError(f"Unknown classifier: {name}")


# Lightweight TabNet wrapper that satisfies sklearn's estimator protocol.
if TABNET_AVAILABLE:
    from sklearn.base import BaseEstimator, ClassifierMixin

    class _TabNetSklearn(ClassifierMixin, BaseEstimator):
        def __init__(self, n_d=8, n_a=8, n_steps=3, max_epochs=30):
            self.n_d = n_d
            self.n_a = n_a
            self.n_steps = n_steps
            self.max_epochs = max_epochs

        def fit(self, X, y):
            self._model = TabNetClassifier(
                n_d=self.n_d, n_a=self.n_a, n_steps=self.n_steps,
                seed=42, verbose=0,
            )
            self.classes_ = np.unique(y)
            X32 = np.asarray(X, dtype=np.float32)
            y_int = np.asarray(y).astype(np.int64)
            self._model.fit(
                X32, y_int,
                max_epochs=self.max_epochs,
                patience=10,
                batch_size=min(256, len(X32)),
                virtual_batch_size=min(64, len(X32)),
                drop_last=False,
            )
            return self

        def predict(self, X):
            return self._model.predict(np.asarray(X, dtype=np.float32))

        def predict_proba(self, X):
            return self._model.predict_proba(np.asarray(X, dtype=np.float32))


def make_decision_boundary(clf, X, scaler, resolution=160):
    """Create a meshgrid and predict class labels for background coloring."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid)

    if hasattr(clf, "predict_proba"):
        try:
            probs = clf.predict_proba(grid_scaled)
            Z = probs.argmax(axis=1)
        except Exception:
            Z = clf.predict(grid_scaled)
    else:
        Z = clf.predict(grid_scaled)

    return xx, yy, Z.reshape(xx.shape)


def plot_decision_boundary(clf, X_train, X_test, y_train, y_test, scaler,
                           title="", feature_names=("Feature 1", "Feature 2"),
                           height=480, show_legend=True):
    classes = np.unique(np.concatenate([y_train, y_test]))

    bg_colors = [
        "rgba(99, 110, 250, 0.15)",
        "rgba(239, 85, 59, 0.15)",
        "rgba(0, 204, 150, 0.15)",
        "rgba(255, 161, 90, 0.15)",
    ]
    point_colors = ["#636EFA", "#EF553B", "#00CC96", "#FFA15A"]

    X_all = np.vstack([X_train, X_test])
    xx, yy, Z = make_decision_boundary(clf, X_all, scaler)

    fig = go.Figure()
    for c in classes:
        fig.add_trace(go.Contour(
            x=xx[0],
            y=yy[:, 0],
            z=(Z == c).astype(float),
            showscale=False,
            colorscale=[[0, "rgba(0,0,0,0)"], [1, bg_colors[int(c) % len(bg_colors)]]],
            contours=dict(start=0.5, end=1.5, size=1),
            hoverinfo="skip",
            showlegend=False,
        ))
    for c in classes:
        mask = y_train == c
        fig.add_trace(go.Scatter(
            x=X_train[mask, 0], y=X_train[mask, 1],
            mode="markers",
            marker=dict(color=point_colors[int(c) % len(point_colors)], size=7,
                        symbol="circle", line=dict(width=1, color="white")),
            name=f"Train class {c}", legendgroup=f"class_{c}",
            showlegend=show_legend,
        ))
    for c in classes:
        mask = y_test == c
        fig.add_trace(go.Scatter(
            x=X_test[mask, 0], y=X_test[mask, 1],
            mode="markers",
            marker=dict(color=point_colors[int(c) % len(point_colors)], size=10,
                        symbol="diamond", line=dict(width=2, color="black")),
            name=f"Test class {c}", legendgroup=f"class_{c}",
            showlegend=False,
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title=feature_names[0],
        yaxis_title=feature_names[1],
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,249,250,1)",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", height=350):
    cm = confusion_matrix(y_true, y_pred)
    labels = np.unique(y_true)
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=[str(l) for l in labels],
        y=[str(l) for l in labels],
        color_continuous_scale="Blues",
        title=title,
        text_auto=True,
    )
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def plot_cv_scores(scores, title="5-Fold CV Accuracy"):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"Fold {i+1}" for i in range(len(scores))],
        y=scores,
        marker_color="rgba(99, 110, 250, 0.8)",
        text=[f"{s:.3f}" for s in scores],
        textposition="outside",
    ))
    fig.add_hline(
        y=scores.mean(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {scores.mean():.3f}",
        annotation_position="right",
    )
    fig.update_layout(
        title=title,
        yaxis=dict(range=[0, 1.05]),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def hyperparam_ui(clf_name: str, suffix: str) -> dict:
    """Render hyperparameter widgets for the given classifier and return the param dict."""
    p = {}
    k = lambda s: f"{s}_{suffix}"
    if clf_name == "dt":
        p["max_depth"] = st.slider("Max depth", 1, 20, 5, key=k("dt_depth"))
        p["min_samples_split"] = st.slider("Min samples split", 2, 20, 2, key=k("dt_mss"))
    elif clf_name in ("rf", "gb"):
        p["n_estimators"] = st.slider("N estimators", 10, 300, 100, key=k("ne"))
        p["max_depth"] = st.slider("Max depth", 1, 20, 5, key=k("md"))
        if clf_name == "gb":
            p["learning_rate"] = st.slider("Learning rate", 0.01, 0.5, 0.1, key=k("lr"))
    elif clf_name in ("svm", "lr"):
        p["C"] = st.slider("C (regularization)", 0.01, 10.0, 1.0, key=k("c"))
    elif clf_name == "knn":
        p["n_neighbors"] = st.slider("K neighbors", 1, 30, 5, key=k("knn"))
    elif clf_name == "mlp":
        p["hidden_layer_sizes"] = st.slider("Hidden units", 8, 256, 64, key=k("mlp"))
    elif clf_name in ("xgb", "lgbm", "cat"):
        p["n_estimators"] = st.slider("N estimators", 50, 500, 200, key=k("boost_ne"))
        p["max_depth"] = st.slider("Max depth", 2, 12, 5, key=k("boost_md"))
        p["learning_rate"] = st.slider("Learning rate", 0.01, 0.5, 0.1, key=k("boost_lr"))
    elif clf_name == "tabnet":
        p["n_d"] = st.slider("n_d (decision width)", 4, 32, 8, key=k("tn_nd"))
        p["n_a"] = st.slider("n_a (attention width)", 4, 32, 8, key=k("tn_na"))
        p["n_steps"] = st.slider("Steps", 1, 8, 3, key=k("tn_steps"))
        p["max_epochs"] = st.slider("Epochs", 5, 100, 30, key=k("tn_epochs"))
    elif clf_name == "sdt":
        p["depth"] = st.slider("Tree depth", 2, 7, 4, key=k("sdt_depth"))
        p["max_epochs"] = st.slider("Epochs", 10, 60, 30, key=k("sdt_epochs"))
        p["penalty_coef"] = st.select_slider("Penalty",
                                             options=[1e-4, 1e-3, 1e-2],
                                             value=1e-3, key=k("sdt_pen"))
    return p


def default_params(clf_name: str) -> dict:
    """Sensible defaults used by Benchmark All so the user doesn't have to tune 10 sliders."""
    return {
        "dt":     {"max_depth": 5, "min_samples_split": 2},
        "rf":     {"n_estimators": 100, "max_depth": 5},
        "gb":     {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
        "svm":    {"C": 1.0},
        "knn":    {"n_neighbors": 5},
        "lr":     {"C": 1.0},
        "mlp":    {"hidden_layer_sizes": 64},
        "nb":     {},
        "xgb":    {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1},
        "lgbm":   {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1},
        "cat":    {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1},
        "tabnet": {"n_d": 8, "n_a": 8, "n_steps": 3, "max_epochs": 30},
        "sdt":    {"depth": 4, "max_epochs": 30, "penalty_coef": 1e-3},
    }.get(clf_name, {})


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎮 ML Playground")
    st.markdown("---")

    # ── Dataset ──
    st.markdown("### 📊 Dataset")
    dataset_label = st.selectbox("Choose dataset", list(DATASETS.keys()))
    dataset_name = DATASETS[dataset_label]

    if dataset_name in ("moons", "circles", "blobs2", "blobs3"):
        n_samples = st.slider("Samples", 100, 1000, 400, step=50)
        noise = st.slider("Noise", 0.0, 0.5, 0.2, step=0.05)
    else:
        n_samples, noise = 500, 0.2

    test_size = st.slider("Test split", 0.1, 0.4, 0.25, step=0.05)
    random_state = st.number_input("Random seed", value=42, step=1)

    st.markdown("---")

    # ── Mode ──
    mode = st.radio(
        "Mode",
        ["🔬 Single Model", "⚔️ Compare Two Models", "🏆 Benchmark All Models"],
    )

    st.markdown("---")

    # Single & Compare modes need explicit model selection; Benchmark All does not.
    params_a = params_b = {}
    clf_name_a = clf_name_b = None
    clf_label_a = clf_label_b = None

    if mode in ("🔬 Single Model", "⚔️ Compare Two Models"):
        st.markdown("### 🤖 Model A")
        clf_label_a = st.selectbox("Algorithm A", list(CLASSIFIERS.keys()), key="clf_a")
        clf_name_a = CLASSIFIERS[clf_label_a]
        with st.expander("Hyperparameters A"):
            params_a = hyperparam_ui(clf_name_a, "a")

    if mode == "⚔️ Compare Two Models":
        st.markdown("### 🤖 Model B")
        # Default Model B to a different algorithm than Model A
        b_default = min(4, len(CLASSIFIERS) - 1)
        clf_label_b = st.selectbox("Algorithm B", list(CLASSIFIERS.keys()),
                                   index=b_default, key="clf_b")
        clf_name_b = CLASSIFIERS[clf_label_b]
        with st.expander("Hyperparameters B"):
            params_b = hyperparam_ui(clf_name_b, "b")

    if mode == "🏆 Benchmark All Models":
        st.info(
            f"Will train **{len(CLASSIFIERS)}** classifiers with default "
            "hyperparameters on the chosen dataset and render a side-by-side grid."
        )

    st.markdown("---")
    run_btn = st.button("▶ Run", use_container_width=True, type="primary")

    # Availability footer so the user understands which backends are wired up
    with st.expander("Installed backends"):
        st.write(
            f"- XGBoost: {'✅' if XGBOOST_AVAILABLE else '❌'}\n"
            f"- LightGBM: {'✅' if LIGHTGBM_AVAILABLE else '❌'}\n"
            f"- CatBoost: {'✅' if CATBOOST_AVAILABLE else '❌'}\n"
            f"- TabNet: {'✅' if TABNET_AVAILABLE else '❌'}\n"
            f"- neural-trees (SDT): {'✅' if NEURAL_TREES_AVAILABLE else '❌'}\n"
            f"- mlxtend (significance tests): {'✅' if MLXTEND_AVAILABLE else '❌'}"
        )

# ── Main area ─────────────────────────────────────────────────────────────────

st.markdown('<h1 class="main-header">ML Playground</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">'
    'Pick an algorithm, pick a dataset, see what happens. Compare two models head to head, '
    'or benchmark every classifier at once with the same evaluation protocol used in the '
    '<a href="https://github.com/cgrtml/neural-trees" target="_blank">neural-trees</a> paper.'
    '</p>',
    unsafe_allow_html=True,
)

if not run_btn:
    st.markdown("""
    Pick a dataset from the sidebar, pick a mode, hit **Run**.

    - **Single Model** — decision boundary, confusion matrix, and 5-fold CV for one classifier.
    - **Compare Two Models** — same plots side by side for two classifiers, plus statistical
      significance tests (paired *t*, McNemar, Alpaydın's 5×2cv *F*-test) on the same train/test split.
    - **Benchmark All Models** — train every available classifier with default hyperparameters
      and compare them on a single decision-boundary grid, a confusion-matrix grid, and a
      cross-validated accuracy bar chart.

    Train points are circles, test points are diamonds. Colored regions are class predictions.
    """)
    st.stop()

# ── Data loading & preprocessing ──────────────────────────────────────────────
X, y, feature_names = load_dataset(dataset_name, n_samples, noise, int(random_state))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(random_state), stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


def fit_and_score(clf_name, params):
    """Fit a fresh classifier on the current split and return predictions, metrics, and CV."""
    clf = build_classifier(clf_name, params)
    clf.fit(X_train_s, y_train)
    pred_train = clf.predict(X_train_s)
    pred_test = clf.predict(X_test_s)
    out = {
        "clf": clf,
        "pred_train": pred_train,
        "pred_test": pred_test,
        "acc_train": accuracy_score(y_train, pred_train),
        "acc_test": accuracy_score(y_test, pred_test),
    }
    try:
        cv = cross_val_score(
            build_classifier(clf_name, params), X_train_s, y_train, cv=5,
        )
    except Exception:
        cv = np.full(5, np.nan)
    out["cv"] = cv
    return out


# ── Single Model ──────────────────────────────────────────────────────────────

if mode == "🔬 Single Model":
    with st.spinner(f"Training {clf_label_a}..."):
        a = fit_and_score(clf_name_a, params_a)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train Accuracy", f"{a['acc_train']:.3f}")
    c2.metric("Test Accuracy", f"{a['acc_test']:.3f}",
              delta=f"{a['acc_test'] - a['acc_train']:.3f} vs train")
    c3.metric("CV Mean ± Std", f"{a['cv'].mean():.3f} ± {a['cv'].std():.3f}")
    c4.metric("Dataset size", f"{len(X)} samples / {len(np.unique(y))} classes")

    st.markdown("---")
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.plotly_chart(
            plot_decision_boundary(
                a["clf"], X_train, X_test, y_train, y_test, scaler,
                title=f"{clf_label_a} — Decision Boundary",
                feature_names=feature_names,
            ),
            use_container_width=True,
        )
    with col_right:
        st.plotly_chart(
            plot_confusion_matrix(y_test, a["pred_test"], "Confusion Matrix (Test Set)"),
            use_container_width=True,
        )
        st.plotly_chart(
            plot_cv_scores(a["cv"], "5-Fold CV Accuracy"),
            use_container_width=True,
        )

# ── Compare Two Models ────────────────────────────────────────────────────────

elif mode == "⚔️ Compare Two Models":
    with st.spinner(f"Training {clf_label_a} and {clf_label_b}..."):
        a = fit_and_score(clf_name_a, params_a)
        b = fit_and_score(clf_name_b, params_b)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"#### Model A: {clf_label_a}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Train Acc", f"{a['acc_train']:.3f}")
        m2.metric("Test Acc", f"{a['acc_test']:.3f}")
        m3.metric("CV Mean", f"{a['cv'].mean():.3f}")
        st.plotly_chart(
            plot_decision_boundary(
                a["clf"], X_train, X_test, y_train, y_test, scaler,
                title="Model A — Decision Boundary",
                feature_names=feature_names,
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            plot_confusion_matrix(y_test, a["pred_test"], "Model A — Confusion Matrix"),
            use_container_width=True,
        )

    with col_b:
        st.markdown(f"#### Model B: {clf_label_b}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Train Acc", f"{b['acc_train']:.3f}")
        m2.metric("Test Acc", f"{b['acc_test']:.3f}")
        m3.metric("CV Mean", f"{b['cv'].mean():.3f}")
        st.plotly_chart(
            plot_decision_boundary(
                b["clf"], X_train, X_test, y_train, y_test, scaler,
                title="Model B — Decision Boundary",
                feature_names=feature_names,
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            plot_confusion_matrix(y_test, b["pred_test"], "Model B — Confusion Matrix"),
            use_container_width=True,
        )

    # ── Head-to-head summary ──
    st.markdown("---")
    st.markdown("### Head-to-Head Comparison")
    winner = clf_label_a if a["acc_test"] >= b["acc_test"] else clf_label_b
    diff = abs(a["acc_test"] - b["acc_test"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Winner (Test Acc)", winner)
    c2.metric("Accuracy difference", f"{diff:.3f}")
    c3.metric("A CV vs B CV", f"{a['cv'].mean():.3f} vs {b['cv'].mean():.3f}")
    c4.metric("Overfit (A vs B)",
              f"{a['acc_train'] - a['acc_test']:.3f} vs {b['acc_train'] - b['acc_test']:.3f}")

    df_comp = pd.DataFrame({
        "Metric": ["Train Accuracy", "Test Accuracy", "CV Mean"],
        clf_label_a: [a["acc_train"], a["acc_test"], a["cv"].mean()],
        clf_label_b: [b["acc_train"], b["acc_test"], b["cv"].mean()],
    })
    fig_comp = px.bar(
        df_comp.melt(id_vars="Metric", var_name="Model", value_name="Score"),
        x="Metric", y="Score", color="Model", barmode="group",
        color_discrete_sequence=["#636EFA", "#EF553B"],
        title="Model Comparison",
        range_y=[0, 1.05],
    )
    fig_comp.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_comp, use_container_width=True)

    # ── Statistical significance tests ──
    st.markdown("---")
    st.markdown("### Statistical Significance Tests")
    if not MLXTEND_AVAILABLE:
        st.warning(
            "`mlxtend` is not installed; significance tests are unavailable. "
            "Install it with `pip install mlxtend` to enable paired *t*, "
            "McNemar, and 5×2cv *F*-tests."
        )
    else:
        with st.spinner("Running paired *t*, McNemar, and 5×2cv *F*-tests..."):
            try:
                est_a = build_classifier(clf_name_a, params_a)
                est_b = build_classifier(clf_name_b, params_b)

                # Paired t-test on aligned 5-fold CV scores
                t_stat, t_p = paired_ttest_5x2cv(
                    estimator1=est_a, estimator2=est_b,
                    X=X_train_s, y=y_train, random_seed=int(random_state),
                )
                # 5x2cv F-test (Alpaydin)
                f_stat, f_p = combined_ftest_5x2cv(
                    estimator1=build_classifier(clf_name_a, params_a),
                    estimator2=build_classifier(clf_name_b, params_b),
                    X=X_train_s, y=y_train, random_seed=int(random_state),
                )
                # McNemar on the held-out test set
                tb = mcnemar_table(
                    y_target=y_test,
                    y_model1=a["pred_test"],
                    y_model2=b["pred_test"],
                )
                chi2, mc_p = mcnemar(ary=tb, corrected=True)

                alpha = 0.05
                rows = [
                    ("Paired *t*-test (5×2cv)", f"{t_stat:.3f}", f"{t_p:.4f}",
                     "reject H₀" if t_p < alpha else "fail to reject H₀"),
                    ("Alpaydın 5×2cv *F*-test", f"{f_stat:.3f}", f"{f_p:.4f}",
                     "reject H₀" if f_p < alpha else "fail to reject H₀"),
                    ("McNemar (test split, corrected)", f"{chi2:.3f}", f"{mc_p:.4f}",
                     "reject H₀" if mc_p < alpha else "fail to reject H₀"),
                ]
                df_stats = pd.DataFrame(rows, columns=["Test", "Statistic", "p-value", f"Decision (α=0.05)"])
                st.table(df_stats)
                st.caption(
                    "H₀: the two models have equal generalization performance. "
                    "*Reject* means the observed difference is unlikely under H₀."
                )
            except Exception as e:
                st.error(f"Significance test failed: {e}")

# ── Benchmark All Models ──────────────────────────────────────────────────────

else:
    items = list(CLASSIFIERS.items())  # [(label, name), ...]
    n = len(items)
    progress = st.progress(0.0, text="Training classifiers...")
    results = {}
    for i, (label, name) in enumerate(items, start=1):
        try:
            results[label] = fit_and_score(name, default_params(name))
        except Exception as e:
            results[label] = {"error": str(e)}
        progress.progress(i / n, text=f"Trained {label} ({i}/{n})")
    progress.empty()

    # ── Summary metrics table ──
    rows = []
    for label, r in results.items():
        if "error" in r:
            rows.append({"Model": label, "Train Acc": np.nan, "Test Acc": np.nan,
                         "CV Mean": np.nan, "CV Std": np.nan, "Status": f"❌ {r['error']}"})
        else:
            rows.append({
                "Model": label,
                "Train Acc": round(r["acc_train"], 4),
                "Test Acc": round(r["acc_test"], 4),
                "CV Mean": round(np.nanmean(r["cv"]), 4),
                "CV Std": round(np.nanstd(r["cv"]), 4),
                "Status": "✅",
            })
    df = pd.DataFrame(rows).sort_values("Test Acc", ascending=False).reset_index(drop=True)
    st.markdown("### Leaderboard")
    st.dataframe(df, use_container_width=True, hide_index=True)

    ok = [(label, r) for label, r in results.items() if "error" not in r]
    if not ok:
        st.error("All classifiers failed to train. Check your dataset and installed backends.")
        st.stop()

    # ── CV accuracy bar chart (Figure 4 / 5 style) ──
    st.markdown("### Cross-validated accuracy")
    bar_df = pd.DataFrame([
        {"Model": label, "CV Mean": np.nanmean(r["cv"]), "CV Std": np.nanstd(r["cv"])}
        for label, r in ok
    ]).sort_values("CV Mean", ascending=False)
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=bar_df["Model"],
        y=bar_df["CV Mean"],
        error_y=dict(type="data", array=bar_df["CV Std"]),
        marker_color="rgba(99, 110, 250, 0.85)",
        text=[f"{v:.3f}" for v in bar_df["CV Mean"]],
        textposition="outside",
    ))
    fig_bar.update_layout(
        yaxis=dict(range=[0, 1.05], title="5-fold CV accuracy"),
        height=420, margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Decision-boundary grid (Figure 3 style) ──
    st.markdown("### Decision boundaries")
    cols_per_row = 3
    rows_iter = [ok[i:i + cols_per_row] for i in range(0, len(ok), cols_per_row)]
    for chunk in rows_iter:
        cols = st.columns(len(chunk))
        for col, (label, r) in zip(cols, chunk):
            with col:
                st.plotly_chart(
                    plot_decision_boundary(
                        r["clf"], X_train, X_test, y_train, y_test, scaler,
                        title=f"{label} (acc {r['acc_test']:.3f})",
                        feature_names=feature_names,
                        height=320, show_legend=False,
                    ),
                    use_container_width=True,
                )

    # ── Confusion-matrix grid (Figure 6 style) ──
    st.markdown("### Confusion matrices")
    for chunk in rows_iter:
        cols = st.columns(len(chunk))
        for col, (label, r) in zip(cols, chunk):
            with col:
                st.plotly_chart(
                    plot_confusion_matrix(
                        y_test, r["pred_test"],
                        title=f"{label} (acc {r['acc_test']:.3f})",
                        height=280,
                    ),
                    use_container_width=True,
                )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>ML Playground · "
    "Algorithms from scikit-learn, XGBoost, LightGBM, CatBoost, TabNet, neural-trees · "
    "Significance tests via mlxtend (Alpaydın 5×2cv *F*-test, McNemar, paired *t*) · "
    "Inspired by *Introduction to Machine Learning* (Alpaydın, MIT Press, 2020) · "
    "[neural-trees](https://github.com/cgrtml/neural-trees)</small>",
    unsafe_allow_html=True,
)
