"""
ML Playground — Interactive Machine Learning Visualizer
========================================================
Explore any classifier on any dataset with live decision boundary plots,
confusion matrices, head-to-head comparisons, multi-model benchmarks,
and statistical significance tests. Available in 6 languages.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from sklearn.datasets import (
    make_moons, make_circles, make_classification,
    load_iris, load_wine, load_breast_cancer,
)
from sklearn.model_selection import train_test_split, cross_val_score
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

from translations import T, LANGUAGES, RTL_LANGS, t

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

# Language must be picked BEFORE the rest of the UI is rendered, so we
# pull it from session state as soon as possible.
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"


def apply_theme(lang: str) -> None:
    """Inject CSS — gradient header, info-card styling, and RTL when needed."""
    direction = "rtl" if lang in RTL_LANGS else "ltr"
    text_align = "right" if direction == "rtl" else "left"
    st.markdown(f"""
    <style>
        .stApp {{ direction: {direction}; }}
        .main-header {{
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
            text-align: {text_align};
        }}
        .sub-header {{
            color: #888;
            font-size: 1rem;
            margin-top: 0;
            text-align: {text_align};
        }}
        .stMetric {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 0.5rem;
        }}
        .info-card {{
            background: #f1f3f7;
            border-left: 4px solid #667eea;
            padding: 0.9rem 1.2rem;
            border-radius: 6px;
            margin: 0.4rem 0 1rem 0;
            font-size: 0.95rem;
            line-height: 1.45;
        }}
    </style>
    """, unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

# Internal IDs are stable; their human-readable labels are looked up via t().
DATASET_IDS = ["moons", "circles", "blobs2", "blobs3", "iris", "wine", "cancer"]

# Classical sklearn zoo — always available
CLASSIFIER_IDS = ["dt", "rf", "gb", "svm", "knn", "lr", "mlp", "nb"]
if XGBOOST_AVAILABLE:
    CLASSIFIER_IDS.append("xgb")
if LIGHTGBM_AVAILABLE:
    CLASSIFIER_IDS.append("lgbm")
if CATBOOST_AVAILABLE:
    CLASSIFIER_IDS.append("cat")
if TABNET_AVAILABLE:
    CLASSIFIER_IDS.append("tabnet")
if NEURAL_TREES_AVAILABLE:
    CLASSIFIER_IDS.append("sdt")


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
        feature_names = tuple(n.replace(" (cm)", "").title() for n in data.feature_names[:2])
    elif name == "wine":
        data = load_wine()
        X, y = data.data[:, :2], data.target
        feature_names = tuple(n.replace("_", " ").title() for n in data.feature_names[:2])
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
        return SVC(C=params.get("C", 1.0), gamma=params.get("gamma", "scale"),
                   probability=True, random_state=42)
    elif name == "knn":
        return KNeighborsClassifier(n_neighbors=params.get("n_neighbors", 5))
    elif name == "lr":
        return LogisticRegression(C=params.get("C", 1.0), max_iter=1000, random_state=42)
    elif name == "mlp":
        hidden = params.get("hidden_layer_sizes", 64)
        return MLPClassifier(hidden_layer_sizes=(hidden, hidden // 2),
                             max_iter=500, random_state=42)
    elif name == "nb":
        return GaussianNB()
    elif name == "xgb" and XGBOOST_AVAILABLE:
        return XGBClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.1),
            random_state=42, eval_metric="mlogloss", verbosity=0, n_jobs=1,
        )
    elif name == "lgbm" and LIGHTGBM_AVAILABLE:
        return LGBMClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.1),
            random_state=42, n_jobs=1, num_threads=1, force_row_wise=True, verbose=-1,
        )
    elif name == "cat" and CATBOOST_AVAILABLE:
        return CatBoostClassifier(
            iterations=params.get("n_estimators", 200),
            depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.1),
            random_state=42, verbose=0, allow_writing_files=False,
        )
    elif name == "tabnet" and TABNET_AVAILABLE:
        return _TabNetSklearn(
            n_d=params.get("n_d", 8), n_a=params.get("n_a", 8),
            n_steps=params.get("n_steps", 3), max_epochs=params.get("max_epochs", 30),
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
            self._model = TabNetClassifier(n_d=self.n_d, n_a=self.n_a,
                                           n_steps=self.n_steps, seed=42, verbose=0)
            self.classes_ = np.unique(y)
            X32 = np.asarray(X, dtype=np.float32)
            y_int = np.asarray(y).astype(np.int64)
            self._model.fit(X32, y_int, max_epochs=self.max_epochs, patience=10,
                            batch_size=min(256, len(X32)),
                            virtual_batch_size=min(64, len(X32)), drop_last=False)
            return self

        def predict(self, X):
            return self._model.predict(np.asarray(X, dtype=np.float32))

        def predict_proba(self, X):
            return self._model.predict_proba(np.asarray(X, dtype=np.float32))


def make_decision_boundary(clf, X, scaler, resolution=160):
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
            Z = clf.predict_proba(grid_scaled).argmax(axis=1)
        except Exception:
            Z = clf.predict(grid_scaled)
    else:
        Z = clf.predict(grid_scaled)
    return xx, yy, Z.reshape(xx.shape)


def plot_decision_boundary(clf, X_train, X_test, y_train, y_test, scaler,
                           title="", feature_names=("Feature 1", "Feature 2"),
                           height=480, show_legend=True):
    classes = np.unique(np.concatenate([y_train, y_test]))
    bg_colors = ["rgba(99, 110, 250, 0.15)", "rgba(239, 85, 59, 0.15)",
                 "rgba(0, 204, 150, 0.15)", "rgba(255, 161, 90, 0.15)"]
    point_colors = ["#636EFA", "#EF553B", "#00CC96", "#FFA15A"]

    X_all = np.vstack([X_train, X_test])
    xx, yy, Z = make_decision_boundary(clf, X_all, scaler)

    fig = go.Figure()
    for c in classes:
        fig.add_trace(go.Contour(
            x=xx[0], y=yy[:, 0], z=(Z == c).astype(float),
            showscale=False,
            colorscale=[[0, "rgba(0,0,0,0)"], [1, bg_colors[int(c) % len(bg_colors)]]],
            contours=dict(start=0.5, end=1.5, size=1),
            hoverinfo="skip", showlegend=False,
        ))
    for c in classes:
        mask = y_train == c
        fig.add_trace(go.Scatter(
            x=X_train[mask, 0], y=X_train[mask, 1], mode="markers",
            marker=dict(color=point_colors[int(c) % len(point_colors)], size=7,
                        symbol="circle", line=dict(width=1, color="white")),
            name=f"Train class {c}", legendgroup=f"class_{c}", showlegend=show_legend,
        ))
    for c in classes:
        mask = y_test == c
        fig.add_trace(go.Scatter(
            x=X_test[mask, 0], y=X_test[mask, 1], mode="markers",
            marker=dict(color=point_colors[int(c) % len(point_colors)], size=10,
                        symbol="diamond", line=dict(width=2, color="black")),
            name=f"Test class {c}", legendgroup=f"class_{c}", showlegend=False,
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title=feature_names[0], yaxis_title=feature_names[1],
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,249,250,1)",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", height=350):
    cm = confusion_matrix(y_true, y_pred)
    labels = np.unique(y_true)
    fig = px.imshow(
        cm, labels=dict(x="Predicted", y="Actual", color="Count"),
        x=[str(l) for l in labels], y=[str(l) for l in labels],
        color_continuous_scale="Blues", title=title, text_auto=True,
    )
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def plot_cv_scores(scores, title):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"Fold {i+1}" for i in range(len(scores))], y=scores,
        marker_color="rgba(99, 110, 250, 0.8)",
        text=[f"{s:.3f}" for s in scores], textposition="outside",
    ))
    fig.add_hline(y=scores.mean(), line_dash="dash", line_color="red",
                  annotation_text=f"μ = {scores.mean():.3f}",
                  annotation_position="right")
    fig.update_layout(title=title, yaxis=dict(range=[0, 1.05]),
                      height=300, margin=dict(l=20, r=20, t=40, b=20),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def hyperparam_ui(clf_name: str, suffix: str) -> dict:
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


def info_card(text: str) -> None:
    """Render an info-card (used for dataset/mode/model explanations)."""
    st.markdown(f'<div class="info-card">{text}</div>', unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    # Language selector — at the very top so the rest of the sidebar updates
    lang_codes = list(LANGUAGES.keys())
    lang_idx = lang_codes.index(st.session_state["lang"])
    lang_label = st.selectbox(
        "🌍 Language / Dil / Idioma / Lingua / اللغة / Язык",
        options=lang_codes, index=lang_idx,
        format_func=lambda c: LANGUAGES[c],
    )
    st.session_state["lang"] = lang_label
    L = lang_label  # short alias used everywhere below

    st.markdown("---")

    # ── Dataset ──
    st.markdown(f"### {t('sidebar.dataset.title', L)}")
    ds_label_map = {ds_id: t(f"ds.{ds_id}.label", L) for ds_id in DATASET_IDS}
    dataset_id = st.selectbox(
        t("sidebar.dataset.choose", L),
        options=DATASET_IDS,
        format_func=lambda x: ds_label_map[x],
        key="dataset_id",
    )
    with st.expander(t("sidebar.about_dataset", L)):
        info_card(t(f"ds.{dataset_id}.desc", L))

    if dataset_id in ("moons", "circles", "blobs2", "blobs3"):
        n_samples = st.slider(t("sidebar.dataset.samples", L), 100, 1000, 400, step=50)
        noise = st.slider(t("sidebar.dataset.noise", L), 0.0, 0.5, 0.2, step=0.05)
    else:
        n_samples, noise = 500, 0.2

    test_size = st.slider(t("sidebar.dataset.test_size", L), 0.1, 0.4, 0.25, step=0.05)
    random_state = st.number_input(t("sidebar.dataset.seed", L), value=42, step=1)

    st.markdown("---")

    # ── Mode ──
    st.markdown(f"### {t('sidebar.mode.title', L)}")
    mode_options = [
        ("single",    t("sidebar.mode.single", L)),
        ("compare",   t("sidebar.mode.compare", L)),
        ("benchmark", t("sidebar.mode.benchmark", L)),
    ]
    mode_id = st.radio(
        t("sidebar.mode.title", L),
        options=[m[0] for m in mode_options],
        format_func=lambda m: dict(mode_options)[m],
        label_visibility="collapsed",
        key="mode_id",
    )
    with st.expander(t("sidebar.about_mode", L)):
        info_card(t(f"mode.{mode_id}.desc", L))

    st.markdown("---")

    params_a = params_b = {}
    clf_id_a = clf_id_b = None
    clf_label_a = clf_label_b = None
    clf_label_map = {cid: t(f"mdl.{cid}.label", L) for cid in CLASSIFIER_IDS}

    if mode_id in ("single", "compare"):
        st.markdown(f"### {t('sidebar.model_a', L)}")
        clf_id_a = st.selectbox(
            t("sidebar.algorithm_a", L),
            options=CLASSIFIER_IDS,
            format_func=lambda c: clf_label_map[c],
            key="clf_a",
        )
        clf_label_a = clf_label_map[clf_id_a]
        with st.expander(t("sidebar.about_model", L)):
            info_card(t(f"mdl.{clf_id_a}.desc", L))
        with st.expander(t("sidebar.hyperparameters_a", L)):
            params_a = hyperparam_ui(clf_id_a, "a")

    if mode_id == "compare":
        st.markdown(f"### {t('sidebar.model_b', L)}")
        b_default_idx = min(4, len(CLASSIFIER_IDS) - 1)
        clf_id_b = st.selectbox(
            t("sidebar.algorithm_b", L),
            options=CLASSIFIER_IDS, index=b_default_idx,
            format_func=lambda c: clf_label_map[c],
            key="clf_b",
        )
        clf_label_b = clf_label_map[clf_id_b]
        with st.expander(t("sidebar.about_model", L)):
            info_card(t(f"mdl.{clf_id_b}.desc", L))
        with st.expander(t("sidebar.hyperparameters_b", L)):
            params_b = hyperparam_ui(clf_id_b, "b")

    if mode_id == "benchmark":
        st.info(t("sidebar.benchmark_info", L, n=len(CLASSIFIER_IDS)))

    st.markdown("---")
    run_btn = st.button(t("sidebar.run", L), use_container_width=True, type="primary")

    with st.expander(t("sidebar.installed_backends", L)):
        st.write(
            f"- XGBoost: {'✅' if XGBOOST_AVAILABLE else '❌'}\n"
            f"- LightGBM: {'✅' if LIGHTGBM_AVAILABLE else '❌'}\n"
            f"- CatBoost: {'✅' if CATBOOST_AVAILABLE else '❌'}\n"
            f"- TabNet: {'✅' if TABNET_AVAILABLE else '❌'}\n"
            f"- neural-trees (SDT): {'✅' if NEURAL_TREES_AVAILABLE else '❌'}\n"
            f"- mlxtend: {'✅' if MLXTEND_AVAILABLE else '❌'}"
        )

# Apply CSS / RTL after language is resolved
apply_theme(L)

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown(f'<h1 class="main-header">{t("app.title", L)}</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-header">{t("app.subtitle", L)}</p>', unsafe_allow_html=True)

if not run_btn:
    st.markdown(f"### {t('landing.title', L)}")
    st.markdown(t("landing.body", L))
    st.stop()

# ── Data loading & preprocessing ──────────────────────────────────────────────
X, y, feature_names = load_dataset(dataset_id, n_samples, noise, int(random_state))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(random_state), stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


def fit_and_score(clf_name, params):
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
        cv = cross_val_score(build_classifier(clf_name, params),
                             X_train_s, y_train, cv=5)
    except Exception:
        cv = np.full(5, np.nan)
    out["cv"] = cv
    return out


# ── Single Model ──────────────────────────────────────────────────────────────

if mode_id == "single":
    with st.spinner(t("single.spinner", L, model=clf_label_a)):
        a = fit_and_score(clf_id_a, params_a)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("single.train_acc", L), f"{a['acc_train']:.3f}")
    c2.metric(t("single.test_acc", L), f"{a['acc_test']:.3f}",
              delta=t("single.delta_train", L, d=a['acc_test'] - a['acc_train']))
    c3.metric(t("single.cv_mean_std", L), f"{a['cv'].mean():.3f} ± {a['cv'].std():.3f}")
    c4.metric(t("single.dataset_size", L),
              t("single.dataset_size_value", L, n=len(X), k=len(np.unique(y))))

    st.markdown("---")
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.plotly_chart(
            plot_decision_boundary(
                a["clf"], X_train, X_test, y_train, y_test, scaler,
                title=t("single.boundary_title", L, model=clf_label_a),
                feature_names=feature_names,
            ),
            use_container_width=True,
        )
    with col_right:
        st.plotly_chart(
            plot_confusion_matrix(y_test, a["pred_test"], t("single.cm_title", L)),
            use_container_width=True,
        )
        st.plotly_chart(
            plot_cv_scores(a["cv"], t("single.cv_title", L)),
            use_container_width=True,
        )

# ── Compare Two Models ────────────────────────────────────────────────────────

elif mode_id == "compare":
    with st.spinner(t("compare.spinner", L, a=clf_label_a, b=clf_label_b)):
        a = fit_and_score(clf_id_a, params_a)
        b = fit_and_score(clf_id_b, params_b)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"#### {t('compare.model_a_panel', L, label=clf_label_a)}")
        m1, m2, m3 = st.columns(3)
        m1.metric(t("compare.train_acc", L), f"{a['acc_train']:.3f}")
        m2.metric(t("compare.test_acc", L), f"{a['acc_test']:.3f}")
        m3.metric(t("compare.cv_mean", L), f"{a['cv'].mean():.3f}")
        st.plotly_chart(
            plot_decision_boundary(
                a["clf"], X_train, X_test, y_train, y_test, scaler,
                title=t("compare.boundary_a", L), feature_names=feature_names,
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            plot_confusion_matrix(y_test, a["pred_test"], t("compare.cm_a", L)),
            use_container_width=True,
        )

    with col_b:
        st.markdown(f"#### {t('compare.model_b_panel', L, label=clf_label_b)}")
        m1, m2, m3 = st.columns(3)
        m1.metric(t("compare.train_acc", L), f"{b['acc_train']:.3f}")
        m2.metric(t("compare.test_acc", L), f"{b['acc_test']:.3f}")
        m3.metric(t("compare.cv_mean", L), f"{b['cv'].mean():.3f}")
        st.plotly_chart(
            plot_decision_boundary(
                b["clf"], X_train, X_test, y_train, y_test, scaler,
                title=t("compare.boundary_b", L), feature_names=feature_names,
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            plot_confusion_matrix(y_test, b["pred_test"], t("compare.cm_b", L)),
            use_container_width=True,
        )

    # ── Head-to-head summary ──
    st.markdown("---")
    st.markdown(f"### {t('compare.head_to_head', L)}")
    winner = clf_label_a if a["acc_test"] >= b["acc_test"] else clf_label_b
    diff = abs(a["acc_test"] - b["acc_test"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("compare.winner", L), winner)
    c2.metric(t("compare.acc_diff", L), f"{diff:.3f}")
    c3.metric(t("compare.cv_compare", L),
              f"{a['cv'].mean():.3f} vs {b['cv'].mean():.3f}")
    c4.metric(t("compare.overfit_compare", L),
              f"{a['acc_train'] - a['acc_test']:.3f} vs {b['acc_train'] - b['acc_test']:.3f}")

    df_comp = pd.DataFrame({
        t("compare.metric", L): [
            t("compare.metric.train_acc", L),
            t("compare.metric.test_acc", L),
            t("compare.metric.cv_mean", L),
        ],
        clf_label_a: [a["acc_train"], a["acc_test"], a["cv"].mean()],
        clf_label_b: [b["acc_train"], b["acc_test"], b["cv"].mean()],
    })
    fig_comp = px.bar(
        df_comp.melt(id_vars=t("compare.metric", L),
                     var_name=t("compare.model", L),
                     value_name=t("compare.score", L)),
        x=t("compare.metric", L),
        y=t("compare.score", L),
        color=t("compare.model", L), barmode="group",
        color_discrete_sequence=["#636EFA", "#EF553B"],
        title=t("compare.bar_title", L), range_y=[0, 1.05],
    )
    fig_comp.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_comp, use_container_width=True)

    # ── Statistical significance tests ──
    st.markdown("---")
    st.markdown(f"### {t('sig.title', L)}")
    if not MLXTEND_AVAILABLE:
        st.warning(t("sig.unavailable", L))
    else:
        with st.spinner(t("sig.spinner", L)):
            try:
                est_a = build_classifier(clf_id_a, params_a)
                est_b = build_classifier(clf_id_b, params_b)
                t_stat, t_p = paired_ttest_5x2cv(
                    estimator1=est_a, estimator2=est_b,
                    X=X_train_s, y=y_train, random_seed=int(random_state),
                )
                f_stat, f_p = combined_ftest_5x2cv(
                    estimator1=build_classifier(clf_id_a, params_a),
                    estimator2=build_classifier(clf_id_b, params_b),
                    X=X_train_s, y=y_train, random_seed=int(random_state),
                )
                tb = mcnemar_table(y_target=y_test,
                                   y_model1=a["pred_test"], y_model2=b["pred_test"])
                chi2, mc_p = mcnemar(ary=tb, corrected=True)

                alpha = 0.05
                rows = [
                    (t("sig.t_test", L), f"{t_stat:.3f}", f"{t_p:.4f}",
                     t("sig.reject", L) if t_p < alpha else t("sig.keep", L)),
                    (t("sig.f_test", L), f"{f_stat:.3f}", f"{f_p:.4f}",
                     t("sig.reject", L) if f_p < alpha else t("sig.keep", L)),
                    (t("sig.mcnemar", L), f"{chi2:.3f}", f"{mc_p:.4f}",
                     t("sig.reject", L) if mc_p < alpha else t("sig.keep", L)),
                ]
                df_stats = pd.DataFrame(rows, columns=[
                    t("sig.col_test", L), t("sig.col_stat", L),
                    t("sig.col_p", L), t("sig.col_decision", L),
                ])
                st.table(df_stats)
                st.caption(t("sig.h0_caption", L))
            except Exception as e:
                st.error(t("sig.error", L, err=str(e)))

# ── Benchmark All Models ──────────────────────────────────────────────────────

else:
    items = [(cid, clf_label_map[cid]) for cid in CLASSIFIER_IDS]
    n = len(items)
    progress = st.progress(0.0, text=t("bench.spinner_init", L))
    results = {}
    for i, (cid, label) in enumerate(items, start=1):
        try:
            results[label] = fit_and_score(cid, default_params(cid))
        except Exception as e:
            results[label] = {"error": str(e)}
        progress.progress(i / n, text=t("bench.spinner_step", L, label=label, i=i, n=n))
    progress.empty()

    rows = []
    for label, r in results.items():
        if "error" in r:
            rows.append({
                t("bench.col_model", L): label,
                t("bench.col_train_acc", L): np.nan,
                t("bench.col_test_acc", L): np.nan,
                t("bench.col_cv_mean", L): np.nan,
                t("bench.col_cv_std", L): np.nan,
                t("bench.col_status", L): f"❌ {r['error']}",
            })
        else:
            rows.append({
                t("bench.col_model", L): label,
                t("bench.col_train_acc", L): round(r["acc_train"], 4),
                t("bench.col_test_acc", L): round(r["acc_test"], 4),
                t("bench.col_cv_mean", L): round(np.nanmean(r["cv"]), 4),
                t("bench.col_cv_std", L): round(np.nanstd(r["cv"]), 4),
                t("bench.col_status", L): "✅",
            })
    df = (pd.DataFrame(rows)
            .sort_values(t("bench.col_test_acc", L), ascending=False)
            .reset_index(drop=True))
    st.markdown(f"### {t('bench.leaderboard', L)}")
    st.dataframe(df, use_container_width=True, hide_index=True)

    ok = [(label, r) for label, r in results.items() if "error" not in r]
    if not ok:
        st.error(t("bench.all_failed", L))
        st.stop()

    st.markdown(f"### {t('bench.cv_chart', L)}")
    bar_df = pd.DataFrame([
        {"Model": label, "CV Mean": np.nanmean(r["cv"]), "CV Std": np.nanstd(r["cv"])}
        for label, r in ok
    ]).sort_values("CV Mean", ascending=False)
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=bar_df["Model"], y=bar_df["CV Mean"],
        error_y=dict(type="data", array=bar_df["CV Std"]),
        marker_color="rgba(99, 110, 250, 0.85)",
        text=[f"{v:.3f}" for v in bar_df["CV Mean"]],
        textposition="outside",
    ))
    fig_bar.update_layout(
        yaxis=dict(range=[0, 1.05], title=t("bench.cv_axis", L)),
        height=420, margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown(f"### {t('bench.boundaries', L)}")
    cols_per_row = 3
    rows_iter = [ok[i:i + cols_per_row] for i in range(0, len(ok), cols_per_row)]
    for chunk in rows_iter:
        cols = st.columns(len(chunk))
        for col, (label, r) in zip(cols, chunk):
            with col:
                st.plotly_chart(
                    plot_decision_boundary(
                        r["clf"], X_train, X_test, y_train, y_test, scaler,
                        title=t("bench.panel_label", L, label=label, acc=r["acc_test"]),
                        feature_names=feature_names,
                        height=320, show_legend=False,
                    ),
                    use_container_width=True,
                )

    st.markdown(f"### {t('bench.confusions', L)}")
    for chunk in rows_iter:
        cols = st.columns(len(chunk))
        for col, (label, r) in zip(cols, chunk):
            with col:
                st.plotly_chart(
                    plot_confusion_matrix(
                        y_test, r["pred_test"],
                        title=t("bench.panel_label", L, label=label, acc=r["acc_test"]),
                        height=280,
                    ),
                    use_container_width=True,
                )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<small>{t('footer.text', L)} · "
    "<a href='https://github.com/cgrtml/neural-trees' target='_blank'>neural-trees</a></small>",
    unsafe_allow_html=True,
)
