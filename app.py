"""
ML Playground — Interactive Machine Learning Visualizer
========================================================
Explore any sklearn classifier on any dataset with live
decision boundary plots, training metrics, and model comparison.
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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

# neural-trees — optional, graceful fallback if not installed
try:
    from neural_trees import SoftDecisionTree
    NEURAL_TREES_AVAILABLE = True
except ImportError:
    NEURAL_TREES_AVAILABLE = False

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
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #333;
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

if NEURAL_TREES_AVAILABLE:
    CLASSIFIERS["⚡ Soft Decision Tree"] = "sdt"


def load_dataset(name: str, n_samples: int = 500, noise: float = 0.2, random_state: int = 42):
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
    elif name == "sdt" and NEURAL_TREES_AVAILABLE:
        return SoftDecisionTree(
            depth=params.get("depth", 4),
            max_epochs=params.get("max_epochs", 30),
            penalty_coef=params.get("penalty_coef", 1e-3),
        )
    raise ValueError(f"Unknown classifier: {name}")


def make_decision_boundary(clf, X, y, scaler, resolution=200):  # y unused, kept for signature clarity
    """Create a meshgrid and predict class probabilities for background coloring."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid)

    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(grid_scaled)
        Z = probs.argmax(axis=1)
    else:
        Z = clf.predict(grid_scaled)

    return xx, yy, Z.reshape(xx.shape)


def plot_decision_boundary(clf, X_train, X_test, y_train, y_test, scaler, title="", feature_names=("Feature 1", "Feature 2")):
    """Build a Plotly figure with decision boundary + train/test scatter."""
    classes = np.unique(np.concatenate([y_train, y_test]))
    n_classes = len(classes)

    # Color palettes
    bg_colors = [
        "rgba(99, 110, 250, 0.15)",
        "rgba(239, 85, 59, 0.15)",
        "rgba(0, 204, 150, 0.15)",
        "rgba(255, 161, 90, 0.15)",
    ]
    point_colors = ["#636EFA", "#EF553B", "#00CC96", "#FFA15A"]

    X_all = np.vstack([X_train, X_test])
    xx, yy, Z = make_decision_boundary(clf, X_all, None, scaler)

    fig = go.Figure()

    # Decision region contours
    for c in classes:
        mask = (Z == c)
        if mask.any():
            fig.add_trace(go.Contour(
                x=xx[0],
                y=yy[:, 0],
                z=(Z == c).astype(float),
                showscale=False,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, bg_colors[c % len(bg_colors)]]],
                contours=dict(start=0.5, end=1.5, size=1),
                hoverinfo="skip",
                name=f"Region {c}",
            ))

    # Training points
    for c in classes:
        mask = y_train == c
        fig.add_trace(go.Scatter(
            x=X_train[mask, 0],
            y=X_train[mask, 1],
            mode="markers",
            marker=dict(
                color=point_colors[c % len(point_colors)],
                size=7,
                symbol="circle",
                line=dict(width=1, color="white"),
            ),
            name=f"Train class {c}",
            legendgroup=f"class_{c}",
        ))

    # Test points (bigger, with black border)
    for c in classes:
        mask = y_test == c
        fig.add_trace(go.Scatter(
            x=X_test[mask, 0],
            y=X_test[mask, 1],
            mode="markers",
            marker=dict(
                color=point_colors[c % len(point_colors)],
                size=10,
                symbol="diamond",
                line=dict(width=2, color="black"),
            ),
            name=f"Test class {c}",
            legendgroup=f"class_{c}",
            showlegend=False,
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title=feature_names[0],
        yaxis_title=feature_names[1],
        height=480,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,249,250,1)",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
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
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
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
    mode = st.radio("Mode", ["🔬 Single Model", "⚔️ Compare Two Models"])

    st.markdown("---")

    # ── Model 1 ──
    st.markdown("### 🤖 Model A")
    clf_label_a = st.selectbox("Algorithm A", list(CLASSIFIERS.keys()), key="clf_a")
    clf_name_a = CLASSIFIERS[clf_label_a]
    params_a = {}

    with st.expander("Hyperparameters A"):
        if clf_name_a == "dt":
            params_a["max_depth"] = st.slider("Max depth", 1, 20, 5, key="dt_depth_a")
            params_a["min_samples_split"] = st.slider("Min samples split", 2, 20, 2, key="dt_mss_a")
        elif clf_name_a in ("rf", "gb"):
            params_a["n_estimators"] = st.slider("N estimators", 10, 300, 100, key="ne_a")
            params_a["max_depth"] = st.slider("Max depth", 1, 20, 5, key="md_a")
            if clf_name_a == "gb":
                params_a["learning_rate"] = st.slider("Learning rate", 0.01, 0.5, 0.1, key="lr_a")
        elif clf_name_a in ("svm", "lr"):
            params_a["C"] = st.slider("C (regularization)", 0.01, 10.0, 1.0, key="c_a")
        elif clf_name_a == "knn":
            params_a["n_neighbors"] = st.slider("K neighbors", 1, 30, 5, key="knn_a")
        elif clf_name_a == "mlp":
            params_a["hidden_layer_sizes"] = st.slider("Hidden units", 8, 256, 64, key="mlp_a")
        elif clf_name_a == "sdt":
            params_a["depth"] = st.slider("Tree depth", 2, 7, 4, key="sdt_depth_a")
            params_a["max_epochs"] = st.slider("Epochs", 10, 60, 30, key="sdt_epochs_a")
            params_a["penalty_coef"] = st.select_slider("Penalty", [1e-4, 1e-3, 1e-2], value=1e-3, key="sdt_pen_a")

    # ── Model 2 (compare mode) ──
    if mode == "⚔️ Compare Two Models":
        st.markdown("### 🤖 Model B")
        clf_label_b = st.selectbox("Algorithm B", list(CLASSIFIERS.keys()), index=4, key="clf_b")
        clf_name_b = CLASSIFIERS[clf_label_b]
        params_b = {}

        with st.expander("Hyperparameters B"):
            if clf_name_b == "dt":
                params_b["max_depth"] = st.slider("Max depth", 1, 20, 5, key="dt_depth_b")
                params_b["min_samples_split"] = st.slider("Min samples split", 2, 20, 2, key="dt_mss_b")
            elif clf_name_b in ("rf", "gb"):
                params_b["n_estimators"] = st.slider("N estimators", 10, 300, 100, key="ne_b")
                params_b["max_depth"] = st.slider("Max depth", 1, 20, 5, key="md_b")
                if clf_name_b == "gb":
                    params_b["learning_rate"] = st.slider("Learning rate", 0.01, 0.5, 0.1, key="lr_b")
            elif clf_name_b in ("svm", "lr"):
                params_b["C"] = st.slider("C (regularization)", 0.01, 10.0, 1.0, key="c_b")
            elif clf_name_b == "knn":
                params_b["n_neighbors"] = st.slider("K neighbors", 1, 30, 5, key="knn_b")
            elif clf_name_b == "mlp":
                params_b["hidden_layer_sizes"] = st.slider("Hidden units", 8, 256, 64, key="mlp_b")
            elif clf_name_b == "sdt":
                params_b["depth"] = st.slider("Tree depth", 2, 7, 4, key="sdt_depth_b")
                params_b["max_epochs"] = st.slider("Epochs", 10, 60, 30, key="sdt_epochs_b")
                params_b["penalty_coef"] = st.select_slider("Penalty", [1e-4, 1e-3, 1e-2], value=1e-3, key="sdt_pen_b")

    st.markdown("---")
    run_btn = st.button("▶ Run", use_container_width=True, type="primary")

# ── Main area ─────────────────────────────────────────────────────────────────

st.markdown('<h1 class="main-header">ML Playground</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Pick an algorithm, pick a dataset, see what happens.</p>', unsafe_allow_html=True)

if not run_btn:
    st.markdown("""
    Pick a dataset from the sidebar, pick an algorithm, hit **Run**.

    The colored regions show where each model draws the line between classes.
    Train points are circles, test points are diamonds.
    If the boundary looks perfect on training data but messy on test points, that's overfitting.

    Switch to **Compare** mode to run two algorithms on the exact same split.

    Supported algorithms: Decision Tree · Random Forest · Gradient Boosting · SVM · KNN · Logistic Regression · MLP · Naive Bayes
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

# ── Train models ──────────────────────────────────────────────────────────────
clf_a = build_classifier(clf_name_a, params_a)
clf_a.fit(X_train_s, y_train)
pred_a_train = clf_a.predict(X_train_s)
pred_a_test = clf_a.predict(X_test_s)
acc_a_train = accuracy_score(y_train, pred_a_train)
acc_a_test = accuracy_score(y_test, pred_a_test)
cv_a = cross_val_score(clf_a, X_train_s, y_train, cv=5)

if mode == "⚔️ Compare Two Models":
    clf_b = build_classifier(clf_name_b, params_b)
    clf_b.fit(X_train_s, y_train)
    pred_b_train = clf_b.predict(X_train_s)
    pred_b_test = clf_b.predict(X_test_s)
    acc_b_train = accuracy_score(y_train, pred_b_train)
    acc_b_test = accuracy_score(y_test, pred_b_test)
    cv_b = cross_val_score(clf_b, X_train_s, y_train, cv=5)

# ── Display ───────────────────────────────────────────────────────────────────

if mode == "🔬 Single Model":
    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train Accuracy", f"{acc_a_train:.3f}")
    c2.metric("Test Accuracy", f"{acc_a_test:.3f}",
              delta=f"{acc_a_test - acc_a_train:.3f} vs train")
    c3.metric("CV Mean ± Std", f"{cv_a.mean():.3f} ± {cv_a.std():.3f}")
    c4.metric("Dataset size", f"{len(X)} samples / {len(np.unique(y))} classes")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.plotly_chart(
            plot_decision_boundary(
                clf_a, X_train, X_test, y_train, y_test, scaler,
                title=f"{clf_label_a} — Decision Boundary",
                feature_names=feature_names,
            ),
            use_container_width=True,
        )

    with col_right:
        st.plotly_chart(
            plot_confusion_matrix(y_test, pred_a_test, "Confusion Matrix (Test Set)"),
            use_container_width=True,
        )
        st.plotly_chart(
            plot_cv_scores(cv_a, f"5-Fold CV Accuracy"),
            use_container_width=True,
        )

else:
    # ── Compare mode ──
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"#### Model A: {clf_label_a}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Train Acc", f"{acc_a_train:.3f}")
        m2.metric("Test Acc", f"{acc_a_test:.3f}")
        m3.metric("CV Mean", f"{cv_a.mean():.3f}")

        st.plotly_chart(
            plot_decision_boundary(
                clf_a, X_train, X_test, y_train, y_test, scaler,
                title="Model A — Decision Boundary",
                feature_names=feature_names,
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            plot_confusion_matrix(y_test, pred_a_test, "Model A — Confusion Matrix"),
            use_container_width=True,
        )

    with col_b:
        st.markdown(f"#### Model B: {clf_label_b}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Train Acc", f"{acc_b_train:.3f}")
        m2.metric("Test Acc", f"{acc_b_test:.3f}")
        m3.metric("CV Mean", f"{cv_b.mean():.3f}")

        st.plotly_chart(
            plot_decision_boundary(
                clf_b, X_train, X_test, y_train, y_test, scaler,
                title="Model B — Decision Boundary",
                feature_names=feature_names,
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            plot_confusion_matrix(y_test, pred_b_test, "Model B — Confusion Matrix"),
            use_container_width=True,
        )

    # ── Head-to-head comparison ──
    st.markdown("---")
    st.markdown("### Head-to-Head Comparison")

    winner = clf_label_a if acc_a_test >= acc_b_test else clf_label_b
    diff = abs(acc_a_test - acc_b_test)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Winner (Test Acc)", winner)
    c2.metric("Accuracy difference", f"{diff:.3f}")
    c3.metric("A CV vs B CV", f"{cv_a.mean():.3f} vs {cv_b.mean():.3f}")
    c4.metric("A overfit vs B overfit",
              f"{acc_a_train - acc_a_test:.3f} vs {acc_b_train - acc_b_test:.3f}")

    # Bar chart comparison
    comparison_data = {
        "Metric": ["Train Accuracy", "Test Accuracy", "CV Mean"],
        clf_label_a: [acc_a_train, acc_a_test, cv_a.mean()],
        clf_label_b: [acc_b_train, acc_b_test, cv_b.mean()],
    }
    df_comp = pd.DataFrame(comparison_data)
    fig_comp = px.bar(
        df_comp.melt(id_vars="Metric", var_name="Model", value_name="Score"),
        x="Metric", y="Score", color="Model", barmode="group",
        color_discrete_sequence=["#636EFA", "#EF553B"],
        title="Model Comparison",
        range_y=[0, 1.05],
    )
    fig_comp.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_comp, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>ML Playground · "
    "Algorithms from scikit-learn · "
    "Inspired by *Introduction to Machine Learning* (Alpaydın, MIT Press, 2020) · "
    "[neural-trees](https://github.com/cgrtml/neural-trees)</small>",
    unsafe_allow_html=True,
)
