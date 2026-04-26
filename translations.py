"""
Translations for ML Playground.

Each language is a flat dict of key -> string. Missing keys fall back to
English. Keys use dot.separated.namespacing only as a convention; the
lookup is exact-match.
"""

from typing import Dict

LANGUAGES: Dict[str, str] = {
    "en": "English",
    "tr": "Türkçe",
    "es": "Español",
    "it": "Italiano",
    "ar": "العربية",
    "ru": "Русский",
}

# Right-to-left languages — UI is mirrored for these.
RTL_LANGS = {"ar"}


T: Dict[str, Dict[str, str]] = {}

# ─────────────────────────────────────────────────────────────────────────────
# English (canonical)
# ─────────────────────────────────────────────────────────────────────────────
T["en"] = {
    # — App header —
    "app.title": "ML Playground",
    "app.subtitle": "Pick an algorithm, pick a dataset, see what happens. Compare two models head to head, or benchmark every classifier at once with the same evaluation protocol used in the neural-trees paper.",

    # — Landing —
    "landing.title": "Welcome",
    "landing.body": (
        "This playground lets you train classical machine-learning models on "
        "well-known datasets, see exactly **where each model draws the line "
        "between classes**, and compare models head-to-head with statistical "
        "rigor.\n\n"
        "**How to use it:**\n"
        "1. **Pick a dataset** in the sidebar (each one has an info panel).\n"
        "2. **Pick a mode** — single model, two-model duel, or full benchmark.\n"
        "3. **Pick the model(s)**, optionally tune their hyperparameters.\n"
        "4. Hit **▶ Run**.\n\n"
        "Train points are circles, test points are diamonds. Colored regions "
        "are class predictions on a fine grid."
    ),

    # — Sidebar generic —
    "sidebar.language": "🌍 Language",
    "sidebar.run": "▶ Run",
    "sidebar.about_dataset": "ℹ About this dataset",
    "sidebar.about_mode": "ℹ About this mode",
    "sidebar.about_model": "ℹ About this model",
    "sidebar.installed_backends": "Installed backends",

    # — Sidebar dataset —
    "sidebar.dataset.title": "📊 Dataset",
    "sidebar.dataset.choose": "Choose dataset",
    "sidebar.dataset.samples": "Samples",
    "sidebar.dataset.noise": "Noise",
    "sidebar.dataset.test_size": "Test split",
    "sidebar.dataset.seed": "Random seed",

    # — Sidebar mode —
    "sidebar.mode.title": "🎯 Mode",
    "sidebar.mode.single": "🔬 Single Model",
    "sidebar.mode.compare": "⚔️ Compare Two Models",
    "sidebar.mode.benchmark": "🏆 Benchmark All Models",

    # — Sidebar model —
    "sidebar.model_a": "🤖 Model A",
    "sidebar.model_b": "🤖 Model B",
    "sidebar.algorithm_a": "Algorithm A",
    "sidebar.algorithm_b": "Algorithm B",
    "sidebar.hyperparameters_a": "Hyperparameters A",
    "sidebar.hyperparameters_b": "Hyperparameters B",
    "sidebar.benchmark_info": "Will train **{n}** classifiers with default hyperparameters on the chosen dataset and render a side-by-side grid.",

    # — Mode descriptions —
    "mode.single.title": "Single Model",
    "mode.single.desc": (
        "Train one classifier on one dataset and inspect its decision "
        "boundary, confusion matrix, and 5-fold cross-validated accuracy. "
        "Best for understanding **how a single algorithm behaves** before "
        "comparing it to others."
    ),
    "mode.compare.title": "Compare Two Models",
    "mode.compare.desc": (
        "Train two classifiers on the **same** train/test split and run "
        "three statistical tests on top: paired *t*-test, McNemar's test, "
        "and Alpaydın's 5×2cv *F*-test. Use this to answer 'is model A "
        "really better than model B, or did it just get lucky?'"
    ),
    "mode.benchmark.title": "Benchmark All Models",
    "mode.benchmark.desc": (
        "Train every available classifier with sensible default hyperparameters "
        "and render a leaderboard, a CV-accuracy bar chart, a decision-boundary "
        "grid, and a confusion-matrix grid. Use this for a quick sanity check "
        "of how the entire model zoo performs on your dataset."
    ),

    # — Datasets —
    "ds.moons.label": "🌙 Moons",
    "ds.moons.desc": (
        "Two interleaving half-moons. Classic non-linear separation test — "
        "linear models (logistic regression) fail; tree-based and kernel "
        "models succeed."
    ),
    "ds.circles.label": "⭕ Circles",
    "ds.circles.desc": (
        "Two concentric circles. Designed to be impossible for any linear "
        "boundary; great for showcasing kernel SVMs and tree ensembles."
    ),
    "ds.blobs2.label": "🔵 Blobs (2 classes)",
    "ds.blobs2.desc": (
        "Two well-separated Gaussian clusters. Almost any classifier solves "
        "this; useful as a sanity check that your pipeline isn't broken."
    ),
    "ds.blobs3.label": "🔵 Blobs (3 classes)",
    "ds.blobs3.desc": (
        "Three Gaussian clusters — multi-class baseline. Tests whether a "
        "model handles more than two classes correctly."
    ),
    "ds.iris.label": "🌸 Iris",
    "ds.iris.desc": (
        "150 iris flowers, 3 species (setosa, versicolor, virginica). The "
        "first two features (sepal length & width) are shown; setosa is "
        "linearly separable, the other two overlap."
    ),
    "ds.wine.label": "🍷 Wine",
    "ds.wine.desc": (
        "178 Italian wines from 3 cultivars, 13 chemical features. Only the "
        "first two features are visualized here; the full feature set is "
        "what the benchmark CV scores actually use."
    ),
    "ds.cancer.label": "🔬 Breast Cancer",
    "ds.cancer.desc": (
        "569 breast-mass samples (malignant vs benign) with 30 features. "
        "Reduced to 2D via PCA for visualization; the underlying classifier "
        "still uses the original 30 features."
    ),

    # — Models —
    "mdl.dt.label": "Decision Tree",
    "mdl.dt.desc": (
        "Splits the feature space with a sequence of axis-aligned if/else "
        "rules. Easy to read, prone to overfit if you let it grow deep."
    ),
    "mdl.rf.label": "Random Forest",
    "mdl.rf.desc": (
        "Bags many decorrelated decision trees and averages their votes. "
        "Robust default — works well on most tabular datasets out of the box."
    ),
    "mdl.gb.label": "Gradient Boosting",
    "mdl.gb.desc": (
        "Builds shallow trees sequentially, each one correcting the previous "
        "one's errors. Strong on tabular data; slower to train than Random Forest."
    ),
    "mdl.svm.label": "SVM (RBF)",
    "mdl.svm.desc": (
        "Support Vector Machine with a radial basis function kernel. Maximum-"
        "margin classifier in an implicit infinite-dimensional space — handles "
        "non-linear boundaries well, scales poorly to >100k samples."
    ),
    "mdl.knn.label": "K-Nearest Neighbors",
    "mdl.knn.desc": (
        "Predicts the majority class among the *k* closest training points. "
        "No training phase, but every prediction scans the whole dataset."
    ),
    "mdl.lr.label": "Logistic Regression",
    "mdl.lr.desc": (
        "Linear model that outputs class probabilities via the sigmoid. The "
        "simplest serious baseline — fast, interpretable, but limited to "
        "linearly separable problems."
    ),
    "mdl.mlp.label": "Neural Network (MLP)",
    "mdl.mlp.desc": (
        "Multi-layer perceptron — a small fully-connected neural network. "
        "Universal approximator, but needs more data and tuning than tree "
        "ensembles to shine on tabular data."
    ),
    "mdl.nb.label": "Naive Bayes",
    "mdl.nb.desc": (
        "Applies Bayes' rule under the (often wrong) assumption that features "
        "are conditionally independent given the class. Surprisingly strong "
        "on text and high-dimensional sparse data."
    ),
    "mdl.xgb.label": "⚡ XGBoost",
    "mdl.xgb.desc": (
        "Extreme Gradient Boosting — the gradient-boosted-tree library that "
        "won most Kaggle competitions in the late 2010s. Strong out-of-the-"
        "box performance on tabular data."
    ),
    "mdl.lgbm.label": "⚡ LightGBM",
    "mdl.lgbm.desc": (
        "Microsoft's gradient-boosted trees with leaf-wise growth and "
        "histogram-based splits. Faster than XGBoost on large datasets, "
        "comparable accuracy."
    ),
    "mdl.cat.label": "⚡ CatBoost",
    "mdl.cat.desc": (
        "Yandex's gradient-boosted trees, designed to handle categorical "
        "features without one-hot encoding and to reduce prediction shift "
        "via ordered boosting."
    ),
    "mdl.tabnet.label": "🧠 TabNet",
    "mdl.tabnet.desc": (
        "Attention-based neural network specifically designed for tabular "
        "data. Uses sequential decision steps with sparse feature selection. "
        "Requires PyTorch."
    ),
    "mdl.sdt.label": "🌳 Soft Decision Tree (neural-trees)",
    "mdl.sdt.desc": (
        "A differentiable decision tree from the *neural-trees* package: "
        "every internal node is a sigmoid gate, every leaf is a softmax over "
        "classes, and the whole tree is trained end-to-end with gradient descent."
    ),

    # — Single mode panels —
    "single.train_acc": "Train Accuracy",
    "single.test_acc": "Test Accuracy",
    "single.cv_mean_std": "CV Mean ± Std",
    "single.dataset_size": "Dataset size",
    "single.dataset_size_value": "{n} samples / {k} classes",
    "single.delta_train": "{d:+.3f} vs train",
    "single.spinner": "Training {model}...",
    "single.boundary_title": "{model} — Decision Boundary",
    "single.cm_title": "Confusion Matrix (Test Set)",
    "single.cv_title": "5-Fold CV Accuracy",

    # — Compare mode —
    "compare.spinner": "Training {a} and {b}...",
    "compare.model_a_panel": "Model A: {label}",
    "compare.model_b_panel": "Model B: {label}",
    "compare.train_acc": "Train Acc",
    "compare.test_acc": "Test Acc",
    "compare.cv_mean": "CV Mean",
    "compare.boundary_a": "Model A — Decision Boundary",
    "compare.boundary_b": "Model B — Decision Boundary",
    "compare.cm_a": "Model A — Confusion Matrix",
    "compare.cm_b": "Model B — Confusion Matrix",
    "compare.head_to_head": "Head-to-Head Comparison",
    "compare.winner": "Winner (Test Acc)",
    "compare.acc_diff": "Accuracy difference",
    "compare.cv_compare": "A CV vs B CV",
    "compare.overfit_compare": "Overfit (A vs B)",
    "compare.bar_title": "Model Comparison",
    "compare.metric": "Metric",
    "compare.metric.train_acc": "Train Accuracy",
    "compare.metric.test_acc": "Test Accuracy",
    "compare.metric.cv_mean": "CV Mean",
    "compare.score": "Score",
    "compare.model": "Model",

    # — Significance —
    "sig.title": "Statistical Significance Tests",
    "sig.unavailable": "`mlxtend` is not installed; significance tests are unavailable. Install it with `pip install mlxtend` to enable paired *t*, McNemar, and 5×2cv *F*-tests.",
    "sig.spinner": "Running paired *t*, McNemar, and 5×2cv *F*-tests...",
    "sig.error": "Significance test failed: {err}",
    "sig.col_test": "Test",
    "sig.col_stat": "Statistic",
    "sig.col_p": "p-value",
    "sig.col_decision": "Decision (α=0.05)",
    "sig.t_test": "Paired *t*-test (5×2cv)",
    "sig.f_test": "Alpaydın 5×2cv *F*-test",
    "sig.mcnemar": "McNemar (test split, corrected)",
    "sig.reject": "reject H₀",
    "sig.keep": "fail to reject H₀",
    "sig.h0_caption": "H₀: the two models have equal generalization performance. *Reject* means the observed difference is unlikely under H₀.",

    # — Benchmark —
    "bench.spinner_init": "Training classifiers...",
    "bench.spinner_step": "Trained {label} ({i}/{n})",
    "bench.leaderboard": "Leaderboard",
    "bench.cv_chart": "Cross-validated accuracy",
    "bench.cv_axis": "5-fold CV accuracy",
    "bench.boundaries": "Decision boundaries",
    "bench.confusions": "Confusion matrices",
    "bench.col_model": "Model",
    "bench.col_train_acc": "Train Acc",
    "bench.col_test_acc": "Test Acc",
    "bench.col_cv_mean": "CV Mean",
    "bench.col_cv_std": "CV Std",
    "bench.col_status": "Status",
    "bench.all_failed": "All classifiers failed to train. Check your dataset and installed backends.",
    "bench.panel_label": "{label} (acc {acc:.3f})",

    # — Footer —
    "footer.text": "ML Playground · Algorithms from scikit-learn, XGBoost, LightGBM, CatBoost, TabNet, neural-trees · Significance tests via mlxtend (Alpaydın 5×2cv *F*-test, McNemar, paired *t*) · Inspired by *Introduction to Machine Learning* (Alpaydın, MIT Press, 2020)",
}

# ─────────────────────────────────────────────────────────────────────────────
# Türkçe
# ─────────────────────────────────────────────────────────────────────────────
T["tr"] = {
    "app.title": "ML Playground",
    "app.subtitle": "Bir algoritma seç, bir veri kümesi seç, ne olduğunu gör. İki modeli kafa kafaya kıyasla veya neural-trees makalesindeki değerlendirme protokolüyle tüm sınıflandırıcıları aynı anda kıyaslayın.",

    "landing.title": "Hoş geldiniz",
    "landing.body": (
        "Bu playground, klasik makine öğrenmesi modellerini iyi bilinen veri "
        "kümeleri üzerinde eğitmenizi, her modelin **sınıflar arasındaki "
        "sınırı tam olarak nereye çizdiğini** görmenizi ve modelleri "
        "istatistiksel olarak kafa kafaya kıyaslamanızı sağlar.\n\n"
        "**Nasıl kullanılır:**\n"
        "1. Sol kenardan bir **veri kümesi seç** (her birinin bilgi paneli vardır).\n"
        "2. Bir **mod seç** — tek model, ikili düello veya tam benchmark.\n"
        "3. **Modelleri seç** ve istersen hiperparametrelerini ayarla.\n"
        "4. **▶ Çalıştır**'a bas.\n\n"
        "Eğitim noktaları daire, test noktaları elmas. Renkli bölgeler, "
        "modelin ince bir ızgara üzerindeki sınıf tahminleridir."
    ),

    "sidebar.language": "🌍 Dil",
    "sidebar.run": "▶ Çalıştır",
    "sidebar.about_dataset": "ℹ Bu veri kümesi hakkında",
    "sidebar.about_mode": "ℹ Bu mod hakkında",
    "sidebar.about_model": "ℹ Bu model hakkında",
    "sidebar.installed_backends": "Kurulu paketler",

    "sidebar.dataset.title": "📊 Veri kümesi",
    "sidebar.dataset.choose": "Veri kümesi seç",
    "sidebar.dataset.samples": "Örnek sayısı",
    "sidebar.dataset.noise": "Gürültü",
    "sidebar.dataset.test_size": "Test oranı",
    "sidebar.dataset.seed": "Rastgele tohum",

    "sidebar.mode.title": "🎯 Mod",
    "sidebar.mode.single": "🔬 Tek Model",
    "sidebar.mode.compare": "⚔️ İki Modeli Kıyasla",
    "sidebar.mode.benchmark": "🏆 Tüm Modelleri Karşılaştır",

    "sidebar.model_a": "🤖 Model A",
    "sidebar.model_b": "🤖 Model B",
    "sidebar.algorithm_a": "Algoritma A",
    "sidebar.algorithm_b": "Algoritma B",
    "sidebar.hyperparameters_a": "Hiperparametreler A",
    "sidebar.hyperparameters_b": "Hiperparametreler B",
    "sidebar.benchmark_info": "Seçilen veri kümesi üzerinde **{n}** sınıflandırıcı varsayılan hiperparametrelerle eğitilecek ve yan yana grid olarak gösterilecek.",

    "mode.single.title": "Tek Model",
    "mode.single.desc": (
        "Bir veri kümesi üzerinde tek bir sınıflandırıcı eğit ve karar "
        "sınırını, karışıklık matrisini ve 5-katlı çapraz doğrulama "
        "doğruluğunu incele. Bir algoritmayı diğerleriyle kıyaslamadan önce "
        "**nasıl davrandığını anlamak** için en iyi mod."
    ),
    "mode.compare.title": "İki Modeli Kıyasla",
    "mode.compare.desc": (
        "İki sınıflandırıcıyı **aynı** eğitim/test bölmesi üzerinde eğitir "
        "ve üst üste üç istatistiksel test çalıştırır: eşli *t*-testi, "
        "McNemar testi ve Alpaydın'ın 5×2cv *F*-testi. 'Model A gerçekten "
        "B'den daha mı iyi, yoksa şanslı mı geldi?' sorusuna cevap verir."
    ),
    "mode.benchmark.title": "Tüm Modelleri Karşılaştır",
    "mode.benchmark.desc": (
        "Mevcut her sınıflandırıcıyı makul varsayılan hiperparametrelerle "
        "eğitir; bir liderlik tablosu, CV doğruluk grafiği, karar sınırı "
        "grid'i ve karışıklık matrisi grid'i üretir. Veri kümeniz üzerinde "
        "tüm modellerin nasıl performans gösterdiğine hızlı bir bakış için."
    ),

    "ds.moons.label": "🌙 Aylar",
    "ds.moons.desc": (
        "İç içe geçmiş iki yarım ay. Klasik doğrusal olmayan ayrım testi — "
        "doğrusal modeller (lojistik regresyon) başarısız olur; ağaç tabanlı "
        "ve çekirdek modelleri başarır."
    ),
    "ds.circles.label": "⭕ Çemberler",
    "ds.circles.desc": (
        "İç içe iki çember. Hiçbir doğrusal sınırla ayrılamaz; çekirdek "
        "SVM'leri ve ağaç toplulukları için harika bir vitrin."
    ),
    "ds.blobs2.label": "🔵 Kümeler (2 sınıf)",
    "ds.blobs2.desc": (
        "İyi ayrılmış iki Gauss kümesi. Hemen hemen her sınıflandırıcı "
        "çözer; pipeline'ın bozulmadığını doğrulamak için kullanışlı."
    ),
    "ds.blobs3.label": "🔵 Kümeler (3 sınıf)",
    "ds.blobs3.desc": (
        "Üç Gauss kümesi — çok-sınıflı temel test. Modelin ikiden fazla "
        "sınıfı doğru ele alıp almadığını kontrol eder."
    ),
    "ds.iris.label": "🌸 İris",
    "ds.iris.desc": (
        "150 iris çiçeği, 3 tür (setosa, versicolor, virginica). Burada ilk "
        "iki özellik (çanak yaprak uzunluğu ve genişliği) gösterilir; "
        "setosa doğrusal olarak ayrılabilir, diğer ikisi örtüşür."
    ),
    "ds.wine.label": "🍷 Şarap",
    "ds.wine.desc": (
        "178 İtalyan şarabı, 3 üzüm türü, 13 kimyasal özellik. Burada sadece "
        "ilk iki özellik görselleştirilir; benchmark CV skorları tüm "
        "özellikleri kullanır."
    ),
    "ds.cancer.label": "🔬 Meme Kanseri",
    "ds.cancer.desc": (
        "569 meme kütle örneği (kötü huylu vs iyi huylu), 30 özellik. "
        "Görselleştirme için PCA ile 2D'ye indirgenmiştir; alttaki "
        "sınıflandırıcı yine de orijinal 30 özelliği kullanır."
    ),

    "mdl.dt.label": "Karar Ağacı",
    "mdl.dt.desc": (
        "Özellik uzayını eksen-hizalı if/else kurallarıyla böler. Okunması "
        "kolay; derin büyürse aşırı uyuma eğilimlidir."
    ),
    "mdl.rf.label": "Rastgele Orman",
    "mdl.rf.desc": (
        "Birbirinden bağımsızlaştırılmış çok sayıda karar ağacının oylarını "
        "ortalar. Dayanıklı bir varsayılan — çoğu tablo verisinde kutudan "
        "çıkar çıkmaz iyi çalışır."
    ),
    "mdl.gb.label": "Gradient Boosting",
    "mdl.gb.desc": (
        "Sığ ağaçları sırayla inşa eder; her ağaç bir öncekinin hatalarını "
        "düzeltir. Tablo verisinde güçlü; eğitim Random Forest'tan yavaştır."
    ),
    "mdl.svm.label": "SVM (RBF)",
    "mdl.svm.desc": (
        "Radyal taban fonksiyonlu çekirdek ile destek vektör makinesi. "
        "Örtük sonsuz boyutlu uzayda maksimum-marjlı sınıflandırıcı — "
        "doğrusal olmayan sınırlarla iyi başa çıkar, >100bin örnekte yavaşlar."
    ),
    "mdl.knn.label": "K-En Yakın Komşu",
    "mdl.knn.desc": (
        "En yakın *k* eğitim noktası arasındaki çoğunluk sınıfı tahmin eder. "
        "Eğitim aşaması yok, ama her tahmin tüm veri kümesini tarar."
    ),
    "mdl.lr.label": "Lojistik Regresyon",
    "mdl.lr.desc": (
        "Sigmoid ile sınıf olasılıkları üreten doğrusal model. En basit "
        "ciddi temel — hızlı, yorumlanabilir, ama doğrusal olarak ayrılabilir "
        "problemlerle sınırlı."
    ),
    "mdl.mlp.label": "Yapay Sinir Ağı (MLP)",
    "mdl.mlp.desc": (
        "Çok katmanlı algılayıcı — küçük tam bağlı sinir ağı. Evrensel "
        "yaklaşımcı; ama tablo verisinde ağaç topluluklarını geçmek için "
        "daha çok veri ve ayara ihtiyaç duyar."
    ),
    "mdl.nb.label": "Naive Bayes",
    "mdl.nb.desc": (
        "Özelliklerin sınıf verildiğinde koşullu bağımsız olduğu (genelde "
        "yanlış) varsayımıyla Bayes kuralını uygular. Metin ve seyrek "
        "yüksek boyutlu veriler üzerinde şaşırtıcı şekilde güçlü."
    ),
    "mdl.xgb.label": "⚡ XGBoost",
    "mdl.xgb.desc": (
        "Extreme Gradient Boosting — 2010'ların sonunda Kaggle'ı domine "
        "eden gradient boosted tree kütüphanesi. Tablo verisinde kutudan "
        "çıkar çıkmaz güçlü performans."
    ),
    "mdl.lgbm.label": "⚡ LightGBM",
    "mdl.lgbm.desc": (
        "Microsoft'un yaprak-bazlı büyüme ve histogram-bazlı bölme kullanan "
        "gradient boosted ağaçları. Büyük veri kümelerinde XGBoost'tan "
        "hızlı, doğruluk benzer."
    ),
    "mdl.cat.label": "⚡ CatBoost",
    "mdl.cat.desc": (
        "Yandex'in gradient boosted ağaçları; one-hot kodlama olmadan "
        "kategorik özellikleri ele almak ve sıralı boosting ile tahmin "
        "kaymasını azaltmak için tasarlanmıştır."
    ),
    "mdl.tabnet.label": "🧠 TabNet",
    "mdl.tabnet.desc": (
        "Tablo verisi için özel olarak tasarlanmış dikkat-tabanlı sinir ağı. "
        "Sıralı karar adımları ve seyrek özellik seçimi kullanır. PyTorch gerektirir."
    ),
    "mdl.sdt.label": "🌳 Soft Decision Tree (neural-trees)",
    "mdl.sdt.desc": (
        "*neural-trees* paketinden türevlenebilir bir karar ağacı: her iç "
        "düğüm bir sigmoid kapı, her yaprak sınıflar üzerinde bir softmax "
        "ve tüm ağaç gradient descent ile uçtan uca eğitilir."
    ),

    "single.train_acc": "Eğitim Doğruluğu",
    "single.test_acc": "Test Doğruluğu",
    "single.cv_mean_std": "CV Ortalama ± Std",
    "single.dataset_size": "Veri kümesi boyutu",
    "single.dataset_size_value": "{n} örnek / {k} sınıf",
    "single.delta_train": "{d:+.3f} eğitime göre",
    "single.spinner": "{model} eğitiliyor...",
    "single.boundary_title": "{model} — Karar Sınırı",
    "single.cm_title": "Karışıklık Matrisi (Test)",
    "single.cv_title": "5-Katlı CV Doğruluğu",

    "compare.spinner": "{a} ve {b} eğitiliyor...",
    "compare.model_a_panel": "Model A: {label}",
    "compare.model_b_panel": "Model B: {label}",
    "compare.train_acc": "Eğitim Doğr.",
    "compare.test_acc": "Test Doğr.",
    "compare.cv_mean": "CV Ortalama",
    "compare.boundary_a": "Model A — Karar Sınırı",
    "compare.boundary_b": "Model B — Karar Sınırı",
    "compare.cm_a": "Model A — Karışıklık Matrisi",
    "compare.cm_b": "Model B — Karışıklık Matrisi",
    "compare.head_to_head": "Kafa Kafaya Karşılaştırma",
    "compare.winner": "Kazanan (Test Doğr.)",
    "compare.acc_diff": "Doğruluk farkı",
    "compare.cv_compare": "A CV vs B CV",
    "compare.overfit_compare": "Aşırı uyum (A vs B)",
    "compare.bar_title": "Model Karşılaştırması",
    "compare.metric": "Metrik",
    "compare.metric.train_acc": "Eğitim Doğruluğu",
    "compare.metric.test_acc": "Test Doğruluğu",
    "compare.metric.cv_mean": "CV Ortalama",
    "compare.score": "Skor",
    "compare.model": "Model",

    "sig.title": "İstatistiksel Anlamlılık Testleri",
    "sig.unavailable": "`mlxtend` yüklü değil; anlamlılık testleri kullanılamıyor. Eşli *t*, McNemar ve 5×2cv *F* testlerini etkinleştirmek için `pip install mlxtend` çalıştırın.",
    "sig.spinner": "Eşli *t*, McNemar ve 5×2cv *F* testleri çalıştırılıyor...",
    "sig.error": "Anlamlılık testi başarısız: {err}",
    "sig.col_test": "Test",
    "sig.col_stat": "İstatistik",
    "sig.col_p": "p-değeri",
    "sig.col_decision": "Karar (α=0.05)",
    "sig.t_test": "Eşli *t*-testi (5×2cv)",
    "sig.f_test": "Alpaydın 5×2cv *F*-testi",
    "sig.mcnemar": "McNemar (test bölmesi, düzeltilmiş)",
    "sig.reject": "H₀ reddedilir",
    "sig.keep": "H₀ reddedilemez",
    "sig.h0_caption": "H₀: iki modelin genelleme performansı eşittir. *Reddedilir* ise gözlenen fark H₀ altında olası değildir.",

    "bench.spinner_init": "Sınıflandırıcılar eğitiliyor...",
    "bench.spinner_step": "{label} eğitildi ({i}/{n})",
    "bench.leaderboard": "Liderlik Tablosu",
    "bench.cv_chart": "Çapraz doğrulama doğruluğu",
    "bench.cv_axis": "5-katlı CV doğruluğu",
    "bench.boundaries": "Karar sınırları",
    "bench.confusions": "Karışıklık matrisleri",
    "bench.col_model": "Model",
    "bench.col_train_acc": "Eğitim Doğr.",
    "bench.col_test_acc": "Test Doğr.",
    "bench.col_cv_mean": "CV Ortalama",
    "bench.col_cv_std": "CV Std",
    "bench.col_status": "Durum",
    "bench.all_failed": "Hiçbir sınıflandırıcı eğitilemedi. Veri kümenizi ve kurulu paketleri kontrol edin.",
    "bench.panel_label": "{label} (doğr. {acc:.3f})",

    "footer.text": "ML Playground · scikit-learn, XGBoost, LightGBM, CatBoost, TabNet, neural-trees algoritmaları · mlxtend ile anlamlılık testleri (Alpaydın 5×2cv *F*-testi, McNemar, eşli *t*) · *Introduction to Machine Learning* (Alpaydın, MIT Press, 2020)'den ilham alınmıştır",
}

# ─────────────────────────────────────────────────────────────────────────────
# Español
# ─────────────────────────────────────────────────────────────────────────────
T["es"] = {
    "app.title": "ML Playground",
    "app.subtitle": "Elige un algoritmo, elige un conjunto de datos, mira qué pasa. Compara dos modelos cara a cara, o evalúa todos los clasificadores a la vez con el mismo protocolo del artículo neural-trees.",

    "landing.title": "Bienvenido",
    "landing.body": (
        "Este playground te permite entrenar modelos clásicos de aprendizaje "
        "automático en conjuntos de datos conocidos, ver exactamente **dónde "
        "cada modelo traza la frontera entre clases**, y comparar modelos "
        "cara a cara con rigor estadístico.\n\n"
        "**Cómo usarlo:**\n"
        "1. **Elige un conjunto de datos** en la barra lateral (cada uno tiene un panel informativo).\n"
        "2. **Elige un modo** — modelo único, duelo de dos modelos o benchmark completo.\n"
        "3. **Elige el/los modelo(s)** y opcionalmente ajusta sus hiperparámetros.\n"
        "4. Pulsa **▶ Ejecutar**.\n\n"
        "Los puntos de entrenamiento son círculos, los de prueba son rombos. "
        "Las regiones coloreadas son las predicciones de clase sobre una rejilla fina."
    ),

    "sidebar.language": "🌍 Idioma",
    "sidebar.run": "▶ Ejecutar",
    "sidebar.about_dataset": "ℹ Sobre este conjunto de datos",
    "sidebar.about_mode": "ℹ Sobre este modo",
    "sidebar.about_model": "ℹ Sobre este modelo",
    "sidebar.installed_backends": "Backends instalados",

    "sidebar.dataset.title": "📊 Conjunto de datos",
    "sidebar.dataset.choose": "Elige conjunto de datos",
    "sidebar.dataset.samples": "Muestras",
    "sidebar.dataset.noise": "Ruido",
    "sidebar.dataset.test_size": "Proporción de prueba",
    "sidebar.dataset.seed": "Semilla aleatoria",

    "sidebar.mode.title": "🎯 Modo",
    "sidebar.mode.single": "🔬 Modelo Único",
    "sidebar.mode.compare": "⚔️ Comparar Dos Modelos",
    "sidebar.mode.benchmark": "🏆 Benchmark de Todos",

    "sidebar.model_a": "🤖 Modelo A",
    "sidebar.model_b": "🤖 Modelo B",
    "sidebar.algorithm_a": "Algoritmo A",
    "sidebar.algorithm_b": "Algoritmo B",
    "sidebar.hyperparameters_a": "Hiperparámetros A",
    "sidebar.hyperparameters_b": "Hiperparámetros B",
    "sidebar.benchmark_info": "Se entrenarán **{n}** clasificadores con hiperparámetros por defecto sobre el conjunto de datos elegido y se mostrarán en una rejilla.",

    "mode.single.title": "Modelo Único",
    "mode.single.desc": (
        "Entrena un clasificador en un conjunto de datos e inspecciona su "
        "frontera de decisión, matriz de confusión y precisión por validación "
        "cruzada de 5 pliegues. Ideal para **entender cómo se comporta un "
        "algoritmo** antes de compararlo con otros."
    ),
    "mode.compare.title": "Comparar Dos Modelos",
    "mode.compare.desc": (
        "Entrena dos clasificadores en la **misma** división entrenamiento/"
        "prueba y ejecuta tres pruebas estadísticas: *t* pareada, prueba de "
        "McNemar y *F*-test 5×2cv de Alpaydın. Útil para responder '¿es "
        "realmente mejor el modelo A o solo tuvo suerte?'"
    ),
    "mode.benchmark.title": "Benchmark de Todos los Modelos",
    "mode.benchmark.desc": (
        "Entrena todos los clasificadores disponibles con hiperparámetros "
        "por defecto razonables y produce una tabla de clasificación, una "
        "gráfica de barras de precisión CV, una rejilla de fronteras de "
        "decisión y otra de matrices de confusión."
    ),

    "ds.moons.label": "🌙 Lunas",
    "ds.moons.desc": "Dos medias lunas entrelazadas. Prueba clásica de separación no lineal — los modelos lineales (regresión logística) fallan; los basados en árboles y kernel triunfan.",
    "ds.circles.label": "⭕ Círculos",
    "ds.circles.desc": "Dos círculos concéntricos. Imposible para cualquier frontera lineal; perfecto para mostrar SVMs con kernel y conjuntos de árboles.",
    "ds.blobs2.label": "🔵 Cúmulos (2 clases)",
    "ds.blobs2.desc": "Dos cúmulos gaussianos bien separados. Casi cualquier clasificador resuelve esto; útil como verificación de que tu pipeline no está roto.",
    "ds.blobs3.label": "🔵 Cúmulos (3 clases)",
    "ds.blobs3.desc": "Tres cúmulos gaussianos — base multiclase. Comprueba si un modelo maneja correctamente más de dos clases.",
    "ds.iris.label": "🌸 Iris",
    "ds.iris.desc": "150 flores iris, 3 especies (setosa, versicolor, virginica). Se muestran las dos primeras características; setosa es linealmente separable, las otras dos se solapan.",
    "ds.wine.label": "🍷 Vino",
    "ds.wine.desc": "178 vinos italianos de 3 cultivares, 13 características químicas. Aquí solo se visualizan las dos primeras; el conjunto completo es lo que usan las puntuaciones CV.",
    "ds.cancer.label": "🔬 Cáncer de Mama",
    "ds.cancer.desc": "569 muestras de masas mamarias (maligno vs benigno) con 30 características. Reducidas a 2D mediante PCA para visualización; el clasificador subyacente sigue usando las 30 originales.",

    "mdl.dt.label": "Árbol de Decisión",
    "mdl.dt.desc": "Divide el espacio de características con reglas if/else alineadas a los ejes. Fácil de leer, propenso a sobreajustar si crece mucho.",
    "mdl.rf.label": "Bosque Aleatorio",
    "mdl.rf.desc": "Combina muchos árboles de decisión decorrelacionados y promedia sus votos. Robusto por defecto — funciona bien en la mayoría de datos tabulares.",
    "mdl.gb.label": "Gradient Boosting",
    "mdl.gb.desc": "Construye árboles superficiales secuencialmente, cada uno corrigiendo los errores del anterior. Fuerte en datos tabulares; más lento que Random Forest.",
    "mdl.svm.label": "SVM (RBF)",
    "mdl.svm.desc": "Máquina de vectores soporte con kernel de función de base radial. Clasificador de margen máximo en un espacio implícito infinito-dimensional — maneja fronteras no lineales bien.",
    "mdl.knn.label": "K-Vecinos más Cercanos",
    "mdl.knn.desc": "Predice la clase mayoritaria entre los *k* puntos de entrenamiento más cercanos. Sin fase de entrenamiento, pero cada predicción recorre todo el dataset.",
    "mdl.lr.label": "Regresión Logística",
    "mdl.lr.desc": "Modelo lineal que produce probabilidades de clase vía sigmoide. La línea de base seria más simple — rápida, interpretable, limitada a problemas linealmente separables.",
    "mdl.mlp.label": "Red Neuronal (MLP)",
    "mdl.mlp.desc": "Perceptrón multicapa — pequeña red neuronal totalmente conectada. Aproximador universal, pero necesita más datos y ajuste que los conjuntos de árboles en datos tabulares.",
    "mdl.nb.label": "Naive Bayes",
    "mdl.nb.desc": "Aplica la regla de Bayes asumiendo (a menudo erróneamente) que las características son condicionalmente independientes dada la clase. Sorprendentemente fuerte en texto.",
    "mdl.xgb.label": "⚡ XGBoost",
    "mdl.xgb.desc": "Extreme Gradient Boosting — la librería que dominó Kaggle a finales de los 2010. Rendimiento fuerte de fábrica en datos tabulares.",
    "mdl.lgbm.label": "⚡ LightGBM",
    "mdl.lgbm.desc": "Árboles de gradient boosting de Microsoft con crecimiento por hojas y splits basados en histogramas. Más rápido que XGBoost en datasets grandes.",
    "mdl.cat.label": "⚡ CatBoost",
    "mdl.cat.desc": "Árboles de gradient boosting de Yandex, diseñados para manejar características categóricas sin one-hot encoding y reducir el sesgo de predicción.",
    "mdl.tabnet.label": "🧠 TabNet",
    "mdl.tabnet.desc": "Red neuronal basada en atención diseñada específicamente para datos tabulares. Usa pasos de decisión secuenciales con selección dispersa de características.",
    "mdl.sdt.label": "🌳 Soft Decision Tree (neural-trees)",
    "mdl.sdt.desc": "Un árbol de decisión diferenciable del paquete *neural-trees*: cada nodo interno es una puerta sigmoide, cada hoja un softmax sobre clases, todo entrenado de extremo a extremo con descenso de gradiente.",

    "single.train_acc": "Precisión Entrenamiento",
    "single.test_acc": "Precisión Prueba",
    "single.cv_mean_std": "Media CV ± Std",
    "single.dataset_size": "Tamaño del dataset",
    "single.dataset_size_value": "{n} muestras / {k} clases",
    "single.delta_train": "{d:+.3f} vs entrenamiento",
    "single.spinner": "Entrenando {model}...",
    "single.boundary_title": "{model} — Frontera de Decisión",
    "single.cm_title": "Matriz de Confusión (Prueba)",
    "single.cv_title": "Precisión CV de 5 pliegues",

    "compare.spinner": "Entrenando {a} y {b}...",
    "compare.model_a_panel": "Modelo A: {label}",
    "compare.model_b_panel": "Modelo B: {label}",
    "compare.train_acc": "Prec. Entr.",
    "compare.test_acc": "Prec. Prueba",
    "compare.cv_mean": "Media CV",
    "compare.boundary_a": "Modelo A — Frontera de Decisión",
    "compare.boundary_b": "Modelo B — Frontera de Decisión",
    "compare.cm_a": "Modelo A — Matriz de Confusión",
    "compare.cm_b": "Modelo B — Matriz de Confusión",
    "compare.head_to_head": "Comparación Cara a Cara",
    "compare.winner": "Ganador (Prec. Prueba)",
    "compare.acc_diff": "Diferencia de precisión",
    "compare.cv_compare": "CV A vs CV B",
    "compare.overfit_compare": "Sobreajuste (A vs B)",
    "compare.bar_title": "Comparación de Modelos",
    "compare.metric": "Métrica",
    "compare.metric.train_acc": "Precisión Entrenamiento",
    "compare.metric.test_acc": "Precisión Prueba",
    "compare.metric.cv_mean": "Media CV",
    "compare.score": "Puntuación",
    "compare.model": "Modelo",

    "sig.title": "Pruebas de Significancia Estadística",
    "sig.unavailable": "`mlxtend` no está instalado; las pruebas de significancia no están disponibles. Instálalo con `pip install mlxtend`.",
    "sig.spinner": "Ejecutando *t* pareada, McNemar y *F*-test 5×2cv...",
    "sig.error": "La prueba falló: {err}",
    "sig.col_test": "Prueba",
    "sig.col_stat": "Estadístico",
    "sig.col_p": "p-valor",
    "sig.col_decision": "Decisión (α=0.05)",
    "sig.t_test": "*t*-test pareado (5×2cv)",
    "sig.f_test": "*F*-test 5×2cv (Alpaydın)",
    "sig.mcnemar": "McNemar (prueba, corregido)",
    "sig.reject": "rechazar H₀",
    "sig.keep": "no rechazar H₀",
    "sig.h0_caption": "H₀: los dos modelos tienen igual rendimiento de generalización. *Rechazar* significa que la diferencia observada es improbable bajo H₀.",

    "bench.spinner_init": "Entrenando clasificadores...",
    "bench.spinner_step": "{label} entrenado ({i}/{n})",
    "bench.leaderboard": "Tabla de Clasificación",
    "bench.cv_chart": "Precisión por validación cruzada",
    "bench.cv_axis": "Precisión CV de 5 pliegues",
    "bench.boundaries": "Fronteras de decisión",
    "bench.confusions": "Matrices de confusión",
    "bench.col_model": "Modelo",
    "bench.col_train_acc": "Prec. Entr.",
    "bench.col_test_acc": "Prec. Prueba",
    "bench.col_cv_mean": "Media CV",
    "bench.col_cv_std": "Std CV",
    "bench.col_status": "Estado",
    "bench.all_failed": "Todos los clasificadores fallaron. Comprueba tu dataset y los backends instalados.",
    "bench.panel_label": "{label} (prec. {acc:.3f})",

    "footer.text": "ML Playground · Algoritmos de scikit-learn, XGBoost, LightGBM, CatBoost, TabNet, neural-trees · Pruebas vía mlxtend · Inspirado en *Introduction to Machine Learning* (Alpaydın, MIT Press, 2020)",
}

# ─────────────────────────────────────────────────────────────────────────────
# Italiano
# ─────────────────────────────────────────────────────────────────────────────
T["it"] = {
    "app.title": "ML Playground",
    "app.subtitle": "Scegli un algoritmo, scegli un dataset, guarda cosa succede. Confronta due modelli faccia a faccia, o valuta tutti i classificatori contemporaneamente con lo stesso protocollo dell'articolo neural-trees.",

    "landing.title": "Benvenuto",
    "landing.body": (
        "Questo playground ti permette di addestrare modelli classici di "
        "machine learning su dataset noti, vedere esattamente **dove ogni "
        "modello traccia il confine tra le classi** e confrontare modelli "
        "faccia a faccia con rigore statistico.\n\n"
        "**Come usarlo:**\n"
        "1. **Scegli un dataset** nella barra laterale (ognuno ha un pannello info).\n"
        "2. **Scegli una modalità** — modello singolo, duello tra due modelli o benchmark completo.\n"
        "3. **Scegli il/i modello/i**, opzionalmente regola gli iperparametri.\n"
        "4. Premi **▶ Esegui**.\n\n"
        "I punti di addestramento sono cerchi, quelli di test sono rombi. "
        "Le regioni colorate sono le predizioni di classe su una griglia fine."
    ),

    "sidebar.language": "🌍 Lingua",
    "sidebar.run": "▶ Esegui",
    "sidebar.about_dataset": "ℹ Informazioni sul dataset",
    "sidebar.about_mode": "ℹ Informazioni sulla modalità",
    "sidebar.about_model": "ℹ Informazioni sul modello",
    "sidebar.installed_backends": "Backend installati",

    "sidebar.dataset.title": "📊 Dataset",
    "sidebar.dataset.choose": "Scegli dataset",
    "sidebar.dataset.samples": "Campioni",
    "sidebar.dataset.noise": "Rumore",
    "sidebar.dataset.test_size": "Frazione di test",
    "sidebar.dataset.seed": "Seme casuale",

    "sidebar.mode.title": "🎯 Modalità",
    "sidebar.mode.single": "🔬 Modello Singolo",
    "sidebar.mode.compare": "⚔️ Confronta Due Modelli",
    "sidebar.mode.benchmark": "🏆 Benchmark di Tutti",

    "sidebar.model_a": "🤖 Modello A",
    "sidebar.model_b": "🤖 Modello B",
    "sidebar.algorithm_a": "Algoritmo A",
    "sidebar.algorithm_b": "Algoritmo B",
    "sidebar.hyperparameters_a": "Iperparametri A",
    "sidebar.hyperparameters_b": "Iperparametri B",
    "sidebar.benchmark_info": "Verranno addestrati **{n}** classificatori con iperparametri di default sul dataset scelto e mostrati in una griglia.",

    "mode.single.title": "Modello Singolo",
    "mode.single.desc": "Addestra un classificatore su un dataset e ispeziona il suo confine di decisione, matrice di confusione e accuratezza in cross-validation a 5 fold. Ideale per **capire come si comporta un algoritmo** prima di confrontarlo con altri.",
    "mode.compare.title": "Confronta Due Modelli",
    "mode.compare.desc": "Addestra due classificatori sulla **stessa** divisione train/test ed esegue tre test statistici: *t*-test appaiato, test di McNemar e *F*-test 5×2cv di Alpaydın. Risponde alla domanda 'il modello A è davvero migliore di B, o è stato fortunato?'",
    "mode.benchmark.title": "Benchmark di Tutti i Modelli",
    "mode.benchmark.desc": "Addestra ogni classificatore disponibile con iperparametri di default ragionevoli e produce una classifica, un grafico a barre di accuratezza CV, una griglia di confini di decisione e una di matrici di confusione.",

    "ds.moons.label": "🌙 Lune",
    "ds.moons.desc": "Due mezzelune intrecciate. Test classico di separazione non lineare — i modelli lineari (regressione logistica) falliscono; gli alberi e i kernel riescono.",
    "ds.circles.label": "⭕ Cerchi",
    "ds.circles.desc": "Due cerchi concentrici. Impossibile per qualsiasi confine lineare; ottimo per mostrare SVM con kernel e ensemble di alberi.",
    "ds.blobs2.label": "🔵 Blob (2 classi)",
    "ds.blobs2.desc": "Due cluster gaussiani ben separati. Quasi ogni classificatore lo risolve; utile come verifica che la pipeline funzioni.",
    "ds.blobs3.label": "🔵 Blob (3 classi)",
    "ds.blobs3.desc": "Tre cluster gaussiani — baseline multiclasse. Verifica se un modello gestisce correttamente più di due classi.",
    "ds.iris.label": "🌸 Iris",
    "ds.iris.desc": "150 fiori iris, 3 specie (setosa, versicolor, virginica). Vengono mostrate le prime due caratteristiche; setosa è linearmente separabile, le altre due si sovrappongono.",
    "ds.wine.label": "🍷 Vino",
    "ds.wine.desc": "178 vini italiani di 3 cultivar, 13 caratteristiche chimiche. Qui sono visualizzate solo le prime due; il set completo è quello usato per i punteggi CV.",
    "ds.cancer.label": "🔬 Tumore al Seno",
    "ds.cancer.desc": "569 campioni di masse mammarie (maligne vs benigne) con 30 caratteristiche. Ridotte a 2D tramite PCA per la visualizzazione; il classificatore usa comunque le 30 originali.",

    "mdl.dt.label": "Albero di Decisione",
    "mdl.dt.desc": "Divide lo spazio delle caratteristiche con regole if/else allineate agli assi. Facile da leggere, soggetto a overfitting se cresce troppo.",
    "mdl.rf.label": "Random Forest",
    "mdl.rf.desc": "Combina molti alberi di decisione decorrelati e media i loro voti. Default robusto — funziona bene su gran parte dei dati tabellari.",
    "mdl.gb.label": "Gradient Boosting",
    "mdl.gb.desc": "Costruisce alberi superficiali in sequenza, ognuno corregge gli errori del precedente. Forte sui dati tabellari; più lento da addestrare di Random Forest.",
    "mdl.svm.label": "SVM (RBF)",
    "mdl.svm.desc": "Support Vector Machine con kernel a funzione di base radiale. Classificatore a margine massimo in uno spazio implicito infinito-dimensionale — gestisce bene confini non lineari.",
    "mdl.knn.label": "K-Nearest Neighbors",
    "mdl.knn.desc": "Predice la classe maggioritaria tra i *k* punti di training più vicini. Nessuna fase di addestramento, ma ogni predizione scansiona tutto il dataset.",
    "mdl.lr.label": "Regressione Logistica",
    "mdl.lr.desc": "Modello lineare che produce probabilità di classe via sigmoide. La baseline seria più semplice — veloce, interpretabile, limitata a problemi linearmente separabili.",
    "mdl.mlp.label": "Rete Neurale (MLP)",
    "mdl.mlp.desc": "Perceptron multistrato — piccola rete neurale fully connected. Approssimatore universale, ma richiede più dati e tuning degli ensemble di alberi sui dati tabellari.",
    "mdl.nb.label": "Naive Bayes",
    "mdl.nb.desc": "Applica la regola di Bayes assumendo (spesso erroneamente) che le caratteristiche siano condizionalmente indipendenti data la classe. Sorprendentemente forte sul testo.",
    "mdl.xgb.label": "⚡ XGBoost",
    "mdl.xgb.desc": "Extreme Gradient Boosting — la libreria che ha dominato Kaggle alla fine degli anni 2010. Prestazioni forti out-of-the-box su dati tabellari.",
    "mdl.lgbm.label": "⚡ LightGBM",
    "mdl.lgbm.desc": "Alberi di gradient boosting di Microsoft con crescita leaf-wise e split basati su istogrammi. Più veloce di XGBoost su dataset grandi.",
    "mdl.cat.label": "⚡ CatBoost",
    "mdl.cat.desc": "Alberi di gradient boosting di Yandex, progettati per gestire feature categoriche senza one-hot encoding e ridurre il prediction shift.",
    "mdl.tabnet.label": "🧠 TabNet",
    "mdl.tabnet.desc": "Rete neurale basata su attenzione progettata specificamente per dati tabellari. Usa step di decisione sequenziali con selezione sparsa delle feature.",
    "mdl.sdt.label": "🌳 Soft Decision Tree (neural-trees)",
    "mdl.sdt.desc": "Un albero di decisione differenziabile dal pacchetto *neural-trees*: ogni nodo interno è una porta sigmoide, ogni foglia un softmax sulle classi, l'intero albero è addestrato end-to-end con discesa del gradiente.",

    "single.train_acc": "Accuratezza Train",
    "single.test_acc": "Accuratezza Test",
    "single.cv_mean_std": "Media CV ± Std",
    "single.dataset_size": "Dimensione dataset",
    "single.dataset_size_value": "{n} campioni / {k} classi",
    "single.delta_train": "{d:+.3f} vs train",
    "single.spinner": "Addestramento {model}...",
    "single.boundary_title": "{model} — Confine di Decisione",
    "single.cm_title": "Matrice di Confusione (Test)",
    "single.cv_title": "Accuratezza CV 5-fold",

    "compare.spinner": "Addestramento {a} e {b}...",
    "compare.model_a_panel": "Modello A: {label}",
    "compare.model_b_panel": "Modello B: {label}",
    "compare.train_acc": "Acc. Train",
    "compare.test_acc": "Acc. Test",
    "compare.cv_mean": "Media CV",
    "compare.boundary_a": "Modello A — Confine di Decisione",
    "compare.boundary_b": "Modello B — Confine di Decisione",
    "compare.cm_a": "Modello A — Matrice di Confusione",
    "compare.cm_b": "Modello B — Matrice di Confusione",
    "compare.head_to_head": "Confronto Faccia a Faccia",
    "compare.winner": "Vincitore (Acc. Test)",
    "compare.acc_diff": "Differenza accuratezza",
    "compare.cv_compare": "CV A vs CV B",
    "compare.overfit_compare": "Overfit (A vs B)",
    "compare.bar_title": "Confronto Modelli",
    "compare.metric": "Metrica",
    "compare.metric.train_acc": "Accuratezza Train",
    "compare.metric.test_acc": "Accuratezza Test",
    "compare.metric.cv_mean": "Media CV",
    "compare.score": "Punteggio",
    "compare.model": "Modello",

    "sig.title": "Test di Significatività Statistica",
    "sig.unavailable": "`mlxtend` non installato; i test non sono disponibili. Installa con `pip install mlxtend`.",
    "sig.spinner": "Esecuzione *t*-test appaiato, McNemar e *F*-test 5×2cv...",
    "sig.error": "Test fallito: {err}",
    "sig.col_test": "Test",
    "sig.col_stat": "Statistica",
    "sig.col_p": "valore-p",
    "sig.col_decision": "Decisione (α=0.05)",
    "sig.t_test": "*t*-test appaiato (5×2cv)",
    "sig.f_test": "*F*-test 5×2cv (Alpaydın)",
    "sig.mcnemar": "McNemar (test, corretto)",
    "sig.reject": "rifiuta H₀",
    "sig.keep": "non rifiuta H₀",
    "sig.h0_caption": "H₀: i due modelli hanno uguale prestazione di generalizzazione. *Rifiuta* significa che la differenza osservata è improbabile sotto H₀.",

    "bench.spinner_init": "Addestramento classificatori...",
    "bench.spinner_step": "{label} addestrato ({i}/{n})",
    "bench.leaderboard": "Classifica",
    "bench.cv_chart": "Accuratezza in cross-validation",
    "bench.cv_axis": "Accuratezza CV 5-fold",
    "bench.boundaries": "Confini di decisione",
    "bench.confusions": "Matrici di confusione",
    "bench.col_model": "Modello",
    "bench.col_train_acc": "Acc. Train",
    "bench.col_test_acc": "Acc. Test",
    "bench.col_cv_mean": "Media CV",
    "bench.col_cv_std": "Std CV",
    "bench.col_status": "Stato",
    "bench.all_failed": "Tutti i classificatori hanno fallito. Controlla il dataset e i backend installati.",
    "bench.panel_label": "{label} (acc. {acc:.3f})",

    "footer.text": "ML Playground · Algoritmi da scikit-learn, XGBoost, LightGBM, CatBoost, TabNet, neural-trees · Test via mlxtend · Ispirato a *Introduction to Machine Learning* (Alpaydın, MIT Press, 2020)",
}

# ─────────────────────────────────────────────────────────────────────────────
# العربية (Arabic) — RTL
# ─────────────────────────────────────────────────────────────────────────────
T["ar"] = {
    "app.title": "ML Playground",
    "app.subtitle": "اختر خوارزمية، اختر مجموعة بيانات، وشاهد ما يحدث. قارن نموذجين وجهًا لوجه، أو قيِّم جميع المصنفات معًا بنفس البروتوكول المستخدم في ورقة neural-trees.",

    "landing.title": "أهلاً بك",
    "landing.body": (
        "تتيح لك هذه المنصة تدريب نماذج تعلم الآلة الكلاسيكية على مجموعات "
        "بيانات معروفة، ورؤية **أين يرسم كل نموذج الحد بين الفئات بدقة**، "
        "ومقارنة النماذج وجهًا لوجه بصرامة إحصائية.\n\n"
        "**كيفية الاستخدام:**\n"
        "1. **اختر مجموعة بيانات** من الشريط الجانبي (لكل واحدة لوحة معلومات).\n"
        "2. **اختر وضعًا** — نموذج واحد، أو مبارزة بين نموذجين، أو مقارنة شاملة.\n"
        "3. **اختر النموذج/النماذج**، واضبط المعلمات الفائقة اختيارياً.\n"
        "4. اضغط **▶ تشغيل**.\n\n"
        "نقاط التدريب دوائر، ونقاط الاختبار معينات. المناطق الملونة هي "
        "تنبؤات الفئة على شبكة دقيقة."
    ),

    "sidebar.language": "🌍 اللغة",
    "sidebar.run": "▶ تشغيل",
    "sidebar.about_dataset": "ℹ عن مجموعة البيانات",
    "sidebar.about_mode": "ℹ عن هذا الوضع",
    "sidebar.about_model": "ℹ عن هذا النموذج",
    "sidebar.installed_backends": "الحزم المثبتة",

    "sidebar.dataset.title": "📊 مجموعة البيانات",
    "sidebar.dataset.choose": "اختر مجموعة بيانات",
    "sidebar.dataset.samples": "عدد العينات",
    "sidebar.dataset.noise": "الضوضاء",
    "sidebar.dataset.test_size": "نسبة الاختبار",
    "sidebar.dataset.seed": "البذرة العشوائية",

    "sidebar.mode.title": "🎯 الوضع",
    "sidebar.mode.single": "🔬 نموذج واحد",
    "sidebar.mode.compare": "⚔️ مقارنة نموذجين",
    "sidebar.mode.benchmark": "🏆 مقارنة جميع النماذج",

    "sidebar.model_a": "🤖 النموذج A",
    "sidebar.model_b": "🤖 النموذج B",
    "sidebar.algorithm_a": "الخوارزمية A",
    "sidebar.algorithm_b": "الخوارزمية B",
    "sidebar.hyperparameters_a": "المعلمات الفائقة A",
    "sidebar.hyperparameters_b": "المعلمات الفائقة B",
    "sidebar.benchmark_info": "سيتم تدريب **{n}** مصنفًا بمعلمات افتراضية على مجموعة البيانات المختارة وعرضها في شبكة جنبًا إلى جنب.",

    "mode.single.title": "نموذج واحد",
    "mode.single.desc": "درّب مصنفًا واحدًا على مجموعة بيانات وافحص حدود قراره ومصفوفة الالتباس ودقة التحقق المتقاطع 5-طيات. الأمثل **لفهم سلوك خوارزمية واحدة** قبل مقارنتها بأخريات.",
    "mode.compare.title": "مقارنة نموذجين",
    "mode.compare.desc": "درّب مصنفين على **نفس** تقسيم التدريب/الاختبار وشغّل ثلاثة اختبارات إحصائية: اختبار *t* المزدوج، اختبار McNemar، واختبار *F* 5×2cv من Alpaydın. للإجابة على سؤال 'هل النموذج A أفضل فعلاً، أم أنه محظوظ؟'",
    "mode.benchmark.title": "مقارنة جميع النماذج",
    "mode.benchmark.desc": "درّب كل مصنف متاح بمعلمات افتراضية معقولة وأنتج لوحة قيادة، ومخطط دقة CV، وشبكة حدود قرار، وشبكة مصفوفات التباس.",

    "ds.moons.label": "🌙 الأقمار",
    "ds.moons.desc": "نصفا قمر متشابكان. اختبار كلاسيكي للفصل غير الخطي — النماذج الخطية (الانحدار اللوجستي) تفشل؛ النماذج المعتمدة على الشجرة والنواة تنجح.",
    "ds.circles.label": "⭕ الدوائر",
    "ds.circles.desc": "دائرتان متحدتا المركز. مستحيل لأي حد خطي؛ مثالي لإظهار SVM مع نواة وتجمعات الأشجار.",
    "ds.blobs2.label": "🔵 تجمعات (فئتان)",
    "ds.blobs2.desc": "تجمعان غاوسيان منفصلان جيدًا. تقريبًا أي مصنف يحلها؛ مفيد للتحقق من أن خط الأنابيب لديك يعمل.",
    "ds.blobs3.label": "🔵 تجمعات (3 فئات)",
    "ds.blobs3.desc": "ثلاثة تجمعات غاوسية — أساس متعدد الفئات. يختبر ما إذا كان النموذج يتعامل بشكل صحيح مع أكثر من فئتين.",
    "ds.iris.label": "🌸 السوسن",
    "ds.iris.desc": "150 زهرة سوسن، 3 أنواع (setosa, versicolor, virginica). تُعرض الميزتان الأوليان؛ setosa قابلة للفصل خطيًا، الأخريان تتداخلان.",
    "ds.wine.label": "🍷 النبيذ",
    "ds.wine.desc": "178 نبيذًا إيطاليًا من 3 أصناف، 13 ميزة كيميائية. يتم عرض الميزتين الأوليين فقط؛ المجموعة الكاملة هي ما تستخدمه درجات CV.",
    "ds.cancer.label": "🔬 سرطان الثدي",
    "ds.cancer.desc": "569 عينة كتلة ثدي (خبيث مقابل حميد) مع 30 ميزة. مُختزلة إلى 2D عبر PCA للتصور؛ المصنف الأساسي يستخدم الميزات الأصلية الـ 30.",

    "mdl.dt.label": "شجرة قرار",
    "mdl.dt.desc": "تقسم فضاء الميزات بقواعد if/else متعامدة مع المحاور. سهلة القراءة، عرضة للإفراط في التعلم إذا نمت بعمق.",
    "mdl.rf.label": "غابة عشوائية",
    "mdl.rf.desc": "تجمع العديد من أشجار القرار غير المترابطة وتحسب متوسط أصواتها. افتراضي قوي — يعمل بشكل جيد على معظم البيانات الجدولية.",
    "mdl.gb.label": "Gradient Boosting",
    "mdl.gb.desc": "يبني أشجارًا ضحلة بالتسلسل، كل واحدة تصحح أخطاء السابقة. قوي على البيانات الجدولية؛ أبطأ من Random Forest في التدريب.",
    "mdl.svm.label": "SVM (RBF)",
    "mdl.svm.desc": "آلة المتجهات الداعمة بنواة الدالة الشعاعية. مصنف هامش أقصى في فضاء ضمني لانهائي الأبعاد — يتعامل مع الحدود غير الخطية جيدًا.",
    "mdl.knn.label": "أقرب K جار",
    "mdl.knn.desc": "يتنبأ بفئة الأغلبية بين أقرب *k* نقطة تدريب. لا توجد مرحلة تدريب، لكن كل توقع يمسح المجموعة بأكملها.",
    "mdl.lr.label": "الانحدار اللوجستي",
    "mdl.lr.desc": "نموذج خطي ينتج احتمالات الفئة عبر السيغمويد. أبسط خط أساس جاد — سريع، قابل للتفسير، محدود بالمشاكل القابلة للفصل خطيًا.",
    "mdl.mlp.label": "شبكة عصبية (MLP)",
    "mdl.mlp.desc": "perceptron متعدد الطبقات — شبكة عصبية صغيرة كاملة الترابط. مقرّب عام، لكنه يحتاج إلى بيانات وضبط أكثر من تجمعات الأشجار للتفوق على البيانات الجدولية.",
    "mdl.nb.label": "Naive Bayes",
    "mdl.nb.desc": "يطبق قاعدة بايز بافتراض (غالبًا الخاطئ) أن الميزات مستقلة شرطيًا بالنظر إلى الفئة. قوي بشكل مفاجئ على النص.",
    "mdl.xgb.label": "⚡ XGBoost",
    "mdl.xgb.desc": "Extreme Gradient Boosting — المكتبة التي هيمنت على Kaggle في أواخر العقد الثاني. أداء قوي مباشرة على البيانات الجدولية.",
    "mdl.lgbm.label": "⚡ LightGBM",
    "mdl.lgbm.desc": "أشجار gradient boosting من مايكروسوفت بنمو حسب الورقة وتقسيمات معتمدة على الرسوم البيانية. أسرع من XGBoost على البيانات الكبيرة.",
    "mdl.cat.label": "⚡ CatBoost",
    "mdl.cat.desc": "أشجار gradient boosting من Yandex، مصممة للتعامل مع الميزات الفئوية بدون one-hot وتقليل انحياز التنبؤ.",
    "mdl.tabnet.label": "🧠 TabNet",
    "mdl.tabnet.desc": "شبكة عصبية تعتمد على الانتباه مصممة خصيصًا للبيانات الجدولية. تستخدم خطوات قرار متسلسلة مع اختيار متناثر للميزات.",
    "mdl.sdt.label": "🌳 Soft Decision Tree (neural-trees)",
    "mdl.sdt.desc": "شجرة قرار قابلة للاشتقاق من حزمة *neural-trees*: كل عقدة داخلية بوابة سيغمويد، وكل ورقة softmax على الفئات، والشجرة بأكملها مدربة من البداية للنهاية بانحدار التدرج.",

    "single.train_acc": "دقة التدريب",
    "single.test_acc": "دقة الاختبار",
    "single.cv_mean_std": "متوسط CV ± الانحراف",
    "single.dataset_size": "حجم البيانات",
    "single.dataset_size_value": "{n} عينة / {k} فئة",
    "single.delta_train": "{d:+.3f} مقابل التدريب",
    "single.spinner": "جارٍ تدريب {model}...",
    "single.boundary_title": "{model} — حد القرار",
    "single.cm_title": "مصفوفة الالتباس (الاختبار)",
    "single.cv_title": "دقة CV 5-طيات",

    "compare.spinner": "جارٍ تدريب {a} و {b}...",
    "compare.model_a_panel": "النموذج A: {label}",
    "compare.model_b_panel": "النموذج B: {label}",
    "compare.train_acc": "دقة التدريب",
    "compare.test_acc": "دقة الاختبار",
    "compare.cv_mean": "متوسط CV",
    "compare.boundary_a": "النموذج A — حد القرار",
    "compare.boundary_b": "النموذج B — حد القرار",
    "compare.cm_a": "النموذج A — مصفوفة الالتباس",
    "compare.cm_b": "النموذج B — مصفوفة الالتباس",
    "compare.head_to_head": "مقارنة وجهًا لوجه",
    "compare.winner": "الفائز (دقة الاختبار)",
    "compare.acc_diff": "فرق الدقة",
    "compare.cv_compare": "CV A مقابل CV B",
    "compare.overfit_compare": "الإفراط (A مقابل B)",
    "compare.bar_title": "مقارنة النماذج",
    "compare.metric": "المقياس",
    "compare.metric.train_acc": "دقة التدريب",
    "compare.metric.test_acc": "دقة الاختبار",
    "compare.metric.cv_mean": "متوسط CV",
    "compare.score": "الدرجة",
    "compare.model": "النموذج",

    "sig.title": "اختبارات الدلالة الإحصائية",
    "sig.unavailable": "`mlxtend` غير مثبت؛ اختبارات الدلالة غير متاحة. ثبّتها عبر `pip install mlxtend`.",
    "sig.spinner": "جارٍ تشغيل اختبارات *t* المزدوج، McNemar و *F* 5×2cv...",
    "sig.error": "فشل الاختبار: {err}",
    "sig.col_test": "الاختبار",
    "sig.col_stat": "الإحصاءة",
    "sig.col_p": "قيمة p",
    "sig.col_decision": "القرار (α=0.05)",
    "sig.t_test": "اختبار *t* المزدوج (5×2cv)",
    "sig.f_test": "اختبار *F* 5×2cv (Alpaydın)",
    "sig.mcnemar": "McNemar (الاختبار، مصحح)",
    "sig.reject": "رفض H₀",
    "sig.keep": "عدم رفض H₀",
    "sig.h0_caption": "H₀: للنموذجين أداء تعميم متساوٍ. *الرفض* يعني أن الفرق المرصود غير محتمل تحت H₀.",

    "bench.spinner_init": "جارٍ تدريب المصنفات...",
    "bench.spinner_step": "تم تدريب {label} ({i}/{n})",
    "bench.leaderboard": "لوحة المتصدرين",
    "bench.cv_chart": "دقة التحقق المتقاطع",
    "bench.cv_axis": "دقة CV 5-طيات",
    "bench.boundaries": "حدود القرار",
    "bench.confusions": "مصفوفات الالتباس",
    "bench.col_model": "النموذج",
    "bench.col_train_acc": "دقة التدريب",
    "bench.col_test_acc": "دقة الاختبار",
    "bench.col_cv_mean": "متوسط CV",
    "bench.col_cv_std": "انحراف CV",
    "bench.col_status": "الحالة",
    "bench.all_failed": "فشل جميع المصنفات في التدريب. تحقق من البيانات والحزم المثبتة.",
    "bench.panel_label": "{label} (الدقة {acc:.3f})",

    "footer.text": "ML Playground · خوارزميات من scikit-learn، XGBoost، LightGBM، CatBoost، TabNet، neural-trees · اختبارات عبر mlxtend · مستوحى من *Introduction to Machine Learning* (Alpaydın, MIT Press, 2020)",
}

# ─────────────────────────────────────────────────────────────────────────────
# Русский (Russian)
# ─────────────────────────────────────────────────────────────────────────────
T["ru"] = {
    "app.title": "ML Playground",
    "app.subtitle": "Выберите алгоритм, выберите датасет, посмотрите, что получится. Сравните две модели лоб в лоб или прогоните все классификаторы по тому же протоколу, что используется в статье neural-trees.",

    "landing.title": "Добро пожаловать",
    "landing.body": (
        "Этот playground позволяет обучать классические модели машинного "
        "обучения на известных датасетах, видеть, **где именно каждая модель "
        "проводит границу между классами**, и сравнивать модели лоб в лоб "
        "со статистической строгостью.\n\n"
        "**Как пользоваться:**\n"
        "1. **Выберите датасет** в боковой панели (у каждого есть инфо-панель).\n"
        "2. **Выберите режим** — одна модель, дуэль двух моделей или полный бенчмарк.\n"
        "3. **Выберите модель(и)** и при желании настройте гиперпараметры.\n"
        "4. Нажмите **▶ Запустить**.\n\n"
        "Точки обучения — кружки, тестовые точки — ромбы. Цветные области — "
        "предсказания классов на мелкой сетке."
    ),

    "sidebar.language": "🌍 Язык",
    "sidebar.run": "▶ Запустить",
    "sidebar.about_dataset": "ℹ Об этом датасете",
    "sidebar.about_mode": "ℹ Об этом режиме",
    "sidebar.about_model": "ℹ Об этой модели",
    "sidebar.installed_backends": "Установленные пакеты",

    "sidebar.dataset.title": "📊 Датасет",
    "sidebar.dataset.choose": "Выберите датасет",
    "sidebar.dataset.samples": "Количество образцов",
    "sidebar.dataset.noise": "Шум",
    "sidebar.dataset.test_size": "Доля теста",
    "sidebar.dataset.seed": "Случайное зерно",

    "sidebar.mode.title": "🎯 Режим",
    "sidebar.mode.single": "🔬 Одна модель",
    "sidebar.mode.compare": "⚔️ Сравнить две модели",
    "sidebar.mode.benchmark": "🏆 Бенчмарк всех моделей",

    "sidebar.model_a": "🤖 Модель A",
    "sidebar.model_b": "🤖 Модель B",
    "sidebar.algorithm_a": "Алгоритм A",
    "sidebar.algorithm_b": "Алгоритм B",
    "sidebar.hyperparameters_a": "Гиперпараметры A",
    "sidebar.hyperparameters_b": "Гиперпараметры B",
    "sidebar.benchmark_info": "На выбранном датасете будет обучено **{n}** классификаторов с дефолтными гиперпараметрами и показано в виде сетки.",

    "mode.single.title": "Одна модель",
    "mode.single.desc": "Обучите один классификатор на одном датасете и изучите его границу решения, матрицу ошибок и точность 5-кратной кросс-валидации. Лучше всего, чтобы **понять, как ведёт себя один алгоритм** перед сравнением с другими.",
    "mode.compare.title": "Сравнить две модели",
    "mode.compare.desc": "Обучите два классификатора на **одной и той же** разбивке train/test и запустите три статистических теста: парный *t*-тест, тест Макнемара и *F*-тест 5×2cv Альпайдына. Отвечает на вопрос: 'действительно ли модель A лучше B, или ей просто повезло?'",
    "mode.benchmark.title": "Бенчмарк всех моделей",
    "mode.benchmark.desc": "Обучите каждый доступный классификатор с разумными дефолтными гиперпараметрами и получите таблицу лидеров, столбчатую диаграмму CV-точности, сетку границ решений и сетку матриц ошибок.",

    "ds.moons.label": "🌙 Луны",
    "ds.moons.desc": "Два переплетающихся полумесяца. Классический тест нелинейного разделения — линейные модели (логистическая регрессия) не справляются; деревья и ядерные методы — справляются.",
    "ds.circles.label": "⭕ Круги",
    "ds.circles.desc": "Два концентрических круга. Невозможно для любой линейной границы; идеально показывает возможности SVM с ядром и ансамблей деревьев.",
    "ds.blobs2.label": "🔵 Блобы (2 класса)",
    "ds.blobs2.desc": "Два хорошо разделённых гауссовых кластера. Почти любой классификатор справится; полезно как проверка работоспособности pipeline.",
    "ds.blobs3.label": "🔵 Блобы (3 класса)",
    "ds.blobs3.desc": "Три гауссовых кластера — мультиклассовая база. Проверяет, корректно ли модель обрабатывает больше двух классов.",
    "ds.iris.label": "🌸 Ирис",
    "ds.iris.desc": "150 цветков ириса, 3 вида (setosa, versicolor, virginica). Показаны первые две характеристики; setosa линейно разделима, две другие пересекаются.",
    "ds.wine.label": "🍷 Вино",
    "ds.wine.desc": "178 итальянских вин 3 сортов, 13 химических признаков. Здесь визуализированы только первые два; полный набор используется для CV-оценок.",
    "ds.cancer.label": "🔬 Рак груди",
    "ds.cancer.desc": "569 образцов опухолей груди (злокачественные/доброкачественные) с 30 признаками. Сведены к 2D через PCA для визуализации; классификатор использует все 30 оригинальных признаков.",

    "mdl.dt.label": "Дерево решений",
    "mdl.dt.desc": "Делит пространство признаков последовательностью if/else правил, выровненных по осям. Легко читается, склонно к переобучению при большой глубине.",
    "mdl.rf.label": "Случайный лес",
    "mdl.rf.desc": "Усредняет голоса множества декоррелированных деревьев решений. Надёжный default — хорошо работает на большинстве табличных данных из коробки.",
    "mdl.gb.label": "Gradient Boosting",
    "mdl.gb.desc": "Строит неглубокие деревья последовательно, каждое исправляет ошибки предыдущего. Сильный на табличных данных; обучается медленнее Random Forest.",
    "mdl.svm.label": "SVM (RBF)",
    "mdl.svm.desc": "Метод опорных векторов с радиально-базисным ядром. Классификатор максимального зазора в неявном бесконечномерном пространстве — хорошо обрабатывает нелинейные границы.",
    "mdl.knn.label": "K ближайших соседей",
    "mdl.knn.desc": "Предсказывает мажоритарный класс среди *k* ближайших обучающих точек. Нет фазы обучения, но каждое предсказание сканирует весь датасет.",
    "mdl.lr.label": "Логистическая регрессия",
    "mdl.lr.desc": "Линейная модель, выдающая вероятности классов через сигмоиду. Простейшая серьёзная база — быстрая, интерпретируемая, ограничена линейно разделимыми задачами.",
    "mdl.mlp.label": "Нейросеть (MLP)",
    "mdl.mlp.desc": "Многослойный персептрон — небольшая полносвязная нейросеть. Универсальный аппроксиматор, но требует больше данных и тюнинга, чем ансамбли деревьев на табличных данных.",
    "mdl.nb.label": "Naive Bayes",
    "mdl.nb.desc": "Применяет правило Байеса в предположении (часто неверном), что признаки условно независимы при заданном классе. Удивительно силён на тексте и разреженных данных.",
    "mdl.xgb.label": "⚡ XGBoost",
    "mdl.xgb.desc": "Extreme Gradient Boosting — библиотека градиентного бустинга, доминировавшая на Kaggle в конце 2010-х. Сильная производительность из коробки на табличных данных.",
    "mdl.lgbm.label": "⚡ LightGBM",
    "mdl.lgbm.desc": "Деревья градиентного бустинга от Microsoft с ростом по листьям и разбиениями на основе гистограмм. Быстрее XGBoost на больших датасетах.",
    "mdl.cat.label": "⚡ CatBoost",
    "mdl.cat.desc": "Деревья градиентного бустинга от Yandex, спроектированные для категориальных признаков без one-hot и снижения сдвига предсказаний.",
    "mdl.tabnet.label": "🧠 TabNet",
    "mdl.tabnet.desc": "Нейросеть на основе внимания, специально для табличных данных. Использует последовательные шаги решения с разреженным выбором признаков.",
    "mdl.sdt.label": "🌳 Soft Decision Tree (neural-trees)",
    "mdl.sdt.desc": "Дифференцируемое дерево решений из пакета *neural-trees*: каждый внутренний узел — сигмоидный гейт, каждый лист — softmax по классам, всё дерево обучается end-to-end градиентным спуском.",

    "single.train_acc": "Точность обучения",
    "single.test_acc": "Точность теста",
    "single.cv_mean_std": "Среднее CV ± Std",
    "single.dataset_size": "Размер датасета",
    "single.dataset_size_value": "{n} образцов / {k} классов",
    "single.delta_train": "{d:+.3f} к обучению",
    "single.spinner": "Обучение {model}...",
    "single.boundary_title": "{model} — Граница решения",
    "single.cm_title": "Матрица ошибок (тест)",
    "single.cv_title": "Точность CV 5 фолдов",

    "compare.spinner": "Обучение {a} и {b}...",
    "compare.model_a_panel": "Модель A: {label}",
    "compare.model_b_panel": "Модель B: {label}",
    "compare.train_acc": "Точн. обуч.",
    "compare.test_acc": "Точн. тест",
    "compare.cv_mean": "Среднее CV",
    "compare.boundary_a": "Модель A — Граница решения",
    "compare.boundary_b": "Модель B — Граница решения",
    "compare.cm_a": "Модель A — Матрица ошибок",
    "compare.cm_b": "Модель B — Матрица ошибок",
    "compare.head_to_head": "Сравнение лоб в лоб",
    "compare.winner": "Победитель (точн. тест)",
    "compare.acc_diff": "Разница точностей",
    "compare.cv_compare": "CV A против CV B",
    "compare.overfit_compare": "Переобучение (A против B)",
    "compare.bar_title": "Сравнение моделей",
    "compare.metric": "Метрика",
    "compare.metric.train_acc": "Точность обучения",
    "compare.metric.test_acc": "Точность теста",
    "compare.metric.cv_mean": "Среднее CV",
    "compare.score": "Значение",
    "compare.model": "Модель",

    "sig.title": "Тесты статистической значимости",
    "sig.unavailable": "`mlxtend` не установлен; тесты значимости недоступны. Установите через `pip install mlxtend`.",
    "sig.spinner": "Запуск парного *t*, McNemar и *F*-теста 5×2cv...",
    "sig.error": "Тест не удался: {err}",
    "sig.col_test": "Тест",
    "sig.col_stat": "Статистика",
    "sig.col_p": "p-значение",
    "sig.col_decision": "Решение (α=0.05)",
    "sig.t_test": "Парный *t*-тест (5×2cv)",
    "sig.f_test": "*F*-тест 5×2cv (Альпайдын)",
    "sig.mcnemar": "Макнемар (тест, скорректированный)",
    "sig.reject": "отвергнуть H₀",
    "sig.keep": "не отвергнуть H₀",
    "sig.h0_caption": "H₀: две модели имеют равную обобщающую способность. *Отвергнуть* означает, что наблюдаемая разница маловероятна при H₀.",

    "bench.spinner_init": "Обучение классификаторов...",
    "bench.spinner_step": "Обучен {label} ({i}/{n})",
    "bench.leaderboard": "Таблица лидеров",
    "bench.cv_chart": "Точность кросс-валидации",
    "bench.cv_axis": "Точность CV 5 фолдов",
    "bench.boundaries": "Границы решения",
    "bench.confusions": "Матрицы ошибок",
    "bench.col_model": "Модель",
    "bench.col_train_acc": "Точн. обуч.",
    "bench.col_test_acc": "Точн. тест",
    "bench.col_cv_mean": "Среднее CV",
    "bench.col_cv_std": "Std CV",
    "bench.col_status": "Статус",
    "bench.all_failed": "Все классификаторы не смогли обучиться. Проверьте датасет и установленные пакеты.",
    "bench.panel_label": "{label} (точн. {acc:.3f})",

    "footer.text": "ML Playground · Алгоритмы из scikit-learn, XGBoost, LightGBM, CatBoost, TabNet, neural-trees · Тесты через mlxtend · Вдохновлено *Introduction to Machine Learning* (Альпайдын, MIT Press, 2020)",
}


def t(key: str, lang: str = "en", **fmt) -> str:
    """Look up a translation key. Falls back to English if missing, then to the key itself."""
    txt = T.get(lang, {}).get(key)
    if txt is None:
        txt = T["en"].get(key, key)
    if fmt:
        try:
            return txt.format(**fmt)
        except (KeyError, IndexError):
            return txt
    return txt
