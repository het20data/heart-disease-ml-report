"""
Heart Disease UCI — Statistical Machine Learning Analysis
DG4MLEL: Machine Learning for Data Science and AI

Reproduces all statistics, figures, and model results referenced in the
accompanying report. Outputs are written to ./outputs/figs and ./outputs/tables.

Usage:
    python heart_disease_analysis.py

Requirements:
    pip install -r requirements.txt

Dataset:
    Place 'heart_disease_uci.csv' in the same folder as this script
    (download from https://archive.ics.uci.edu/dataset/45/heart+disease
    or the Kaggle mirror: https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score,
                              roc_curve)
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

np.random.seed(42)
sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 10})

DATA_PATH = "heart_disease_uci.csv"
FIG_DIR = "outputs/figs"
TABLE_DIR = "outputs/tables"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


def load_and_clean(path=DATA_PATH):
    """Load the dataset and apply the data-quality fixes described in the report."""
    df = pd.read_csv(path)
    df["target"] = (df["num"] > 0).astype(int)

    # chol == 0 / trestbps == 0 are physiologically impossible -> treat as missing
    df.loc[df["chol"] == 0, "chol"] = np.nan
    df.loc[df["trestbps"] == 0, "trestbps"] = np.nan

    for c in ["fbs", "exang"]:
        df[c] = df[c].astype(str).map(
            {"True": 1, "False": 0, "TRUE": 1, "FALSE": 0, "nan": np.nan}
        )
    return df


def build_preprocessor(num_feats, cat_feats):
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([("num", num_pipe, num_feats), ("cat", cat_pipe, cat_feats)])


def section_1_eda(df):
    """Section 1: descriptive statistics, correlation, and figures 1-2."""
    num_cols = ["age", "trestbps", "chol", "thalch", "oldpeak"]
    desc = df[num_cols].describe(percentiles=[.25, .5, .75]).T
    desc["skew"] = df[num_cols].skew()
    desc.to_csv(f"{TABLE_DIR}/descriptive_stats.csv")

    corr = df[num_cols + ["ca", "target"]].corr(method="pearson")
    corr.to_csv(f"{TABLE_DIR}/correlation.csv")

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": .8})
    plt.title("Correlation Matrix (Numeric Features & Target)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig1_correlation_heatmap.png", dpi=150)
    plt.close()

    plt.figure(figsize=(5.5, 4))
    sns.boxplot(x="target", y="age", data=df, hue="target",
                palette=["#4C72B0", "#DD8452"], legend=False)
    plt.xlabel("Heart disease present (0=No, 1=Yes)")
    plt.title("Age Distribution by Disease Status")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig2_age_by_target.png", dpi=150)
    plt.close()

    print("[1] EDA complete -> outputs/tables, outputs/figs")
    return desc, corr


def section_1_5_pca(X, num_feats, cat_feats):
    """Section 1.5: PCA dimensionality-reduction analysis (figures 3-4)."""
    pre = build_preprocessor(num_feats, cat_feats)
    Xt = pre.fit_transform(X)

    pca = PCA(n_components=Xt.shape[1], random_state=42)
    pcs = pca.fit_transform(Xt)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    n80, n90 = int(np.argmax(cum >= 0.80) + 1), int(np.argmax(cum >= 0.90) + 1)

    plt.figure(figsize=(6, 4.5))
    plt.plot(range(1, len(evr) + 1), cum, marker="o", color="#4C72B0")
    plt.axhline(0.9, color="red", linestyle="--", linewidth=1, label="90% variance")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA: Cumulative Explained Variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig3_pca_variance.png", dpi=150)
    plt.close()

    with open(f"{TABLE_DIR}/pca_results.json", "w") as f:
        json.dump({"evr": evr.tolist(), "cum": cum.tolist(), "n80": n80, "n90": n90,
                   "total_dims": int(Xt.shape[1])}, f, indent=2)

    print(f"[1.5] PCA complete -> {n80} components for 80% variance, {n90} for 90%")
    return pcs, evr


def section_2_4_models(X, y, num_feats, cat_feats):
    """Sections 2-4: train Model A (Logistic Regression) & Model B (Random Forest),
    evaluate, and produce comparison figures/tables."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    pre = build_preprocessor(num_feats, cat_feats)

    logreg = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, random_state=42))])
    logreg.fit(X_train, y_train)

    rf = Pipeline([("pre", pre), ("clf", RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=42))])
    rf.fit(X_train, y_train)

    def evaluate(model):
        out = {}
        for split, Xs, ys in [("train", X_train, y_train), ("test", X_test, y_test)]:
            pred = model.predict(Xs)
            proba = model.predict_proba(Xs)[:, 1]
            out[split] = {
                "accuracy": accuracy_score(ys, pred),
                "precision": precision_score(ys, pred),
                "recall": recall_score(ys, pred),
                "f1": f1_score(ys, pred),
                "roc_auc": roc_auc_score(ys, proba),
            }
        out["confusion_test"] = confusion_matrix(y_test, model.predict(X_test)).tolist()
        return out

    results = {"logreg": evaluate(logreg), "rf": evaluate(rf)}
    with open(f"{TABLE_DIR}/model_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[2-4] Model results:")
    print(json.dumps(results, indent=2))

    # --- ROC curves (Figure 8) ---
    plt.figure(figsize=(5.5, 5))
    for model, name, color in [(logreg, "Logistic Regression", "#4C72B0"),
                                (rf, "Random Forest", "#DD8452")]:
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves: Test Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig8_roc_curves.png", dpi=150)
    plt.close()

    # --- Logistic regression coefficients (Figure 5) ---
    ohe_names = logreg.named_steps["pre"].named_transformers_["cat"].named_steps["ohe"]\
        .get_feature_names_out(cat_feats)
    all_names = num_feats + list(ohe_names)
    coefs = logreg.named_steps["clf"].coef_[0]
    coef_df = pd.DataFrame({"feature": all_names, "coefficient": coefs}).sort_values("coefficient")
    coef_df.to_csv(f"{TABLE_DIR}/logreg_coefficients.csv", index=False)

    plt.figure(figsize=(6, 6))
    colors = ["#C44E52" if v < 0 else "#4C72B0" for v in coef_df["coefficient"]]
    plt.barh(coef_df["feature"], coef_df["coefficient"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Standardised Logistic Regression Coefficient")
    plt.title("Feature Influence (Model A: Logistic Regression)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig5_logreg_coefficients.png", dpi=150)
    plt.close()

    # --- Random Forest importances (Figure 6) ---
    rf_imp = rf.named_steps["clf"].feature_importances_
    rf_imp_df = pd.DataFrame({"feature": all_names, "importance": rf_imp})\
        .sort_values("importance", ascending=False)
    rf_imp_df.to_csv(f"{TABLE_DIR}/rf_importances.csv", index=False)

    plt.figure(figsize=(6, 6))
    top = rf_imp_df.head(12).sort_values("importance")
    plt.barh(top["feature"], top["importance"], color="#55A868")
    plt.xlabel("Gini Importance")
    plt.title("Feature Importance (Model B: Random Forest)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig6_rf_importance.png", dpi=150)
    plt.close()

    # --- Confusion matrices (Figure 9) ---
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (name, key, cmap) in zip(axes, [("Logistic Regression", "logreg", "Blues"),
                                             ("Random Forest", "rf", "Oranges")]):
        cm = np.array(results[key]["confusion_test"])
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False, ax=ax,
                    xticklabels=["No Disease", "Disease"],
                    yticklabels=["No Disease", "Disease"])
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.suptitle("Confusion Matrices (Test Set)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig9_confusion_matrices.png", dpi=150)
    plt.close()

    # --- Learning curves (Figure 7) ---
    plt.figure(figsize=(6, 5))
    for model, name, color in [(logreg, "Logistic Regression", "#4C72B0"),
                                (rf, "Random Forest", "#DD8452")]:
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=5, scoring="accuracy",
            train_sizes=np.linspace(0.1, 1.0, 8), random_state=42)
        plt.plot(train_sizes, train_scores.mean(axis=1), "--", color=color, label=f"{name} (train)")
        plt.plot(train_sizes, test_scores.mean(axis=1), "-", color=color, label=f"{name} (CV)")
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig7_learning_curves.png", dpi=150)
    plt.close()

    return logreg, rf, X_train, X_test, y_train, y_test, results


def section_4_4_aic_bic(X, y, num_feats, cat_feats):
    """Section 4.4: AIC/BIC for the logistic regression model (full-data fit)."""
    pre = build_preprocessor(num_feats, cat_feats)
    Xt = pre.fit_transform(X)
    Xt_const = sm.add_constant(Xt)
    model = sm.Logit(y, Xt_const).fit(disp=0)
    print(f"[4.4] AIC={model.aic:.2f}  BIC={model.bic:.2f}  LogLik={model.llf:.2f}")
    return model.aic, model.bic, model.llf


def section_3_gradient_descent_demo():
    """Section 3.2: illustrative hand-worked gradient descent example."""
    X = np.array([
        [0.06, 1.30],
        [-1.65, -0.81],
        [1.43, 0.57],
        [-0.59, -0.81],
    ])
    y = np.array([1, 0, 1, 0])
    n = len(y)
    w, b, eta = np.array([0.0, 0.0]), 0.0, 0.5

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def bce(w, b):
        p = sigmoid(X @ w + b)
        eps = 1e-9
        return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

    print(f"[3.2] iter 0: w={w}, b={b:.3f}, loss={bce(w, b):.4f}")
    for i in range(5):
        p = sigmoid(X @ w + b)
        grad_w = (1 / n) * X.T @ (p - y)
        grad_b = (1 / n) * np.sum(p - y)
        w = w - eta * grad_w
        b = b - eta * grad_b
        print(f"[3.2] iter {i+1}: w={w}, b={b:.4f}, loss={bce(w, b):.4f}")


def main():
    df = load_and_clean(DATA_PATH)
    section_1_eda(df)

    features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalch", "exang", "oldpeak", "slope"]
    num_feats = ["age", "trestbps", "chol", "thalch", "oldpeak"]
    cat_feats = ["sex", "cp", "fbs", "restecg", "exang", "slope"]

    X, y = df[features].copy(), df["target"].copy()

    section_1_5_pca(X, num_feats, cat_feats)
    section_2_4_models(X, y, num_feats, cat_feats)
    section_4_4_aic_bic(X, y, num_feats, cat_feats)
    section_3_gradient_descent_demo()

    print("\nAll outputs written to ./outputs/figs and ./outputs/tables")


if __name__ == "__main__":
    main()
