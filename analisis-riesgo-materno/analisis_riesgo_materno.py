"""
analisis_riesgo_materno.py
Proyecto: Análisis y Modelado Predictivo de Riesgo de Muerte Materna (dataset sintético)

Requerimientos cubiertos:
- Generación de DataFrame sintético con variables:
  Edad, Atencion_Prenatal, Presion_Arterial, Riesgo_Previo, Muerte_Materna
- Train/Test split 70/30
- Enfoque de inferencia: LogisticRegression + coeficientes + Odds Ratio de Riesgo_Previo
- Enfoque de predicción: 5-fold CV (AUC promedio) + métricas en test (AUC, Sensibilidad, VPP)

Ejecución:
  python analisis_riesgo_materno.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, precision_score


def make_synthetic_dataset(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Crea un dataset sintético para riesgo de muerte materna (evento raro)."""
    rng = np.random.default_rng(seed)

    # Features
    edad = np.clip(rng.normal(28, 6, n), 14, 48).round(0)
    atencion_prenatal = np.clip(rng.gamma(shape=2.0, scale=10.0, size=n), 0, 90).round(0)  # días de retraso
    presion = np.clip(
        rng.normal(118, 15, n) + 0.15 * atencion_prenatal + 0.35 * (edad - 28),
        85, 200
    ).round(0)
    riesgo_previo = rng.binomial(
        1,
        p=np.clip(0.12 + 0.01 * (edad - 28) / 10 + 0.003 * atencion_prenatal, 0.05, 0.35),
        size=n
    )

    # Proceso generador de datos (logístico) para producir un outcome raro (~1%)
    b0 = -5.4
    b_edad = 0.03
    b_prenatal = 0.025
    b_pa = 0.018
    b_riesgo = 1.25

    lin = (
        b0
        + b_edad * (edad - 28)
        + b_prenatal * (atencion_prenatal - 10)
        + b_pa * (presion - 120)
        + b_riesgo * riesgo_previo
    )
    p = 1 / (1 + np.exp(-lin))
    muerte = rng.binomial(1, p=np.clip(p, 0, 0.95), size=n)

    return pd.DataFrame(
        {
            "Edad": edad.astype(int),
            "Atencion_Prenatal": atencion_prenatal.astype(int),
            "Presion_Arterial": presion.astype(int),
            "Riesgo_Previo": riesgo_previo.astype(int),
            "Muerte_Materna": muerte.astype(int),
        }
    )


def main() -> None:
    # 1) Datos
    df = make_synthetic_dataset()

    X = df[["Edad", "Atencion_Prenatal", "Presion_Arterial", "Riesgo_Previo"]]
    y = df["Muerte_Materna"].astype(int)

    print("\n=== Resumen del dataset ===")
    print(df.head())
    print("\nPrevalencia (Muerte_Materna=1):", f"{y.mean():.4f}")

    # 2) Split 70/30
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # 3A) Enfoque de inferencia (modelo base)
    # class_weight="balanced" ayuda en escenarios de evento raro.
    model = LogisticRegression(max_iter=3000, solver="lbfgs", class_weight="balanced")
    model.fit(X_train, y_train)

    coefs = model.coef_[0]
    feature_names = X.columns.tolist()
    coef_map = dict(zip(feature_names, coefs))

    # Odds Ratio (OR) para Riesgo_Previo
    or_riesgo_previo = float(np.exp(coef_map["Riesgo_Previo"]))

    print("\n=== Enfoque de Inferencia (LogisticRegression) ===")
    print("Coeficientes (log-odds):")
    for name in feature_names:
        print(f"  - {name}: {coef_map[name]:.4f}")

    print(f"\nOdds Ratio para Riesgo_Previo: {or_riesgo_previo:.3f}")

    # 3B) Enfoque de predicción (robustez + test)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    print("\n=== Enfoque de Predicción ===")
    print(f"AUC promedio (CV 5-fold): {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

    # Modelo final + métricas en test
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc_test = roc_auc_score(y_test, y_pred_proba)
    sensibilidad = recall_score(y_test, y_pred)
    vpp = precision_score(y_test, y_pred, zero_division=0)

    print("\nMétricas en Test (30% hold-out):")
    print(f"  - AUC ROC: {auc_test:.3f}")
    print(f"  - Sensibilidad (Recall): {sensibilidad:.3f}")
    print(f"  - VPP (Precision): {vpp:.3f}")

    print("\nNota: Si el objetivo clínico es maximizar sensibilidad, normalmente se ajusta el umbral (<0.5).")


if __name__ == "__main__":
    main()
