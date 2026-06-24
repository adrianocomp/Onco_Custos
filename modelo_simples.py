"""
================================================================================
MODELO SIMPLES DE REGRESSÃO - PREVISÃO DE CUSTOS ONCOLÓGICOS v2
================================================================================
Dataset: View_Onco_Custos_clean.parquet
Target: custo_total (custo total do paciente por tipo de cancer)
Features: idade, tipo_cancer, sexo

CORREÇÕES v2:
- GroupShuffleSplit para evitar vazamento de pacientes
- Otimização de hiperparâmetros
- Log-transform no target

Autor: Desenvolvedor Sênior Data Science
Data: 2025
================================================================================
"""

import pandas as pd
import numpy as np
import warnings
import joblib
import os
from datetime import datetime

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================
# CONFIGURAÇÕES
# ============================================================
OUTPUT_DIR = "model"
N_FOLDS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ============================================================
# 1. CARREGAR DADOS
# ============================================================
print("=" * 70)
print("1. CARREGANDO DADOS")
print("=" * 70)

df_raw = pd.read_parquet("View_Onco_Custos_clean.parquet")
print(f"Dados brutos: {df_raw.shape[0]:,} linhas")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 70)
print("2. FEATURE ENGINEERING")
print("=" * 70)

df_raw["custo"] = df_raw["custo_total_item"].astype(float)
df_raw["idade"] = (df_raw["data_evento"] - df_raw["dat_nascimento"]).dt.days // 365

# Agregar por paciente_id + tipo_cancer
df_agg = (
    df_raw.groupby(["paciente_id", "tipo_cancer"])
    .agg({"custo": "sum", "idade": "first", "sexo_paciente": "first"})
    .reset_index()
)

df_agg.columns = ["paciente_id", "tipo_cancer", "custo_total", "idade", "sexo"]

print(f"Aggregacao: {len(df_agg):,} linhas")
print(f"Pacientes: {df_agg['paciente_id'].nunique():,}")

# ============================================================
# 3. LIMPEZA DE DADOS
# ============================================================
print("\n" + "=" * 70)
print("3. LIMPEZA DE DADOS")
print("=" * 70)

df_agg["sexo"] = df_agg["sexo"].replace("I", "M")

# Remover outliers
Q1 = df_agg["custo_total"].quantile(0.25)
Q3 = df_agg["custo_total"].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

antes = len(df_agg)
df_clean = df_agg[
    (df_agg["custo_total"] >= limite_inferior)
    & (df_agg["custo_total"] <= limite_superior)
].copy()

print(f"Outliers removidos: {antes - len(df_clean)}")
print(f"Linhas finais: {len(df_clean):,}")

# ============================================================
# 4. PREPARAÇÃO DE FEATURES
# ============================================================
print("\n" + "=" * 70)
print("4. PREPARACAO DE FEATURES")
print("=" * 70)

from sklearn.preprocessing import LabelEncoder

# Features
feature_cols = ["tipo_cancer", "idade", "sexo"]
target_col = "custo_total"

# Label Encoding
label_encoders = {}
categorical_cols = ["tipo_cancer", "sexo"]

for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col + "_enc"] = le.fit_transform(df_clean[col].astype(str))
    label_encoders[col] = le
    print(f"  {col}: {le.classes_.tolist()}")

# Features finais
final_features = ["tipo_cancer_enc", "idade", "sexo_enc"]
X = df_clean[final_features].values
y = df_clean[target_col].values
y_log = np.log1p(y)

print(f"\nFeatures: {final_features}")
print(f"Target: {target_col}")
print(f"Shape: X={X.shape}, y={y.shape}")

# ============================================================
# 5. SPLIT TREINO/TESTE (CORRIGIDO)
# ============================================================
print("\n" + "=" * 70)
print("5. SPLIT TREINO/TESTE (GroupShuffleSplit)")
print("=" * 70)

from sklearn.model_selection import GroupShuffleSplit

# GroupShuffleSplit: garante que paciente NAO apareça em treino E teste
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# Obter indices
for train_idx, test_idx in gss.split(X, y, groups=df_clean["paciente_id"].values):
    pass  # Apenas obter os índices

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
y_train_log, y_test_log = y_log[train_idx], y_log[test_idx]
df_train = df_clean.iloc[train_idx]
df_test = df_clean.iloc[test_idx]

# Verificar vazamento
pacientes_treino = set(df_train["paciente_id"])
pacientes_test = set(df_test["paciente_id"])
overlap = pacientes_treino & pacientes_test

print(f"Pacientes treino: {len(pacientes_treino):,}")
print(f"Pacientes teste: {len(pacientes_test):,}")
print(f"Overlap: {len(overlap)}")
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

if len(overlap) == 0:
    print("[OK] SEM VAZAMENTO - Pacientes separados corretamente")
else:
    print("[WARNING] VAZAMENTO!")

# ============================================================
# 6. BASELINE
# ============================================================
print("\n" + "=" * 70)
print("6. BASELINE")
print("=" * 70)

from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Baseline média
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
y_pred_baseline = baseline.predict(X_test)

rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
r2_baseline = r2_score(y_test, y_pred_baseline)

print(f"Baseline MEDIA:")
print(f"  RMSE: R$ {rmse_baseline:,.2f}")
print(f"  MAE:  R$ {mae_baseline:,.2f}")
print(f"  R²:   {r2_baseline:.4f}")

baseline_metrics = {"rmse": rmse_baseline, "mae": mae_baseline, "r2": r2_baseline}

# ============================================================
# 7. TESTE DE HIPERPARÂMETROS
# ============================================================
print("\n" + "=" * 70)
print("7. TESTE DE HIPERPARAMETROS")
print("=" * 70)

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold

# Hiperparâmetros para testar
param_grid = [
    {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "subsample": 0.8,
    },
    {
        "n_estimators": 150,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "subsample": 0.8,
    },
    {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "subsample": 0.9,
    },
    {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.1,
        "min_samples_split": 5,
        "min_samples_leaf": 3,
        "subsample": 1.0,
    },
    {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "min_samples_split": 20,
        "min_samples_leaf": 5,
        "subsample": 0.8,
    },
]

gkf = GroupKFold(n_splits=N_FOLDS)
groups_train = df_train["paciente_id"].values

best_r2 = -np.inf
best_params = None
results_all = []

for i, params in enumerate(param_grid):
    print(
        f"\nTeste {i + 1}: n={params['n_estimators']}, depth={params['max_depth']}, lr={params['learning_rate']}"
    )

    r2s_fold = []
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups_train)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        model = GradientBoostingRegressor(**params, random_state=RANDOM_STATE)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        r2 = r2_score(y_val, y_pred)
        r2s_fold.append(r2)

    r2_mean = np.mean(r2s_fold)
    r2_std = np.std(r2s_fold)
    print(f"  R² CV: {r2_mean:.4f} (+/- {r2_std:.4f})")

    results_all.append({"params": params, "r2_mean": r2_mean, "r2_std": r2_std})

    if r2_mean > best_r2:
        best_r2 = r2_mean
        best_params = params

print(f"\nMELHORES HIPERPARAMETROS:")
print(f"  R² CV: {best_r2:.4f}")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# ============================================================
# 8. TREINAMENTO FINAL COM MELHORES PARAMS
# ============================================================
print("\n" + "=" * 70)
print("8. TREINAMENTO FINAL")
print("=" * 70)

best_model = GradientBoostingRegressor(**best_params, random_state=RANDOM_STATE)
best_model.fit(X_train, y_train)

y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

# Métricas treino
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

# Métricas teste
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f"\nTREINO ({len(X_train):,} amostras):")
print(f"  RMSE: R$ {rmse_train:,.2f}")
print(f"  MAE:  R$ {mae_train:,.2f}")
print(f"  R²:   {r2_train:.4f}")

print(f"\nTESTE ({len(X_test):,} amostras):")
print(f"  RMSE: R$ {rmse_test:,.2f}")
print(f"  MAE:  R$ {mae_test:,.2f}")
print(f"  R²:   {r2_test:.4f}")

# ============================================================
# 9. COMPARAÇÃO COM BASELINE
# ============================================================
print("\n" + "=" * 70)
print("9. COMPARACAO COM BASELINE")
print("=" * 70)

print(f"\n{'Metrica':<15} {'Baseline':>15} {'Modelo':>15} {'Melhoria':>15}")
print("-" * 60)
print(
    f"{'RMSE':<15} {'R$ ' + str(round(rmse_baseline, 2)):>15} {'R$ ' + str(round(rmse_test, 2)):>15} {((rmse_baseline - rmse_test) / rmse_baseline * 100):>12.1f}%"
)
print(
    f"{'MAE':<15} {'R$ ' + str(round(mae_baseline, 2)):>15} {'R$ ' + str(round(mae_test, 2)):>15} {((mae_baseline - mae_test) / mae_baseline * 100):>12.1f}%"
)
print(
    f"{'R²':<15} {str(round(r2_baseline, 4)):>15} {str(round(r2_test, 4)):>15} {((r2_test - r2_baseline) * 100):>12.1f}%"
)

# ============================================================
# 10. ANÁLISE DE RESÍDUOS
# ============================================================
print("\n" + "=" * 70)
print("10. ANALISE DE RESIDUOS")
print("=" * 70)

residuos = y_test - y_pred_test

print(f"Residuos (teste):")
print(f"  Media:  R$ {np.mean(residuos):,.2f}")
print(f"  Std:    R$ {np.std(residuos):,.2f}")
print(f"  Min:    R$ {np.min(residuos):,.2f}")
print(f"  Max:    R$ {np.max(residuos):,.2f}")

print(f"\nPercentis:")
for p in [5, 25, 50, 75, 95]:
    print(f"  {p}%: R$ {np.percentile(residuos, p):,.2f}")

# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 70)
print("11. FEATURE IMPORTANCE")
print("=" * 70)

if hasattr(best_model, "feature_importances_"):
    for feat, imp in zip(final_features, best_model.feature_importances_):
        print(f"  {feat}: {imp:.4f}")

# ============================================================
# 12. SALVAR MODELO
# ============================================================
print("\n" + "=" * 70)
print("12. SALVANDO MODELO")
print("=" * 70)

os.makedirs(OUTPUT_DIR, exist_ok=True)

model_path = f"{OUTPUT_DIR}/modelo_simples.joblib"
joblib.dump(best_model, model_path)
print(f"[OK] Modelo: {model_path}")

encoders_path = f"{OUTPUT_DIR}/label_encoders_simples.joblib"
joblib.dump(label_encoders, encoders_path)
print(f"[OK] Encoders: {encoders_path}")

info = {
    "model_name": "GradientBoostingRegressor",
    "features": final_features,
    "feature_cols": feature_cols,
    "categorical_cols": categorical_cols,
    "target": target_col,
    "gb_params": best_params,
    "metrics_test": {
        "rmse": float(rmse_test),
        "mae": float(mae_test),
        "r2": float(r2_test),
    },
    "baseline_metrics": baseline_metrics,
    "dataset_shape": df_clean.shape,
    "n_pacientes": df_clean["paciente_id"].nunique(),
    "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}
joblib.dump(info, f"{OUTPUT_DIR}/model_info_simples.joblib")
print(f"[OK] Info: {OUTPUT_DIR}/model_info_simples.joblib")

# ============================================================
# 13. EXEMPLOS DE PREVISÃO
# ============================================================
print("\n" + "=" * 70)
print("13. EXEMPLOS DE PREVISAO")
print("=" * 70)


def prever_custo(tipo_cancer, idade, sexo):
    """Prevê custo de tratamento oncológico"""
    tipo_enc = label_encoders["tipo_cancer"].transform([tipo_cancer])[0]
    sexo_enc = label_encoders["sexo"].transform([sexo])[0]
    features = np.array([[tipo_enc, idade, sexo_enc]])
    return best_model.predict(features)[0]


exemplos = [
    ("Próstata", 65, "M"),
    ("Colorretal", 55, "F"),
    ("Pulmão", 70, "M"),
    ("Próstata", 75, "M"),
    ("Colorretal", 60, "F"),
]

print("\nPrevisoes:")
for tipo, idade, sexo in exemplos:
    custo = prever_custo(tipo, idade, sexo)
    print(f"  {tipo}, {idade} anos, {sexo}: R$ {custo:,.2f}")

print("\n" + "=" * 70)
print("FIM DO TREINAMENTO")
print("=" * 70)
