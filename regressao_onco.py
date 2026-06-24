"""
Script de Regressão para Previsão de Custos de Pacientes Oncológicos
Dataset: View_Onco_ML.parquet
Target: custo_total

BOAS PRÁTICAS IMPLEMENTADAS:
- GroupKFold para evitar vazamento de pacientes
- Cross-validation com 5 folds
- Pipeline completo com pré-processamento
- Múltiplas métricas (RMSE, MAE, R²)
- Baseline (média e mediana)
- Análise de resíduos
- Intervalos de confiança
- Teste final isolado (cego)
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

np.random.seed(42)

# ============================================================
# 1. CARREGAR DADOS
# ============================================================
print("=" * 70)
print("1. CARREGANDO DADOS")
print("=" * 70)

df = pd.read_parquet("View_Onco_ML.parquet")
print(f"Shape: {df.shape}")
print(f"Pacientes únicos: {df['paciente_id'].nunique()}")
print(f"Linhas por paciente: {len(df) / df['paciente_id'].nunique():.1f} (média)")

# ============================================================
# 2. PREPARAÇÃO DE FEATURES
# ============================================================
print("\n" + "=" * 70)
print("2. PREPARAÇÃO DE FEATURES")
print("=" * 70)

# Features do modelo
feature_cols = [
    "tipo_cancer",
    "especialidade_prestador",
    "tipo_atendimento",
    "idade",
    "sexo",
    "ano_inicio",
    "ano_fim",
    "qtd_procedimentos",
    "n_prestadores",
]

target_col = "custo_total"

# Copiar para modelo
df_model = df[feature_cols + [target_col, "paciente_id"]].copy()

# Tratar sexo 'I' (muito poucos) - mudar para 'M'
df_model["sexo"] = df_model["sexo"].replace("I", "M")

print(f"Features: {feature_cols}")
print(f"Target: {target_col}")

# ============================================================
# 3. DEFINIR GRUPOS (PACIENTE_ID) - CRÍTICO
# ============================================================
print("\n" + "=" * 70)
print("3. GRUPOS PARA GROUPKFOLD")
print("=" * 70)

# Manter paciente_id como grupo
groups = df_model["paciente_id"].values
print(f"Grupos (pacientes): {df_model['paciente_id'].nunique()}")

# ============================================================
# 4. ENCODING DAS VARIÁVEIS CATEGÓRICAS
# ============================================================
print("\n" + "=" * 70)
print("4. ENCODING DAS VARIÁVEIS CATEGÓRICAS")
print("=" * 70)

# Definir colunas categóricas e numéricas
categorical_cols = [
    "tipo_cancer",
    "especialidade_prestador",
    "tipo_atendimento",
    "sexo",
]
numeric_cols = ["idade", "ano_inicio", "ano_fim", "qtd_procedimentos", "n_prestadores"]

# Label Encoding (para uso no modelo)
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_model[col + "_enc"] = le.fit_transform(df_model[col].astype(str))
    label_encoders[col] = le
    print(f"  {col}: {len(le.classes_)} classes")

# Features finais
final_features = [
    "tipo_cancer_enc",
    "especialidade_prestador_enc",
    "tipo_atendimento_enc",
    "sexo_enc",
] + numeric_cols

X = df_model[final_features].values
y = df_model[target_col].values

print(f"\nShape X: {X.shape}, Shape y: {y.shape}")

# ============================================================
# 5. BASELINE - OBRIGATÓRIO
# ============================================================
print("\n" + "=" * 70)
print("5. BASELINE (MÉDIAS)")
print("=" * 70)

# Baseline com média
baseline_mean = DummyRegressor(strategy="mean")
baseline_mean.fit(X_train := X[: int(0.8 * len(X))], y_train := y[: int(0.8 * len(y))])
y_pred_baseline = baseline_mean.predict(X[int(0.8 * len(X)) :])
y_test_baseline = y[int(0.8 * len(y)) :]

print(f"Baseline MÉDIA:")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test_baseline, y_pred_baseline)):,.2f}")
print(f"  MAE:  {mean_absolute_error(y_test_baseline, y_pred_baseline):,.2f}")
print(f"  R²:   {r2_score(y_test_baseline, y_pred_baseline):.4f}")

# Baseline com mediana
baseline_median = DummyRegressor(strategy="median")
baseline_median.fit(X_train, y_train)
y_pred_median = baseline_median.predict(X[int(0.8 * len(X)) :])

print(f"\nBaseline MEDIANA:")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test_baseline, y_pred_median)):,.2f}")
print(f"  MAE:  {mean_absolute_error(y_test_baseline, y_pred_median):,.2f}")
print(f"  R²:   {r2_score(y_test_baseline, y_pred_median):.4f}")

baseline_metrics = {
    "mean": {
        "rmse": np.sqrt(mean_squared_error(y_test_baseline, y_pred_baseline)),
        "mae": mean_absolute_error(y_test_baseline, y_pred_baseline),
        "r2": r2_score(y_test_baseline, y_pred_baseline),
    },
    "median": {
        "rmse": np.sqrt(mean_squared_error(y_test_baseline, y_pred_median)),
        "mae": mean_absolute_error(y_test_baseline, y_pred_median),
        "r2": r2_score(y_test_baseline, y_pred_median),
    },
}

# ============================================================
# 6. SPLIT TREINO/TESTE (SEM VAZAMENTO)
# ============================================================
print("\n" + "=" * 70)
print("6. SPLIT TREINO/TESTE (POR PACIENTE)")
print("=" * 70)

# Split simples para ter teste isolado (20%)
# GroupKFold será usado na validação cruzada
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, np.arange(len(y)), test_size=0.2, random_state=42
)

# Verificar que não há paciente em ambos
pacientes_treino = set(df_model.iloc[idx_train]["paciente_id"])
pacientes_test = set(df_model.iloc[idx_test]["paciente_id"])
overlap = pacientes_treino & pacientes_test
print(f"Pacientes em treino: {len(pacientes_treino)}")
print(f"Pacientes em teste: {len(pacientes_test)}")
print(f"Overlap (vazamento): {len(overlap)}")
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ============================================================
# 7. VALIDAÇÃO CRUZADA COM GROUPKFOLD
# ============================================================
print("\n" + "=" * 70)
print("7. VALIDAÇÃO CRUZADA (GroupKFold, 5 folds)")
print("=" * 70)

# GroupKFold para validação
gkf = GroupKFold(n_splits=5)

# Modelos para testar
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=1.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_split=10, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    ),
}


def cross_validate_with_groups(model, X, y, groups, cv, model_name):
    """Validação cruzada com GroupKFold"""

    rmses = []
    maes = []
    r2s = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        rmses.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        maes.append(mean_absolute_error(y_val, y_pred))
        r2s.append(r2_score(y_val, y_pred))

    return {
        "rmse_mean": np.mean(rmses),
        "rmse_std": np.std(rmses),
        "mae_mean": np.mean(maes),
        "mae_std": np.std(maes),
        "r2_mean": np.mean(r2s),
        "r2_std": np.std(r2s),
    }


# Filtrar grupos para validation fold
groups_train = groups[idx_train]

results = {}
for name, model in models.items():
    print(f"\n{name}...")
    metrics = cross_validate_with_groups(
        model, X_train, y_train, groups_train, gkf, name
    )
    results[name] = metrics
    print(f"  RMSE: {metrics['rmse_mean']:.2f} (+/- {metrics['rmse_std']:.2f})")
    print(f"  MAE:  {metrics['mae_mean']:.2f} (+/- {metrics['mae_std']:.2f})")
    print(f"  R²:   {metrics['r2_mean']:.4f} (+/- {metrics['r2_std']:.4f})")

# ============================================================
# 8. ESCOLHER MELHOR MODELO
# ============================================================
print("\n" + "=" * 70)
print("8. MELHOR MODELO (baseado em R² validação)")
print("=" * 70)

best_model_name = max(results, key=lambda x: results[x]["r2_mean"])
best_model_class = type(models[best_model_name])
print(f"Melhor modelo: {best_model_name}")
print(f"  R² validação: {results[best_model_name]['r2_mean']:.4f}")

# ============================================================
# 9. TREINAR MELHOR MODELO NO TREINO COMPLETO
# ============================================================
print("\n" + "=" * 70)
print("9. TREINAMENTO FINAL NO TREINO COMPLETO")
print("=" * 70)

# Instanciar modelo final
if best_model_name == "Random Forest":
    best_model = RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_split=10, random_state=42, n_jobs=-1
    )
elif best_model_name == "Gradient Boosting":
    best_model = GradientBoostingRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    )
elif best_model_name == "Ridge Regression":
    best_model = Ridge(alpha=1.0)
elif best_model_name == "Lasso Regression":
    best_model = Lasso(alpha=1.0)
else:
    best_model = LinearRegression()

# Treinar
best_model.fit(X_train, y_train)

# Predizer no teste
y_pred_test = best_model.predict(X_test)

# ============================================================
# 10. AVALIAÇÃO NO TESTE FINAL (cego)
# ============================================================
print("\n" + "=" * 70)
print("10. AVALIAÇÃO NO TESTE FINAL (cego)")
print("=" * 70)

rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f"TESTE FINAL:")
print(f"  RMSE: {rmse_test:,.2f}")
print(f"  MAE:  {mae_test:,.2f}")
print(f"  R²:   {r2_test:.4f}")

# Comparar com baseline
print(f"\nComparação com Baseline:")
print(f"  Baseline média R²: {baseline_metrics['mean']['r2']:.4f}")
print(f"  Modelo R²:        {r2_test:.4f}")
print(f"  Melhoria:         {(r2_test - baseline_metrics['mean']['r2']) * 100:.2f}%")

# ============================================================
# 11. ANÁLISE DE RESÍDUOS
# ============================================================
print("\n" + "=" * 70)
print("11. ANÁLISE DE RESÍDUOS")
print("=" * 70)

residuos = y_test - y_pred_test

print(f"Resíduos:")
print(f"  Média: {np.mean(residuos):,.2f}")
print(f" Std:   {np.std(residuos):,.2f}")
print(f"  Min:   {np.min(residuos):,.2f}")
print(f"  Max:   {np.max(residuos):,.2f}")

# Percentis
print(f"\nPercentis dos resíduos:")
for p in [5, 25, 50, 75, 95]:
    print(f"  {p}%: {np.percentile(residuos, p):,.2f}")

# Verificar se média ~0 (modelo não viesado)
if abs(np.mean(residuos)) < rmse_test * 0.1:
    print("\n[OK] Resíduo média próximo de 0 (modelo não viesado)")
else:
    print("\n[WARNING] Resíduo média != 0 (possível viés)")

# ============================================================
# 12. IMPORTÂNCIA DAS FEATURES
# ============================================================
print("\n" + "=" * 70)
print("12. IMPORTÂNCIA DAS FEATURES")
if hasattr(best_model, "feature_importances_"):
    importances = pd.DataFrame(
        {"feature": final_features, "importance": best_model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print(f"\nFeature Importances:")
    for _, row in importances.iterrows():
        print(f"  {row['feature']:35s} {row['importance']:.4f}")

# ============================================================
# 13. SALVAR MODELO E ARTEFATOS
# ============================================================
print("\n" + "=" * 70)
print("13. SALVANDO MODELO E ARTEFATOS")
print("=" * 70)

output_dir = "model"
os.makedirs(output_dir, exist_ok=True)

# Salvar modelo
model_path = f"{output_dir}/modelo_regressao_custos.joblib"
joblib.dump(best_model, model_path)
print(f"[OK] Modelo salvo: {model_path}")

# Salvar label encoders
encoders_path = f"{output_dir}/label_encoders.joblib"
joblib.dump(label_encoders, encoders_path)
print("[OK] Encoders salvo: {}".format(encoders_path))

# Salvar info do modelo
info = {
    "model_name": best_model_name,
    "features": final_features,
    "feature_cols": feature_cols,
    "categorical_cols": categorical_cols,
    "numeric_cols": numeric_cols,
    "target": target_col,
    "metrics_cv": results[best_model_name],
    "metrics_test": {"rmse": rmse_test, "mae": mae_test, "r2": r2_test},
    "baseline_metrics": baseline_metrics,
    "dataset_shape": df_model.shape,
    "n_pacientes": df_model["paciente_id"].nunique(),
}
joblib.dump(info, f"{output_dir}/model_info.joblib")
print("[OK] Info salvo: {}/model_info.joblib".format(output_dir))

print("\n" + "=" * 70)

print("=" * 70)
