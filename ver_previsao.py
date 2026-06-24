"""
Script para ver exemplos de predições do modelo
"""

import pandas as pd
import numpy as np
import joblib

# Carregar modelo e artefatos
model = joblib.load("model/modelo_regressao_custos.joblib")
label_encoders = joblib.load("model/label_encoders.joblib")
info = joblib.load("model/model_info.joblib")

# Carregar dados originais
df = pd.read_parquet("View_Onco_Custos_clean.parquet")

# Calcular idade
df["idade"] = (df["data_evento"] - df["dat_nascimento"]).dt.days // 365

# Agregar como no modelo
df_agg = (
    df.groupby(
        ["paciente_id", "tipo_cancer", "especialidade_prestador", "tipo_atendimento"]
    )
    .agg(
        {
            "custo_total_item": "sum",
            "idade": "first",
            "sexo_paciente": "first",
            "ano_evento": ["min", "max"],
            "quantidade": "sum",
            "prestador_id": "nunique",
        }
    )
    .reset_index()
)

df_agg.columns = [
    "paciente_id",
    "tipo_cancer",
    "especialidade_prestador",
    "tipo_atendimento",
    "custo_total",
    "idade",
    "sexo",
    "ano_inicio",
    "ano_fim",
    "qtd_procedimentos",
    "n_prestadores",
]

df_agg = df_agg.dropna()

# Tratar sexo 'I'
df_agg["sexo"] = df_agg["sexo"].replace("I", "M")

print("=" * 70)
print("EXEMPLOS DE PREDIÇÕES DO MODELO")
print("=" * 70)

# Codificar
for col in ["tipo_cancer", "especialidade_prestador", "tipo_atendimento", "sexo"]:
    le = label_encoders[col]
    df_agg[col + "_enc"] = le.transform(df_agg[col].astype(str))

# Features
features = [
    "tipo_cancer_enc",
    "especialidade_prestador_enc",
    "tipo_atendimento_enc",
    "sexo_enc",
    "idade",
    "ano_inicio",
    "ano_fim",
    "qtd_procedimentos",
    "n_prestadores",
]

X = df_agg[features].values
y_real = df_agg["custo_total"].values
y_pred = model.predict(X)

# Criar dataframe de resultados
df_result = pd.DataFrame(
    {
        "paciente_id": df_agg["paciente_id"],
        "tipo_cancer": df_agg["tipo_cancer"],
        "especialidade": df_agg["especialidade_prestador"],
        "tipo_atendimento": df_agg["tipo_atendimento"],
        "idade": df_agg["idade"],
        "custo_real": y_real,
        "custo_predito": y_pred,
        "diferenca": y_real - y_pred,
        "erro_pct": ((y_pred - y_real) / y_real * 100).round(2),
    }
)

print(f"\nTotal de registros: {len(df_result)}")
print(f"\nErro médio: R$ {df_result['diferenca'].abs().mean():.2f}")
print(f"Erro médio %: {df_result['erro_pct'].abs().mean():.2f}%")

print("\n" + "=" * 70)
print("10 EXEMPLOS ALEATÓRIOS")
print("=" * 70)

amostras = df_result.sample(10, random_state=42)
for i, row in amostras.iterrows():
    print(f"\nPaciente ID: {row['paciente_id']}")
    print(f"  Cancer: {row['tipo_cancer']}")
    print(f"  Especialidade: {row['especialidade']}")
    print(f"  Atendimento: {row['tipo_atendimento']}")
    print(f"  Idade: {row['idade']}")
    print(f"  CUSTO REAL:    R$ {row['custo_real']:>12,.2f}")
    print(f"  CUSTO PREVISTO: R$ {row['custo_predito']:>12,.2f}")
    print(f"  DIFERENÇA:     R$ {row['diferenca']:>12,.2f} ({row['erro_pct']:>6,.2f}%)")

print("\n" + "=" * 70)
print("EXEMPLOS COM BOM E MAU DESEMPENHO")
print("=" * 70)

# Melhor prediction
df_result["erro_abs"] = df_result["diferenca"].abs()
melhores = df_result.nsmallest(5, "erro_abs")
piores = df_result.nlargest(5, "erro_abs")

print("\n[BEST] previsões mais próximas do real:")
for _, row in melhores.iterrows():
    print(
        f"  Paciente {row['paciente_id']}: Real R${row['custo_real']:,.0f} | Previsto R${row['custo_predito']:,.0f} | Erro R${row['diferenca']:,.0f}"
    )

print("\n[WORST] previsões mais distantes do real:")
for _, row in piores.iterrows():
    print(
        f"  Paciente {row['paciente_id']}: Real R${row['custo_real']:,.0f} | Previsto R${row['custo_predito']:,.0f} | Erro R${row['diferenca']:,.0f}"
    )
