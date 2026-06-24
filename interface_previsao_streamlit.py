"""
================================================================================
INTERFACE STREAMLIT - PREVISÃO DE CUSTOS ONCOLÓGICOS
================================================================================
Modelos disponíveis:
- Modelo Simples: idade + tipo_cancer + sexo -> custo total por paciente
- Modelo Completo: todas as features -> custo por especialidade/atendimento

Uso: streamlit run interface_previsao_streamlit.py
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Previsão de Custos Oncológicos", page_icon="🏥", layout="wide"
)


# ============================================================
# CARREGAR MODELOS
# ============================================================
@st.cache_resource
def carregar_modelos():
    """Carrega os modelos e encoders"""

    # Modelo Completo
    modelo_completo = joblib.load("model/modelo_regressao_custos.joblib")
    encoders_completo = joblib.load("model/label_encoders.joblib")
    info_completo = joblib.load("model/model_info.joblib")

    # Modelo Simples
    modelo_simples = joblib.load("model/modelo_simples.joblib")
    encoders_simples = joblib.load("model/label_encoders_simples.joblib")
    info_simples = joblib.load("model/model_info_simples.joblib")

    return {
        "completo": {
            "modelo": modelo_completo,
            "encoders": encoders_completo,
            "info": info_completo,
        },
        "simples": {
            "modelo": modelo_simples,
            "encoders": encoders_simples,
            "info": info_simples,
        },
    }


# ============================================================
# FUNÇÕES DE PREVISÃO
# ============================================================
def prever_modelo_simples(modelos, tipo_cancer, idade, sexo):
    """Previsão com modelo simples (idade + tipo_cancer + sexo)"""

    encoders = modelos["simples"]["encoders"]
    modelo = modelos["simples"]["modelo"]

    # Codificar inputs
    tipo_enc = encoders["tipo_cancer"].transform([tipo_cancer])[0]
    sexo_enc = encoders["sexo"].transform([sexo])[0]

    # Features: [tipo_cancer_enc, idade, sexo_enc]
    features = np.array([[tipo_enc, idade, sexo_enc]])

    # Prever
    custo = modelo.predict(features)[0]

    return custo


def prever_modelo_completo(modelos, dados):
    """Previsão com modelo completo (todas as features)"""

    encoders = modelos["completo"]["encoders"]
    modelo = modelos["completo"]["modelo"]

    # Features na ordem: tipo_cancer_enc, especialidade_prestador_enc,
    # tipo_atendimento_enc, sexo_enc, idade, ano_inicio, ano_fim,
    # qtd_procedimentos, n_prestadores

    features = np.array(
        [
            [
                encoders["tipo_cancer"].transform([dados["tipo_cancer"]])[0],
                encoders["especialidade_prestador"].transform([dados["especialidade"]])[
                    0
                ],
                encoders["tipo_atendimento"].transform([dados["tipo_atendimento"]])[0],
                encoders["sexo"].transform([dados["sexo"]])[0],
                dados["idade"],
                dados["ano_inicio"],
                dados["ano_fim"],
                dados["qtd_procedimentos"],
                dados["n_prestadores"],
            ]
        ]
    )

    # Prever
    custo = modelo.predict(features)[0]

    return custo


# ============================================================
# INTERFACE
# ============================================================
def main():

    # Título principal
    st.title("🏥 Previsão de Custos Oncológicos")
    st.markdown("---")

    # Carregar modelos
    try:
        modelos = carregar_modelos()
    except Exception as e:
        st.error(f"Erro ao carregar modelos: {e}")
        st.info("Certifique-se de que os arquivos dos modelos estão na pasta 'model/'")
        return

    # Seleção de modelo
    st.markdown("## Selecione o Tipo de Previsão")

    tipo_previsao = st.radio(
        "Tipo de Previsão:",
        options=["simples", "completo"],
        format_func=lambda x: {
            "simples": "Modelo Simples (idade + tipo cancer + sexo)",
            "completo": "Modelo Completo (todos os campos)",
        }[x],
        horizontal=True,
        help="Escolha entre o modelo simples (3 campos) ou completo (9 campos)",
    )

    st.markdown("---")

    # ============================================================
    # MODELO SIMPLES
    # ============================================================
    if tipo_previsao == "simples":
        st.markdown("## Modelo Simples")
        st.markdown("*Previsão de custo total do paciente por tipo de câncer*")

        with st.expander("ℹ️ Sobre este modelo", expanded=False):
            st.info("""
            **Modelo Simples** usa apenas dados demográficos básicos:
            - Idade do paciente
            - Tipo de cancer
            - Sexo
            
            **Ideal para:** Estimativas gerais de custo para o setor de planejamento financeiro.
            
            **Limitação:** Por usar poucas features, a precisão é menor (R² ~0.05).
            """)

        st.markdown("### Campos Obrigatórios")

        col1, col2, col3 = st.columns(3)

        with col1:
            tipo_cancer = st.selectbox(
                "**Tipo de Câncer** *",
                options=["Pr\u00f3stata", "Colorretal", "Pulm\u00e3o"],
                help="Tipo de cancer do paciente",
            )

        with col2:
            idade = st.number_input(
                "**Idade** *",
                min_value=0,
                max_value=120,
                value=65,
                help="Idade do paciente em anos",
            )

        with col3:
            sexo = st.selectbox(
                "**Sexo** *",
                options=["M", "F"],
                format_func=lambda x: {"M": "Masculino", "F": "Feminino"}[x],
                help="Sexo do paciente",
            )

        st.markdown("---")

        # Botão de previsão
        if st.button(" Prever Custo", type="primary", use_container_width=True):
            with st.spinner("Calculando..."):
                custo_previsto = prever_modelo_simples(
                    modelos, tipo_cancer, idade, sexo
                )

                # Intervalo de confiança (~±30%)
                intervalo_inferior = custo_previsto * 0.7
                intervalo_superior = custo_previsto * 1.3

                # Exibir resultado
                st.markdown("### Resultado")

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.metric(label="Custo Previsto", value=f"R$ {custo_previsto:,.2f}")

                with col2:
                    st.metric(
                        label="Intervalo de Confiança",
                        value=f"R$ {intervalo_inferior:,.0f} - R$ {intervalo_superior:,.0f}",
                    )

                # Informações do modelo
                info = modelos["simples"]["info"]

                st.markdown("---")
                st.markdown("### Informações do Modelo")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"**Modelo:** {info['model_name']}")
                with col2:
                    st.info(f"**R² (teste):** {info['metrics_test']['r2']:.4f}")
                with col3:
                    st.info(f"**MAE:** R$ {info['metrics_test']['mae']:,.2f}")

                st.caption(
                    f"⚠️ Este modelo explica ~5% da variância dos custos. Use para estimativas gerais."
                )

    # ============================================================
    # MODELO COMPLETO
    # ============================================================
    else:
        st.markdown("## Modelo Completo")
        st.markdown("*Previsão de custo por especialidade e tipo de atendimento*")

        with st.expander("ℹ️ Sobre este modelo", expanded=False):
            st.info("""
            **Modelo Completo** usa mais informações do tratamento:
            - Tipo de cancer
            - Especialidade do prestador
            - Tipo de atendimento
            - Idade
            - Sexo
            - Período do tratamento
            - Quantidade de procedimentos
            - Número de prestadores
            
            **Ideal para:** Previsões mais detalhadas quando há informações específicas do tratamento.
            
            **Vantagem:** Maior precisão (R² ~0.85).
            """)

        st.markdown("### Campos do Paciente")

        col1, col2, col3 = st.columns(3)

        with col1:
            tipo_cancer = st.selectbox(
                "**Tipo de Câncer** *",
                options=["Pr\u00f3stata", "Colorretal", "Pulm\u00e3o"],
                help="Tipo de cancer do paciente",
            )

        with col2:
            idade = st.number_input(
                "**Idade** *",
                min_value=0,
                max_value=120,
                value=65,
                help="Idade do paciente em anos",
            )

        with col3:
            sexo = st.selectbox(
                "**Sexo** *",
                options=["M", "F"],
                format_func=lambda x: {"M": "Masculino", "F": "Feminino"}[x],
                help="Sexo do paciente",
            )

        st.markdown("### Campos do Tratamento")

        col1, col2, col3 = st.columns(3)

        with col1:
            especialidade = st.selectbox(
                "**Especialidade** *",
                options=[
                    "ANESTESIOLOGIA",
                    "UROLOGIA",
                    "COLOPROCTOLOGIA",
                    "CIRURGIA GERAL",
                    "CIRURGIA TORACICA",
                    "CLINICA MEDICA",
                    "MEDICINA INTENSIVA",
                    "CARDIOLOGIA",
                    "ONCOLOGIA CLINICA E CANCEROLOGIA",
                    "RADIOLOGIA E DIAGNOSTICO POR IMAGEM",
                    "GASTROENTEROLOGIA",
                    "PNEUMOLOGIA",
                    "GENERICA",
                    "NAO INFORMADO",
                ],
                help="Especialidade do prestador de saúde",
            )

        with col2:
            tipo_atendimento = st.selectbox(
                "**Tipo de Atendimento** *",
                options=[
                    "ATENDIMENTO HOSPITALAR",
                    "CONSULTA ELETIVA",
                    "ATENDIMENTO EM PA",
                    "DEMAIS ATENDIMENTOS AMBULATORIAIS",
                ],
                help="Tipo de atendimento realizado",
            )

        with col3:
            ano_inicio = st.number_input(
                "**Ano Início** *",
                min_value=2015,
                max_value=2025,
                value=2020,
                help="Ano de início do tratamento",
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            ano_fim = st.number_input(
                "**Ano Fim** *",
                min_value=2015,
                max_value=2025,
                value=2020,
                help="Ano de fim do tratamento",
            )

        with col2:
            qtd_procedimentos = st.number_input(
                "**Qtd Procedimentos** *",
                min_value=1,
                max_value=1000,
                value=10,
                help="Quantidade aproximada de procedimentos",
            )

        with col3:
            n_prestadores = st.number_input(
                "**Nº Prestadores** *",
                min_value=1,
                max_value=50,
                value=3,
                help="Número de prestadores envolvidos",
            )

        st.markdown("---")

        # Botão de previsão
        if st.button("Prever Custo", type="primary", use_container_width=True):
            with st.spinner("Calculando..."):
                dados = {
                    "tipo_cancer": tipo_cancer,
                    "especialidade": especialidade,
                    "tipo_atendimento": tipo_atendimento,
                    "idade": idade,
                    "sexo": sexo,
                    "ano_inicio": ano_inicio,
                    "ano_fim": ano_fim,
                    "qtd_procedimentos": qtd_procedimentos,
                    "n_prestadores": n_prestadores,
                }

                custo_previsto = prever_modelo_completo(modelos, dados)

                # Intervalo de confiança (~±15%)
                intervalo_inferior = custo_previsto * 0.85
                intervalo_superior = custo_previsto * 1.15

                # Exibir resultado
                st.markdown("### 💰 Resultado")

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.metric(label="Custo Previsto", value=f"R$ {custo_previsto:,.2f}")

                with col2:
                    st.metric(
                        label="Intervalo de Confiança",
                        value=f"R$ {intervalo_inferior:,.0f} - R$ {intervalo_superior:,.0f}",
                    )

                # Informações do modelo
                info = modelos["completo"]["info"]

                st.markdown("---")
                st.markdown("### 📋 Informações do Modelo")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"**Modelo:** {info['model_name']}")
                with col2:
                    st.info(f"**R² (teste):** {info['metrics_test']['r2']:.4f}")
                with col3:
                    st.info(f"**MAE:** R$ {info['metrics_test']['mae']:,.2f}")

                st.caption(f"✅ Este modelo explica ~85% da variância dos custos.")

    # ============================================================
    # RODAPÉ
    # ============================================================
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: gray;">
            <p>🏥 Sistema de Previsão de Custos Oncológicos</p>
            <p>Modelos baseados em Gradient Boosting Regressor</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# EXECUTAR
# ============================================================
if __name__ == "__main__":
    main()
