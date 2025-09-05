import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from tsm_model import evaluate_model, load_data, preprocess_data, split_train_test_holdout

# ------------------------------
# Função auxiliar para exportar CSV
# ------------------------------
def convert_df_to_csv(df):
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()

# # ------------------------------
# # Aba 1 - Predição TSM
# # ------------------------------
# def prediction_tab():
#     st.header("⚡ Predição de Tempo Remanescente (TSM)")

#     # Seleção do centro de planejamento
#     centro = st.selectbox("Selecione o centro de planejamento:", ["Amarelo", "Vermelho", "Azul", "Verde"])

#     # Escolha entre nota ou ordem
#     tipo = st.radio("Você deseja calcular para:", ["Nota", "Ordem"])

#     if tipo == "Nota":
#         trace = st.text_input("Digite o trace da Nota (ex: A,B,C)")
#     else:
#         ordem_tipo = st.selectbox("Selecione o tipo de Ordem:", ["X", "Y"])
#         trace = st.text_input("Digite o trace da Ordem (ex: A,B,C)")

#     if st.button("Calcular tempo remanescente"):
#         if trace.strip() == "":
#             st.warning("⚠️ Digite um trace válido.")
#         else:
#             try:
#                 trace_list = trace.split(",")
#                 predicted_time = predict_time_remaining(trace_list)

#                 st.success(f"⏳ Tempo remanescente previsto: **{predicted_time:.2f} unidades de tempo**")
#             except Exception as e:
#                 st.error(f"Erro ao calcular predição: {e}")

# ------------------------------
# Aba 2 - Upload & Split (Hold-Out)
# ------------------------------
def data_split_tab():
    st.header("📂 Modelo ATS")

    # Upload do arquivo
    uploaded_file = st.file_uploader("Carregue seu arquivo CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        df = preprocess_data(df)
        st.success("✅ Arquivo carregado com sucesso!")
        
        st.subheader("Visualização inicial dos dados")
        st.dataframe(df.head(50))

        # Armazena o DataFrame na sessão
        st.session_state["df"] = df

        # Escolha da porcentagem de treino
        test_size = st.slider("Escolha a porcentagem de teste", min_value=10, max_value=90, value=30, step=5)
        train_size = 100 - test_size

        st.write(f"➡️ Treino: **{train_size}%** | Teste: **{test_size}%**")

        if st.button("Dividir dados"):
            df_train, df_test = split_train_test_holdout(df, test_size=test_size/100)
            st.session_state["df_train"] = df_train
            st.session_state["df_test"] = df_test
            st.success("✅ Dados divididos com sucesso!")

    # Avaliar modelo
    if st.button("Treinar e Avaliar Modelo"):
        if "df" not in st.session_state or "df_train" not in st.session_state or "df_test" not in st.session_state:
            st.error("⚠️ Você precisa carregar e dividir os dados primeiro!")
        else:
            results = evaluate_model(
                {"default": st.session_state["df"]},
                st.session_state["df_train"],
                st.session_state["df_test"],
                metric="Mean",
                pen=1
            )

        df_all = []

        # Itera pelos produtos dentro de Predict_trace
        for prod_key in results["Predict_trace"].keys():
            preds_folds = results["Predict_trace"][prod_key]
            reals_folds = results["Real_trace"][prod_key]
            test_folds = st.session_state["df_test"][prod_key]

            for fold_idx, (preds, reals, test_idx) in enumerate(zip(preds_folds, reals_folds, test_folds)):
                ids = st.session_state["df"].loc[test_idx]["id_caso"].values

                df_fold = pd.DataFrame({
                    "produto": prod_key,
                    "fold": fold_idx,
                    "id_caso": ids,
                    "real": reals,
                    "pred": preds,
                })

                # Erros
                df_fold["error_abs"] = (df_fold["real"] - df_fold["pred"]).abs()
                df_fold["error_rel"] = df_fold["error_abs"] / df_fold["real"].replace(0, np.nan)

                df_all.append(df_fold)

        df_model = pd.DataFrame(results["model"])

        st.subheader("📈 Resultados Resumidos por Modelo")
        st.dataframe(df_model)
        # Concatenar tudo em uma única tabela
        df_all = pd.concat(df_all, ignore_index=True)

        st.subheader("📊 Resultados Detalhados")
        st.dataframe(df_all.head(50))  # mostra os primeiros 50 para não pesar

        st.download_button(
            "⬇️ Baixar Resultados CSV",
            data=df_all.to_csv(index=False).encode("utf-8"),
            file_name="resultados_detalhados.csv",
            mime="text/csv",
        )
# ------------------------------
# Layout principal
# ------------------------------
def main():
    st.set_page_config(page_title="TSM App", layout="wide")

    data_split_tab()

if __name__ == "__main__":
    main()
