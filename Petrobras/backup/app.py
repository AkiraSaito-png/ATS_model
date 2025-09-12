import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from tsm_model import evaluate_model_new, get_index, evaluate_model, load_data, preprocess_data, split_train_test_holdout

# ------------------------------
# Função auxiliar para exportar CSV
# ------------------------------
def convert_df_to_csv(df):
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()

# ------------------------------
# Aba 1 - Predição TSM
# ------------------------------
def prediction_tab():
    st.header("📂 Predição de Tempo Remanescente (TSM)")

    train_file = st.file_uploader("Carregue seu arquivo de treino (.CSV)", type=["csv"])
    if train_file is not None:
        with st.spinner('Carregando arquivo...'):
            training_dataframe = load_data(train_file)
            training_dataframe = preprocess_data(training_dataframe)
            st.session_state["original_train_df"] = training_dataframe
            training_dataframe = st.session_state["original_train_df"].reset_index(drop=True)
            st.success("✅ Arquivo de treino carregado com sucesso!")
            with st.spinner("Processando dados..."):
                caseId_Index_Train = get_index(training_dataframe)
                st.session_state["df_train"] = caseId_Index_Train
                st.success("✅ Pré-processamento finalizado com sucesso!")
                

    test_file = st.file_uploader("Carregue seu arquivo de teste (.CSV)", type=["csv"])
    if test_file is not None:
            test_dataframe = load_data(test_file)
            test_dataframe = preprocess_data(test_dataframe)
            st.session_state["original_test_df"] = test_dataframe
            test_dataframe = st.session_state["original_test_df"].reset_index(drop=True)
            st.success("✅ Arquivo de teste carregado com sucesso!")
            with st.spinner("Processando dados..."):
                caseId_Index_Test = get_index(test_dataframe)
                st.session_state["df_test"] = caseId_Index_Test
                st.success("✅ Pré-processamento finalizado com sucesso!")

    metric_option = st.radio("Selecione a métrica que deseja usar:", ["Média", "Mediana"])

    if metric_option == "Média":
        metric = "Mean"
    else:
        metric = "Median"

    penalty_option = st.radio("Deseja aplicar penalidade no treino?:", ["Sim", "Não"])

    if penalty_option == "Sim":
        pen = 2
    else:
        pen = 1
    
    # Avaliar modelo
    if st.button("Executar Predição"):
        if "original_train_df" not in st.session_state or "original_test_df" not in st.session_state or "df_train" not in st.session_state or "df_test" not in st.session_state:
            st.error("⚠️ Você precisa carregar e dividir os dados primeiro!")
        else:
            results = evaluate_model_new(
                DF_train = {"default": test_dataframe},
                DF_test = {"default": training_dataframe},
                train_indexes = st.session_state["df_train"],
                test_indexes = st.session_state["df_test"],
                metric = metric,
                pen = pen
            )

        df_all = []

        # Itera pelos produtos dentro de Predict_trace
        for prod_key in results["Predict_trace"].keys():
            preds_folds = results["Predict_trace"][prod_key]
            reals_folds = results["Real_trace"][prod_key]
            trace_folds = results["better_trace"][prod_key]
            test_folds = st.session_state["df_test"][prod_key]

            for fold_idx, (preds, reals, trace, test_idx) in enumerate(zip(preds_folds, reals_folds, trace_folds, test_folds)):
                ids = st.session_state["original_test_df"].loc[test_idx, ["id_caso", "trace"]]

                df_fold = pd.DataFrame({
                    "id_caso": ids["id_caso"].values,
                    "trace": ids["trace"].astype(str).values,  # garante formato legível
                    "find_better": trace,
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

    metric_option = st.radio(" Selecione a métrica que deseja usar", ["Média", "Mediana"])

    if metric_option == "Média":
        metric = "Mean"
    else:
        metric = "Median"

    penalty_option = st.radio(" Deseja aplicar penalidade no treino?", ["Sim", "Não"])

    if penalty_option == "Sim":
        pen = 2
    else:
        pen = 1

    # Avaliar modelo
    if st.button("Treinar e Avaliar Modelo"):
        if "df" not in st.session_state or "df_train" not in st.session_state or "df_test" not in st.session_state:
            st.error("⚠️ Você precisa carregar e dividir os dados primeiro!")
        else:
            results = evaluate_model(
                {"default": st.session_state["df"]},
                st.session_state["df_train"],
                st.session_state["df_test"],
                metric=metric,
                pen=pen
            )

        df_all = []

        # Itera pelos produtos dentro de Predict_trace
        for prod_key in results["Predict_trace"].keys():
            preds_folds = results["Predict_trace"][prod_key]
            reals_folds = results["Real_trace"][prod_key]
            trace_folds = results["better_trace"][prod_key]
            test_folds = st.session_state["df_test"][prod_key]

            for fold_idx, (preds, reals, trace, test_idx) in enumerate(zip(preds_folds, reals_folds, trace_folds, test_folds)):
                ids = st.session_state["original_test_df"].loc[test_idx, ["id_caso", "trace"]]

                df_fold = pd.DataFrame({
                    "id_caso": ids["id_caso"].values,
                    "trace": ids["trace"].astype(str).values,  # garante formato legível
                    "find_better": trace,
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

    tab1, tab2 = st.tabs(["Teste por amostragem", "Teste por CSV"])

    with tab1:
        prediction_tab()

    with tab2:
        data_split_tab()

if __name__ == "__main__":
    main()
