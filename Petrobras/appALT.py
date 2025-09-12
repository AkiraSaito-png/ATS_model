import streamlit as st
import pandas as pd
from tsm_model_ALT import evaluate_model_new # Assumindo que evaluate_model_new será a função principal
from tsm_model_ALT import get_index # Ainda usaremos get_index para organizar o df por 'id_caso'

# Função para carregar o arquivo
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Arquivo carregado com sucesso!")
            return df
        except Exception as e:
            st.error(f"Erro ao carregar o arquivo: {e}")
            return None
    return None

def prediction_tab():
    st.header("Previsão de Tempo de Permanência")

    st.subheader("Carregar Dados de Treino")
    uploaded_file_train = st.file_uploader("Escolha um arquivo Excel para Treino", type=["csv"])
    
    df_train_raw = load_data(uploaded_file_train)
    
    # Armazenar no session state apenas se houver um arquivo carregado
    if df_train_raw is not None:
        st.session_state["original_train_df"] = df_train_raw

    st.subheader("Carregar Dados de Teste")
    uploaded_file_test = st.file_uploader("Escolha um arquivo Excel para Teste", type=["csv"])
    
    df_test_raw = load_data(uploaded_file_test)

    if df_test_raw is not None:
        st.session_state["original_test_df"] = df_test_raw

    # Parâmetros
    metric = st.selectbox("Métrica de Agregação", ["Mean", "Median"])
    pen = st.slider("Penalidade (pen)", min_value=0, max_value=10, value=1) # Ajuste o range conforme necessário

    results = None
    if st.button("Executar Análise"):
        if "original_train_df" in st.session_state and "original_test_df" in st.session_state:
            df_train = st.session_state["original_train_df"]
            df_test = st.session_state["original_test_df"]

            st.write("Dados de Treino carregados:", df_train.head())
            st.write("Dados de Teste carregados:", df_test.head())

            # Resetar os índices ANTES de passar para as funções para garantir consistência
            # E usar get_index para preparar os DataFrames para a avaliação
            # Agora, df_train_processed e df_test_processed serão DataFrames planos,
            # sem a estrutura de dicionário de antes.
            
            # Nota: A função get_index atualmente retorna um dicionário de listas de listas.
            # Se não há mais "folds", get_index precisaria ser adaptada para retornar
            # uma lista plana de índices, ou o DF já "agrupado" por id_caso.
            # POR ENQUANTO, vou assumir que get_index é usada APENAS para criar a estrutura
            # de "cases" para DF_train e DF_test, mas os loops de folds sumirão.
            
            # Vamos simplificar a chamada: get_index era para gerar índices para folds.
            # Se não há folds, os índices são para os "cases" completos de cada DF.
            
            # Vamos precisar de uma versão dos DataFrames com índices resetados para
            # que .iloc funcione corretamente.
            df_train_reset = df_train.reset_index(drop=True)
            df_test_reset = df_test.reset_index(drop=True)
            
            # Se get_index ainda é para agrupar por 'id_caso' para retornar "case_indices",
            # então ele vai retornar um dict com a chave "default" e uma lista de listas
            # (onde cada sublista é os índices de um id_caso). Isso ainda pode ser útil
            # para o test_model_TSM que pode precisar iterar por caso.

            # Agora, DF_train e DF_test na evaluate_model_new serão os próprios DataFrames,
            # não mais um dicionário com chave "default".
            results = evaluate_model_new(
                DF_train = df_train_reset, # Passa o DataFrame diretamente
                DF_test = df_test_reset,   # Passa o DataFrame diretamente
                metric = metric,
                pen = pen
            )
            
            st.subheader("Resultados da Análise")
            # Aqui você vai precisar adaptar como os resultados são exibidos
            # pois 'results' não será mais um dicionário de dicionários de listas.
            # Será um dicionário com valores diretos ou DataFrames.
            
            # Exemplo de exibição (adapte conforme o novo formato de 'results')
            if results:
                st.write("MAE:", results.get("MAE"))
                st.write("RMSE:", results.get("RMSE"))
                st.write("MAPE:", results.get("MAPE"))
                # ... outras métricas
                if "Predict_trace" in results:
                    st.subheader("Previsões de Traços")
                    for key, val in results["Predict_trace"].items():
                        st.write(f"Previsões para {key}:", val)
                if "Real_trace" in results:
                    st.subheader("Traços Reais")
                    for key, val in results["Real_trace"].items():
                        st.write(f"Reais para {key}:", val)
        else:
            st.warning("Por favor, carregue os dados de treino e teste.")

def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("Navegação")
    page = st.sidebar.radio("Ir para", ["Previsão"])

    if page == "Previsão":
        prediction_tab()

if __name__ == "__main__":
    main()