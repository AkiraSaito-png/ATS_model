import pandas as pd
import numpy as np

# A função get_index ainda pode ser útil para identificar os casos individuais ('id_caso')
# em um DataFrame e retornar seus índices agrupados, mesmo que não haja "folds" de k-fold.
# Ela retornará um dicionário onde a chave é "default" e o valor é uma lista de listas de índices,
# onde cada sublista representa os índices de um 'id_caso' específico.
def get_index(dfnew):
    # Não precisa mais do if isinstance(dfnew, pd.DataFrame): dfnew = {"default": dfnew}
    # pois agora esperamos um DataFrame direto.

    # O DataFrame passado para get_index já deve ter o índice resetado (0 a N-1).
    if "id_caso" not in dfnew.columns:
        raise KeyError(f"O DataFrame não contém a coluna 'id_caso'.")
    
    case_indices = []
    # Usamos o dfnew diretamente, que já foi resetado no app.py
    for case_id, group in dfnew.groupby("id_caso"):
        case_indices.append(group.index.tolist())
    
    return {"default": case_indices} # Retorna como um dict para consistência com chamadas anteriores, se necessário.
                                   # Mas o uso será simplificado.

# --- Funções de Criação e Teste do Modelo (Simplificadas) ---

def create_model_sist_trans(acts, times, metric="Mean"):
    """
    Cria o modelo de sistema de transição baseado nos sub-traces.
    Esta função já estava refatorada e continua limpa.
    """
    df_fold = pd.DataFrame({'trace': acts, 'time_remain': times})

    if metric == "Mean":
        model_df = df_fold.groupby('trace')['time_remain'].mean().reset_index()
    elif metric == "Median":
        model_df = df_fold.groupby('trace')['time_remain'].median().reset_index()
    else:
        raise ValueError("Métrica deve ser 'Mean' ou 'Median'")
        
    return model_df

def test_model_TSM(TSM_model, DF_test_processed, case_indices_for_test, metric, pen):
    """
    Testa o modelo TSM.
    DF_test_processed agora é um DataFrame com 'trace' já em formato de string.
    case_indices_for_test é uma lista de listas (índices por 'id_caso').
    """
    erro1 = [] # Erro absoluto
    erro2 = [] # MAPE
    erro3 = [] # Erro não absoluto (diferença)
    erro4 = [] # MAPE não absoluto
    pred = []
    real = []
    show_trace = []

    # Itera por cada caso ('id_caso') no DataFrame de teste
    for indices_do_caso in case_indices_for_test:
        # Fatiar o DataFrame de teste para o caso atual
        df_caso = DF_test_processed.iloc[indices_do_caso].copy()
        
        # O trace para previsão será o último trace do caso (para prever o próximo)
        trace_to_predict = df_caso["trace"].iloc[-1]
        
        # O tempo real restante é o último tempo restante do caso
        real_time_remain = df_caso["time_remain"].iloc[-1]
        
        # Procurar a previsão no modelo TSM
        predicted_row = TSM_model[TSM_model["trace"] == trace_to_predict]
        
        predicted_time_remain = 0 # Valor padrão
        if not predicted_row.empty:
            if metric == "Mean":
                predicted_time_remain = predicted_row["time_remain"].mean()
            elif metric == "Median":
                predicted_time_remain = predicted_row["time_remain"].median()
            
            # Aplica a penalidade (se aplicável, com lógica a ser mantida)
            predicted_time_remain += pen # Assumindo que a penalidade é uma adição simples

        # Calcula os erros
        diff = real_time_remain - predicted_time_remain
        erro1.append(abs(diff)) # MAE
        
        if real_time_remain != 0: # Evitar divisão por zero para MAPE
            erro2.append(abs(diff) / real_time_remain) # MAPE
        else:
            erro2.append(0) # Ou np.nan, dependendo da sua preferência

        erro3.append(diff) # Erro não absoluto
        if real_time_remain != 0:
            erro4.append(diff / real_time_remain)
        else:
            erro4.append(0)

        pred.append(predicted_time_remain)
        real.append(real_time_remain)
        show_trace.append(trace_to_predict)

    # Retorna as listas de erros e as predições/reais
    # A agregação final das métricas será feita na evaluate_model_new
    return erro1, erro2, erro3, erro4, pred, real, show_trace


def evaluate_model_new(DF_train, DF_test, metric="Mean", pen=1):
    """
    Avalia modelos de Sistema de Transição (TSM).
    DF_train e DF_test agora são os DataFrames completos (com índices resetados).
    """
    
    # 1. Pré-processar DF_train e DF_test para ter a coluna 'trace' como string.
    #    Isso foi feito no app.py, mas é bom ter certeza ou fazer aqui se o app.py for mais simples.
    #    Vamos manter a lógica de que o app.py já passa 'DF_train' e 'DF_test' com índices resetados.
    df_train_processed = DF_train.copy()
    df_test_processed = DF_test.copy()

    # Converter a coluna 'trace' para string para ambos os DataFrames
    df_train_processed['trace'] = df_train_processed['trace'].apply(lambda x: '_'.join(map(str, x)))
    df_test_processed['trace'] = df_test_processed['trace'].apply(lambda x: '_'.join(map(str, x)))

    # 2. Criar o TSM_model a partir dos dados de treino.
    TSM_model = create_model_sist_trans(
        acts=df_train_processed["trace"],
        times=df_train_processed["time_remain"],
        metric=metric
    )

    # 3. Gerar os índices dos casos para o DF_test (para testar caso a caso)
    #    get_index retorna um dict com chave "default" e uma lista de listas de índices.
    #    Pegamos a primeira (e única) lista de índices.
    test_case_indices_dict = get_index(df_test_processed)
    test_case_indices_list = test_case_indices_dict["default"] # Lista de listas, cada sublista é um id_caso

    # 4. Testar o modelo
    # test_model_TSM agora vai iterar sobre os casos do conjunto de teste
    erro1_list, erro2_list, erro3_list, erro4_list, pred_list, real_list, trace_list = test_model_TSM(
        TSM_model,
        df_test_processed,
        test_case_indices_list, # Passa a lista de listas de índices de casos
        metric,
        pen
    )

    # 5. Calcular as métricas finais agregadas
    # Como não temos mais folds, as métricas serão calculadas diretamente sobre as listas de erros
    MAE = np.mean(erro1_list)
    RMSE = np.sqrt(np.mean(np.square(erro1_list)))
    MAPE = 100 * np.mean(erro2_list)
    MAPE_median = 100 * np.median(erro2_list) # Não estava no seu exemplo, mas deixei

    MAE_not_abs_mean = np.mean(erro3_list)
    MAE_not_abs_median = np.median(erro3_list)
    MAPE_not_abs_mean = 100 * np.mean(erro4_list)
    MAPE_not_abs_median = 100 * np.median(erro4_list)

    # 6. Retornar os resultados
    return {
        "MAE": MAE,
        "RMSE": RMSE,
        "MAPE": MAPE,
        "MAPE_median": MAPE_median,
        "MAE_not_abs_mean": MAE_not_abs_mean,
        "MAPE_not_abs_mean": MAPE_not_abs_mean,
        "MAE_not_abs_median": MAE_not_abs_median,
        "MAPE_not_abs_median": MAPE_not_abs_median,
        "Predict_trace": {"default": pred_list}, # Mantendo a estrutura de dicionário para saída
        "Real_trace": {"default": real_list},    # para compatibilidade com a exibição do app.py
        "model": TSM_model,
        "better_trace": {"default": trace_list}
    }