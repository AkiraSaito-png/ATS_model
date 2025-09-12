import pandas as pd
# import joblib
import numpy as np
from sklearn.model_selection import train_test_split


# =====================================
# Carregar CSV
# =====================================
def load_data(csv_path: str) -> pd.DataFrame:
    """
    Carrega os dados a partir de um arquivo CSV fornecido pelo usuário.
    """
    df = pd.read_csv(csv_path)

    # 🔹 Verifica colunas mínimas necessárias
    required_cols = ["id_caso", "timestamp", "abreviacao_status"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"O CSV precisa conter a coluna '{col}'")

    return df


# =====================================
# Pré-processamento
# =====================================
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza o pré-processamento: calcula inicio/fim, tempo restante, sub-traces.
    """

    # Conversão de timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["inicio"] = df["timestamp"]
    df["fim"] = df.groupby("id_caso")["timestamp"].shift(-1)
    if df["fim"].isnull().any():
        df["fim"] = df["fim"].fillna(method="ffill")

    # Tempo remanescente (em horas)
    last_event_per_case = df.groupby("id_caso")["timestamp"].transform("max")
    df["time_remain"] = (last_event_per_case - df["timestamp"]).dt.total_seconds() / 3600
    # cria uma coluna com a sequencia de atividades
    df["trace"] = df.groupby("id_caso")["abreviacao_status"].transform(lambda x: list(x))
    # cria uma coluna com o tipo de nota de ordem
    df["tipo_nota_ordem"] = df["tipo_nota_ordem"].astype(str)
    
    # Criação dos sub-traces
    all_sequences = []
    current_activities = []
    order = df["id_caso"].iloc[0]
    for j in range(len(df)):
        current_order_id = df["id_caso"].iloc[j]
        current_activity = df["abreviacao_status"].iloc[j]
        if current_order_id == order:
            current_activities.append(current_activity)
        else:
            order = current_order_id
            current_activities = [current_activity]
        all_sequences.append(current_activities.copy())
    df["trace"] = all_sequences

    return df


# =====================================
# Modelo TSM
# =====================================
def create_model_sist_trans_descartado(acts, times, train_ind, metric="Mean"):
    """
    Cria o modelo de sistema de transição baseado nos sub-traces.
    """
    prod = []
    prod_aux = []
    trace = []
    acts_str = acts.astype(str)
    train_aux = []

    print(f"acts_str has {len(acts_str)} rows.") # Debug: Quantas linhas acts_str tem
    print(f"train_ind has {len(train_ind)} elements.")

    for i in train_ind:  # itera nos indices de treino
        if i not in train_aux:
            case1 = acts_str.iloc[i]
            trace.append(acts.iloc[i])
            ind = acts_str[acts_str == case1].index
            for j in ind:
                train_aux.append(j)
                prod_aux.append(times.iloc[j])

            if metric == "Mean":
                val = np.mean(prod_aux)
            elif metric == "Median":
                val = np.median(prod_aux)
            prod.append(val)
            prod_aux = []

    return pd.DataFrame(list(zip(trace, prod)),
                        columns=["trace", "time_remain"])


def create_model_sist_trans(acts, times, metric="Mean"):
    """
    Cria o modelo de sistema de transição baseado nos sub-traces.
    Esta é uma versão refatorada que usa groupby para eficiência e simplicidade.
    
    Args:
        acts (pd.Series): Série contendo os traces (atividades).
        times (pd.Series): Série contendo os tempos restantes.
        metric (str): Métrica de agregação ("Mean" ou "Median").
        
    Returns:
        pd.DataFrame: DataFrame do modelo com as colunas ["trace", "time_remain"].
    """
    # Combina os dados de entrada em um único DataFrame temporário
    df_fold = pd.DataFrame({'trace': acts, 'time_remain': times})

    # Agrupa por 'trace' e calcula a agregação desejada
    if metric == "Mean":
        model_df = df_fold.groupby('trace')['time_remain'].mean().reset_index()
    elif metric == "Median":
        model_df = df_fold.groupby('trace')['time_remain'].median().reset_index()
    else:
        # Lança um erro se a métrica for inválida
        raise ValueError("Métrica deve ser 'Mean' ou 'Median'")
        
    return model_df

# =====================================
# Treinar modelo e salvar
# =====================================
# def train_and_save_model(df: pd.DataFrame, path="tsm_model.pkl"):
#     """
#     Pré-processa os dados, treina o modelo TSM e salva no disco
#     """
#     df = preprocess_data(df)

#     # Índices de treino: todos os registros
#     train_ind = list(df.index)

#     # Cria modelo
#     model = create_model_sist_trans(df["trace"], df["time_remain"], train_ind, metric="Mean")

#     joblib.dump(model, path)
#     print(f"Modelo salvo em {path}")
#     return model


# =====================================
# Carregar modelo salvo
# =====================================
# def load_model(path="tsm_model.pkl"):
#     return joblib.load(path)


# =====================================
# Predição
# =====================================
def predict_time_remaining(model: pd.DataFrame, trace: list, metric="Mean"):
    """
    Faz a previsão do tempo remanescente para um único trace (lista de atividades).
    """
    act_to_value = dict(zip(model['trace'].astype(str), model['time_remain']))
    act_str = str(trace)

    if act_str in act_to_value:
        return act_to_value[act_str]

    # Se não encontrou, tenta achar sub-traces semelhantes
    similar_acts = find_better(model["trace"], trace, pen=1)
    if similar_acts:
        similar_times = [act_to_value[str(a)] for a in similar_acts if str(a) in act_to_value]
        if similar_times:
            if metric == "Mean":
                return np.mean(similar_times)
            elif metric == "Median":
                return np.median(similar_times)

    # fallback: retorna média geral
    return model["time_remain"].mean()


# =====================================
# Função auxiliar
# =====================================
def find_better(acts, activ, pen=1):
    indelist = []
    inde = -1
    max_seq = 0
    n_act = len(activ)
    if pen == 1:
        for ki in acts:
            seq = 0
            taman1 = max(n_act, len(ki))
            taman2 = min(n_act, len(ki))
            for ji in range(taman2):
                if activ[ji] == ki[ji]:
                    seq = seq + 1 / taman1
                    if seq >= max_seq:
                        max_seq = seq
                        inde = str(ki)
            if inde == str(ki):
                indelist.append((max_seq, ki))
    elif pen == 2:
        for ki in acts:
            seq = 0
            taman1 = max(n_act, len(ki))
            taman2 = min(n_act, len(ki))
            if taman1 != taman2:
                seq = -1 / taman1
            for ji in range(taman2):
                if activ[ji] == ki[ji]:
                    seq = seq + 1 / taman1
                    if seq >= max_seq:
                        max_seq = seq
                        inde = str(ki)
            if inde == str(ki):
                indelist.append((max_seq, ki))

    real_inde = [k[1] for k in indelist if k[0] == max_seq]
    return real_inde


# =====================================
# Preparar CSV de teste
# =====================================
def prepare_test_data(csv_path: str) -> pd.DataFrame:
    """
    Carrega e pré-processa um CSV de teste bruto.
    Retorna DataFrame pronto para usar em predict_time_remaining.
    """
    df = load_data(csv_path)
    df = preprocess_data(df)
    return df


# =====================================
# Split Holdout
# =====================================
def split_train_test_holdout(dfnew, test_size: float = 0.3, random_state: int = 42):
    """
    Faz o holdout split em treino e teste (nível de casos),
    suportando DataFrame único ou dict de DataFrames.
    
    Args:
        dfnew (dict | DataFrame): dados de entrada
        test_size (float): proporção do conjunto de teste (0.0 - 1.0)
        random_state (int): seed para reprodutibilidade
        
    Returns:
        df_train (dict): índices de treino no formato {key: [[indices]]}
        df_test (dict): índices de teste no formato {key: [[indices]]}
    """
    df_train = {}
    df_test = {}

    # 🔹 Se for um único DataFrame, converte para dict
    if isinstance(dfnew, pd.DataFrame):
        dfnew = {"default": dfnew}

    for key, df in dfnew.items():
        if "id_caso" not in df.columns:
            raise KeyError(f"O DataFrame '{key}' não contém a coluna 'id_caso'.")

        # Casos únicos
        cases = df["id_caso"].unique()

        # Split de casos (não de linhas)
        train_cases, test_cases = train_test_split(
            cases, test_size=test_size, random_state=random_state
        )

        # Índices correspondentes
        train_idx = df.index[df["id_caso"].isin(train_cases)].tolist()
        test_idx = df.index[df["id_caso"].isin(test_cases)].tolist()

        # Mantém o mesmo formato esperado: listas de listas
        df_train[key] = [train_idx]
        df_test[key] = [test_idx]

    return df_train, df_test


# =====================================
# Avaliar modelo no teste
# =====================================
def evaluate_model(DF, train_indexes, test_indexes, metric="Mean", pen=1):
    """
    Avalia modelos de Sistema de Transição (TSM) usando treino e teste definidos.

    Args:
        DF (dict): Dicionário de DataFrames, onde cada chave representa um produto/condição.
        train_indexes (dict): Índices de treino para cada chave de DF.
        test_indexes (dict): Índices de teste para cada chave de DF.
        metric (str): Métrica de agregação usada para o modelo ("Mean" ou "Median").
        pen (int, optional): Penalidade usada no teste do modelo. Default = 1.

    Returns:
        dict: Resultados contendo erros (MAE, RMSE, MAPE, etc.) e previsões reais/preditas.
    """
    MAEs = {}
    RMSEs = {}
    MAPEs = {}
    MAPEs2 = {}

    MAEs_not_abs_mean = {}
    MAEs_not_abs_median = {}
    MAPEs_not_abs_mean = {}
    MAPEs_not_abs_median = {}

    predict_trace = {}
    real_trace = {}
    better_trace = {}

    for i in DF.keys():  # itera nos dataframes
        erro_test1 = []
        erro_test2 = []
        erro_test3 = []
        erro_test4 = []
        p = []
        r = []
        show_trace = []

        for j in range(len(train_indexes[i])):  # itera nos grupos de treino
            print(f'Sample {j} of prod {i[0]} with {i[1]} prob rep          ', end='\r')

            # cria o modelo de sistema de transição
            TSM_model = create_model_sist_trans(
                DF[i]["trace"],
                DF[i]["time_remain"],
                train_indexes[i][j],
                metric
            )

            # testa o modelo de sistema de transição
            err1, err2, err3, err4, pred, real, trace = test_model_TSM(
                TSM_model,
                DF[i],
                test_indexes[i][j],
                metric,
                pen
            )

            erro_test1.append(err1)  # salva o erro absoluto
            erro_test2.append(err2)  # salva o erro relativo
            erro_test3.append(err3)  # salva o erro absoluto (sem abs)
            erro_test4.append(err4)  # salva o erro relativo (sem abs)
            p.append(pred)
            r.append(real)
            show_trace.append(trace)

        MAEs[i] = [np.mean(k) for k in erro_test1]
        RMSEs[i] = [np.sqrt(np.mean(np.square(k))) for k in erro_test1]
        MAPEs[i] = [100 * np.mean(k) for k in erro_test2]
        MAPEs2[i] = [100 * np.median(k) for k in erro_test2]

        MAEs_not_abs_mean[i] = [np.mean(k) for k in erro_test3]
        MAEs_not_abs_median[i] = [np.median(k) for k in erro_test3]
        MAPEs_not_abs_mean[i] = [100 * np.mean(k) for k in erro_test4]
        MAPEs_not_abs_median[i] = [100 * np.median(k) for k in erro_test4]

        predict_trace[i] = p
        real_trace[i] = r
        better_trace[i] = show_trace


    return {
        "MAE": MAEs,
        "RMSE": RMSEs,
        "MAPE": MAPEs,
        "MAPE_median": MAPEs2,
        "MAE_not_abs_mean": MAEs_not_abs_mean,
        "MAPE_not_abs_mean": MAPEs_not_abs_mean,
        "MAE_not_abs_median": MAEs_not_abs_median,
        "MAPE_not_abs_median": MAPEs_not_abs_median,
        "Predict_trace": predict_trace,
        "Real_trace": real_trace,
        "model": TSM_model,
        "better_trace": better_trace
    }



def evaluate_model_new(DF_train, DF_test, train_indexes, test_indexes, metric="Mean", pen=1):
    """
    Avalia modelos de Sistema de Transição (TSM) usando treino e teste vindos
    de bases diferentes, mas ainda controlados por índices.

    Args:
        DF_train (dict): Dicionário de DataFrames de treino.
                         Estrutura: {"trace", "time_remain"}
        DF_test (dict): Dicionário de DataFrames de teste.
                        Estrutura: {"trace", "time_remain"}
        train_indexes (dict): Índices (listas de listas) de treino para cada chave.
        test_indexes (dict): Índices (listas de listas) de teste para cada chave.
        metric (str): Métrica de agregação ("Mean" ou "Median").
        pen (int): Penalidade usada no teste do modelo.

    Returns:
        dict: Resultados contendo métricas de erro e previsões reais/preditas.
    """
    MAEs = {}
    RMSEs = {}
    MAPEs = {}
    MAPEs2 = {}

    MAEs_not_abs_mean = {}
    MAEs_not_abs_median = {}
    MAPEs_not_abs_mean = {}
    MAPEs_not_abs_median = {}

    predict_trace = {}
    real_trace = {}
    better_trace = {}

    for i in DF_train.keys():  # itera nos produtos/condições
        erro_test1 = []
        erro_test2 = []
        erro_test3 = []
        erro_test4 = []
        p = []
        r = []
        show_trace = []

        for j in range(len(train_indexes[i])):  # itera nos folds/grupos de treino

            current_train_indices = train_indexes[i][j]
        
            train_fold_df = DF_train[i].iloc[current_train_indices].copy()

            train_fold_df['trace'] = train_fold_df['trace'].apply(lambda x: '_'.join(map(str, x)))

            TSM_model = create_model_sist_trans(
                acts=train_fold_df["trace"],
                times=train_fold_df["time_remain"],
                metric=metric
            )

            current_test_indices = test_indexes[i][j]
            
            # 2. Fatie o DataFrame de teste
            test_fold_df = DF_test[i].iloc[current_test_indices].copy()

            # 3. Converta a coluna 'trace' para string no DataFrame de teste também,
            #    para que ele tenha o mesmo formato que o TSM_model espera para a comparação.
            test_fold_df['trace'] = test_fold_df['trace'].apply(lambda x: '_'.join(map(str, x)))
            

            # testa o modelo com os índices de teste sobre DF_test
            err1, err2, err3, err4, pred, real, trace = test_model_TSM(
                TSM_model,
                test_fold_df,
                current_test_indices,
                metric,
                pen
            )

            erro_test1.append(err1)
            erro_test2.append(err2)
            erro_test3.append(err3)
            erro_test4.append(err4)
            p.append(pred)
            r.append(real)
            show_trace.append(trace)

        # calcula métricas por fold
        MAEs[i] = [np.mean(k) for k in erro_test1]
        RMSEs[i] = [np.sqrt(np.mean(np.square(k))) for k in erro_test1]
        MAPEs[i] = [100 * np.mean(k) for k in erro_test2]
        MAPEs2[i] = [100 * np.median(k) for k in erro_test2]

        MAEs_not_abs_mean[i] = [np.mean(k) for k in erro_test3]
        MAEs_not_abs_median[i] = [np.median(k) for k in erro_test3]
        MAPEs_not_abs_mean[i] = [100 * np.mean(k) for k in erro_test4]
        MAPEs_not_abs_median[i] = [100 * np.median(k) for k in erro_test4]

        predict_trace[i] = p
        real_trace[i] = r
        better_trace[i] = show_trace

    return {
        "MAE": MAEs,
        "RMSE": RMSEs,
        "MAPE": MAPEs,
        "MAPE_median": MAPEs2,
        "MAE_not_abs_mean": MAEs_not_abs_mean,
        "MAPE_not_abs_mean": MAPEs_not_abs_mean,
        "MAE_not_abs_median": MAEs_not_abs_median,
        "MAPE_not_abs_median": MAPEs_not_abs_median,
        "Predict_trace": predict_trace,
        "Real_trace": real_trace,
        "model": TSM_model,
        "better_trace": better_trace
    }


# =====================================
# test_model_TSM
# =====================================

def test_model_TSM(model, df_test, test_ind, metric="Mean", find=1):
    """
    Args:
      modelo: modelo de sistema de transição.
      df_teste: dataframe do log de teste.
      teste_ind: indices dos sub-traces de teste.
      metric: metrica usada para agragar os valores de sub-traces iguais. Pode ser "Mean" ou "Median".
      find: para a função find_better(). Se find=1, não aplica uma penalidade a sub-traces de tamanhos diferentes do desconhecido. Se find=2, aplica uma penalidade.

    Returns: erros absolutos e relativos do modelo nos sub-traces de teste.
    """

    # Pré-processamento inicial
    # Cria dicionário de mapeamento atividade -> valor previsto
    act_to_value = dict(zip(model['trace'].astype(str),
                           model['time_remain']))

    # Pré-filtra o dataframe de teste
    test_df = df_test.loc[test_ind]

    # Inicializa listas de resultados
    error_pred1 = []  # Erro absoluto
    error_pred2 = []  # Erro relativo absoluto
    error_pred3 = []  # Erro bruto
    error_pred4 = []  # Erro relativo bruto
    predicted = []
    real1 = []
    find_better_traces = []

    # Processa cada caso de teste
    for act, real in zip(test_df['trace'], test_df['time_remain']):
        act_str = str(act)

        if act_str in act_to_value:
            pred = act_to_value[act_str]

            traces = None
        else:
            # Encontra atividades similares
            similar_acts = find_better(model["trace"], act, find)

            traces = similar_acts
            
            if similar_acts:
                # Obtém valores previstos para atividades similares
                similar_times = [act_to_value[str(a)] for a in similar_acts if str(a) in act_to_value]

                if similar_times:
                    if metric == "Mean":
                        pred = np.mean(similar_times)
                    elif metric == "Median":
                        pred = np.median(similar_times)
                else:
                    pred = model['time_remain'].mean()
            else:
                pred = model['time_remain'].mean()

        predicted.append(pred)
        real1.append(real)
        find_better_traces.append(traces)

        # Calcula todos os erros de uma vez
        error_raw = real - pred
        error_abs = abs(error_raw)

        error_pred1.append(error_abs)
        error_pred3.append(error_raw)

        if real != 0:
            error_rel = error_raw / real
            error_pred2.append(abs(error_rel))
            error_pred4.append(error_rel)

    return error_pred1, error_pred2, error_pred3, error_pred4, predicted, real1, find_better_traces

def get_index(dfnew):
    if isinstance(dfnew, pd.DataFrame):
        dfnew = {"default": dfnew}
    
    index_df = {}

    for key, df in dfnew.items():
        if "id_caso" not in df.columns:
            raise KeyError(f"O DataFrame '{key}' não contém a coluna 'id_caso'.")
        df_reset = df.reset_index(drop=True)
        # 🔹 pega os índices de cada caso
        case_indices = []
        for case_id, group in df_reset.groupby("id_caso"):
            case_indices.append(group.index.tolist())

        index_df[key] = case_indices

    return index_df