import os
import pandas as pd
import numpy as np
from datetime import timedelta
import joblib
import holidays

# ======================================
# 1. Configurações
# ======================================
PROCESSED_PATH = "data/raw/PRODUCTIONS.xlsx"   # base já tratada e com features
MODEL_PATH = "models/model_tweedie.pkl"     # modelo salvo
FORECAST_PATH = "forecast.csv"
FORECAST_DAYS = 180  # horizonte de previsão

# ======================================
# 2. Funções auxiliares
# ======================================


def create_features(df):
    """Cria features temporais e evita leakage"""
    df['mean_bank'] = (
        df.groupby('bank')['production']
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    df['mean_product'] = (
        df.groupby('product')['production']
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    df['mean_bp'] = (
        df.groupby(['bank','product'])['production']
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    df['std_bank'] = (
        df.groupby('bank')['production']
        .transform(lambda x: x.shift(1).expanding().std())
    )

    df['std_product'] = (
        df.groupby('product')['production']
        .transform(lambda x: x.shift(1).expanding().std())
    )

    df['std_bp'] = (
        df.groupby(['bank','product'])['production']
        .transform(lambda x: x.shift(1).expanding().std())
    )

    # Lags
    df['lag_1'] = df.groupby(['bank', 'product'])['production'].shift(1).fillna(df['mean_bp'])
    df['lag_7'] = df.groupby(['bank', 'product'])['production'].shift(7).fillna(df['mean_bp'])
    df['lag_30'] = df.groupby(['bank', 'product'])['production'].shift(30).fillna(df['mean_bp'])

    # Médias móveis
    df['rolling_mean_7'] = df.groupby(['bank', 'product'])['production'].transform(lambda x: x.shift(1).rolling(7).mean()).fillna(df['mean_bp'])
    df['rolling_mean_30'] = df.groupby(['bank', 'product'])['production'].transform(lambda x: x.shift(1).rolling(30).mean()).fillna(df['mean_bp'])
    df['rolling_mean_60'] = df.groupby(['bank', 'product'])['production'].transform(lambda x: x.shift(1).rolling(60).mean()).fillna(df['mean_bp'])

    # Desvio padrão
    df['rolling_std_7'] = df.groupby(['bank', 'product'])['production'].transform(lambda x: x.shift(1).rolling(7).std()).fillna(df['std_bp'])
    df['rolling_std_30'] = df.groupby(['bank', 'product'])['production'].transform(lambda x: x.shift(1).rolling(30).std()).fillna(df['std_bp'])
    df['rolling_std_60'] = df.groupby(['bank', 'product'])['production'].transform(lambda x: x.shift(1).rolling(60).std()).fillna(df['std_bp'])

    # Features de calendário
    df['weekday'] = df['date'].dt.dayofweek
    df['day_of_week'] = df['date'].dt.day_name()
    df['month'] = df['date'].dt.month
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['fortnight'] = df['date'].apply(lambda x: 'First Fortnight' if x.day <= 15 else 'Second Fortnight')
    df["is_weekend"] = df['date'].dt.dayofweek > 4
    df["is_weekend"] = df["is_weekend"].astype(int)

    # Feriados (externa)
    br_holidays = holidays.Brazil()
    df['is_holiday'] = df['date'].isin(br_holidays).astype(int)
    df['is_pre_holiday'] = df['date'].shift(1).isin(br_holidays).astype(int)
    df['is_post_holiday'] = df['date'].shift(-1).isin(br_holidays).astype(int)

    # Variáveis Produção
    # OBS: o shift/lag  evitam data leakage

    df['diff_mean_lag1_bank'] = df['lag_1'] - df['mean_bank']
    df['diff_mean_lag7_bank'] = df['lag_7'] - df['mean_bank']
    df['diff_mean_lag30_bank'] = df['lag_30'] - df['mean_bank']

    df['diff_mean_lag1_product'] = df['lag_1'] - df['mean_product']
    df['diff_mean_lag7_product'] = df['lag_7'] - df['mean_product']
    df['diff_mean_lag30_product'] = df['lag_30'] - df['mean_product']

    df['diff_mean_lag1_bank_product'] = df['lag_1'] - df['mean_bp']
    df['diff_mean_lag7_bank_product'] = df['lag_7'] - df['mean_bp']
    df['diff_mean_lag30_bank_product'] = df['lag_30'] - df['mean_bp']

    df['ratio_mean_lag1_bank'] = df['lag_1'] / (df['mean_bank'] + 1e-6)
    df['ratio_mean_lag7_bank'] = df['lag_7'] / (df['mean_bank'] + 1e-6)
    df['ratio_mean_lag30_bank'] = df['lag_30'] / (df['mean_bank'] + 1e-6)

    df['ratio_mean_lag1_product'] = df['lag_1'] / (df['mean_product'] + 1e-6)
    df['ratio_mean_lag7_product'] = df['lag_7'] / (df['mean_product'] + 1e-6)
    df['ratio_mean_lag30_product'] = df['lag_30'] / (df['mean_product'] + 1e-6)

    df['ratio_mean_lag1_bank_product'] = df['lag_1'] / (df['mean_bp'] + 1e-6)
    df['ratio_mean_lag7_bank_product'] = df['lag_7'] / (df['mean_bp'] + 1e-6)
    df['ratio_mean_lag30_bank_product'] = df['lag_30'] / (df['mean_bp'] + 1e-6)

    # Trend
    df['trend'] = df.groupby(['bank','product']).cumcount()

    # Fills
    features_to_fill_zero = [
        # Médias e Desvios Expandidos
        'mean_bank', 'mean_product', 'mean_bp',
        'std_bank', 'std_product', 'std_bp',

        # Lags (já preenchidas com mean_bp/std_bp, mas o 1º valor permanece NaN)
        'lag_1', 'lag_7', 'lag_30',
        'rolling_mean_7', 'rolling_mean_30', 'rolling_mean_60',
        'rolling_std_7', 'rolling_std_30', 'rolling_std_60',

        # Cross-Features (elas herdam os NaN dos lags/means)
        'diff_mean_lag1_bank', 'diff_mean_lag7_bank', 'diff_mean_lag30_bank',
        'diff_mean_lag1_product', 'diff_mean_lag7_product', 'diff_mean_lag30_product',
        'diff_mean_lag1_bank_product', 'diff_mean_lag7_bank_product', 'diff_mean_lag30_bank_product',
        'ratio_mean_lag1_bank', 'ratio_mean_lag7_bank', 'ratio_mean_lag30_bank',
        'ratio_mean_lag1_product', 'ratio_mean_lag7_product', 'ratio_mean_lag30_product',
        'ratio_mean_lag1_bank_product', 'ratio_mean_lag7_bank_product', 'ratio_mean_lag30_bank_product'
    ]

    df[features_to_fill_zero] = df[features_to_fill_zero].fillna(0)

    # As features is_pre_holiday e is_post_holiday usam shift(1) e shift(-1), respectivamente,
    # e podem ter um NaN no primeiro e último registro do DataFrame global.
    df['is_pre_holiday'] = df['is_pre_holiday'].fillna(0)
    df['is_post_holiday'] = df['is_post_holiday'].fillna(0)

    df = df.dropna()
    return df


def rolling_forecast(model, df, features, horizon=180):
    """Faz previsões iterativas (rolling) com base nas features"""
    last_date = df['date'].max()
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=horizon)
    preds_list = []
    current_df = df.copy()
    br_holidays = holidays.Brazil()

    for dt in future_dates:
        new_rows = []
        for (bank, product), group in current_df.groupby(['bank','product']):
            last_row = group.iloc[-1]
            new = {
                'date': dt,
                'bank': bank,
                'product': product,
                'mean_bank': last_row['mean_bank'],
                'mean_product': last_row['mean_product'],
                'mean_bp': last_row['mean_bp'],
                'std_bank': last_row['std_bank'],
                'std_product': last_row['std_product'],
                'std_bp': last_row['std_bp'],

                # Lags e médias com base no histórico (rolling)
                'lag_1': last_row['production'],
                'lag_7': group[group['date'] == dt - timedelta(days=7)]['production'].values[-1]
                         if (dt - timedelta(days=7)) in group['date'].values else last_row['production'],
                'lag_30': group[group['date'] == dt - timedelta(days=30)]['production'].values[-1]
                         if (dt - timedelta(days=30)) in group['date'].values else last_row['production'],
                'rolling_mean_7': group['production'].tail(7).mean(),
                'rolling_std_7': group['production'].tail(7).std(),
                'rolling_mean_30': group['production'].tail(30).mean(),
                'rolling_std_30': group['production'].tail(30).std(),
                'rolling_mean_60': group['production'].tail(60).mean(),
                'rolling_std_60': group['production'].tail(60).std(),
                # Cross-Features
                'diff_mean_lag1_bank': last_row['lag_1'] - last_row['mean_bank'],
                'diff_mean_lag7_bank': (group['production'].tail(7).mean() - last_row['mean_bank']) if not np.isnan(group['production'].tail(7).mean()) else 0,
                'diff_mean_lag30_bank': (group['production'].tail(30).mean() - last_row['mean_bank']) if not np.isnan(group['production'].tail(30).mean()) else 0,
                'diff_mean_lag1_product': last_row['lag_1'] - last_row['mean_product'],
                'diff_mean_lag7_product': (group['production'].tail(7).mean() - last_row['mean_product']) if not np.isnan(group['production'].tail(7).mean()) else 0,
                'diff_mean_lag30_product': (group['production'].tail(30).mean() - last_row['mean_product']) if not np.isnan(group['production'].tail(30).mean()) else 0,
                'diff_mean_lag1_bank_product': last_row['lag_1'] - last_row['mean_bp'],
                'diff_mean_lag7_bank_product': (group['production'].tail(7).mean() - last_row['mean_bp']) if not np.isnan(group['production'].tail(7).mean()) else 0,
                'diff_mean_lag30_bank_product': (group['production'].tail(30).mean() - last_row['mean_bp']) if not np.isnan(group['production'].tail(30).mean()) else 0,
                'ratio_mean_lag1_bank': last_row['lag_1'] / (last_row['mean_bank'] + 1e-6),
                'ratio_mean_lag7_bank': (group['production'].tail(7).mean()   / (last_row['mean_bank'] + 1e-6)) if not np.isnan(group['production'].tail(7).mean()) else 1e-6,
                'ratio_mean_lag30_bank': (group['production'].tail(30).mean() / (last_row['mean_bank'] + 1e-6)) if not np.isnan(group['production'].tail(30).mean()) else 1e-6,
                'ratio_mean_lag1_product': last_row['lag_1'] / (last_row['mean_product'] + 1e-6),
                'ratio_mean_lag7_product': (group['production'].tail(7).mean() / (last_row['mean_product'] + 1e-6)) if not np.isnan(group['production'].tail(7).mean()) else 1e-6,
                'ratio_mean_lag30_product': (group['production'].tail(30).mean() / (last_row['mean_product'] + 1e-6)) if not np.isnan(group['production'].tail(30).mean()) else 1e-6,
                'ratio_mean_lag1_bank_product': last_row['lag_1'] / (last_row['mean_bp'] + 1e-6),
                'ratio_mean_lag7_bank_product': (group['production'].tail(7).mean() / (last_row['mean_bp'] + 1e-6)) if not np.isnan(group['production'].tail(7).mean()) else 1e-6,
                'ratio_mean_lag30_bank_product': (group['production'].tail(30).mean() / (last_row['mean_bp'] + 1e-6)) if not np.isnan(group['production'].tail(30).mean()) else 1e-6,

                'trend': last_row['trend'] + 1,
                'day_of_week': dt.dayofweek,
                'weekday': dt.day_name(),
                'month': dt.month,
                'is_month_end': int(dt.is_month_end),
                'fortnight': 'First Fortnight' if dt.day <= 15 else 'Second Fortnight',

                # novas features de calendário
                'is_weekend': int(dt.dayofweek > 4),

                # feriados
                'is_holiday': int(dt in br_holidays),
                'is_pre_holiday': int((dt + timedelta(days=1)) in br_holidays),
                'is_post_holiday': int((dt - timedelta(days=1)) in br_holidays)
            }
            new_rows.append(new)

        new_df = pd.DataFrame(new_rows)
        # prever produção futura
        
        X_pred = new_df.reindex(columns=features, fill_value=0)

        # 🔹 Substituir inf e NaN
        X_pred = X_pred.replace([np.inf, -np.inf], 0).fillna(0)

        new_df['production'] = model.predict(X_pred)
        new_df['production'] = np.expm1(new_df['production'])
        new_df['production'] = new_df['production'].replace([np.inf, -np.inf], 0).fillna(0)
        new_df['production'] = new_df['production'].clip(lower=0)
        preds_list.append(new_df)
        # atualizar histórico
        current_df = pd.concat([current_df, new_df], ignore_index=True)

    forecast = pd.concat(preds_list, ignore_index=True)
    forecast = forecast[['date','bank','product','production']]
    forecast = forecast.rename(columns={'production':'prediction'})
    return forecast

# ======================================
# 3. Pipeline principal
# ======================================
def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "..", PROCESSED_PATH)
    model_path = os.path.join(BASE_DIR, "..", MODEL_PATH)

    print("🔹 Carregando base de features...")
    print("Lendo de:", file_path)
    df_full = pd.read_excel(file_path, parse_dates=['date'])

    print("🔹 Criando features...")
    df_full = create_features(df_full)  # 🔹 adiciona todas as colunas necessárias

    print("🔹 Carregando modelo treinado...")
    model = joblib.load(model_path)

    features = model.feature_names_in_.tolist()

    print("🔹 Iniciando rolling forecast para os próximos 180 dias...")
    forecast = rolling_forecast(model, df_full, features, horizon=FORECAST_DAYS)

    print(f"🔹 Salvando resultados em {FORECAST_PATH} ...")
    forecast.to_csv(FORECAST_PATH, index=False)

    print("✅ Processo concluído com sucesso!")

# ======================================
# 4. Execução
# ======================================
if __name__ == "__main__":
    main()