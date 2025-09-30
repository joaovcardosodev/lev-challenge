# 📊 Desafio de Previsão de Produção

> Sistema de previsão de produção diária por Banco × Produto utilizando Machine Learning e técnicas de séries temporais.

---

## 📋 Índice

- [Entendimento do Problema](#-entendimento-do-problema)
- [Solução Proposta](#-solução-proposta)
- [Validação](#-validação)
- [Riscos e Limitações](#️-riscos-e-limitações)
- [Reprodução Local](#-reprodução-local)
- [Próximos Passos](#-próximos-passos)
- [Entregáveis](#-entregáveis)

---

## 🎯 Entendimento do Problema

O desafio consiste em **prever a produção diária** por **Banco × Produto** para os próximos **6 meses** (granularidade diária).

### 💼 Relevância do Negócio

Este forecast é essencial para:
- 📈 **Planejamento de metas** estratégicas
- 🏭 **Capacidade operacional** e alocação de recursos
- 💰 **Comissionamento** de equipes comerciais

### 🔍 Hipóteses Assumidas

| Aspecto | Tratamento |
|---------|-----------|
| **Zeros e dados faltantes** | Ausência de registros = tratado com rolling forecast, previsto pares (´bank´ x ´product´) que estiveram ativos em uma janela de 90 dias |
| **Sazonalidade** | Incorporada via features de calendário (dia da semana, fim de mês, feriados) |
| **Mudança de mix** | Séries encerradas mantêm zero após última data real |
| **Choques e picos** | Capturados indiretamente por lags e médias móveis |
| **Dados externos** | Biblioteca `holidays` para feriados nacionais (reprodutível) |

---

## 🚀 Solução Proposta

### 🤖 Modelagem

**Abordagem:** Modelo global baseado em **Tweedie Regressor** treinado em todas as séries disponívels (Banco × Produto), o alvo foi convertido para escala logaritimica a fim de resolver o rpoblema de sparse data

### 🔧 Features Principais

#### **Temporais**
- 📅 **Lags:** `lag_1`, `lag_7`, `lag_30`
- 📊 **Médias móveis:** Rolling mean (7, 30, 60 dias)
- 📉 **Desvio padrão móvel:** Rolling std (7, 30, 60 dias)
- 📈 **Tendência:** Contador incremental (`trend`)

#### **Calendário**
- 🗓️ `day_of_week`, `weekday`, `month`
- 📆 `is_month_end`, `is_weekend`
- 🎉 `is_holiday`, `is_pre_holiday`, `is_post_holiday`
- 🌓 `fortnight` (quinzena)

#### **Agregações Estáticas**
- 🏦 `mean_bank`, `std_bank`
- 📦 `mean_product`, `std_product`
- 🔗 `mean_bp`, `std_bp` (banco × produto)

#### **Features Derivadas**
- ➖ Diferenças: `diff_mean_lag{1,7,30}_{bank,product,bp}`
- ➗ Razões: `ratio_mean_lag{1,7,30}_{bank,product,bp}`

### 🔄 Rolling Forecast

Previsão **iterativa dia a dia**, alimentando previsões anteriores como novos lags:

```
Dia 1 → Prever → Atualizar seed
Dia 2 → Prever (usando Dia 1) → Atualizar seed
...
Dia 180 → Prever final
```
### Resultados do Modelo (escala log)

| Métrica | Valor |
| --------|---------- |
| `RMSE` | 1.9533 |
| `R2` | 0.3966 |
| `MAE` | 1.2956 |
| `MEDAE` | 0.9257 |


### 📤 Saída Final

**Arquivo:** `forecast.csv`  
**Granularidade:** Diária por Banco × Produto

```csv
date,bank,product,prediction,prediction_lo,prediction_hi
2025-10-01,Banco X,Product A,123.4,98.7,148.1
2025-10-01,Banco X,Product B,87.1,69.7,104.5
...
```

**Colunas:**
- `prediction`: Previsão pontual
- `prediction_lo`: Limite inferior (IC 80%)
- `prediction_hi`: Limite superior (IC 120%)

---

## ✅ Validação

### 📊 Estratégia

**Backtesting temporal** com time-series split: janelas deslizantes simulando previsões em diferentes períodos históricos.

### 📈 Métricas Avaliadas

| Métrica | Descrição | Uso |
|---------|-----------|-----|
| **MAE** | Mean Absolute Error | Erro absoluto médio - fácil interpretação |
| **R²** | Coefficient of Determination | Proporção de variância explicada (0-1) |
| **RMSE** | Root Mean Squared Error | Penaliza erros grandes - sensível a outliers |
| **MedAE** | Median Absolute Error | Robusto a outliers - mediana dos erros |

> 💡 **Por quê?** Essas métricas fornecem uma visão completa:
> - **MAE/MedAE**: Magnitude típica dos erros
> - **R²**: Qualidade do ajuste, mostra o quão bem o modelo explica a variância do alvo (quanto maior, melhor)
> - **RMSE**: Penaliza previsões muito distantes

---

## ⚠️ Riscos e Limitações

### 🚨 Principais Riscos

| Risco | Impacto | Mitigação |
|-------|---------|-----------|
| **Mudança abrupta de mix** | Modelo pode prever zeros incorretamente | Flag de descontinuação em produção |
| **Choques não recorrentes** | Pandemias, mudanças monetárias | Variáveis externas + re-treino |
| **Sazonalidades complexas** | Calendário INSS, folhas de pagamento | Features de pagamento customizadas |
| **Valores extremos** | Overflow em cálculos estatísticos | MAD (Median Absolute Deviation) e Reescaling |

### 🛡️ Mitigações em Produção

✅ **Re-treino periódico** (mensal)  
✅ **Inclusão de variáveis externas** (taxas de juros, calendário INSS)  
✅ **Monitoramento em tempo real** com alertas de drift  
✅ **A/B testing** de novos modelos  

---

## 💻 Reprodução Local

### 📦 Pré-requisitos

```bash
Python 3.10+
pip install -r requirements.txt
```

### 📁 Estrutura de Pastas

```
project/
│
├── data/
│   ├── raw/              # 📥 Dados originais
│   ├── processed/        # ⚙️ Dados tratados
│
├── notebooks/
│   ├── 01_eda_and_feature_engineering.ipynb
│   └── 02_model_training.ipynb
│
├── src/
│   └── run_model.py      # 🚀 Script principal
│
├── models/
│   └── tweedie_model.pkl # 🤖 Modelo treinado
│
├── forecast.csv          # 📊 Saída final
├── requirements.txt      # 📋 Dependências
└── README.md             # 📖 Este arquivo
```

### ▶️ Rodando o Pipeline

1. **Garantir que o modelo treinado existe:**
   ```bash
   ls models/tweedie_model.pkl
   ```

2. **Executar o script de previsão:**
   ```bash
   python src/run_model.py
   ```

3. **Verificar a saída:**
   ```bash
   head forecast.csv
   ```

### 🔧 Dependências Principais

```txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
holidays>=0.35
joblib>=1.5.2
```

---

## 🎯 Próximos Passos

### 🔬 Melhorias de Modelo

- [ ] Testar **Prophet** para sazonalidades complexas
- [ ] Experimentar **ensemble** de múltiplos modelos
- [ ] Adicionar **features externas** (macro-econômicas)

### 🏗️ Infraestrutura

- [ ] **Deploy em API REST** (FastAPI)
- [ ] **Pipeline de re-treino** automatizado (Airflow/Prefect)
- [ ] **Dashboard de monitoramento** (Grafana/Streamlit)
- [ ] **Containerização** com Docker

---

## 📦 Entregáveis

### ✅ Arquivos Incluídos

| Arquivo | Descrição |
|---------|-----------|
| 📊 `forecast.csv` | Previsões 6 meses (diário, banco × produto) |
| 📖 `README.md` | Documentação completa (este arquivo) |
| 📓 `notebooks/` | EDA, feature engineering e treinamento |
| 🐍 `src/run_model.py` | Script de inferência reprodutível |
| 🤖 `models/` | Modelos treinados (pickle) |

### 📊 Resultados Esperados

**Formato do CSV de saída:**
- ✅ 180 dias de previsão
- ✅ Todas as combinações Banco × Produto ativas
- ✅ Intervalos de confiança (80%-120%)
- ✅ Valores não-negativos
- ✅ Tratamento de overflow e valores extremos

---

## 🧾 Conclusão

A base de dados com alta granularidade, sparse e com zero-inflation trouxe diversos desafios para o desempenho de um modelo generalista, os resultados na escala original podem maquiar o verdadeiro desempenho do modelo final, com mais tempo de desenvolvimento tentaria seguir por dois caminhos em paralelo:

- Modelos especialistas por banco
- Modelos especificos e potentes para séries temporais como o Prophet

---

## 👥 Contato

📧 **Email:** [joaovictorcs.20@exemplo.com](mailto:joaovictorcs.20@gmail.com)  
💼 **LinkedIn:** [João Victor Cardoso](https://www.linkedin.com/in/jo%C3%A3o-victor-cardoso/)  
🐙 **GitHub:** [@joaovcardosodev](https://github.com/joaovcardosodev)

---