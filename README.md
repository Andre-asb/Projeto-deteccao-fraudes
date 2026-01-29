# Projeto Detecção de Fraudes com K-means

## Visão Geral

Este projeto implementa um sistema de detecção de fraudes utilizando **K-means** como técnica de aprendizado não supervisionado, aplicado a um dataset sintético de transações. O objetivo é identificar transações suspeitas com base em padrões de comportamento (valores, horários, scores de risco, categorias, etc.).

O app foi desenvolvido em Streamlit, permitindo visualização interativa de métricas (precision, recall), distribuição de distâncias aos centróides, projeção PCA dos clusters, além de um ranking das transações mais suspeitas.

---

## Objetivo do Projeto

- Detectar possíveis fraudes em transações financeiras usando clustering não supervisionado.
- Utilizar K-means para agrupar transações em clusters com base em features numéricas e derivadas.
- Selecionar como suspeitas as transações que estão mais distantes dos centróides (top X% por distância).
- Avaliar o desempenho em relação à variável verdadeira de fraude por meio de métricas como **precision** e **recall**.

Em termos práticos, o modelo tenta capturar anomalias: quanto maior a distância de uma transação ao centróide do cluster, maior o potencial de comportamento atípico, logo maior a suspeita de fraude.

---

## Sobre o Algoritmo K-means

K-means é um algoritmo de **clusterização** (aprendizado não supervisionado) que tem como objetivo particionar o conjunto de dados em 3 grupos (clusters), onde cada ponto é atribuído ao cluster com o centróide mais próximo.

### Passos principais do K-means

1. Definir o número de clusters: 3.
2. Inicializar 3 centróides (por exemplo, de forma aleatória).
3. Atribuir cada ponto de dados ao centróide mais próximo (formando os clusters).
4. Recalcular os centróides como a média dos pontos em cada cluster.
5. Repetir os passos 3 e 4 até convergência (mudança mínima nos centróides ou número máximo de iterações).

### Por que usar K-means para fraude?

- É simples, rápido e escalável.
- Permite identificar grupos de comportamento "normal" e, por contraste, pontos distantes dos centróides que podem ser interpretados como anomalias.
- Não exige rótulos de fraude para treinar, o que é útil em cenários em que os rótulos são escassos ou pouco confiáveis.

Neste projeto, após o treino do K-means:

- Calculamos a **distância ao centróide** de cada transação.
- Marcamos como suspeitos os casos no top X% de maior distância.
- Comparamos com `is_fraud` apenas para cálculo de **precision** e **recall** (sem vazamento de informação no modelo).

# Por que ustilizamos 3 clusters?

Mesmo sendo um problema binário (fraude vs. não fraude), o **K-Means** não “aprende” essas duas classes: ele apenas particiona as transações em k grupos pelo centróide mais próximo e minimiza a variância dentro de cada cluster, então k não precisa ser igual (e muitas vezes não é) ao número de classes do alvo . Quando você usa k = 2, pode forçar vários perfis legítimos diferentes se agruparem em só dois centróides, o que tende a aumentar a dispersão/variância dentro dos clusters e deixar muitas transações normais com distância alta ao centróide; como a regra marca “suspeito” pelo top x% de distância, isso aumenta falsos positivos e reduz a precisão. Com k = 3, o modelo consegue separar melhor subgrupos de comportamento normal, reduzindo a variância intra-cluster e tornando a distância ao centróide um escore mais discriminativo, de forma que os pontos realmente atípicos (onde fraudes tendem a cair) ficam mais concentrados no topo das distâncias, elevando a precisão.
---

## Estrutura do Código

O script principal do Streamlit segue a seguinte lógica:

1. **Configuração do app**  
   Define layout, título da página e mensagem informativa com parâmetros principais (k e percentual de top distâncias).

2. **Carregamento e preparação dos dados**
   - Leitura do arquivo `synthetic_fraud_dataset.csv`.
   - Criação de features derivadas:
     - `hora_suspeita`
     - `risco_medio`
     - `amount_log`
     - `amount_extremo`
     - `risco_alto`
     - Conversão de variáveis categóricas (`transaction_type`, `merchant_category`) em numéricas (`type_num`, `cat_num`).

3. **Configuração de hiperparâmetros**
   - `k_clusters`: número de clusters K-means.
   - `percent_top_dist`: percentual de transações mais distantes marcadas como suspeitas.

4. **Padronização e treino do K-means**
   - Padronização com `StandardScaler`.
   - Treino do K-means com `n_clusters=k_clusters` e parâmetros de estabilidade (`n_init`, `max_iter`, etc.).
   - Cálculo da distância de cada ponto ao centróide mais próximo (`dist_centroid`).

5. **Detecção de suspeitos**
   - Cálculo do limiar como quantil superior das distâncias.
   - Criação do indicador `kmeans_suspeito` (1 para suspeito, 0 caso contrário).

6. **Métricas**
   - Cálculo de:
     - Total de suspeitos.
     - Fraudes dentro dos suspeitos.
     - **Precision**: fraudes_detectadas / suspeitos.
     - **Recall**: fraudes_detectadas / fraudes_total.

7. **Visualizações (PCA e gráficos)**
   - PCA 2D para visualizar:
     - Suspeitos vs. não suspeitos.
     - Fraudes reais marcadas com "X".
     - Centros dos clusters no espaço PCA.
   - Histograma de distâncias com limiar.
   - Tamanho dos clusters.
   - Relação distância vs. fraude real.
   - Gráfico de barras: fraudes reais vs. transações suspeitas por cluster.

8. **Tabelas**
   - Top 200 suspeitos ordenados por distância ao centróide.
   - Quantidade por cluster, incluindo:
     - total de transações,
     - número de suspeitos,
     - distância média,
     - percentil 95 da distância,
     - percentual de suspeitos.

---

## Link da Base de dados (fictícia)

https://www.kaggle.com/datasets/umitka/synthetic-financial-fraud-dataset

## Rodar arquivo app.py

python -m streamlit run app.py

## Instalação das Dependências

### Dependencias
```bash
pip install streamlit==1.29.0
pip install pandas==2.1.0
pip install numpy==1.24.3
pip install matplotlib==3.7.2
pip install scikit-learn==1.3.0

