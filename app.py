import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# =========================
# 1) CONFIGURAÇÃO DO APP
# =========================
st.set_page_config(layout="wide", page_title="Detecção de fraudes")
st.title("Projeto Detecção de fraudes")

# =========================
# 2) CARREGAMENTO DO DATASET + FEATURE ENGINEERING
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_fraud_dataset.csv")

    df["hora_suspeita"] = df["hour"].isin([0, 1, 2, 3, 4, 5, 6, 7, 8, 22, 23]).astype(int)
    df["risco_medio"] = (df["device_risk_score"] + df["ip_risk_score"]) / 2
    df["amount_log"] = np.log1p(df["amount"])
    df["amount_extremo"] = (df["amount"] > df["amount"].quantile(0.97)).astype(int)
    df["risco_alto"] = (df["risco_medio"] > 0.84).astype(int)

    # Se quiser usar categorias no modelo, inclua nos features_kmeans.
    df["type_num"] = pd.Categorical(df["transaction_type"]).codes
    df["cat_num"] = pd.Categorical(df["merchant_category"]).codes

    return df

df = load_data()
fraudes_total = int(df["is_fraud"].sum())

# =========================
# 3) CONFIGURAÇÃO (HIPERPARÂMETROS)
# =========================
k_clusters = 3  # diminuiu de 8
percent_top_dist = 5.0  # top X% maiores distâncias = suspeitos (tende a aumentar precisão)

st.info(f"Configuração: k={k_clusters} | distancia de seleção={percent_top_dist:.1f}")

# =========================
# 4) PREPARAÇÃO DOS DADOS
# =========================
features_kmeans = [
    "risco_medio",
    "amount_log",
    "amount_extremo",
    "hora_suspeita",
    "risco_alto",
    "device_risk_score",
    "ip_risk_score",
    "type_num",
    "cat_num"
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features_kmeans])

# =========================
# 5) TREINAMENTO DO KMEANS + SCORE DE DISTÂNCIA
# =========================
kmeans = KMeans(
    n_clusters=k_clusters,
    random_state=42,
    n_init=50,
    max_iter=100,
    tol=1e-4
)

df["cluster"] = kmeans.fit_predict(X_scaled)
df["dist_centroid"] = np.min(kmeans.transform(X_scaled), axis=1)

# =========================
# 6) DETECÇÃO (SEM is_fraud)
# =========================
limiar_dist = df["dist_centroid"].quantile(1 - (percent_top_dist / 100))
df["kmeans_suspeito"] = (df["dist_centroid"] > limiar_dist).astype(int)

suspeitos = df[df["kmeans_suspeito"] == 1]
fraudes_detectadas = int(suspeitos["is_fraud"].sum())

# =========================
# 7) MÉTRICAS
# =========================
st.header("Resultados")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Suspeitos", int(len(suspeitos)))
col2.metric("Fraudes (dentro dos suspeitos)", int(fraudes_detectadas))

precision = (fraudes_detectadas / len(suspeitos)) if len(suspeitos) > 0 else 0.0
recall = (fraudes_detectadas / fraudes_total) if fraudes_total > 0 else 0.0

col3.metric("Precision", f"{precision:.1%}")
col4.metric("Recall", f"{recall:.1%}")

# =========================
# 8) VISUALIZAÇÕES (PCA + NOVO GRÁFICO)
# =========================
st.header("Visualizações")

fig, axes = plt.subplots(2, 3, figsize=(24, 12))

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

def comentarios(ax, txt, y=-0.22):
    ax.text(
        0.5, y, txt,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=12, wrap=True
    )


# (1) PCA: suspeitos em vermelho, normais em azul
colors = np.where(df["kmeans_suspeito"].values == 1, "red", "lightblue")

sc_base = axes[0, 0].scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=colors, alpha=0.7, s=25
)

# Fraudes reais como X preto (apenas visualização / avaliação)
sc_fraudes = axes[0, 0].scatter(
    X_pca[df["is_fraud"] == 1, 0],
    X_pca[df["is_fraud"] == 1, 1],
    c="black", s=200, marker="X",
    label=f"Fraudes reais ({fraudes_total})"
)

# Legenda manual para cores do scatter (azul = normal, vermelho = suspeito)
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Normal',
            markerfacecolor='lightblue', markersize=10, alpha=0.7),
    Line2D([0], [0], marker='o', color='w', label='Suspeitos detectados dentro dos normais',
            markerfacecolor='red', markersize=10, alpha=0.7),
    Line2D([0], [0], marker='X', color='w', label=f'Fraudes reais ({fraudes_total})',
            markerfacecolor='black', markeredgecolor='black', markersize=12),
]

axes[0, 0].set_title("Suspeitos + Fraudes reais")
axes[0, 0].legend(handles=legend_elements, loc="best")

comentarios(
    axes[0, 0],
    "Análise das transações. Cada ponto no gráfico refere-se a uma transação."
)


# (2) Centros dos clusters no PCA
centros_pca = pca.transform(kmeans.cluster_centers_)
scatter2 = axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=df["cluster"], cmap="viridis", alpha=0.6, s=20)
axes[0, 1].scatter(centros_pca[:, 0], centros_pca[:, 1], c="red", s=500, marker="X", linewidth=3)
axes[0, 1].set_title(f"Centros dos {k_clusters} clusters")
legend1 = axes[0, 1].legend(*scatter2.legend_elements(), title="Cluster")
axes[0, 1].add_artist(legend1)

comentarios(
    axes[0, 1],
    "PCA colorido por cluster do KMeans. Cada cor é um cluster; \n o X vermelho é o centróide do cluster no PCA."
)

# (3) Histograma de distâncias + limiar
axes[0, 2].hist(df[df["kmeans_suspeito"] == 1]["dist_centroid"], bins=20, alpha=0.8, color="red",
                label=f"Suspeitos ({len(suspeitos)})")
axes[0, 2].hist(df[df["kmeans_suspeito"] == 0]["dist_centroid"], bins=20, alpha=0.7, label="Normais")
axes[0, 2].axvline(limiar_dist, color="orange", linestyle="--", label=f"Top {percent_top_dist:.1f}% = {limiar_dist:.2f}")
axes[0, 2].set_title("Distância ao centróide (limiar)")
axes[0, 2].legend()

comentarios(
    axes[0, 2],
    "Centróide dos clusteres. Define a distância limite (4,05) \n para o modelo marcar a transação como suspeita."
)

# (4) Tamanho dos clusters (sem usar is_fraud)
cluster_sizes = df["cluster"].value_counts().sort_index()
axes[1, 0].bar(cluster_sizes.index.astype(str), cluster_sizes.values, color="steelblue")
axes[1, 0].set_title("Volume de transações por cluster.")
axes[1, 0].set_ylabel("Qtd transações")
axes[1, 0].set_xlabel("Clusters")


comentarios(
    axes[1, 0],
    "Volume de transações por cluster. Ajuda a ver \n se algum cluster concentra a maior parte do tráfego."
)

# (5) Distância vs fraude real (apenas para avaliar)
axes[1, 1].scatter(df["dist_centroid"], df["is_fraud"], alpha=0.6, s=30)
axes[1, 1].axvline(limiar_dist, color="red", linestyle="--")
axes[1, 1].set_title("Distância vs Fraude real (avaliação)")
axes[1, 1].set_xlabel("Distância ao centróide")
axes[1, 1].set_ylabel("is_fraud")

comentarios(
    axes[1, 1],
    " A linha tracejada marca o valor 4,05. Acima desse valor, \n vê-se que a transação tem alta probabilidade de ser Fraude."
)

# (6) Fraudes reais vs suspeitos por cluster
cluster_fraud_summary = (
    df.groupby("cluster")
        .agg(
            fraudes_reais=("is_fraud", "sum"),
            suspeitos=("kmeans_suspeito", "sum")
        )
        .reset_index()
)

x = np.arange(len(cluster_fraud_summary["cluster"]))
width = 0.35

axes[1, 2].bar(x - width/2, cluster_fraud_summary["fraudes_reais"], width, label="Fraudes reais", color="black")
axes[1, 2].bar(x + width/2, cluster_fraud_summary["suspeitos"], width, label="Transações suspeitas", color="red")
axes[1, 2].set_xticks(x)
axes[1, 2].set_xticklabels(cluster_fraud_summary["cluster"].astype(str))
axes[1, 2].set_title("Fraudes reais vs suspeitos por cluster")
axes[1, 2].set_xlabel("Cluster")
axes[1, 2].set_ylabel("Quantidade")
axes[1, 2].legend()

comentarios(
    axes[1, 2],
    "Visualização da captura de fraudes. \n Comparação por cluster: preto = fraudes reais, vermelho = suspeitos do KMeans."
)

plt.tight_layout(pad=1.2, h_pad=2.0, w_pad=1.0)
st.pyplot(fig)


# =========================
# 9) TABELA FINAL
# =========================
st.header("Top suspeitos")
top_suspeitos = df[df["kmeans_suspeito"] == 1].nlargest(200, "dist_centroid")
st.dataframe(
    top_suspeitos[
        ["transaction_id", "user_id", "amount", "merchant_category", "hour",
            "risco_medio", "dist_centroid", "cluster", "is_fraud"]
    ].round(3)
)

st.header("Quantidades por cluster")

cluster_qtd = (
    df.groupby("cluster")
        .agg(
            total_transacoes=("transaction_id", "count"),
            suspeitos=("kmeans_suspeito", "sum"),
            dist_media=("dist_centroid", "mean"),
            dist_percentil95=("dist_centroid", lambda s: float(np.quantile(s, 0.95)))
        )
        .reset_index()
)

cluster_qtd["pct_suspeitos"] = (
    100 * cluster_qtd["suspeitos"] / cluster_qtd["total_transacoes"]
)

# Ordena para ficar mais fácil ver onde está concentrando suspeitos
cluster_qtd = cluster_qtd.sort_values("suspeitos", ascending=False)

st.dataframe(
    cluster_qtd.round(
        {"dist_media": 3, "dist_p95": 3, "pct_suspeitos": 2}
    ),
    use_container_width=True
)
