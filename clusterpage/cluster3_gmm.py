import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sklearn
from sklearn.preprocessing import MinMaxScaler

# Return Button and Title
col1, col2 = st.columns([1, 5])

with col1:
    if st.button("⬅ Back"):
        st.switch_page("homepage/home.py")

with col2:
    st.title("Mid-Table Teams Analysis menggunakan GMM dengan PCA Clustering")

with st.expander("Penjelasan Klaster"):
    st.markdown("""
    Klaster Mid-Table mengalami perubahan paling signifikan setelah penggunaan GMM dengan PCA. Nilai Rank bergeser dari menengah menjadi rendah–menengah, mencerminkan masuknya tim dengan karakteristik performa yang lebih beragam. Dari sisi ofensif, terjadi peningkatan pada indikator GtS_90 dan SoT_90, yang menunjukkan produktivitas serangan dan akurasi tembakan yang lebih tinggi, bahkan setara dengan tim elit pada variabel SoT_90. Dari aspek defensif, TackleWon_per meningkat dari menengah ke tinggi, mengindikasikan pendekatan pertahanan yang lebih reaktif. Selain itu, indikator Possession juga meningkat dari rendah ke rendah–menengah pada hampir seluruh variabel. Perubahan ini mencerminkan meningkatnya heterogenitas performa tim papan tengah akibat perpindahan tim dari klaster elit dan degradasi.
    """)

with st.sidebar:
    if st.button("⬅ Back to Home"):
        st.switch_page("homepage/home.py")

@st.cache_data
def load_data(path: str):
    data = pd.read_excel(path)
    return data

df_cluster2 = load_data("Data/Cluster/gmm_cluster2.xlsx")
st.dataframe(df_cluster2)

# =========================
# Metric groups
# =========================
METRIC_GROUPS = {
    "General": [
        "Win",
        "Draw",
        "Lose"
    ],
    "Shot and Shot Creation": [
        "Shot_90",
        "SoT_per",
        "GC_90",
        "GtS_per",
        "Goal_90",
        "xG_90"
    ],
    "Defensive  Actions": [
        "TackleWon_per",
        "ChalWon_per",
        "SB_90",
        "Err_90",
        "Int_90",
        "Clr_90"
    ],
    "Possession": [
        "Possession",
        "PassComp_per",
        "ShortPass_per",
        "MediumPass_90",
        "LongPass_90",
        "TakeOn_90",
        "Assist_90",
        "xAG_90"
    ],
    "Goalkeeping Contribution": [
        "KeeperPass_90",
        "Save_per",
        "CS_per"
    ]
}

METRIC_DIRECTION = {
    "Win": "higher",
    "Draw": "higher",
    "Lose": "lower",
    "Shot_90": "higher",
    "SoT_per": "higher",
    "GC_90": "higher",
    "GtS_per": "lower",
    "Goal_90": "higher",
    "xG_90": "higher",
    "TackleWon_per": "higher",
    "ChalWon_per": "higher",
    "SB_90": "higher",
    "Err_90": "lower",
    "Int_90": "higher",
    "Clr_90": "higher",
    "Possession": "higher",
    "PassComp_per": "higher",
    "ShortPass_per": "higher",
    "MediumPass_90": "higher",
    "LongPass_90": "higher",
    "TakeOn_90": "higher",
    "Assist_90": "higher",
    "xAG_90": "higher",
    "KeeperPass_90": "higher",
    "Save_per": "higher",
    "CS_per": "higher"
}


all_metrics = sum(METRIC_GROUPS.values(), [])

# =========================
# Team selection
# =========================
col1, col2 = st.columns(2)
with col1:
    team_a = st.selectbox("Select Team A", df_cluster2["Team"].unique(), index=0)
with col2:
    team_b = st.selectbox("Select Team B", df_cluster2["Team"].unique(), index=1)

# =========================
# Min-Max scaling
# =========================

scaled_df = pd.DataFrame(index=df_cluster2.index)

for col in all_metrics:
    col_min = df_cluster2[col].min()
    col_max = df_cluster2[col].max()
    if col_max - col_min == 0:
        # Zero-variance safeguard
        scaled_df[col] = 0.5
    else:
        scaled_df[col] = (df_cluster2[col] - col_min) / (col_max - col_min)

# =========================
# Invert "lower is better" metrics
# =========================
for metric, direction in METRIC_DIRECTION.items():
    if direction == "lower" and metric in scaled_df.columns:
        scaled_df[metric] = 1 - scaled_df[metric]

# 2) Prevent misleading hard 0 / 1 (visual collapse)
EPS = 0.05
scaled_df[all_metrics] = scaled_df[all_metrics].clip(
    lower=EPS,
    upper=1 - EPS
)

team_a_row = df_cluster2[df_cluster2["Team"] == team_a].iloc[0]
team_b_row = df_cluster2[df_cluster2["Team"] == team_b].iloc[0]

team_a_scaled = scaled_df.loc[df_cluster2["Team"] == team_a].iloc[0]
team_b_scaled = scaled_df.loc[df_cluster2["Team"] == team_b].iloc[0]
# =========================
# League average
# =========================
league_avg_real = df_cluster2[all_metrics].mean()
league_avg_scaled = scaled_df.mean()

# =========================
# Function to create radar
# =========================
def plot_radar(title, metrics):
    metrics_closed = metrics + [metrics[0]]

    a_scaled = [team_a_scaled[m] for m in metrics] + [team_a_scaled[metrics[0]]]
    b_scaled = [team_b_scaled[m] for m in metrics] + [team_b_scaled[metrics[0]]]
    avg_scaled = [league_avg_scaled[m] for m in metrics] + [league_avg_scaled[metrics[0]]]

    a_real = [team_a_row[m] for m in metrics] + [team_a_row[metrics[0]]]
    b_real = [team_b_row[m] for m in metrics] + [team_b_row[metrics[0]]]

    fig = go.Figure()

    # Team A
    fig.add_trace(go.Scatterpolar(
    r=a_scaled,
    theta=metrics_closed,
    mode="lines+markers+text",
    name=team_a,
    line=dict(color="#1f77b4", width=3),   # Strong blue
    fill="toself",
    fillcolor="rgba(31, 119, 180, 0.25)",
    marker=dict(size=7),
    text=[f"{v:.2f}" for v in a_real],
    textfont=dict(color="white"),
    textposition="top center"
))

# Team B
    fig.add_trace(go.Scatterpolar(
    r=b_scaled,
    theta=metrics_closed,
    mode="lines+markers+text",
    name=team_b,
    line=dict(color="#ff7f0e", width=3),   # Orange
    fill="toself",
    fillcolor="rgba(255, 127, 14, 0.25)",
    marker=dict(size=7),
    text=[f"{v:.2f}" for v in b_real],
    textfont=dict(color="white"),
    textposition="top center"
))

# League average
    fig.add_trace(go.Scatterpolar(
    r=avg_scaled,
    theta=metrics_closed,
    mode="lines",
    name="League Average",
    line=dict(
        color="rgba(0,0,0,0.9)",
        dash="dash",
        width=2
    )
))

    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# Draw 5 radar charts
# =========================
for group_name, metrics in METRIC_GROUPS.items():
    st.subheader(group_name)
    plot_radar(group_name, metrics)