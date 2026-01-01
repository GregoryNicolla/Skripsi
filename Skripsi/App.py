import streamlit as st

# Page Setup
home_page = st.Page(
    page="homepage/home.py",
    title="Home",
    icon="🏠",
    default=True,
)

matchpredict_page = st.Page(
    page="homepage/matchpredict.py",
    title="Match Prediction",
    icon="🤖"
)

cluster1_pca_page = st.Page(
    page="clusterpage/cluster1_gmm.py",
    title="GMM Relegation Candidates Analysis",
    icon="📊",
)

cluster2_pca_page = st.Page(
    page="clusterpage/cluster2_gmm.py",
    title="GMM Top Performing Teams Analysis",
    icon="📈",
)

cluster3_pca_page = st.Page(
    page="clusterpage/cluster3_gmm.py",
    title="GMM Mid-Table Teams Analysis",
    icon="📉",
)

cluster1_hier_page = st.Page(
    page="clusterpage/cluster1_hier.py",
    title="Hierarchical Top Performing Teams Analysis",
    icon="📊",
)

cluster2_hier_page = st.Page(
    page="clusterpage/cluster2_hier.py",
    title="Hierarchical Relegation Candidates Analysis",
    icon="📈",
)

cluster3_hier_page = st.Page(
    page="clusterpage/cluster3_hier.py",
    title="Hierarchical Mid-Table Teams Analysis",
    icon="📉"
)

cluster1_kmeans_page = st.Page(
    page="clusterpage/cluster1_kmeans.py",
    title="K-Means Top Performing Teams Analysis",
    icon="📊",
)

cluster2_kmeans_page = st.Page(
    page="clusterpage/cluster2_kmeans.py",
    title="K-Means Relegation Candidates Analysis",
    icon="📈",
)

cluster3_kmeans_page = st.Page(
    page="clusterpage/cluster3_kmeans.py",
    title="K-Means Mid-Table Teams Analysis",
    icon="📉"
)

epl_page = st.Page(
    page="leaguepage/epl.py",
    title="EPL Data",
    icon="⚽",
)

laliga_page = st.Page(
    page="leaguepage/laliga.py",
    title="La Liga Data",
    icon="🏆",
)

seriea_page = st.Page(
    page="leaguepage/seriea.py",
    title="Serie A Data",
    icon="🥅",
)

bundesliga_page = st.Page(
    page="leaguepage/bundesliga.py",
    title="Bundesliga Data",
    icon="🎯",
)

ligue1_page = st.Page(
    page="leaguepage/ligue1.py",
    title="Ligue 1 Data",
    icon="🥖",
)


# Navigation Setup
pg = st.navigation(
    {
        "Info": [home_page , matchpredict_page],
        "GMM PCA": [cluster2_pca_page, cluster3_pca_page, cluster1_pca_page, ],
        "Hierarchical": [cluster1_hier_page,  cluster3_hier_page, cluster2_hier_page],
        "K-Means": [cluster1_kmeans_page, cluster3_kmeans_page, cluster2_kmeans_page, ],
        "League Data": [epl_page, laliga_page, seriea_page, bundesliga_page, ligue1_page]
    }
)

# Run Navigation
pg.run()