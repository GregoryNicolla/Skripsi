import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import statsmodels.api as sm
from scipy.stats import poisson

# =========================
# PAGE SETUP
# =========================
st.set_page_config(layout="wide")
st.title("Football Match Prediction (Poisson + Random Forest)")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data(path: str):
    return pd.read_excel(path)

df_match = load_data("Data/Match Predict/matchpred.xlsx")

st.subheader("Dataset Overview")
st.dataframe(df_match, use_container_width=True)

# League → Teams mapping
league_teams = (
    df_match[["League", "Team"]]
    .drop_duplicates()
    .sort_values(["League", "Team"])
)

# =========================
# CONSTANTS
# =========================
FEATURES = [
    "Shot_90", "SoT_per", "GC_90", "GtS_per", "Goal_90", "xG_90",
    "TackleWon_per","ChalWon_per","SB_90","Err_90","Int_90","Clr_90",
    "Possession","PassComp_per","ShortPass_per","MediumPass_90",
    "LongPass_90","TakeOn_90","Assist_90","xAG_90",
    "KeeperPass_90","Save_per","CS_per"
]

HOME_ADV = {
    "Premier League": 1.15,
    "Serie A": 1.13,
    "La Liga": 1.12,
    "Bundesliga": 1.10,
    "Ligue 1": 1.11
}

# =========================
# FEATURE ENGINEERING
# =========================
def create_match(home, away):
    row = {}
    for f in FEATURES:
        row[f"diff_{f}"] = home[f] - away[f]

    row["home_advantage"] = HOME_ADV.get(home["League"], 1.12)
    return row

# =========================
# MODEL TRAINING
# =========================
@st.cache_resource
def train_models(df):
    np.random.seed(42)
    teams = df.sample(40)

    matches, home_goals, away_goals, results = [], [], [], []

    for _ in range(600):
        home, away = teams.sample(2).to_dict("records")
        row = create_match(home, away)
        matches.append(row)

        adv = HOME_ADV.get(home["League"], 1.12)
        hg = np.random.poisson(max(home["xG_90"] * adv, 0.2))
        ag = np.random.poisson(max(away["xG_90"], 0.2))

        home_goals.append(hg)
        away_goals.append(ag)

        results.append(2 if hg > ag else 0 if hg < ag else 1)

    match_df = pd.DataFrame(matches)
    match_df["home_goals"] = home_goals
    match_df["away_goals"] = away_goals
    match_df["result"] = results

    # ---- POISSON ----
    X_base = match_df.drop(["home_goals", "away_goals", "result"], axis=1)
    X_poisson = sm.add_constant(X_base, has_constant="add")
    POISSON_COLS = X_poisson.columns.tolist()

    poisson_home = sm.GLM(
        match_df["home_goals"], X_poisson, family=sm.families.Poisson()
    ).fit()

    poisson_away = sm.GLM(
        match_df["away_goals"], X_poisson, family=sm.families.Poisson()
    ).fit()

    match_df["exp_home_goals"] = poisson_home.predict(X_poisson)
    match_df["exp_away_goals"] = poisson_away.predict(X_poisson)

    # ---- RANDOM FOREST ----
    X = match_df.drop(["home_goals", "away_goals", "result"], axis=1)
    y = match_df["result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestClassifier(
        max_features= "log2", max_depth = 12, min_samples_leaf = 1, min_samples_split = 5, n_estimators = 300, random_state = 42
    )
    rf.fit(X_train, y_train)

    acc = accuracy_score(y_test, rf.predict(X_test))
    return poisson_home, poisson_away, rf, POISSON_COLS, acc

poisson_home, poisson_away, rf, POISSON_COLS, acc = train_models(df_match)
st.metric("Model Accuracy (RF)", f"{acc:.2%}")

# =========================
# PREDICTION FUNCTIONS
# =========================
def predict_match(home_team, away_team):
    home = df_match[df_match["Team"] == home_team].iloc[0]
    away = df_match[df_match["Team"] == away_team].iloc[0]

    row = create_match(home, away)
    X_new = pd.DataFrame([row])

    Xp = sm.add_constant(X_new, has_constant="add")[POISSON_COLS]
    exp_home = poisson_home.predict(Xp).iloc[0]
    exp_away = poisson_away.predict(Xp).iloc[0]

    X_new["exp_home_goals"] = exp_home
    X_new["exp_away_goals"] = exp_away

    rf_result = rf.predict(X_new)[0]
    outcome_map = {2: "Home Win", 1: "Draw", 0: "Away Win"}

    return exp_home, exp_away, outcome_map[rf_result]

def scoreline_table(home_team, away_team, max_goals=5):
    home = df_match[df_match["Team"] == home_team].iloc[0]
    away = df_match[df_match["Team"] == away_team].iloc[0]

    row = create_match(home, away)
    X_new = pd.DataFrame([row])
    Xp = sm.add_constant(X_new, has_constant="add")[POISSON_COLS]

    lam_home = poisson_home.predict(Xp).iloc[0]
    lam_away = poisson_away.predict(Xp).iloc[0]

    scores = []
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            prob = poisson.pmf(h, lam_home) * poisson.pmf(a, lam_away)
            scores.append({"Score": f"{h}-{a}", "Probability": prob})

    return pd.DataFrame(scores).sort_values("Probability", ascending=False).head(5)

def outcome_probabilities(home_team, away_team, max_goals=7):
    home = df_match[df_match["Team"] == home_team].iloc[0]
    away = df_match[df_match["Team"] == away_team].iloc[0]

    row = create_match(home, away)
    X_new = pd.DataFrame([row])
    Xp = sm.add_constant(X_new, has_constant="add")[POISSON_COLS]

    lam_home = poisson_home.predict(Xp).iloc[0]
    lam_away = poisson_away.predict(Xp).iloc[0]

    home_win = draw = away_win = 0.0

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = poisson.pmf(h, lam_home) * poisson.pmf(a, lam_away)

            if h > a:
                home_win += p
            elif h < a:
                away_win += p
            else:
                draw += p

    total = home_win + draw + away_win

    return {
        "Home Win %": round(home_win / total * 100, 1),
        "Draw %": round(draw / total * 100, 1),
        "Away Win %": round(away_win / total * 100, 1),
    }


# =========================
# UI — MATCH SELECTION
# =========================
st.subheader("Match Selection")
col1, col2 = st.columns(2)

with col1:
    home_league = st.selectbox("Home League", league_teams["League"].unique())
    home_team = st.selectbox(
        "Home Team",
        league_teams[league_teams["League"] == home_league]["Team"]
    )

with col2:
    away_league = st.selectbox("Away League", league_teams["League"].unique(), key="away_lg")
    away_team = st.selectbox(
        "Away Team",
        league_teams[
            (league_teams["League"] == away_league) &
            (league_teams["Team"] != home_team)
        ]["Team"],
        key="away_tm"
    )

# =========================
# RUN PREDICTION
# =========================
if st.button("Predict Match", key="predict_match_btn"):
    exp_home, exp_away, outcome = predict_match(home_team, away_team)

    st.session_state["prediction"] = {
        "exp_home": exp_home,
        "exp_away": exp_away,
        "outcome": outcome,
        "home_team": home_team,
        "away_team": away_team
    }
    # =========================
    # Prediction Result
    # =========================

if "prediction" in st.session_state:
    exp_home = st.session_state["prediction"]["exp_home"]
    exp_away = st.session_state["prediction"]["exp_away"]
    outcome = st.session_state["prediction"]["outcome"]
    home_team = st.session_state["prediction"]["home_team"]
    away_team = st.session_state["prediction"]["away_team"]

    # =========================
    # Prediction Result
    # =========================
    st.subheader("Prediction Result (Round of Expected Goals)")

    c1, c2, c3 = st.columns(3)

    c1.metric("Expected Score", f"{round(exp_home)} - {round(exp_away)}")
    c2.metric("Expected Goals", f"{exp_home:.2f} - {exp_away:.2f}")

    home_score = round(exp_home)
    away_score = round(exp_away)

    if home_score > away_score:
        c3.success(f"{home_team} Win")
    elif away_score > home_score:
        c3.success(f"{away_team} Win")
    else:
        c3.info("Draw")

    # =========================
    # Match Outcome Probabilities
    # =========================
    st.subheader("Match Outcome Probabilities")

    outcomes = outcome_probabilities(home_team, away_team)

    col1, col2, col3 = st.columns(3)
    col1.metric(f"🏠 {home_team} Win", f"{outcomes['Home Win %']}%")
    col2.metric("🤝 Draw", f"{outcomes['Draw %']}%")
    col3.metric(f"✈️ {away_team} Win", f"{outcomes['Away Win %']}%")

    # =========================
    # Top Scoreline Probabilities (ONLY ONCE)
    # =========================
    if st.checkbox(
        "Show Top Scoreline Probabilities",
        key="show_scorelines"
    ):
        st.subheader("Top Scoreline Probabilities")

        df_scores = scoreline_table(home_team, away_team).copy()
        df_scores["Probability"] = (
            df_scores["Probability"] * 100
        ).round(2).astype(str) + "%"

        st.dataframe(df_scores, use_container_width=True)


