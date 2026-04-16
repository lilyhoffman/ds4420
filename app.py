import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Hot Hand Analysis", page_icon="🏀", layout="wide")

page = st.sidebar.radio("Navigate", ["Landing Page", "Interactive Visualization"])

@st.cache_data
def load_data():
    df = pd.read_csv("pbp/pbp2006.csv").copy()
    df = df[df["type"].isin(["Made Shot", "Missed Shot"])].copy()

    for col in ["player", "type", "result", "clock", "desc", "team"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    keep_cols = [
        "gameid", "playerid", "player", "team",
        "period", "clock", "result", "type",
        "season", "desc", "dist", "subtype"
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    df["made"] = (df["result"] == "Made").astype(int)

    if "clock" in df.columns:
        clock_parts = df["clock"].str.extract(r"PT(\d+)M([\d\.]+)S")
        df["clock_sec"] = (
            pd.to_numeric(clock_parts[0], errors="coerce") * 60
            + pd.to_numeric(clock_parts[1], errors="coerce")
        )
    else:
        df["clock_sec"] = np.nan

    sort_cols = [c for c in ["gameid", "player", "period", "clock_sec"] if c in df.columns]
    if sort_cols:
        ascending = [True, True, True, False][:len(sort_cols)]
        df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    df["shot_num"] = df.groupby(["gameid", "player"]).cumcount() + 1
    df["game_date"] = df["gameid"].astype(str).str[:8]

    if "team" in df.columns:
        game_teams = df.groupby("gameid")["team"].unique().reset_index()
        game_teams["matchup"] = game_teams["team"].apply(
            lambda x: " vs ".join(sorted([str(team) for team in x if pd.notna(team)]))
        )
        df = df.merge(game_teams[["gameid", "matchup"]], on="gameid", how="left")
    else:
        df["matchup"] = ""

    eligibility = (
        df.groupby(["gameid", "player", "playerid"], as_index=False)
        .agg(
            n_shots=("made", "size"),
            made_sum=("made", "sum"),
            matchup=("matchup", "first"),
            game_date=("game_date", "first")
        )
    )

    eligibility["can_fit"] = (
        (eligibility["n_shots"] >= 8)
        & (eligibility["made_sum"] > 0)
        & (eligibility["made_sum"] < eligibility["n_shots"])
    )

    valid_pairs = eligibility[eligibility["can_fit"]].copy()
    valid_games = sorted(valid_pairs["gameid"].unique())

    return df, valid_pairs, valid_games


def build_make_streak(values):
    streak = []
    count = 0
    for x in values:
        streak.append(count)
        count = count + 1 if x == 1 else 0
    return streak


def build_model_df(one_game):
    model_df = one_game.copy()
    model_df["lag_1"] = model_df["made"].shift(1)
    model_df["lag_2"] = model_df["made"].shift(2)
    model_df["lag_3"] = model_df["made"].shift(3)
    model_df["hit_rate_5"] = model_df["made"].shift(1).rolling(5, min_periods=5).mean()
    model_df["make_streak"] = build_make_streak(model_df["made"].tolist())
    model_df = model_df.dropna(subset=["lag_1", "lag_2", "lag_3", "hit_rate_5"]).copy()
    return model_df


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def train_logistic_regression(X, y, lr=0.05, epochs=2500):
    n, d = X.shape
    w = np.zeros((d, 1))
    for _ in range(epochs):
        p_hat = sigmoid(X @ w)
        gradient = (X.T @ (p_hat - y)) / n
        w -= lr * gradient
    return w


def run_model(one_game):
    model_df = build_model_df(one_game)
    if len(model_df) < 8:
        return None

    feature_cols = ["lag_1", "lag_2", "lag_3", "hit_rate_5", "make_streak"]

    for col in ["hit_rate_5", "make_streak"]:
        mean = model_df[col].mean()
        std = model_df[col].std()
        if pd.isna(std) or std == 0:
            std = 1.0
        model_df[col] = (model_df[col] - mean) / std

    X = model_df[feature_cols].values.astype(float)
    y = model_df["made"].values.reshape(-1, 1).astype(float)

    X = np.hstack([np.ones((X.shape[0], 1)), X])
    w = train_logistic_regression(X, y)

    pred_prob = sigmoid(X @ w).flatten()
    model_df["pred_prob"] = pred_prob

    return model_df


df, valid_pairs, valid_games = load_data()

if page == "Landing Page":
    st.title("Hot Hand or Illusion? A Data-Driven NBA Analysis")

    st.write(
        """
        This app presents a small interactive component of our project exploring the
        “hot hand” in basketball. The hot hand refers to the idea that a player is more
        likely to make a shot after a streak of successful attempts. While this belief
        is common among players, coaches, and fans, it is still debated whether it reflects
        a real pattern in performance or simply random variation.
        """
    )

    st.write(
        """
        In our full project, we analyze NBA play-by-play data to study shot outcomes
        over time. We treat each shot as a binary event, either made or missed, and
        examine whether recent performance has a meaningful effect on future shots.
        To do this, we apply a handwritten logistic autoregressive model as well as
        a Bayesian logistic regression model.
        """
    )

    st.write(
        """
        The logistic autoregressive model uses recent shot history to estimate the
        probability of making the next shot. This allows us to capture short-term
        dependence and see whether streaks actually change predicted performance.
        The Bayesian model complements this by providing a probabilistic view of
        uncertainty in these effects.
        """
    )

    st.write(
        """
        This app focuses on one part of that analysis. You can select a specific
        game and player to visualize how predicted shot probabilities evolve over
        the course of a game. The goal is to build intuition for how the model
        responds to sequences of made and missed shots, and whether strong streak
        patterns appear in the predictions.
        """
    )

    st.write(
        """
        Overall, this interactive view reflects our broader finding that while there
        may be small changes in probability after recent success, these effects are
        generally weak and not strong enough to suggest a consistent hot hand.
        """
    )

elif page == "Interactive Visualization":
    st.title("Shot Sequence")

    game = st.selectbox("Select Game", valid_games)

    valid_players = (
        valid_pairs[valid_pairs["gameid"] == game]
        .sort_values(["n_shots"], ascending=False)
    )

    player = st.selectbox("Select Player", valid_players["player"].tolist())

    filtered = (
        df[(df["gameid"] == game) & (df["player"] == player)]
        .sort_values(["period", "clock_sec"], ascending=[True, False])
        .copy()
    )

    result_df = run_model(filtered)

    if result_df is None:
        st.warning("Not enough shots for this selection.")
    else:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=result_df["shot_num"],
            y=result_df["pred_prob"],
            mode="lines+markers",
            name="Predicted P(make)"
        ))

        fig.add_trace(go.Scatter(
            x=result_df["shot_num"],
            y=result_df["made"],
            mode="markers",
            name="Actual"
        ))

        fig.update_layout(
            title="Predicted Probability vs Actual",
            xaxis_title="Shot Number",
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1])
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Game Info & Sample Data")

    st.write(
        """
        Explore a sample of the shot data for the selected player and game,
        along with key game details and context.
        """
    )

    # game info
    game_info = filtered.iloc[0]

    st.markdown(
        f"""
        **Matchup:** {game_info.get('matchup', 'N/A')}  
        **Game ID:** {game}  
        **Date:** {game_info.get('game_date', 'N/A')}  
        **Player:** {player}  
        **Total Shots:** {len(filtered)}
        """
    )

    # number of rows to display
    n_rows = st.slider("Number of rows to display", 5, 50, 15)

    # choose useful columns if they exist
    possible_cols = [
        "shot_num", "period", "clock", "result", "made",
        "team", "player", "dist", "subtype"
    ]
    show_cols = [c for c in possible_cols if c in filtered.columns]

    st.dataframe(
        filtered[show_cols]
        .head(n_rows)
        .reset_index(drop=True)
    )