import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# page setup
st.set_page_config(
    page_title="Hot Hand Analysis",
    page_icon="📊",
    layout="wide"
)

# sidebar nav
page = st.sidebar.radio(
    "Navigate",
    ["Landing Page", "Visualization"]
)

@st.cache_data
def load_data():
    df = pd.read_csv("pbp/pbp2006.csv")

    # keep only shots
    df = df[df["type"].isin(["Made Shot", "Missed Shot"])].copy()

    # keep needed columns
    df = df[[
        "gameid", "playerid", "player", "team",
        "period", "clock", "result", "type",
        "season", "desc"
    ]].copy()

    # binary outcome
    df["made"] = (df["result"] == "Made").astype(int)

    # points
    df["points"] = 0
    df.loc[df["result"] == "Made", "points"] = 2
    df.loc[
        (df["result"] == "Made") &
        (
            df["desc"].str.contains("3PT", case=False, na=False) |
            df["desc"].str.contains("3-PT", case=False, na=False)
        ),
        "points"
    ] = 3

    # convert clock to seconds
    clock_parts = df["clock"].str.extract(r"PT(\d+)M([\d\.]+)S")
    df["clock_sec"] = clock_parts[0].astype(float) * 60 + clock_parts[1].astype(float)

    # sort in shot order
    df = df.sort_values(
        ["gameid", "player", "period", "clock_sec"],
        ascending=[True, True, True, False]
    ).reset_index(drop=True)

    # shot number within each player-game
    df["shot_num"] = df.groupby(["gameid", "player"]).cumcount() + 1

    # cumulative fg%
    df["fg_pct"] = (
        df.groupby(["gameid", "player"])["made"]
        .expanding()
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    # lag for AR(1)
    df["fg_pct_lag1"] = df.groupby(["gameid", "player"])["fg_pct"].shift(1)

    # game metadata
    df["game_date"] = df["gameid"].astype(str).str[:8]

    game_teams = df.groupby("gameid")["team"].unique().reset_index()
    game_teams["matchup"] = game_teams["team"].apply(
        lambda x: " vs ".join(sorted([str(team) for team in x if pd.notna(team)]))
    )
    df = df.merge(game_teams[["gameid", "matchup"]], on="gameid", how="left")

    # precompute which player-games can fit logistic AR
    # need enough rows after 3 lags, and both classes present
    eligibility = (
        df.groupby(["gameid", "player", "playerid"], as_index=False)
        .agg(
            n_shots=("made", "size"),
            made_sum=("made", "sum"),
            matchup=("matchup", "first"),
            game_date=("game_date", "first"),
            points=("points", "sum")
        )
    )

    eligibility["can_fit_logistic_ar"] = (
        (eligibility["n_shots"] >= 8) &
        (eligibility["made_sum"] > 0) &
        (eligibility["made_sum"] < eligibility["n_shots"])
    )

    valid_pairs = eligibility[eligibility["can_fit_logistic_ar"]].copy()
    valid_games = sorted(valid_pairs["gameid"].unique())

    return df, valid_pairs, valid_games


def fit_logistic_ar(one_game):
    model_df = one_game.copy()

    model_df["lag1"] = model_df["made"].shift(1)
    model_df["lag2"] = model_df["made"].shift(2)
    model_df["lag3"] = model_df["made"].shift(3)

    model_df = model_df.dropna(subset=["lag1", "lag2", "lag3"]).copy()

    X = model_df[["lag1", "lag2", "lag3"]]
    X = sm.add_constant(X)
    y = model_df["made"]

    model = sm.Logit(y, X).fit(disp=0)
    model_df["pred_prob"] = model.predict(X)

    return model_df


def fit_ar1_prediction(one_game):
    one_game = one_game.dropna(subset=["fg_pct_lag1", "fg_pct"]).copy()

    train_size = int(0.8 * len(one_game))
    train_data = one_game.iloc[:train_size].copy()
    test_data = one_game.iloc[train_size:].copy()

    X_train = np.array(train_data["fg_pct_lag1"]).reshape(-1, 1)
    y_train = np.array(train_data["fg_pct"]).reshape(-1, 1)

    w = np.dot(
        np.linalg.inv(np.dot(X_train.T, X_train)),
        np.dot(X_train.T, y_train)
    )

    y_pred = []
    start = np.array(train_data["fg_pct_lag1"].iloc[-1]).reshape(-1, 1)

    for _ in range(len(test_data)):
        next_pred = w.T.dot(start).flatten()[0]
        y_pred.append(next_pred)
        start = np.array(next_pred).reshape(-1, 1)

    pred_df = pd.DataFrame({
        "shot_num": test_data["shot_num"].values,
        "fg_pct_actual": test_data["fg_pct"].values,
        "fg_pct_pred": y_pred
    })

    return pred_df


df, valid_pairs, valid_games = load_data()

# landing page
if page == "Landing Page":
    st.title("Hot Hand or Randomness?")

    st.write("""

This app is a small, interactive sample of our full project analyzing the “hot hand” in basketball. The complete study uses NBA play-by-play data and applies time series and Bayesian methods to test whether shooting outcomes are dependent over time or largely random.

Here, we showcase two core components of our methodology:
- **Logistic Autoregressive Model:** estimates the probability of making a shot based on recent shot history.
- **AR(1) Model:** predicts shooting performance trends using prior values.

Users can explore selected games and players to visualize how these models behave on real shot sequences and gain intuition for our broader analysis.
 """)

# visualization page
elif page == "Visualization":
    st.title("Time Series Visualizations")

    st.write("""
    Only games with at least one player eligible for the logistic AR model are shown.
    After selecting a game, choose from the eligible players in that game.
    """)

    # only valid games appear
    game = st.selectbox("Select Game", valid_games)

    # only valid players in selected game appear
    valid_players_in_game = (
        valid_pairs[valid_pairs["gameid"] == game]
        .sort_values(["points", "n_shots"], ascending=[False, False])
    )

    player_options = valid_players_in_game["player"].tolist()
    player = st.selectbox("Select Player", player_options)

    filtered = (
        df[(df["gameid"] == game) & (df["player"] == player)]
        .sort_values(["period", "clock_sec"], ascending=[True, False])
        .copy()
    )

    game_info = filtered.iloc[0]
    player_info = valid_players_in_game[valid_players_in_game["player"] == player].iloc[0]

    st.subheader(f"{player}")
    st.caption(f"Matchup: {game_info['matchup']}")
    st.caption(f"Game ID: {game}")
    st.caption(f"Date: {game_info['game_date']}")
    st.caption(f"Points in game: {int(player_info['points'])}")
    st.caption(f"Shot attempts: {int(player_info['n_shots'])}")

    # logistic AR plot
    st.subheader("Logistic AR Model")
    st.caption("Predicted make probability based on the previous three shot outcomes.")

    logistic_df = fit_logistic_ar(filtered)

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(logistic_df["shot_num"].values, logistic_df["pred_prob"].values, label="Predicted Probability")
    ax1.scatter(logistic_df["shot_num"].values, logistic_df["made"].values, alpha=0.4, label="Actual Outcome")
    ax1.set_xlabel("Shot Number")
    ax1.set_ylabel("Probability / Outcome")
    ax1.set_title("Logistic AR Prediction")
    ax1.set_ylim(0, 1)
    ax1.legend()
    st.pyplot(fig1)

    # AR(1) plot
    st.subheader("AR(1) Prediction")
    st.caption("Predicted cumulative FG% on the test portion of the shot sequence.")

    ar1_df = fit_ar1_prediction(filtered)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(ar1_df["shot_num"].values, ar1_df["fg_pct_actual"].values, label="Actual FG%")
    ax2.plot(ar1_df["shot_num"].values, ar1_df["fg_pct_pred"].values, linestyle="--", label="Predicted FG%")
    ax2.set_xlabel("Shot Number")
    ax2.set_ylabel("FG%")
    ax2.set_title("AR(1) Model Prediction")
    ax2.set_ylim(0, 1)
    ax2.legend()
    st.pyplot(fig2)

    # table
    st.dataframe(
        filtered[["shot_num", "period", "clock", "result", "points", "fg_pct"]]
        .reset_index(drop=True)
    )