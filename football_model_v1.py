#!/usr/bin/env python3
"""
Football Betting Model V1
Leagues: Ligue 1 (F1) + Premier League (E0)

Purpose:
- Build pre-match probabilities from information available BEFORE kickoff.
- Compare model probabilities with market odds.
- Flag potential positive-EV bets.
- Keep a paper-trading ledger for later validation.

IMPORTANT:
This is a research/backtesting tool, not a guarantee of profit.
Do not use a model output as a "sure bet".

Data source:
Football-Data.co.uk historical CSV files.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import log_loss, accuracy_score
except ImportError:
    raise SystemExit("Install dependencies first: pip install pandas numpy scikit-learn requests")

import requests


BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
LEAGUES = {"F1": "Ligue 1", "E0": "Premier League"}


def season_code(start_year: int) -> str:
    # 2025-26 -> 2526
    return f"{str(start_year)[-2:]}{str(start_year + 1)[-2:]}"


def download_data(seasons: List[int], cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []

    for year in seasons:
        code = season_code(year)
        for league, league_name in LEAGUES.items():
            cache = cache_dir / f"{code}_{league}.csv"
            if not cache.exists():
                url = BASE_URL.format(season=code, league=league)
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                cache.write_bytes(r.content)

            df = pd.read_csv(cache)
            df["LeagueCode"] = league
            df["League"] = league_name
            df["SeasonStart"] = year
            frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["Date"] = pd.to_datetime(out["Date"], dayfirst=True, errors="coerce")
    out = out.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    out = out.sort_values(["Date"]).reset_index(drop=True)
    return out


@dataclass
class TeamState:
    elo: float = 1500.0
    home_games: int = 0
    away_games: int = 0
    matches: int = 0
    gf: List[float] = None
    ga: List[float] = None
    sot_for: List[float] = None
    sot_against: List[float] = None

    def __post_init__(self):
        self.gf = [] if self.gf is None else self.gf
        self.ga = [] if self.ga is None else self.ga
        self.sot_for = [] if self.sot_for is None else self.sot_for
        self.sot_against = [] if self.sot_against is None else self.sot_against


def mean_last(values, n=8, default=1.0):
    if not values:
        return default
    return float(np.mean(values[-n:]))


def safe_col(row, col, default=0.0):
    x = row.get(col, default)
    return default if pd.isna(x) else float(x)


def make_features(home: TeamState, away: TeamState) -> Dict[str, float]:
    # Features are strictly based on matches already played.
    return {
        "elo_diff": home.elo - away.elo,
        "home_gf": mean_last(home.gf, 8, 1.2),
        "home_ga": mean_last(home.ga, 8, 1.2),
        "away_gf": mean_last(away.gf, 8, 1.0),
        "away_ga": mean_last(away.ga, 8, 1.3),
        "home_sot_for": mean_last(home.sot_for, 8, 4.5),
        "home_sot_against": mean_last(home.sot_against, 8, 4.5),
        "away_sot_for": mean_last(away.sot_for, 8, 4.2),
        "away_sot_against": mean_last(away.sot_against, 8, 4.7),
        "home_experience": min(home.matches, 20),
        "away_experience": min(away.matches, 20),
    }


def update_elo(home: TeamState, away: TeamState, result: int, k=22.0):
    expected = 1 / (1 + 10 ** (-(home.elo - away.elo) / 400))
    actual = 1.0 if result == 0 else 0.0 if result == 2 else 0.5
    delta = k * (actual - expected)
    home.elo += delta
    away.elo -= delta


def update_state(home: TeamState, away: TeamState, row):
    hg, ag = float(row.FTHG), float(row.FTAG)

    # Football-Data uses HST/AST for shots on target in these leagues.
    hst = safe_col(row, "HST", 4.5)
    ast = safe_col(row, "AST", 4.2)

    home.gf.append(hg); home.ga.append(ag)
    away.gf.append(ag); away.ga.append(hg)
    home.sot_for.append(hst); home.sot_against.append(ast)
    away.sot_for.append(ast); away.sot_against.append(hst)

    home.matches += 1
    away.matches += 1
    home.home_games += 1
    away.away_games += 1

    result = 0 if hg > ag else 2 if hg < ag else 1
    update_elo(home, away, result)


def build_pre_match_dataset(df: pd.DataFrame):
    states: Dict[Tuple[str, str], TeamState] = {}
    rows = []

    for _, row in df.iterrows():
        key_h = (row["LeagueCode"], row["HomeTeam"])
        key_a = (row["LeagueCode"], row["AwayTeam"])
        states.setdefault(key_h, TeamState())
        states.setdefault(key_a, TeamState())

        home = states[key_h]
        away = states[key_a]

        # Season reset: retain some Elo, reset short-term form.
        # A separate key by season is used here, so no form leakage across seasons.
        if home.matches == 0 and away.matches == 0:
            pass

        feat = make_features(home, away)
        feat.update({
            "Date": row["Date"],
            "LeagueCode": row["LeagueCode"],
            "League": row["League"],
            "SeasonStart": row["SeasonStart"],
            "HomeTeam": row["HomeTeam"],
            "AwayTeam": row["AwayTeam"],
            "FTHG": row["FTHG"],
            "FTAG": row["FTAG"],
            "Result": 0 if row["FTHG"] > row["FTAG"] else 2 if row["FTHG"] < row["FTAG"] else 1,
        })

        # Market probabilities are NOT used as model inputs.
        # They are only used later to calculate value.
        for col in ["B365H", "B365D", "B365A", "AvgH", "AvgD", "AvgA"]:
            feat[col] = row[col] if col in row.index else np.nan

        rows.append(feat)
        update_state(home, away, row)

    return pd.DataFrame(rows)


FEATURES = [
    "elo_diff",
    "home_gf", "home_ga",
    "away_gf", "away_ga",
    "home_sot_for", "home_sot_against",
    "away_sot_for", "away_sot_against",
    "home_experience", "away_experience",
]


def train_and_test(dataset: pd.DataFrame, test_season: int):
    train = dataset[dataset["SeasonStart"] < test_season].copy()
    test = dataset[dataset["SeasonStart"] == test_season].copy()

    # Remove very early matches where there is essentially no information.
    train = train[(train["home_experience"] >= 3) & (train["away_experience"] >= 3)]
    test = test[(test["home_experience"] >= 3) & (test["away_experience"] >= 3)]

    X_train = train[FEATURES].fillna(train[FEATURES].median())
    X_test = test[FEATURES].fillna(X_train.median())

    model = Pipeline([
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(
            max_iter=2000,
            C=0.5,
            solver="lbfgs",
            random_state=42
        ))
    ])

    model.fit(X_train, train["Result"])
    p = model.predict_proba(X_test)

    # sklearn orders classes numerically: 0=home, 1=draw, 2=away
    out = test.copy()
    for i, c in enumerate(model.named_steps["logit"].classes_):
        out[f"p_{c}"] = p[:, i]

    print("\n=== OUT-OF-SAMPLE TEST ===")
    print(f"Test season: {test_season}-{str(test_season+1)[-2:]}")
    print(f"Matches: {len(out)}")
    print(f"Accuracy: {accuracy_score(out['Result'], model.predict(X_test)):.3f}")
    print(f"Log loss: {log_loss(out['Result'], p):.3f}")

    return model, out


def market_probability(odds: float) -> float:
    return 1.0 / odds if odds and odds > 1 else np.nan


def add_value(out: pd.DataFrame, odds_source="Avg"):
    cols = {"H": f"{odds_source}H", "D": f"{odds_source}D", "A": f"{odds_source}A"}
    probs = {"H": "p_0", "D": "p_1", "A": "p_2"}

    # Remove bookmaker margin before comparing probabilities.
    raw = []
    for _, r in out.iterrows():
        vals = {k: market_probability(r.get(v, np.nan)) for k, v in cols.items()}
        if all(pd.notna(x) for x in vals.values()):
            s = sum(vals.values())
            vals = {k: v / s for k, v in vals.items()}
        else:
            vals = {k: np.nan for k in vals}
        raw.append(vals)

    mp = pd.DataFrame(raw, index=out.index)
    out["market_p_home"] = mp["H"]
    out["market_p_draw"] = mp["D"]
    out["market_p_away"] = mp["A"]

    for side, prob_col in probs.items():
        out[f"ev_{side.lower()}"] = out[prob_col] * out[cols[side]] - 1

    out["best_side"] = out[["ev_h", "ev_d", "ev_a"]].idxmax(axis=1).str[-1].str.upper()
    out["best_ev"] = out[["ev_h", "ev_d", "ev_a"]].max(axis=1)
    return out


def fractional_kelly(prob, odds, fraction=0.25, cap=0.02):
    if not (0 < prob < 1 and odds > 1):
        return 0.0
    b = odds - 1
    q = 1 - prob
    full = (b * prob - q) / b
    return max(0.0, min(cap, fraction * full))


def generate_picks(out: pd.DataFrame, min_ev=0.05, bankroll=500.0):
    picks = []
    for _, r in out.iterrows():
        side = r["best_side"]
        if side == "H":
            prob, odds = r["p_0"], r.get("AvgH", np.nan)
        elif side == "D":
            prob, odds = r["p_1"], r.get("AvgD", np.nan)
        else:
            prob, odds = r["p_2"], r.get("AvgA", np.nan)

        if pd.isna(odds) or pd.isna(prob) or r["best_ev"] < min_ev:
            continue

        stake_pct = fractional_kelly(prob, odds)
        picks.append({
            "Date": r["Date"],
            "League": r["League"],
            "Match": f"{r['HomeTeam']} - {r['AwayTeam']}",
            "Side": side,
            "ModelProb": round(float(prob), 4),
            "MarketOdds": round(float(odds), 3),
            "EV": round(float(r["best_ev"]), 4),
            "StakePct": round(stake_pct, 4),
            "Stake€": round(stake_pct * bankroll, 2),
            "Result": r["Result"],
        })

    return pd.DataFrame(picks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-season", type=int, default=2018,
                        help="First season start year, e.g. 2018 = 2018/19")
    parser.add_argument("--to-season", type=int, default=2025,
                        help="Last season start year, e.g. 2025 = 2025/26")
    parser.add_argument("--test-season", type=int, default=2025)
    parser.add_argument("--bankroll", type=float, default=500.0)
    parser.add_argument("--min-ev", type=float, default=0.05)
    args = parser.parse_args()

    cache = Path("data_cache")
    seasons = list(range(args.from_season, args.to_season + 1))
    raw = download_data(seasons, cache)
    dataset = build_pre_match_dataset(raw)

    model, test = train_and_test(dataset, args.test_season)
    test = add_value(test, odds_source="Avg")

    picks = generate_picks(test, min_ev=args.min_ev, bankroll=args.bankroll)

    Path("outputs").mkdir(exist_ok=True)
    test.to_csv("outputs/test_predictions.csv", index=False)
    picks.to_csv("outputs/paper_picks.csv", index=False)

    print("\n=== POTENTIAL VALUE PICKS (PAPER ONLY) ===")
    if picks.empty:
        print("No picks passed the EV threshold.")
    else:
        print(picks.to_string(index=False))

    print("\nFiles written:")
    print("- outputs/test_predictions.csv")
    print("- outputs/paper_picks.csv")


if __name__ == "__main__":
    main()
