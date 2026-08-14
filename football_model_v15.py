from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
LEAGUES = {"F1": "Ligue 1", "E0": "Premier League"}

FEATURES = [
    "elo_diff", "home_gf", "home_ga", "away_gf", "away_ga",
    "home_sot_for", "home_sot_against", "away_sot_for",
    "away_sot_against", "home_experience", "away_experience"
]


class TeamState:
    def __init__(self):
        self.elo = 1500.0
        self.matches = 0
        self.gf, self.ga = [], []
        self.sot_for, self.sot_against = [], []


def season_code(y):
    return f"{str(y)[-2:]}{str(y+1)[-2:]}"


def download_data(seasons, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for y in seasons:
        code = season_code(y)
        for league, name in LEAGUES.items():
            path = cache_dir / f"{code}_{league}.csv"
            if not path.exists():
                r = requests.get(BASE_URL.format(season=code, league=league), timeout=30)
                r.raise_for_status()
                path.write_bytes(r.content)
            df = pd.read_csv(path)
            df["LeagueCode"] = league
            df["League"] = name
            df["SeasonStart"] = y
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["Date"] = pd.to_datetime(out["Date"], dayfirst=True, errors="coerce")
    out = out.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    return out.sort_values("Date").reset_index(drop=True)


def mean_last(x, n=8, default=1.0):
    return float(np.mean(x[-n:])) if x else default


def features(h, a):
    return {
        "elo_diff": h.elo - a.elo,
        "home_gf": mean_last(h.gf, 8, 1.2),
        "home_ga": mean_last(h.ga, 8, 1.2),
        "away_gf": mean_last(a.gf, 8, 1.0),
        "away_ga": mean_last(a.ga, 8, 1.3),
        "home_sot_for": mean_last(h.sot_for, 8, 4.5),
        "home_sot_against": mean_last(h.sot_against, 8, 4.5),
        "away_sot_for": mean_last(a.sot_for, 8, 4.2),
        "away_sot_against": mean_last(a.sot_against, 8, 4.7),
        "home_experience": min(h.matches, 20),
        "away_experience": min(a.matches, 20),
    }


def elo_update(h, a, result, k=22):
    exp = 1 / (1 + 10 ** (-(h.elo - a.elo) / 400))
    actual = 1 if result == 0 else 0 if result == 2 else 0.5
    d = k * (actual - exp)
    h.elo += d
    a.elo -= d


def update(h, a, row):
    hg, ag = float(row["FTHG"]), float(row["FTAG"])
    hst = float(row.get("HST", 4.5)) if pd.notna(row.get("HST", np.nan)) else 4.5
    ast = float(row.get("AST", 4.2)) if pd.notna(row.get("AST", np.nan)) else 4.2
    h.gf.append(hg); h.ga.append(ag); a.gf.append(ag); a.ga.append(hg)
    h.sot_for.append(hst); h.sot_against.append(ast)
    a.sot_for.append(ast); a.sot_against.append(hst)
    h.matches += 1; a.matches += 1
    result = 0 if hg > ag else 2 if hg < ag else 1
    elo_update(h, a, result)


def build_pre_match_dataset(df):
    states = {}
    rows = []
    for _, r in df[df["FTHG"].notna() & df["FTAG"].notna()].sort_values("Date").iterrows():
        hk = (r["LeagueCode"], r["SeasonStart"], r["HomeTeam"])
        ak = (r["LeagueCode"], r["SeasonStart"], r["AwayTeam"])
        states.setdefault(hk, TeamState()); states.setdefault(ak, TeamState())
        h, a = states[hk], states[ak]
        f = features(h, a)
        f.update({
            "Date": r["Date"], "LeagueCode": r["LeagueCode"], "League": r["League"],
            "SeasonStart": r["SeasonStart"], "HomeTeam": r["HomeTeam"],
            "AwayTeam": r["AwayTeam"],
            "Result": 0 if r["FTHG"] > r["FTAG"] else 2 if r["FTHG"] < r["FTAG"] else 1
        })
        rows.append(f)
        update(h, a, r)
    return pd.DataFrame(rows)


def train_model_all_history(dataset):
    usable = dataset[(dataset.home_experience >= 3) & (dataset.away_experience >= 3)].copy()
    X = usable[FEATURES].fillna(usable[FEATURES].median())
    y = usable["Result"]
    model = Pipeline([
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(max_iter=2000, C=0.5, solver="lbfgs", random_state=42))
    ])
    model.fit(X, y)
    return model


def current_states_from_completed(current_df):
    states = {}
    completed = current_df[current_df["FTHG"].notna() & current_df["FTAG"].notna()].sort_values("Date")
    for _, r in completed.iterrows():
        hk = (r["LeagueCode"], r["HomeTeam"]); ak = (r["LeagueCode"], r["AwayTeam"])
        states.setdefault(hk, TeamState()); states.setdefault(ak, TeamState())
        update(states[hk], states[ak], r)
    return states


def predict_upcoming(model, hist_dataset, current_df):
    states = current_states_from_completed(current_df)
    rows = []
    upcoming = current_df[current_df["FTHG"].isna() | current_df["FTAG"].isna()].copy()
    for _, r in upcoming.sort_values("Date").iterrows():
        hk = (r["LeagueCode"], r["HomeTeam"]); ak = (r["LeagueCode"], r["AwayTeam"])
        states.setdefault(hk, TeamState()); states.setdefault(ak, TeamState())
        f = features(states[hk], states[ak])
        X = pd.DataFrame([f])[FEATURES].fillna(hist_dataset[FEATURES].median())
        p = model.predict_proba(X)[0]
        classes = model.named_steps["logit"].classes_
        probs = {int(c): float(p[i]) for i, c in enumerate(classes)}

        odds = {
            "H": r.get("AvgH", np.nan),
            "D": r.get("AvgD", np.nan),
            "A": r.get("AvgA", np.nan),
        }
        ev = {}
        for side, c in odds.items():
            idx = {"H":0, "D":1, "A":2}[side]
            prob = probs.get(idx, np.nan)
            ev[side] = prob * c - 1 if pd.notna(c) and c > 1 else np.nan

        valid = {k:v for k,v in ev.items() if pd.notna(v)}
        if valid:
            best_side = max(valid, key=valid.get)
            best_ev = valid[best_side]
        else:
            best_side, best_ev = None, np.nan

        best_prob = probs.get({"H":0,"D":1,"A":2}.get(best_side, -1), np.nan)
        best_odds = odds.get(best_side, np.nan) if best_side else np.nan
        stake_pct = 0.0
        if pd.notna(best_prob) and pd.notna(best_odds) and best_odds > 1:
            b = best_odds - 1
            kelly = (b * best_prob - (1-best_prob)) / b
            stake_pct = max(0, min(0.02, 0.25 * kelly))

        rows.append({
            "Date": r["Date"], "LeagueCode": r["LeagueCode"], "League": r["League"],
            "HomeTeam": r["HomeTeam"], "AwayTeam": r["AwayTeam"],
            "p_home": probs.get(0, np.nan), "p_draw": probs.get(1, np.nan),
            "p_away": probs.get(2, np.nan),
            "best_side": best_side, "best_prob": best_prob,
            "best_odds": best_odds, "best_ev": best_ev, "stake_pct": stake_pct,
            "status": "pending"
        })
    return pd.DataFrame(rows)


def load_predictions(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["Date"])


def save_predictions(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def score_predictions(pred, played):
    if pred.empty or played.empty:
        return pred
    p = pred.copy()
    results = played[["LeagueCode","Date","HomeTeam","AwayTeam","FTHG","FTAG"]].copy()
    results["Date"] = pd.to_datetime(results["Date"])
    p["Date"] = pd.to_datetime(p["Date"])
    p = p.merge(results, on=["LeagueCode","Date","HomeTeam","AwayTeam"], how="left")
    p["status"] = np.where(p["FTHG"].notna(), "settled", p["status"])
    p["profit"] = np.nan

    def calc(r):
        if r["status"] != "settled" or pd.isna(r.get("best_odds", np.nan)):
            return np.nan
        side = r["best_side"]
        hit = (side == "H" and r.FTHG > r.FTAG) or (side == "D" and r.FTHG == r.FTAG) or (side == "A" and r.FTHG < r.FTAG)
        stake = float(r.get("stake_pct", 0)) * 500.0
        return stake * (float(r.best_odds) - 1) if hit else -stake

    p["profit"] = p.apply(calc, axis=1)
    return p


def performance_summary(df):
    if df.empty or "profit" not in df:
        return {"roi":0.0,"profit":0.0,"settled":0,"hit_rate":0.0,
                "by_league":pd.DataFrame(),"by_side":pd.DataFrame()}
    s = df[df.status == "settled"].copy()
    if s.empty:
        return {"roi":0.0,"profit":0.0,"settled":0,"hit_rate":0.0,
                "by_league":pd.DataFrame(),"by_side":pd.DataFrame()}
    s["stake"] = s["stake_pct"].fillna(0) * 500.0
    profit = s["profit"].sum()
    stake = s["stake"].sum()
    roi = profit / stake if stake else 0
    hit = (s["profit"] > 0).mean()
    by_league = s.groupby("League").agg(
        Paris=("profit","count"), Profit=("profit","sum"), Mise=("stake","sum")
    ).reset_index()
    by_league["ROI"] = by_league["Profit"] / by_league["Mise"].replace(0, np.nan)
    by_side = s.groupby("best_side").agg(
        Paris=("profit","count"), Profit=("profit","sum"), Mise=("stake","sum")
    ).reset_index()
    by_side["ROI"] = by_side["Profit"] / by_side["Mise"].replace(0, np.nan)
    return {"roi":roi,"profit":profit,"settled":len(s),"hit_rate":hit,
            "by_league":by_league,"by_side":by_side}
