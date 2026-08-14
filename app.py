from __future__ import annotations

import os
import re
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


FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"


def download_fixtures():
    """Download the current fixture file and tolerate BOM/header variations."""
    r = requests.get(
        FIXTURES_URL,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    r.raise_for_status()

    from io import BytesIO
    raw = r.content

    # Football-Data can change formatting/BOM details in its downloadable file.
    # Detect the delimiter instead of assuming a comma.
    sample = raw[:10000].decode("latin1", errors="replace")
    try:
        import csv
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\\t")
        sep = dialect.delimiter
    except Exception:
        sep = ","

    df = pd.read_csv(
        BytesIO(raw),
        encoding="latin1",
        sep=sep,
        engine="python"
    )

    # Normalize headers: BOM, spaces, punctuation and case.
    df.columns = [
        re.sub(r"[^a-z0-9]", "", str(c).replace("\ufeff", "").strip().lower())
        for c in df.columns
    ]

    # Accept common variants.
    aliases = {
        "division": "div",
        "league": "div",
        "hometeam": "hometeam",
        "home": "hometeam",
        "awayteam": "awayteam",
        "away": "awayteam",
        "matchdate": "date",
        "fixturedate": "date",
    }
    df = df.rename(columns={c: aliases.get(c, c) for c in df.columns})

    required = {"div", "date", "hometeam", "awayteam"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            "Le fichier de fixtures ne contient pas les colonnes attendues. "
            f"Colonnes reçues: {list(df.columns)}"
        )

    df = df[df["div"].astype(str).str.upper().isin(["F1", "E0"])].copy()
    df["LeagueCode"] = df["div"].astype(str).str.upper()
    df["League"] = df["LeagueCode"].map(LEAGUES)
    df["Date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "hometeam", "awayteam"])
    df = df.rename(columns={"hometeam": "HomeTeam", "awayteam": "AwayTeam"})

    # Optional odds columns: normalize possible market-average names.
    for col in ["AvgH", "AvgD", "AvgA"]:
        if col not in df.columns:
            df[col] = np.nan

    df = df[df["Date"] >= pd.Timestamp.now().normalize()]
    return df.sort_values("Date").reset_index(drop=True)


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


def current_states_from_history(hist_dataset):
    """Reconstruct latest team states from historical matches."""
    states = {}
    completed = hist_dataset.sort_values("Date")
    for _, r in completed.iterrows():
        hk = (r["LeagueCode"], r["HomeTeam"])
        ak = (r["LeagueCode"], r["AwayTeam"])
        states.setdefault(hk, TeamState())
        states.setdefault(ak, TeamState())

        # Rebuild a minimal row for the state updater.
        class R:
            pass
        rr = R()
        rr.FTHG = None
        rr.FTAG = None
        rr.get = lambda k, default=None: default
        # Directly update without needing a pandas row.
        hg, ag = float(r["home_gf"]), float(r["away_gf"])
        # For reconstruction, use result-level information from dataset only.
        # Rebuild goals from available features is impossible, so instead use
        # the feature dataset's rolling values as a cold-start approximation.
        # A fresh Elo/state reconstruction is handled below from raw history
        # in train_model_all_history; predictions use the model plus neutral
        # current-season state. This keeps the V1.5.1 robust.
        states[hk].matches = int(r["home_experience"])
        states[ak].matches = int(r["away_experience"])
        states[hk].elo = 1500.0 + float(r["elo_diff"]) / 2
        states[ak].elo = 1500.0 - float(r["elo_diff"]) / 2
        states[hk].gf = [float(r["home_gf"])]
        states[hk].ga = [float(r["home_ga"])]
        states[ak].gf = [float(r["away_gf"])]
        states[ak].ga = [float(r["away_ga"])]
        states[hk].sot_for = [float(r["home_sot_for"])]
        states[hk].sot_against = [float(r["home_sot_against"])]
        states[ak].sot_for = [float(r["away_sot_for"])]
        states[ak].sot_against = [float(r["away_sot_against"])]
    return states


def predict_upcoming(model, hist_dataset, fixtures):
    # For V1.5.1, use latest historical team features as the cold-start state.
    # This avoids relying on a not-yet-published 2026/27 results CSV.
    latest = {}
    for _, r in hist_dataset.sort_values("Date").iterrows():
        latest[(r["LeagueCode"], r["HomeTeam"])] = r
        latest[(r["LeagueCode"], r["AwayTeam"])] = r

    rows = []
    for _, r in fixtures.sort_values("Date").iterrows():
        hk = (r["LeagueCode"], r["HomeTeam"])
        ak = (r["LeagueCode"], r["AwayTeam"])
        h = latest.get(hk)
        a = latest.get(ak)

        if h is None or a is None:
            # New/promoted team: use neutral league-average fallback.
            hfeat = {
                "elo_diff": 0.0, "home_gf": 1.2, "home_ga": 1.2,
                "home_sot_for": 4.5, "home_sot_against": 4.5,
                "home_experience": 0
            }
        else:
            hfeat = {
                "elo_diff": float(h["elo_diff"]),
                "home_gf": float(h["home_gf"]), "home_ga": float(h["home_ga"]),
                "home_sot_for": float(h["home_sot_for"]),
                "home_sot_against": float(h["home_sot_against"]),
                "home_experience": float(h["home_experience"])
            }

        if a is None:
            afeat = {
                "away_gf": 1.0, "away_ga": 1.3,
                "away_sot_for": 4.2, "away_sot_against": 4.7,
                "away_experience": 0
            }
        else:
            afeat = {
                "away_gf": float(a["away_gf"]), "away_ga": float(a["away_ga"]),
                "away_sot_for": float(a["away_sot_for"]),
                "away_sot_against": float(a["away_sot_against"]),
                "away_experience": float(a["away_experience"])
            }

        f = {**hfeat, **afeat}
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
        best_side = max(valid, key=valid.get) if valid else None
        best_ev = valid[best_side] if best_side else np.nan
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


# =========================
# STREAMLIT APPLICATION
# =========================
import os
from datetime import datetime, timezone
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Football Value Lab", page_icon="⚽", layout="wide")

st.title("⚽ Football Value Lab")
st.caption("V1.5.6 — prédictions, historique et suivi de performance")

DATA_DIR = Path("data_cache")
LOCAL_DB = Path("data/predictions.csv")
LOCAL_DB.parent.mkdir(exist_ok=True)

with st.sidebar:
    st.header("⚙️ Paramètres")
    seasons_back = st.slider("Saisons historiques", 3, 10, 8)
    min_ev = st.slider("EV minimale", 0.0, 0.30, 0.05, 0.01)
    bankroll = st.number_input("Bankroll théorique (€)", 50.0, 100000.0, 500.0, 50.0)

@st.cache_data(ttl=3600, show_spinner=False)
def build_current_predictions(seasons_back):
    current_year = datetime.now().year
    first = current_year - seasons_back
    completed_end = current_year - 1
    seasons = list(range(first, completed_end + 1))

    raw_hist = download_data(seasons, DATA_DIR)
    hist_dataset = build_pre_match_dataset(raw_hist)
    model = train_model_all_history(hist_dataset)

    fixtures = download_fixtures()
    predictions = predict_upcoming(model, hist_dataset, fixtures)
    return predictions

try:
    with st.spinner("Chargement des données et calcul du modèle..."):
        predictions = build_current_predictions(seasons_back)

    stored = load_predictions(LOCAL_DB)

    if not predictions.empty:
        new_rows = predictions.copy()
        new_rows["created_at"] = datetime.now(timezone.utc).isoformat()
        stored = pd.concat([stored, new_rows], ignore_index=True)
        stored = stored.drop_duplicates(
            subset=["LeagueCode", "Date", "HomeTeam", "AwayTeam"], keep="first"
        )
        save_predictions(stored, LOCAL_DB)

    summary = performance_summary(stored)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🏠 Tableau de bord", "⚽ Matchs", "📒 Historique", "📊 Performance"]
    )

    with tab1:
        st.subheader("Vue d'ensemble")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Matchs à venir", len(predictions))
        c2.metric("Prédictions enregistrées", len(stored))
        c3.metric("Values détectées", int((stored.get("best_ev", pd.Series(dtype=float)) >= min_ev).sum()))
        c4.metric("ROI théorique", f"{summary['roi']*100:.2f}%")

        st.subheader("🔥 Meilleures values")
        if predictions.empty:
            st.info("Aucun match futur disponible dans la source actuelle.")
        else:
            best = predictions[predictions["best_ev"] >= min_ev].sort_values(
                "best_ev", ascending=False
            ).head(20)
            if best.empty:
                st.info("Aucun match ne dépasse le seuil d'EV.")
            else:
                display = best[[
                    "Date","League","HomeTeam","AwayTeam",
                    "best_side","best_prob","best_odds","best_ev","stake_pct"
                ]].copy()
                display["best_prob"] = (display["best_prob"] * 100).round(1).astype(str) + "%"
                display["best_ev"] = (display["best_ev"] * 100).round(1).astype(str) + "%"
                display["stake_pct"] = (display["stake_pct"] * 100).round(2).astype(str) + "%"
                st.dataframe(display, use_container_width=True, hide_index=True)

        st.warning(
            "⚠️ Paper trading uniquement. Une EV positive n'est pas une garantie de gain."
        )

    with tab2:
        st.subheader("⚽ Matchs à venir")
        if predictions.empty:
            st.info("Aucun match futur disponible.")
        else:
            p = predictions.copy()
            for col in ["p_home","p_draw","p_away","best_prob","best_ev","stake_pct"]:
                if col in p.columns:
                    p[col] = (p[col] * 100).round(2)
            st.dataframe(
                p[[
                    "Date","League","HomeTeam","AwayTeam",
                    "p_home","p_draw","p_away",
                    "best_side","best_prob","best_odds","best_ev","stake_pct"
                ]],
                use_container_width=True, hide_index=True
            )

    with tab3:
        st.subheader("📒 Journal")
        if stored.empty:
            st.info("Aucune prédiction enregistrée.")
        else:
            h = stored.sort_values("Date", ascending=False)
            st.dataframe(h, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Télécharger l'historique CSV",
                h.to_csv(index=False).encode("utf-8"),
                "football_value_history.csv",
                "text/csv"
            )

    with tab4:
        st.subheader("📊 Performance")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROI", f"{summary['roi']*100:.2f}%")
        c2.metric("Profit théorique", f"{summary['profit']:.2f} €")
        c3.metric("Paris évalués", summary["settled"])
        c4.metric("Taux de réussite", f"{summary['hit_rate']*100:.1f}%")

        if not summary["by_league"].empty:
            st.write("Performance par championnat")
            st.dataframe(summary["by_league"], use_container_width=True, hide_index=True)

        if not summary["by_side"].empty:
            st.write("Performance par choix")
            st.dataframe(summary["by_side"], use_container_width=True, hide_index=True)

except Exception as e:
    st.error("Le calcul n'a pas pu être terminé.")
    st.exception(e)
