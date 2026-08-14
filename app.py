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
LEAGUES = {"F1": "Ligue 1", "E0": "Premier League", "SP1": "La Liga", "MLS": "MLS"}
HIST_CODES = {"F1":"F1","E0":"E0","SP1":"SP1"}

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


# Upcoming fixtures: ESPN public scoreboard (no API key required).
# Flashscore is used as a human verification link; we do not scrape it automatically.
ESPN_LEAGUES = {"F1":"fra.1", "E0":"eng.1", "SP1":"esp.1", "MLS":"usa.1"}
FLASHSCORE_URLS = {
    "F1":"https://www.flashscore.com/football/france/ligue-1/",
    "E0":"https://www.flashscore.com/football/england/premier-league/",
    "SP1":"https://www.flashscore.com/football/spain/laliga/",
    "MLS":"https://www.flashscore.com/football/usa/mls/",
}

def download_fixtures(days=45):
    from datetime import timedelta
    start_day = datetime.now().date()
    rows=[]
    for code, league in ESPN_LEAGUES.items():
        for offset in range(0, days, 7):
            dates=','.join((start_day+timedelta(days=i)).strftime('%Y%m%d') for i in range(offset,min(offset+7,days)))
            url=f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard?dates={dates}&limit=100"
            try:
                r=requests.get(url,timeout=20,headers={"User-Agent":"Mozilla/5.0"})
                r.raise_for_status(); data=r.json()
            except Exception:
                continue
            for ev in data.get('events',[]):
                comp=(ev.get('competitions') or [{}])[0]
                cs=comp.get('competitors') or []
                home=next((x for x in cs if x.get('homeAway')=='home'),None)
                away=next((x for x in cs if x.get('homeAway')=='away'),None)
                if not home or not away: continue
                dt=pd.to_datetime(ev.get('date'),utc=True,errors='coerce')
                if pd.isna(dt): continue
                rows.append({
                    'Date':dt.tz_convert(None), 'LeagueCode':code, 'League':LEAGUES[code],
                    'HomeTeam':home.get('team',{}).get('displayName',''),
                    'AwayTeam':away.get('team',{}).get('displayName',''),
                    'AvgH':np.nan,'AvgD':np.nan,'AvgA':np.nan,
                    'fixture_source':'ESPN'
                })
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(['LeagueCode','Date','HomeTeam','AwayTeam']).sort_values('Date').reset_index(drop=True)


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
st.caption("V1.6.1 — 4 championnats • calendrier automatique • paper trading")

DATA_DIR = Path("data_cache")
LOCAL_DB = Path("data/predictions.csv")
LOCAL_DB.parent.mkdir(exist_ok=True)

with st.sidebar:
    st.header("⚙️ Paramètres")
    seasons_back = st.slider("Saisons historiques", 3, 10, 8)
    min_ev = st.slider("EV minimale", 0.0, 0.30, 0.05, 0.01)
    bankroll = st.number_input("Bankroll théorique (€)", 50.0, 100000.0, 500.0, 50.0)
    selected = st.multiselect("Championnats", list(LEAGUES), default=list(LEAGUES), format_func=lambda x: LEAGUES[x])
    st.divider()
    st.caption("Validation Flashscore")
    for code in selected:
        st.markdown(f"[{LEAGUES[code]} sur Flashscore]({FLASHSCORE_URLS[code]})")

@st.cache_data(ttl=1800, show_spinner=False)
def build_current_predictions(seasons_back, selected_codes):
    current_year = datetime.now().year
    first = current_year - seasons_back
    completed_end = current_year - 1
    seasons = list(range(first, completed_end + 1))

    hist_codes = [c for c in selected_codes if c in ("F1","E0","SP1")]
    raw_hist = download_data(seasons, DATA_DIR)
    raw_hist = raw_hist[raw_hist["LeagueCode"].isin(hist_codes)].copy()
    hist_dataset = build_pre_match_dataset(raw_hist)
    model = train_model_all_history(hist_dataset)

    fixtures = download_fixtures()
    fixtures = fixtures[fixtures["LeagueCode"].isin(selected_codes)].copy()
    model_fixtures = fixtures[fixtures["LeagueCode"].isin(hist_codes)].copy()
    predictions = predict_upcoming(model, hist_dataset, model_fixtures)
    if not predictions.empty:
        predictions["prediction_status"] = "Modèle disponible"
    mls = fixtures[fixtures["LeagueCode"].eq("MLS")].copy()
    if not mls.empty:
        mls["prediction_status"] = "Calendrier uniquement — historique MLS à connecter"
        for c in ["p_home","p_draw","p_away","best_prob","best_odds","best_ev","stake_pct"]:
            mls[c] = np.nan
        mls["best_side"] = "—"
        predictions = pd.concat([predictions, mls], ignore_index=True, sort=False)
    return predictions

if not selected:
    st.warning("Sélectionne au moins un championnat.")
    st.stop()

try:
    with st.spinner("Chargement des données et calcul du modèle..."):
        predictions = build_current_predictions(seasons_back, selected)

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
    st.info("Calendrier à venir : ESPN public scoreboard. Flashscore est fourni comme source de vérification manuelle. Historique : Football-Data.co.uk. Les cotes ne sont jamais inventées.")

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
            best = predictions[(predictions["prediction_status"] == "Modèle disponible") & (predictions["best_ev"] >= min_ev)].sort_values(
                "best_ev", ascending=False
            ).head(20)
            if best.empty:
                st.info("Aucun match ne dépasse le seuil d'EV.")
            else:
                display = best[[
                    "Date","League","HomeTeam","AwayTeam",
                    "best_side","best_prob","best_odds","best_ev","stake_pct","prediction_status"
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
                    "best_side","best_prob","best_odds","best_ev","stake_pct","prediction_status"
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
