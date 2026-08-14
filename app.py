
import re, csv
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Football Value Lab", page_icon="⚽", layout="wide")

LEAGUES = {"F1":"Ligue 1", "E0":"Premier League", "SP1":"La Liga", "USA":"MLS"}
HIST_CODES = {"F1":"F1", "E0":"E0", "SP1":"SP1", "USA":"USA"}
ESPN_LEAGUES = {
    "F1": ("fra.1", "Ligue 1"),
    "E0": ("eng.1", "Premier League"),
    "SP1": ("esp.1", "La Liga"),
    "USA": ("usa.1", "MLS"),
}
FEATURES = [
    "elo_diff","home_gf","home_ga","away_gf","away_ga",
    "home_sot_for","home_sot_against","away_sot_for",
    "away_sot_against","home_experience","away_experience"
]

class TeamState:
    def __init__(self):
        self.elo=1500.0; self.matches=0
        self.gf=[]; self.ga=[]; self.sot_for=[]; self.sot_against=[]

def season_code(y): return f"{str(y)[-2:]}{str(y+1)[-2:]}"

def safe_float(v, default=np.nan):
    try: return float(v)
    except: return default

@st.cache_data(ttl=3600, show_spinner=False)
def download_history(seasons_back):
    current=datetime.now().year
    years=range(current-seasons_back, current)
    frames=[]; cache=Path("data_cache"); cache.mkdir(exist_ok=True)
    for y in years:
        code=season_code(y)
        for lc, sc in HIST_CODES.items():
            path=cache/f"{code}_{sc}.csv"
            url=f"https://www.football-data.co.uk/mmz4281/{code}/{sc}.csv"
            try:
                if not path.exists():
                    r=requests.get(url,timeout=25,headers={"User-Agent":"Mozilla/5.0"})
                    if r.status_code != 200 or len(r.content)<100: continue
                    path.write_bytes(r.content)
                df=pd.read_csv(path)
                needed={"Date","HomeTeam","AwayTeam","FTHG","FTAG"}
                if not needed.issubset(df.columns): continue
                df["LeagueCode"]=lc; df["League"]=LEAGUES[lc]; df["SeasonStart"]=y
                frames.append(df)
            except Exception: continue
    if not frames: raise RuntimeError("Aucune donnée historique exploitable.")
    out=pd.concat(frames,ignore_index=True)
    out["Date"]=pd.to_datetime(out["Date"],dayfirst=True,errors="coerce")
    out=out.dropna(subset=["Date","HomeTeam","AwayTeam","FTHG","FTAG"])
    return out.sort_values("Date").reset_index(drop=True)

def mean_last(x,n=8,default=1.0): return float(np.mean(x[-n:])) if x else default

def make_features(h,a):
    return {
        "elo_diff":h.elo-a.elo,
        "home_gf":mean_last(h.gf,8,1.2),"home_ga":mean_last(h.ga,8,1.2),
        "away_gf":mean_last(a.gf,8,1.0),"away_ga":mean_last(a.ga,8,1.3),
        "home_sot_for":mean_last(h.sot_for,8,4.5),"home_sot_against":mean_last(h.sot_against,8,4.5),
        "away_sot_for":mean_last(a.sot_for,8,4.2),"away_sot_against":mean_last(a.sot_against,8,4.7),
        "home_experience":min(h.matches,20),"away_experience":min(a.matches,20)
    }

def update_state(h,a,r):
    hg=float(r.FTHG); ag=float(r.FTAG)
    hs=safe_float(r.get("HST",np.nan),4.5) if hasattr(r,"get") else 4.5
    ats=safe_float(r.get("AST",np.nan),4.2) if hasattr(r,"get") else 4.2
    h.gf.append(hg); h.ga.append(ag); a.gf.append(ag); a.ga.append(hg)
    h.sot_for.append(hs); h.sot_against.append(ats); a.sot_for.append(ats); a.sot_against.append(hs)
    h.matches+=1; a.matches+=1
    exp=1/(1+10**(-(h.elo-a.elo)/400))
    result=1 if hg>ag else 0 if hg<ag else .5
    d=22*(result-exp); h.elo+=d; a.elo-=d

def build_dataset(df):
    states={}; rows=[]
    for _,r in df[df.FTHG.notna()&df.FTAG.notna()].sort_values("Date").iterrows():
        hk=(r.LeagueCode,r.SeasonStart,r.HomeTeam); ak=(r.LeagueCode,r.SeasonStart,r.AwayTeam)
        states.setdefault(hk,TeamState()); states.setdefault(ak,TeamState())
        h,a=states[hk],states[ak]; f=make_features(h,a)
        f.update({"Date":r.Date,"LeagueCode":r.LeagueCode,"League":r.League,
                  "SeasonStart":r.SeasonStart,"HomeTeam":r.HomeTeam,"AwayTeam":r.AwayTeam,
                  "Result":0 if r.FTHG>r.FTAG else 2 if r.FTHG<r.FTAG else 1})
        rows.append(f); update_state(h,a,r)
    return pd.DataFrame(rows)

def train_model(ds):
    u=ds[(ds.home_experience>=3)&(ds.away_experience>=3)]
    X=u[FEATURES].fillna(ds[FEATURES].median()); y=u.Result
    return Pipeline([("scale",StandardScaler()),("logit",LogisticRegression(max_iter=2000,C=.5,solver="lbfgs",random_state=42))]).fit(X,y)

def espn_events(league_code, days=45):
    """Fetch future fixtures one calendar day at a time."""
    slug = ESPN_LEAGUES[league_code][0]
    today = pd.Timestamp.now(tz="UTC").normalize()
    rows = []
    for offset in range(days):
        d = today + pd.Timedelta(days=offset)
        url = f"https://site.api.espn.com/apis/site/v2/sports/{slug}/scoreboard?dates={d.strftime('%Y%m%d')}&limit=100"
        try:
            r = requests.get(url, timeout=15, headers={"User-Agent":"Mozilla/5.0 FootballValueLab/1.6.3"})
            if r.status_code != 200:
                continue
            for ev in r.json().get("events", []):
                comps = ev.get("competitions") or []
                if not comps:
                    continue
                teams = comps[0].get("competitors") or []
                if len(teams) < 2:
                    continue
                home = next((t for t in teams if t.get("homeAway")=="home"), teams[0])
                away = next((t for t in teams if t.get("homeAway")=="away"), teams[1])
                hn = (home.get("team") or {}).get("displayName","")
                an = (away.get("team") or {}).get("displayName","")
                if not hn or not an:
                    continue
                dt = pd.to_datetime(ev.get("date"), utc=True, errors="coerce")
                rows.append({
                    "Date": dt.tz_convert(None) if pd.notna(dt) else pd.Timestamp(d),
                    "LeagueCode": league_code, "League": LEAGUES[league_code],
                    "HomeTeam": hn, "AwayTeam": an, "Source": "ESPN"
                })
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["Date","LeagueCode","League","HomeTeam","AwayTeam","Source"])
    return pd.DataFrame(rows).drop_duplicates(["LeagueCode","Date","HomeTeam","AwayTeam"])


@st.cache_data(ttl=1800,show_spinner=False)
def download_upcoming(selected):
    frames = []
    status = []
    for lc in selected:
        f = espn_events(lc, days=45)
        status.append(f"{LEAGUES[lc]}: {len(f)} match(s)")
        if not f.empty:
            frames.append(f)
    if not frames:
        return pd.DataFrame(), status
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(["LeagueCode","Date","HomeTeam","AwayTeam"])
    return out.sort_values("Date").reset_index(drop=True), status


def flashscore_links():
    return {
        "Ligue 1":"https://www.flashscore.com/football/france/ligue-1/",
        "Premier League":"https://www.flashscore.com/football/england/premier-league/",
        "La Liga":"https://www.flashscore.com/football/spain/laliga/",
        "MLS":"https://www.flashscore.com/football/usa/mls/"
    }

def latest_team_rows(ds):
    latest={}
    for _,r in ds.sort_values("Date").iterrows():
        latest[(r.LeagueCode,r.HomeTeam)]=r
        latest[(r.LeagueCode,r.AwayTeam)]=r
    return latest

def predict(model,ds,fixtures):
    latest=latest_team_rows(ds); med=ds[FEATURES].median(); rows=[]
    for _,r in fixtures.iterrows():
        h=latest.get((r.LeagueCode,r.HomeTeam)); a=latest.get((r.LeagueCode,r.AwayTeam))
        f={
          "elo_diff":safe_float(h.elo_diff,0) if h is not None else 0,
          "home_gf":safe_float(h.home_gf,1.2) if h is not None else 1.2,
          "home_ga":safe_float(h.home_ga,1.2) if h is not None else 1.2,
          "away_gf":safe_float(a.away_gf,1.0) if a is not None else 1.0,
          "away_ga":safe_float(a.away_ga,1.3) if a is not None else 1.3,
          "home_sot_for":safe_float(h.home_sot_for,4.5) if h is not None else 4.5,
          "home_sot_against":safe_float(h.home_sot_against,4.5) if h is not None else 4.5,
          "away_sot_for":safe_float(a.away_sot_for,4.2) if a is not None else 4.2,
          "away_sot_against":safe_float(a.away_sot_against,4.7) if a is not None else 4.7,
          "home_experience":safe_float(h.home_experience,0) if h is not None else 0,
          "away_experience":safe_float(a.away_experience,0) if a is not None else 0
        }
        X=pd.DataFrame([f])[FEATURES].fillna(med)
        p=model.predict_proba(X)[0]; classes=model.named_steps["logit"].classes_
        probs={int(c):float(p[i]) for i,c in enumerate(classes)}
        rows.append({**r.to_dict(),"p_home":probs.get(0,np.nan),"p_draw":probs.get(1,np.nan),"p_away":probs.get(2,np.nan),
                     "best_side":["1","N","2"][int(np.argmax([probs.get(0,0),probs.get(1,0),probs.get(2,0)]))],
                     "best_prob":max(probs.values())})
    return pd.DataFrame(rows)

def load_history():
    p=Path("data/predictions.csv")
    if not p.exists(): return pd.DataFrame()
    try: return pd.read_csv(p,parse_dates=["Date"])
    except: return pd.DataFrame()

def save_history(df):
    p=Path("data/predictions.csv"); p.parent.mkdir(exist_ok=True); df.to_csv(p,index=False)

st.title("⚽ Football Value Lab")
st.caption("V1.6.4 — 4 championnats • calendrier automatique • paper trading")

with st.sidebar:
    st.header("⚙️ Paramètres")
    selected=st.multiselect("Championnats",list(LEAGUES.keys()),default=list(LEAGUES.keys()),format_func=lambda x:LEAGUES[x])
    seasons=st.slider("Saisons historiques",3,8,6)
    st.divider()
    st.caption("📅 Calendrier : ESPN")
    st.caption("🔎 Vérification : Flashscore")
    for name,url in flashscore_links().items():
        st.markdown(f"[{name}]({url})")

try:
    with st.spinner("Chargement de l'historique et du calendrier..."):
        hist=download_history(seasons)
        ds=build_dataset(hist)
        model=train_model(ds)
        fixtures, fixture_status=download_upcoming(selected)
        predictions=predict(model,ds,fixtures) if not fixtures.empty else pd.DataFrame()

    stored=load_history()
    if not predictions.empty:
        new=predictions.copy()
        new["created_at"]=datetime.now(timezone.utc).isoformat()
        stored=pd.concat([stored,new],ignore_index=True).drop_duplicates(["LeagueCode","Date","HomeTeam","AwayTeam"],keep="first")
        save_history(stored)

    tabs=st.tabs(["🏠 Tableau de bord","⚽ Matchs","📒 Historique","📊 Performance"])

    with tabs[0]:
        c1,c2,c3=st.columns(3)
        c1.metric("Matchs à venir",len(predictions))
        c2.metric("Prédictions enregistrées",len(stored))
        c3.metric("Championnats",len(selected))
        st.info("⚠️ Paper trading uniquement : aucune garantie de bénéfice.")
        if predictions.empty:
            st.warning("Aucun match automatique récupéré.")
            st.caption("Diagnostic : " + " • ".join(fixture_status))
        else:
            st.subheader("Prochains matchs")
            st.dataframe(predictions[["Date","League","HomeTeam","AwayTeam","p_home","p_draw","p_away","best_side","best_prob"]],use_container_width=True,hide_index=True)

    with tabs[1]:
        st.subheader("⚽ Matchs à venir")
        if predictions.empty:
            st.info("Aucun match futur disponible.")
            st.caption("Diagnostic : " + " • ".join(fixture_status))
        else:
            d=predictions.copy()
            for c in ["p_home","p_draw","p_away","best_prob"]: d[c]=(d[c]*100).round(1)
            st.dataframe(d[["Date","League","HomeTeam","AwayTeam","p_home","p_draw","p_away","best_side","best_prob","Source"]],use_container_width=True,hide_index=True)

    with tabs[2]:
        st.subheader("📒 Historique")
        if stored.empty: st.info("Aucune prédiction enregistrée pour le moment.")
        else:
            st.dataframe(stored.sort_values("Date",ascending=False),use_container_width=True,hide_index=True)
            st.download_button("⬇️ Télécharger CSV",stored.to_csv(index=False).encode(),"football_value_history.csv","text/csv")

    with tabs[3]:
        st.subheader("📊 Performance")
        settled=stored[stored.get("status","pending")=="settled"] if not stored.empty else pd.DataFrame()
        if settled.empty: st.info("Pas encore assez de résultats pour calculer une performance. C'est normal au lancement.")
        else: st.dataframe(settled,use_container_width=True,hide_index=True)

except Exception as e:
    st.error("Le calcul n'a pas pu être terminé.")
    st.exception(e)
