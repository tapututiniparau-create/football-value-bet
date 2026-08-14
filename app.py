import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Reuse the V1 engine from the same folder.
sys.path.append(str(Path(__file__).parent))
from football_model_v1 import (
    download_data, build_pre_match_dataset, train_and_test,
    add_value, generate_picks
)

st.set_page_config(
    page_title="Football Value Lab V1",
    page_icon="⚽",
    layout="centered"
)

st.title("⚽ Football Value Lab")
st.caption("Ligue 1 + Premier League — V1 recherche / paper trading")

with st.sidebar:
    st.header("Paramètres")
    from_season = st.number_input("Première saison", 2018, 2025, 2018)
    to_season = st.number_input("Dernière saison", 2018, 2025, 2025)
    test_season = st.number_input("Saison à tester", from_season, to_season, to_season)
    bankroll = st.number_input("Bankroll théorique (€)", 50.0, 100000.0, 500.0, step=50.0)
    min_ev = st.slider("EV minimale", 0.0, 0.30, 0.05, 0.01)

@st.cache_data(show_spinner=False)
def load_predictions(from_season, to_season, test_season):
    cache = Path("data_cache")
    seasons = list(range(int(from_season), int(to_season) + 1))
    raw = download_data(seasons, cache)
    dataset = build_pre_match_dataset(raw)
    model, test = train_and_test(dataset, int(test_season))
    test = add_value(test, odds_source="Avg")
    return test

if st.button("🔄 Charger / recalculer le modèle", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

try:
    with st.spinner("Chargement des données et calcul du modèle..."):
        predictions = load_predictions(from_season, to_season, test_season)

    st.success(f"{len(predictions)} matchs analysés hors-échantillon.")

    picks = generate_picks(predictions, min_ev=min_ev, bankroll=bankroll)

    st.subheader("🔎 Value détectée")

    if picks.empty:
        st.info("Aucun pari ne dépasse actuellement le seuil d'EV sélectionné.")
    else:
        view = picks.copy()
        view["ModelProb"] = (view["ModelProb"] * 100).round(1).astype(str) + "%"
        view["EV"] = (view["EV"] * 100).round(1).astype(str) + "%"
        view["StakePct"] = (view["StakePct"] * 100).round(2).astype(str) + "%"
        view = view.rename(columns={
            "League": "Championnat",
            "Match": "Match",
            "Side": "Choix",
            "ModelProb": "Proba modèle",
            "MarketOdds": "Cote",
            "EV": "EV",
            "StakePct": "Mise %",
            "Stake€": "Mise €"
        })
        st.dataframe(
            view[["Championnat","Match","Choix","Proba modèle","Cote","EV","Mise %","Mise €"]],
            use_container_width=True,
            hide_index=True
        )

    st.subheader("📊 Qualité du test")
    c1, c2 = st.columns(2)
    c1.metric("Matchs", len(predictions))
    c2.metric("EV max", f"{predictions[['ev_h','ev_d','ev_a']].max().max()*100:.1f}%")

    st.warning(
        "⚠️ V1 est un outil de recherche. Une EV positive dépend entièrement de la "
        "qualité des probabilités du modèle. Aucun pari n'est garanti."
    )

except Exception as e:
    st.error("Le calcul n'a pas pu être terminé.")
    st.exception(e)
