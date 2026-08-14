import os
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import streamlit as st

from football_model_v15 import (
    download_data,
    build_pre_match_dataset,
    train_model_all_history,
    predict_upcoming,
    score_predictions,
    performance_summary,
    save_predictions,
    load_predictions,
)

st.set_page_config(page_title="Football Value Lab", page_icon="⚽", layout="wide")

st.title("⚽ Football Value Lab")
st.caption("V1.5.1 — prédictions, historique et suivi de performance")

DATA_DIR = Path("data_cache")
LOCAL_DB = Path("data/predictions.csv")
LOCAL_DB.parent.mkdir(exist_ok=True)

with st.sidebar:
    st.header("⚙️ Paramètres")
    seasons_back = st.slider("Saisons historiques", 3, 10, 8)
    min_ev = st.slider("EV minimale", 0.0, 0.30, 0.05, 0.01)
    bankroll = st.number_input("Bankroll théorique (€)", 50.0, 100000.0, 500.0, 50.0)
    st.divider()
    st.write("**Persistance**")
    if os.getenv("GITHUB_TOKEN"):
        st.success("☁️ Sauvegarde GitHub activée")
    else:
        st.warning("💾 Sauvegarde locale uniquement")

@st.cache_data(ttl=3600, show_spinner=False)
def build_current_predictions(seasons_back):
    current_year = datetime.now().year
    # In August, current season starts in the current calendar year.
    # Use previous completed seasons for training.
    first = current_year - seasons_back
    completed_end = current_year - 1

    seasons = list(range(first, completed_end + 1))
    raw_hist = download_data(seasons, DATA_DIR)

    # Football-Data publishes the latest fixture list separately.
    # We use it for upcoming matches instead of assuming the 2026/27
    # historical season CSV already exists.
    fixtures = download_fixtures()

    hist_dataset = build_pre_match_dataset(raw_hist)
    model = train_model_all_history(hist_dataset)

    predictions = predict_upcoming(model, hist_dataset, fixtures)
    played = pd.DataFrame()
    return predictions, played

try:
    predictions, played = build_current_predictions(seasons_back)

    # Load stored paper predictions.
    stored = load_predictions(LOCAL_DB)

    # Add new predictions to history, deduplicated by league/date/teams.
    if not predictions.empty:
        new_rows = predictions.copy()
        new_rows["created_at"] = datetime.now(timezone.utc).isoformat()
        stored = pd.concat([stored, new_rows], ignore_index=True)
        stored = stored.drop_duplicates(
            subset=["LeagueCode", "Date", "HomeTeam", "AwayTeam"],
            keep="first"
        )

        # Update completed results from current season.
        stored = score_predictions(stored, played)
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
        c3.metric("Paris value", int((stored.get("best_ev", pd.Series(dtype=float)) >= min_ev).sum()))
        c4.metric("ROI théorique", f"{summary['roi']*100:.2f}%")

        st.subheader("🔥 Meilleures values")
        if predictions.empty:
            st.info("Aucun match futur trouvé dans les données de la saison en cours.")
        else:
            best = predictions[predictions["best_ev"] >= min_ev].copy()
            best = best.sort_values("best_ev", ascending=False).head(20)
            if best.empty:
                st.info("Aucun match ne dépasse le seuil d'EV.")
            else:
                st.dataframe(
                    best[[
                        "Date","League","HomeTeam","AwayTeam",
                        "best_side","best_prob","best_odds","best_ev","stake_pct"
                    ]].rename(columns={
                        "Date":"Date","League":"Championnat","HomeTeam":"Domicile",
                        "AwayTeam":"Extérieur","best_side":"Choix",
                        "best_prob":"Proba modèle","best_odds":"Cote",
                        "best_ev":"EV","stake_pct":"Mise %"
                    }),
                    use_container_width=True, hide_index=True
                )

        st.info(
            "⚠️ Les valeurs affichées sont des estimations statistiques. "
            "V1.5 fonctionne en paper trading : aucune mise réelle n'est recommandée."
        )

    with tab2:
        st.subheader("Matchs à venir")
        if predictions.empty:
            st.info("Aucun match futur disponible.")
        else:
            p = predictions.copy()
            for col in ["p_home","p_draw","p_away","best_prob","best_ev","stake_pct"]:
                if col in p:
                    p[col] = p[col] * 100
            st.dataframe(
                p[[
                    "Date","League","HomeTeam","AwayTeam",
                    "p_home","p_draw","p_away",
                    "best_side","best_prob","best_odds","best_ev","stake_pct"
                ]].rename(columns={
                    "League":"Championnat","HomeTeam":"Domicile","AwayTeam":"Extérieur",
                    "p_home":"1 %","p_draw":"N %","p_away":"2 %",
                    "best_side":"Choix","best_prob":"Proba choix %",
                    "best_odds":"Cote","best_ev":"EV %",
                    "stake_pct":"Mise %"
                }),
                use_container_width=True, hide_index=True
            )

    with tab3:
        st.subheader("📒 Journal des prédictions")
        if stored.empty:
            st.info("Aucune prédiction enregistrée.")
        else:
            h = stored.copy()
            st.dataframe(h.sort_values("Date", ascending=False), use_container_width=True, hide_index=True)
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

        if not stored.empty:
            st.write("**Performance par championnat**")
            st.dataframe(summary["by_league"], use_container_width=True, hide_index=True)
            st.write("**Performance par choix**")
            st.dataframe(summary["by_side"], use_container_width=True, hide_index=True)

except Exception as e:
    st.error("Le calcul n'a pas pu être terminé.")
    st.exception(e)
