# Football Value Lab V1 — mobile

Cette version transforme le moteur V1 en interface web adaptée au téléphone avec Streamlit.

## Déploiement recommandé

Streamlit Community Cloud permet de publier gratuitement une app Streamlit et de lui attribuer une URL `streamlit.app`.

1. Créer un compte GitHub.
2. Créer un dépôt.
3. Ajouter tous les fichiers de ce dossier.
4. Aller sur Streamlit Community Cloud.
5. Connecter GitHub.
6. Choisir `app.py` comme fichier principal.
7. Déployer.

L'application sera ensuite accessible depuis Safari/Chrome sur téléphone.

## Attention

La V1 est destinée au paper trading et au backtest. Elle ne constitue pas une garantie de rentabilité.

## Architecture

app.py
    -> moteur football_model_v1.py
    -> données historiques
    -> probabilités
    -> EV
    -> sélection de value
    -> mise théorique

La prochaine V2 pourra ajouter une base persistante des prédictions, les closing odds, CLV,
xG, blessures/compositions et un vrai tableau de suivi ROI/drawdown.
