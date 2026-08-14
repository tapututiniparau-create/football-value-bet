# Football Value Lab V1.5.1

Correction importante de V1.5:
- ne tente plus de lire un fichier historique 2026/27 qui n'est pas encore publié comme saison complète;
- utilise le fichier de fixtures dédié de Football-Data.co.uk pour les matchs à venir;
- conserve l'historique pour entraîner le modèle;
- gère les équipes absentes de l'historique avec un fallback neutre.

Le site Football-Data indique que sa page de fixtures publie séparément les dernières rencontres et les cotes, ce qui est la source utilisée ici.

## Fichiers à mettre sur GitHub
- app.py
- football_model_v15.py
- requirements.txt
- README.md

## Déploiement
Choisir `app.py` comme fichier principal dans Streamlit Community Cloud.

## Important
Les prédictions sont du paper trading et ne garantissent aucun bénéfice.
