# Football Value Lab V1.5

V1.5 ajoute:
- tableau de bord
- matchs à venir
- journal des prédictions
- mise à jour des résultats
- statistiques de performance
- téléchargement CSV
- sauvegarde locale des prédictions

## Déploiement
Mettre `app.py`, `football_model_v15.py` et `requirements.txt` dans le dépôt GitHub.
Dans Streamlit Community Cloud, choisir `app.py` comme fichier principal.

## Persistance
La sauvegarde locale fonctionne pendant l'exécution de l'app, mais Streamlit Community Cloud
peut réinitialiser le stockage local lors d'un redéploiement/restart. Pour une vraie persistance
longue durée, V1.6 pourra connecter la base à Supabase ou une autre base externe.

## Important
Les prédictions sont du paper trading. Elles ne garantissent aucun bénéfice.
