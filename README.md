# Football Value Lab V1.6.5

Correction majeure du calendrier :
- ESPN reste une source de calendrier.
- Ajout d'un fallback robuste sur `https://www.football-data.co.uk/fixtures.csv`.
- Football-Data met à jour ce fichier avec les rencontres à venir et, lorsqu'elles sont disponibles, les cotes.
- 45 jours de fenêtre.
- Ligue 1, Premier League, La Liga, MLS.
- Historique séparé des fixtures.
- Paper trading uniquement.

Flashscore reste disponible comme source de vérification manuelle ; aucun scraping fragile de Flashscore n'est utilisé.
