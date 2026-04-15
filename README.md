# Prédiction du Taux de Réussite Universitaire

## Description
Ce projet utilise le Machine Learning pour prédire le taux de réussite des établissements universitaires tunisiens à partir de données officielles du Portail Open Data Tunisien.

## Auteur
Chaima CHATTI - Polytechnique de Sousse (2025-2026)

## Données
- **Source :** data.gov.tn (Ministère de l'Enseignement Supérieur)
- **Dataset 1 :** Étudiants diplômés (2021-2022) - 2 083 lignes
- **Dataset 2 :** Effectifs inscrits (2021-2023) - 19 878 lignes

## Modèles testés
- Régression Linéaire (R² = 0.9125)
- Random Forest (R² = 0.9767)
- Gradient Boosting (R² = 0.9758)

## Meilleur modèle
**Random Forest** avec :
- R² = 0.9767 (97.67%)
- MAE = 1.5%

## Variables les plus importantes
1. Efficacité féminine : 48.8%
2. Efficacité masculine : 45.7%
3. Écart de réussite : 1.1%

## Application Streamlit
```bash
streamlit run app.py