import pandas as pd

maladies = pd.read_csv('./data/Training_processed.csv')

# Calculer la matrice de corrélation
correlation_matrix = maladies.corr()

# Afficher la corrélation avec la colonne 'Avg Salary'
correlation_with_avg_salary = correlation_matrix['prognosis'].sort_values(ascending=False)
print(correlation_with_avg_salary)
