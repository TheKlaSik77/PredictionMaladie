import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


maladies_training = pd.read_csv('./data/maladies_training_processed.csv')
maladies_test = pd.read_csv('./data/maladies_test_processed.csv')

X_train = maladies_training.drop(columns=['prognosis'])
y_train = maladies_training['prognosis']

X_test = maladies_test.drop(columns=['prognosis'])
y_test = maladies_test['prognosis']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R2: {r2}")

feature_importance = pd.DataFrame({
    'feature': X_test.columns,
    'importance': model.feature_importances_
})
print("\nFeature Importance:")
print(feature_importance.sort_values(by='importance', ascending=False))


# Validation croisée avec 5 folds qui confirme que le modele est performant est n'est pas en cas de surrapprentissage
scores = cross_val_score(model, X_train, y_train, scoring='r2', cv=5)
print(f"R² moyen avec validation croisée : {np.mean(scores)}")
print(f"Scores R² pour chaque fold : {scores}")