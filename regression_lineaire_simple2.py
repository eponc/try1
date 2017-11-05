# Regression lineaire simple

# Importer les librairies
import numpy as np # pour fonctions mathematiques
import matplotlib.pyplot as plt #pour les courbes
import pandas as pd #pour gerer des datasets

# Importer le data set
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Gérer les données manquantes (les Nan dans certaines cellules)
# Non necessaire en l'absence de données manquantes

# Gérer les variables catégoriques
# Correspond aux variables à manipuler pour corréler les résultats 
# Pas de données catégoriques

# Divisr le data set entre le training set et le test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 1.0/3, random_state = 0) 

# Feature scaling
# Pas necessaire en regression lineaire

# Construction du modèle de regression lineaire simple 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# X_train est le vecteur utilisé pour calculer le modèle
regressor.fit(X_train, y_train)

# Faire de nouvelles prédictions
# X_test est le vecteur de test utilisé pour faire la prédiction utilisant le modèle précédent
y_pred = regressor.predict(X_test)
# Exemple de prédiction sur une valeur en dehors du dataset
regressor.predict(15)

# Visualisr les resultats
plt.scatter(X_test, y_test, color  = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salaire vs Expérience')
plt.xlabel('Expérience')
plt.ylabel('Salaire')
plt.show()


