#!/usr/bin/env python
# coding: utf-8

# In[149]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource, HoverTool

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

output_notebook() 


# In[150]:


#je commence par importer le DataFrame
df = pd.read_csv('/Users/octoberone/Desktop/DataFrame GitHub/DataFrame/Prévision des prix de immobilier canadien.csv', encoding='ISO-8859-1')


# In[151]:


#affichage des 5 premières lignes
df.head()


# In[152]:


#voici le nombre de lignes et de colonnes sans le détail
df.shape


# In[153]:


#voici les détails des colonnes avec beaucoup plus d'informations (variables, types, le nombre de valeurs nulles)

df.info()


# In[154]:


#j'ai utilisé cette commande pour être sûr qu'il n'y a pas de valeurs manquantes
df.isnull().sum()
#par conséquent, j'en déduis que ce DataFrame est plutôt agréable à utiliser dans un premier temps


# In[155]:


#Afin d'avoir plus de détails statistiques, j'affiche les colonnes numériques

#count%:nombre de valeurs nulles
#mean%:la moyenne des valeurs
#std%:l'écart type (dispersion valeurs)
#min%:le min de la colonne 
#25%:1er quatile à moins de 25
#50%:2er quatile à moins de 25
#75%:3er quatile à moins de 25
#max:le maxi de la colonne 

print(df.describe())


# In[156]:


#je commence par ce premier affichage
#je regroupe les provinces dans la variable province 
#Ppuis grâce à une visualisation interactive, j'affiche le prix des villes regroupées dans les provinces
#la visualisation n'est la plus agréable par rapport aux données variées et volumineuses du DataFrame
#mais je trouve intéressant de réaliser cet affichage interactif

plt.figure(figsize=(10,15))
unique_provinces = list(set(df["Province"]))
source = ColumnDataSource(data=df)
 
hover = HoverTool(
        tooltips=[
            ("City", "@City"),
            ("Price", "@Price")])

p = figure(width=950, height = 400, title ="Prévision des prix de l'immobilier canadien", 
           x_axis_label = "Province",
           y_axis_label = "Paliers de prix",
           x_range = unique_provinces)
p.circle(x="Province", y="Price", source = source, size = 5, color ="navy", alpha = 0.5)
p.add_tools(hover)


show(p)


# In[157]:


#dans cet histogramme plutôt classique
#je peux observer une tendance plus significative sur le prix de l'immobilier en fonction des ville au Canada
top_city = df.sort_values(by="Price", ascending=False)

plt.figure(figsize=(10,15))

plt.barh(top_city["City"], top_city["Price"], label="Prix (price)", color= "g") 
plt.title("Affichage des prix croissant de l’immobilier au Canada")
plt.xlabel("Prix de l'immobiler")
plt.ylabel("Ville")
plt.legend(title="Légende")

plt.show()


# In[158]:


#ans cet histogramme des Top 5 des prix de l'immobilier canadien
#je peux voir que la province de Barrie est en tête avec le prix le plus élevé et Maple Ridge est à la dernière place du top 
top_city = df.sort_values(by='Price', ascending=False).head(13)


plt.figure(figsize=(10,5))
plt.barh(top_city['City'], top_city['Price'], label="Price") 
plt.legend()
plt.title('Top des prix de l’immobilier au Canada')
plt.show()


# In[159]:


#cette partie a juste été utilisée pour observer les variables que je pourrais supprimer en les affichant 
#j'ai rencontré plusieurs problèmes car premièrement j'avais réalisé l'encodage avant la suppression
#ce qui m'a donné des erreurs puis je me suis dit qu'il vaut mieux supprimer les variables inutiles en premier
print(df.columns)


# In[160]:


df.drop(["Address", "Latitude", "Longitude"], axis= 1, inplace=True)
df.head()


# In[161]:


#afin d'avoir le meilleur résultat possible, j'ai réalisé un encodage pour garantir une homogénéité des données en numérique 
label  = LabelEncoder()
df["City"] = label.fit_transform(df["City"])
df["Province"] = label.fit_transform(df["Province"])
df.head()


# In[162]:


#e vais séparer les colonnes dans la variable X sauf "Price" dans y
#pour isoler les caractéristiques de l'entraînement
X = df.drop("Price", axis=1)
y = df["Price"]


# In[163]:


#maintenant, je vais fractionner le jeu de données
#pour tester la performance de la prédiction
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[164]:


#Ensuite, une standardisation
#pour améliorer les résultats
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[165]:


#et enfin j'effectue une régression linéaire pour réaliser la prédiction
model = LinearRegression()
model.fit(X_train, y_train)


# In[166]:


#afin de confirmer la prédiction je vais calculer l'erreur quadratique moyenne et le coefficient
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(mse)
print(r2)
#'observe une performance MSE beaucoup trop élevée (j'aurais préféré avoir un chiffre plus proche de 0)
#j'observe une performance R² beaucoup trop basse de 24 % (j'aurais préféré avoir un chiffre dans les 90 %)


# In[167]:


#même si je pense que cette prédiction est assez compliquée à identifier, j'ai souhaité réaliser un tableau 
#e peux apercevoir une tendance plutôt linéaire dans l'ensemble 
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--k")

plt.xlabel("Prix actuel")
plt.ylabel("Prix prédit")
plt.title("prix actuel vs Prix prédit")
plt.show()

