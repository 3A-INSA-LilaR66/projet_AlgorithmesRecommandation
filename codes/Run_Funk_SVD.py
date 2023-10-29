# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 00:24:04 2021

@author: Lila
"""
### importation des librairies nécessaires
#***********************************************
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import pandas as pd

### importer la bonne version de data si on veut supprimer les users qui n'ont rien noté et items sans note
import data3 as data 
#import data

import Funk_SVD_movielens_2 as Funk_SVD
import efficacite_test

#***********************************************
# Récupération et mise en forme des données
#***********************************************
data.create_movie_dict() # retourne le dictionnaire movie_dict 
                         # contient le movieID en clé et son indice dans le fichier en valeur 
data.csv_to_matrix() # charge les données et les met dans la matrice R

#R_train, test_set, list_U, list_I = data.set_train_test()
R_train, test_set, list_U, list_I, size_test_set, nb_ratings = data.set_train_test()


"""
- R_train  : matrice R pour laquelle on a enlevé des notations afin verifier que notre algo les retrouve bien.
- test_set : liste des valeurs R[i,j] enlevées de R (ie valeurs manquantes de R_train)
- list_U   : liste des indices des utilisateurs enlevés de R
- list_I   : liste des indices des items enlevés de R
"""

#***********************************************
# Détermination de K
#***********************************************
# on choisit de prendre K comme étant le nombre de valeurs singulières dominantes de R_train
#-----------------------------------------------

u, s, vh  = npl.svd(R_train,full_matrices=False) #reduced SVD
x = np.arange(1,min(np.shape(R_train))+1)

fig = plt.figure(0)
plt.figure(figsize=((16,4)))
plt.plot(x,s)
plt.title(u"Valeurs singulières de la matrice des notations")
plt.xlabel("composantes")
plt.ylabel("valeur singulière")
plt.xticks(np.arange(0,min(np.shape(R_train))+1, 20))
plt.grid()
plt.savefig('Movielens_singularValue.png', bbox_inches='tight', dpi = 300, format = 'png') 
plt.show #à mettre après le savefig !! 

# il semblerait que rien ne sert de garder au delà de 200 valeurs singulières. 

"""
# affinons notre recherche en traçant le graphe sur [0,50]
x = np.arange(1,101,1)

fig = plt.figure(1)
plt.figure(figsize=((16,4)))
plt.plot(x,s[0:100])
plt.title(u"Valeurs singulières de la matrice des notations")
plt.xlabel("composante")
plt.ylabel("valeur singulière")
plt.xticks(np.arange(0,100, 5))
plt.grid()
plt.show
"""
# d'après le graphique, nous observons une cassure à environ 200. On garde donc K = 200. 

#***********************************************
# Calcul de la Funk_SVD
#***********************************************
# la litterature affirme que l'algo converge entre 1 et 10 itérations
R_predicted, training_process = Funk_SVD.compute_Funk_SVD(R_train, K=200, alpha=0.095, beta=0.02, nb_iterations=10)


#***********************************************
# Tracé de l'erreur à chaque étape pour vérifier qu'elle diminue et donc que l'algo converge bien.
#***********************************************
x = [x for x, y in training_process]
y = [y for x, y in training_process]
plt.plot(x, y)
plt.xticks(x, x)
plt.xlabel("Iterations")
plt.ylabel("Erreur totale")
plt.grid(axis="y")


#***********************************************
# Détermination du meilleur learning rate (ie. du meilleur alpha)
#***********************************************
# J'ai pu remarquer qu'un bon learning rate est de l'ordre de 0.001
# faisons tourner notre programme avec différents learnings rate et traçons l'ereur à chaque itération
nb_iterations = 10 #Une fois qu'on aura le bon learning rate, on pourra mettre + d'itérations 
#learning_rates = np.linspace(0.09,0.2,11)
learning_rates = [0.09,0.095,0.01,0.015,0.020,0.025]
all_training_process = np.zeros((nb_iterations,len(learning_rates))) 
x = np.arange(0,nb_iterations) 

for i in range(0,len(learning_rates)):
    R_predicted2,training_process2 = Funk_SVD.compute_Funk_SVD(R_train, K=200, alpha=learning_rates[i], beta=0.02, nb_iterations=nb_iterations)
    all_training_process[:,i] = [y for x, y in training_process2]
    
plt.figure(1) 
plt.figure(figsize=((10,4)))
ax = plt.subplot(111)
for i in range(0,len(learning_rates)):
    ax.plot(x,all_training_process[:,i], label = str(learning_rates[i]))

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# add labels and title
plt.xlabel("Iterations")
plt.ylabel("Erreur totale")
plt.xticks(x, x)
plt.savefig('Movielens_bestLearningRate.png', bbox_inches='tight', dpi = 300, format = 'png') 
plt.show()

# exportation de all_training_processes en fichier excel pour mieux visualiser le meilleur alpha
columns = list(map(str,np.round(learning_rates, decimals=4))) 
df = pd.DataFrame( all_training_process,columns=columns)
df.to_excel("Movielens_all_training_process.xlsx")


# on trouve que le meilleur alpha est 0.095 avec K = 200
# on peut maintenant modifier le nombre d'itérations


#***********************************************
# Calcul de l'efficacité de l'algorithme
#***********************************************
# On met nos prédictions dans une liste
list_predic = R_predicted[list_U,list_I]
# Liste des notes supprimées : test_set
# Appel de efficacite_test.MAE() sur test_set et list_predic
print()
print ("erreur MAE de l'algorithme Funk_SVD: ")
print(efficacite_test.MAE(test_set,list_predic))
print()
print ("erreur RMSE de l'algorithme Funk_SVD: ")
print(efficacite_test.RMSE(test_set,list_predic))

"""
RESULTATS

Base de données Movielens
----------------------------
Si on ne supprime pas les utilisateurs qui n'ont rien noté et les items qui n'ont pas de note :
- paramètres : K=200, alpha=0.095, beta=0.02, nb_iterations=10  => donnent les meilleurs résultats
- MAE = 0.7067538806953435; RMSE = 0.9267258032158023
graphiques et excel obtenus: 
    - Movielens_singularValue.png
    - Movielens_bestLearningRate.png
    - Movielens_all_training_process.xlsx

Si on supprime les utilisateurs qui n'ont rien noté et les items qui n'ont pas de note :
- paramètres : K=200, alpha=0.095, beta=0.02, nb_iterations=10  => donnent les meilleurs résultats
- MAE = 0.6619731842010032; RMSE = 0.8660061247617047

"""