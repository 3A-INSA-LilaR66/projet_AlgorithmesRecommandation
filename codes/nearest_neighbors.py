#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 17:30:16 2021

@author: julie
"""
import numpy as np 
import data
import baseline_predictor
import efficacite_test

nb_movies = 9742
nb_users = 610 
nb_ratings = 100836
movie_dict = dict() #va contenir le movieID en clé et son indice en valeur 
R = np.zeros((nb_users,nb_movies),dtype=float) #matrice contenant les ratings
size_test_set = int(0.2*nb_ratings)

#matrice pour stocker les similarités : 
#matrice triangulaire supérieure sans la diagonale
S = np.triu(np.zeros((nb_users,nb_users),dtype=float),k=1) 
moy = list() 
ecart = list() #contient sqrt(somme(R[u,i]-mu)²)
T = 50 #nb requis d'utilisateurs communs

#initialisation matrice des voisins 
N = list()
compteur = 0

#chargement des données 
movie_dict = data.create_movie_dict()
R = data.csv_to_matrix()
R_train, test_set, list_U, list_I = data.set_train_test() 
m = np.sum(R_train)/(nb_ratings-size_test_set) #moyenne

#from baseline predictor
list_bi=list()
list_bu=list()
baseline_predictor.compute_biais(20)


#calcule la moyenne et variance des notes de chaque utilisateur et les stocke dans une liste
#evite des calculs inutiles dans la fonction similarite
def list_moy_ecart() :
    for u in range(nb_users) :
        somme = 0
        notes = np.where(R_train[u,:]>0)[0]
        mu = np.mean(R_train[u,notes]) #moyenne des notes données par l'utilisateur u
        moy.append(mu) 
        
        for i in range(nb_movies) :
            if R_train[u,i] != 0 :
                somme += (R_train[u,i]-mu)**2
        ecart.append(np.sqrt(somme)) #variance des notes données par l'utilisateur u
        
    
# Calcule la similarité entre deux utilisateurs
# On l'utilisera pour calculer une matrice triangulaire supérieure des similarités entre utilisateurs 
# ATTENTION : pour ne pas fausser le résultat, il faut que les deux utilisateurs 
# aient noté quelques films en commun (sinon s'ils ont un seul film en commun et qu'ils
# lui ont attribué la même note alors sim(u,v)=1 mais en réalité ils ne seront pas 
# vraiment similaires et ça faussera les n plus proches voisins) => on introduit un seuil T
# qui indique le nombre de films qu'ils devraient au moins avoir en commun, sinon on met 
# leur similarité à 0

def similarity(u,v) :
    #u1 et v1 sont les sets des indices des cases des lignes u et v respectivement qui sont > 0
    #la longueur de l'intersection de deux sets = nb de films en communs
    u1 = set(np.where(R_train[u,:] >0)[0])
    v1 = set(np.where(R_train[v,:] >0)[0])
    intersection = list(u1 & v1)
    # rq : on met le [0] car la fonction np.where renvoie un tuple dont le premier élément est le np.darray qui
    # nous interesse et le deuxième élément est vide. Et set ne peut pas convertir des tuples
    if len(intersection)<T or ecart[u]==0 or ecart[v]==0:
        sim = 0
    else :
        somme = 0
        for i in intersection : #si i fait parti des items notés par u et v 
            somme += (R_train[u,i] - moy[u])*(R_train[v,i] - moy[v])
        sim = somme/(ecart[u]*ecart[v])
    return sim 

#### PB ! ecart[u] peut être = 0 ! 
#### si pour un utilisateur donné on a supprimé toutes les notes qu'il a donné dans R_train
#### dans ce cas là on a pas d'info sur cet utilisateur donc on renvoie une similarité de 0

def sim_matrix() : 
    for u in range(nb_users-1) :
        for v in range(u+1,nb_users) :
            S[u,v]=similarity(u,v)


# renvoie une liste des plus proches voisins de u 
def neighbors(u) :
    Nu = list() 
    ind = np.argsort(S[u,:])[::-1] #renvoie les indices classés par ordre décroissant
    k = 0
    while S[u,ind[k]]>0 :
        Nu.append(ind[k])
        k += 1 
    return Nu

#on stocke pour chaque u ses voisins (peut servir au cas où on appelle le même voisin plusieurs fois dans le set de test)
def neighbors_matrix() :
    global N
    N=[neighbors(u) for u in range(nb_users)]
    return N

### PB POUR LA FONCTION SCORE : si parmi les voisins de u, aucun n'a noté l'item i que faire ?? 
### le set Nui est vide => somme2=0 => on a une division par 0
### comment faire une prédiction dans ce cas ? est ce qu'on renvoie simplement la moyenne de l'utilisateur ? 
### est ce que pour ces "trous" de nearest neighbours on appelle le baseline predictor pour faire la prediction à sa place ? 
### est ce que qd on a un jeu de donné si petit c'est pertinent de regarder Nui plutot que Nu ?? 

def score(u,i) : #N = neighbors_matrix()
    global compteur
    Nui = list()
    taille_Nui_max = 24
    k = 0
    j = 0
    #tant qu'on a pas atteint la taille maximale pour le set Nui et qu'on n'a pas fini de parourir la liste des voisins
    while j < taille_Nui_max and k < len(N[u]) :
        v = N[u][k]
        if R_train[v,i]>0: #si l'utilisateur v a noté l'item i c'est un plus proche voisin
            Nui.append(v)
            j+=1
        k += 1
    somme1 = 0
    somme2 = 0
    if len(Nui) == 0:
        compteur = compteur +1
    for v in Nui :
        somme1 += S[u,v]*(R_train[v,i]-moy[v])
        somme2 += abs(S[u,v])
    if len(Nui) == 0:
        return baseline_predictor.score(u,i)
    else :
        return (moy[u]+(somme1/somme2))
    
    
def test_score() :
    global S
    list_moy_ecart()
    sim_matrix()
    S = S + S.T #rend S symétrique => pour pouvoir accéder à ses indices de manière équivalente
    neighbors_matrix() #calcule N
    
    predic = list()
    for x in range(len(list_U)) :
        score_x = score(list_U[x],list_I[x])
        if score_x > 5 :
            score_x = 5 
        if score_x < 0.5 :
            score_x = 0.5
        predic.append(score_x)
    print("MAE = ", efficacite_test.MAE(test_set, predic))
    print("RMSE = ", efficacite_test.RMSE(test_set, predic))

baseline_predictor.test_score()
test_score()
print(compteur)