#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 12:45:04 2021

@author: julie
"""
import numpy as np 

# Les 2 fonctions suivantes prennent en argument une liste contenant les notes réelles
# et une liste contenant les prédictions réalisées par l'algorithme

#calcule le coefficient MAE
def MAE(notes, predic) :
    somme = 0
    for x in range(len(predic)) :
        somme += abs(predic[x]-notes[x])
    return somme/len(predic)

#calcule le coefficient RMSE
def RMSE(notes, predic) :
    somme = 0
    for x in range(len(predic)) :
        somme += ((predic[x]-notes[x])**2)
    return np.sqrt(somme/len(predic))


# Quand vous voulez utiliser ces fonctions, vous devez 
#- Mettre vos prédictions dans une liste
#- Récupérer la liste test_set renvoyée par la fonction set_train_test => c'est la liste des notes supprimées
#- Appeler efficacite_test.MAE(...) sur ces listes