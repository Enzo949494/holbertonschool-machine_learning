#!/usr/bin/env python3
"""
Module contenant la fonction poly_integral.

Cette fonction calcule l'intégrale d'un polynôme défini
par une liste de coefficients, avec constante d'intégration.
"""


def poly_integral(poly, C=0):
    """
    Calcule l'intégrale d'un polynôme donné sous forme de liste de
    coefficients.

    Arguments :
        poly : list
            Liste des coefficients du polynôme, où poly[i] correspond
            au coefficient du terme x**i.
        C : int ou float
            Constante d'intégration (ajoutée au coefficient de degré 0).

    Retour :
        list :
            Nouvelle liste des coefficients représentant l'intégrale du
            polynôme.
            Les coefficients entiers sont renvoyés comme int, sinon en 
            float.
            La liste est réduite au minimum (pas de zéros inutiles en 
            fin).

        None :
            Si poly ou C ne sont pas valides.
    """
    # Vérification des types
    if (not isinstance(poly, list) or
            not isinstance(C, (int, float))):
        return None
    
    # Vérification que poly n'est pas vide
    if not poly:
        return None
    
    # Vérification que tous les éléments de poly sont int ou float
    if not all(isinstance(c, (int, float)) for c in poly):
        return None

    # Liste résultante commençant par la constante
    res = [int(C) if isinstance(C, int) or C.is_integer() else C]

    # Intégration terme par terme
    for i, coeff in enumerate(poly):
        val = coeff / (i + 1)
        if isinstance(val, float) and val.is_integer():
            val = int(val)
        res.append(val)

    # Retirer les zéros de fin inutiles (mais garder au moins un élément)
    while len(res) > 1 and res[-1] == 0:
        res.pop()

    return res
