def error_empi(y, y_predict):
    n = len(y_predict)
    erreur = 0
    
    for i in range(n):
        if y[i] != y_predict[i]:
            erreur += 1
    return erreur/n