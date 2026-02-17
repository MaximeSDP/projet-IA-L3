def fitFromHisto(X, y, algo):
    model = algo
    model.fit(X, y)
    return model