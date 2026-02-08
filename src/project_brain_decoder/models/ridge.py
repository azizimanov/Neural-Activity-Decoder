from sklearn.linear_model import Ridge

def get_ridge(alpha=1.0):
    ridge = Ridge(alpha=alpha)
    return ridge