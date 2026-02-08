from sklearn.linear_model import Ridge

def get_ridge(alpha):
    ridge = Ridge(alpha=alpha)
    return ridge