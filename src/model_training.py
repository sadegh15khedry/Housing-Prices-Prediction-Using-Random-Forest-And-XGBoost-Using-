from sklearn.ensemble import RandomForestRegressor 


def train_model(x_train, y_train, estimators=100, max_depth=5):
    model = RandomForestRegressor (n_estimators=estimators, random_state=50,  max_depth = max_depth)
    model.fit(x_train, y_train)
    return model

