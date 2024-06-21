from sklearn.ensemble import RandomForestClassifier

def train_model(x_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=50)
    model.fit(x_train, y_train)
    return model