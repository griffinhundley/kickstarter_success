from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

def tts(df):
    X = df.drop(columns = 'state_successful')
    y = df.state_successful
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=69
    )
    data = (X_train, X_test, y_train, y_test)
    return data

def fitmodel(model, data):
    X_train = data[0]
    X_test = data[1]
    y_train = data[2]
    y_test = data[3]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = round(f1_score(y_test, y_pred), 3)
    acc = round(accuracy_score(y_test, y_pred),3)
    print(f'F1:  {f1}')
    print(f'Acc: {acc}')
    return model

def ft_importance(model,cols):
    return pd.DataFrame(
        model.feature_importances_,
        index = cols).sort_values(by= 0,
                                       ascending = False).head(10)