from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
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
    return model

def get_scores(model, data):
    X_train = data[0]
    X_test = data[1]
    y_train = data[2]
    y_test = data[3]
    
    test_y_pred = model.predict(X_test)
    test_f1 = round(f1_score(y_test, test_y_pred), 3)
    test_acc = round(accuracy_score(y_test, test_y_pred),3)
    test_rec = round(recall_score(y_test, test_y_pred),3)
    test_prec = round(precision_score(y_test, test_y_pred),3)
    
    train_y_pred = model.predict(X_train)
    train_f1 = round(f1_score(y_train, train_y_pred), 3)
    train_acc = round(accuracy_score(y_train, train_y_pred),3)
    train_rec = round(recall_score(y_train, train_y_pred),3)
    train_prec = round(precision_score(y_train, train_y_pred),3)
    
    print(f'Test F1:  {test_f1}         Train F1: {train_f1}')
    print(f'Test Acc: {test_acc}         Train Acc: {train_acc}')
    print(f'Test Recall: {test_rec}      Train Recall: {train_rec}')
    print(f'Test Precision: {test_prec}   Train Precision: {train_prec}')
    return


def ft_importance(model,cols):
    df = pd.DataFrame(
        model.feature_importances_,
        index = cols).sort_values(by= 0,
                                       ascending = False).head(10)
    df.rename(columns = {0: 'Importance'})
    return df

# roc curve and auc
import seaborn as sns
sns.set_style('darkgrid')
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
def generate_ROC(data, md1, md2, md3, title):
    # generate 2 class dataset
    fig, ax = plt.subplots(figsize = (8,8))
    # split into train/test sets
    trainX, testX, trainy, testy = data
    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(testy))]
    # fit a model
    # predict probabilities
    md1_probs = md1.predict_proba(testX)
    md2_probs = md2.predict_proba(testX)
    md3_probs = md3.predict_proba(testX)
    # keep probabilities for the positive outcome only
    md1_probs = md1_probs[:, 1]
    md2_probs = md2_probs[:, 1]
    md3_probs = md3_probs[:, 1]
    # calculate scores
    ns_auc = roc_auc_score(testy, ns_probs)
    md1_auc = roc_auc_score(testy, md1_probs)
    md2_auc = roc_auc_score(testy, md2_probs)
    md3_auc = roc_auc_score(testy, md3_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (md1_auc))
    print('Random Forest: ROC AUC=%.3f' % (md2_auc))
    print('XGBoost: ROC AUC=%.3f' % (md3_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
    md1_fpr, md1_tpr, _ = roc_curve(testy, md1_probs)
    md2_fpr, md2_tpr, _ = roc_curve(testy, md2_probs)
    md3_fpr, md3_tpr, _ = roc_curve(testy, md3_probs)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(md1_fpr, md1_tpr, marker='.', label='Logistic')
    plt.plot(md2_fpr, md2_tpr, marker='.', label='Forest')
    plt.plot(md3_fpr, md3_tpr, marker='.', label='XGBoost')
    # axis labels
    ax.set_xlabel('False Positive Rate', fontsize = 14)
    ax.set_ylabel('True Positive Rate', fontsize = 14)
    ax.set_title(f'{title}', fontsize = 20)
    # show the legend
    ax.legend();