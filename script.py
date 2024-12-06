import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report, ConfusionMatrixDisplay
from visualization import vizualise_parameter
import joblib
import os

from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.datasets import load_iris

scaler = StandardScaler()
df = pd.read_csv("Jeu de données 1 - Californie-20241119/alt_acsincome_ca_features_85(1).csv")
df_values = pd.read_csv("Jeu de données 1 - Californie-20241119/alt_acsincome_ca_labels_85.csv")

#for col in df.columns.tolist() :
    #vizualise_parameter(str(col))

#Decide the train and test sets (70% train, 30% test)
x_train,x_test,y_train,y_test = train_test_split(df, 
df_values,test_size = 0.3, shuffle=True)
#print("Train set before scaling : ")
#print(x_train.head())

x_train_scaled = scaler.fit_transform(x_train)
# On ne réevalue pas les paramètre de la transformation sur x_test, juste transformer le x_test
x_test_scaled = scaler.transform(x_test)


#print("Train set after scaling : ")
#print(x_train_scaled)
#print("Test set after transformation : ")
#print(x_test_scaled)

#################################################################################################
#   DIFFERENT METHODS :
##################################################################################################

## Random Forest
param_grid_rf = { 
    'n_estimators': [200,500],
    'max_features': ['log2','sqrt'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini','entropy']
}

param_grid_aBoost = { 
    'n_estimators': [10, 50, 100, 500],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0],
    'algorithm' : ['SAMME']
}

def printMetrics(y_test, y_pred):

    #Accuracy : % of correct predictions
    accuracy = accuracy_score(y_test,y_pred)
    #Precision : % of TP over "predicted as positive"
    precision = precision_score(y_test, y_pred)
    #Recall : % of TP over "really positive"
    recall = recall_score(y_test, y_pred)
    print("Accuracy : ", accuracy)
    print("Precision : ", precision)
    print("Recall : ", recall)

    # Create the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix : ")
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

# Standard performance metrics
def standard(model):
    model.fit(x_train_scaled, y_train )
    y_pred_std = model.predict(x_test_scaled)

    print(f"\nMetrics for {model} parameters\n")
    printMetrics(y_test, y_pred_std)

# Uses crossValidation to asses performance
def standardCrossValidation(model):
    # Obtenir les prédictions avec validation croisée
    y_pred = cross_val_predict(model, x_test_scaled, y_test, cv=5)
    # Calculer les scores de validation croisée (accuracy)
    cv_accuracy = cross_val_score(model, x_test_scaled, y_test, cv=5, scoring='accuracy')
    cv_precision = cross_val_score(model, x_test_scaled, y_test, cv=5, scoring='precision')
    cv_recall = cross_val_score(model, x_test_scaled, y_test, cv=5, scoring='recall')

    # Afficher les résultats
    print("\nMetrics for Standard parameters - Cross validation\n")
    print("Accuracy moyenne (validation croisée) :", cv_accuracy.mean())
    print("Précision moyenne (validation croisée) :", cv_precision.mean())
    print("Recall moyen (validation croisée) :", cv_recall.mean())

    print("\nClassification Report :")
    print(classification_report(y_test, y_pred))

    # Create the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix : ")
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

def searchBestRandomForest():
    rf = RandomForestClassifier()
    CV_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv= 5, verbose=4)
    CV_rf.fit(x_train_scaled, y_train)
    best_params = CV_rf.best_params_
    print("Best parameters : ", best_params)

    rf_best = RandomForestClassifier(random_state=None,max_features=best_params["max_features"],
                                        n_estimators=best_params["n_estimators"],max_depth=best_params["max_depth"],
                                        criterion=best_params["criterion"])
    joblib.dump(CV_rf.best_estimator_,'RandomForest_BestModel_05890.joblib')
    rf_best.fit(x_train_scaled,y_train)
    y_pred = rf_best.predict(x_test)

    print("\nMetrics for Random Forest Best parameters\n")
    printMetrics(y_test, y_pred)

def searchBestAdaBoost():
    aBoost = AdaBoostClassifier()
    CV_aB = GridSearchCV(estimator=aBoost, param_grid=param_grid_aBoost, n_jobs=-1, cv=5, verbose=4)
    # scoring='accuracy') ?
    CV_aB.fit(x_train_scaled, y_train )
    best_params = CV_aB.best_params_

    aB_best = AdaBoostClassifier(random_state=None, 
                                 n_estimators=best_params["n_estimators"],
                                 learning_rate=best_params["learning_rate"],
                                 algorithm=best_params["algorithm"])

    joblib.dump(CV_aB.best_estimator_,'AdaBoost_BestModel_XXXXX.joblib')
    aB_best.fit(x_train_scaled,y_train)
    y_pred = aB_best.predict(x_test)

    print("\nMetrics for AdaBoost Best parameters\n")
    printMetrics(y_test, y_pred)

#standard(RandomForestClassifier())
#standardCrossValidation(RandomForestClassifier())

#standard(AdaBoostClassifier())
#standardCrossValidation(AdaBoostClassifier())

searchBestAdaBoost()
searchBestModel()
