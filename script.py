import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from visualization import vizualise_parameter
import os

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

#Pas de scaler sur le x_test, juste transformer le x_test
x_test_scaled = scaler.transform(x_test)

#print("Train set after scaling : ")
#print(x_train_scaled)
#print("Test set after transformation : ")
#print(x_test_scaled)

### DIFFERENT METHODS :

## Random Forest
param_grid = { 
    'n_estimators': [200],
    'max_features': ['sqrt'],
    'max_depth' : [8],
    'criterion' :['gini']
}

rf = RandomForestClassifier()

CV_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5)
CV_rf.fit(x_train, y_train)
best_params = CV_rf.best_params_

rf_best = RandomForestClassifier(random_state=None,max_features=best_params["max_features"],
                                    n_estimators=best_params["n_estimators"],max_depth=best_params["max_depth"],
                                    criterion=best_params["criterion"])

rf_best.fit(x_train_scaled,y_train)
y_pred = rf_best.predict(x_test)

# Metrics
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
