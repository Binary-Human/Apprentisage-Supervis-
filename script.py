import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from visualization import vizualise_parameter
import os

scaler = StandardScaler()
df = pd.read_csv("Jeu de données 1 - Californie-20241119/alt_acsincome_ca_features_85(1).csv")
df_values = pd.read_csv("Jeu de données 1 - Californie-20241119/alt_acsincome_ca_labels_85.csv")

#for col in df.columns.tolist() :
    #vizualise_parameter(str(col))

#Decide the train and test sets
x_train,x_test,y_train,y_test = train_test_split(df, 
df_values,test_size = 0.3, shuffle=True)
print("Train set before scaling : ")
print(x_train.head())

x_train_scaled = scaler.fit_transform(x_train)

#Pas de scaler sur le x_test, juste transformer le x_test
x_test_scaled = scaler.transform(x_test)

print("Train set after scaling : ")
print(x_train_scaled)
print("Test set after transformation : ")
print(x_test_scaled)