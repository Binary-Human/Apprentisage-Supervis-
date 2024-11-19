
import matplotlib.pyplot as plt 
import pandas as pd

def vizualise_parameter(PARAMETER) :
    x = [] 
    y = [] 
    columns = [PARAMETER]

    df = pd.read_csv("Jeu de données 1 - Californie-20241119/alt_acsincome_ca_features_85(1).csv", usecols=columns)
    plt.hist(df, color = 'lightgreen', ec='black', bins=20, label = PARAMETER) 
    plt.xlabel('Distribution of '+PARAMETER) 
    plt.legend() 
    plt.show() 