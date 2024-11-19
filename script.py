import pandas as pd
import numpy as np
from visualization import vizualise_parameter
import os

df = pd.read_csv("Jeu de données 1 - Californie-20241119/alt_acsincome_ca_features_85(1).csv")

for col in df.columns.tolist() :
    vizualise_parameter(str(col))

