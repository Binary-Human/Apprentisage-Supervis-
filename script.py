import pandas as pd
import numpy as np
import os

df = pd.read_csv("Jeu de données 1 - Californie-20241119/alt_acsincome_ca_features_85(1).csv")

# 1. Basic Summary Statistics
print(df.describe())
