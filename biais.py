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
from sklearn.inspection import permutation_importance
import seaborn as sns


scaler = StandardScaler()
df = pd.read_csv("Jeu de données 1 - Californie-20241119/alt_acsincome_ca_features_85(1).csv")
df_values = pd.read_csv("Jeu de données 1 - Californie-20241119/alt_acsincome_ca_labels_85.csv")

x_train,x_test,y_train,y_test = train_test_split(df, 
df_values,test_size = 0.3, shuffle=True)

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


#  Analyse des corrélations entre attributs et labels
def analyze_correlations(df, labels, model=None):
    print("Initial Correlations between attributes and labels:")
    correlations = df.corrwith(labels.squeeze(), method='pearson')
    print(correlations)
    print("\nTop correlated features with the label:")
    print(correlations.sort_values(ascending=False).head(10))

    if model:
        print("\nCorrelations as produced by the model's feature importance (if available):")
        try:
            feature_importances = model.feature_importances_
            model_correlations = pd.Series(feature_importances, index=df.columns)
            print(model_correlations.sort_values(ascending=False).head(10))
        except AttributeError:
            print("Model does not provide feature importances directly.")
    
    # Plot correlations
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.show()

# Classement des attributs par importance
def feature_importance_analysis(model, x_train_scaled, df):
    print("\nFeature Importance Analysis:")
    try:
        # Feature importance specific to the model
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': df.columns,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            print(feature_importance_df)
        
        # Permutation importance
        print("\nPermutation Importance (Generic Method):")
        perm_importance = permutation_importance(model, x_train_scaled, y_train, n_repeats=10, random_state=42)
        perm_importance_df = pd.DataFrame({
            'Feature': df.columns,
            'Importance': perm_importance.importances_mean
        }).sort_values(by='Importance', ascending=False)
        print(perm_importance_df)

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
        plt.title("Feature Importance from Model")
        plt.gca().invert_yaxis()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.barh(perm_importance_df['Feature'], perm_importance_df['Importance'])
        plt.title("Permutation Importance")
        plt.gca().invert_yaxis()
        plt.show()

    except Exception as e:
        print("Error computing feature importance:", e)

# 3. Inférence manuelle
def manual_inference(model, x_test_scaled, df, sample_size=5):
    print("\nManual Inference on Test Data:")
    sample_indices = np.random.choice(len(x_test_scaled), sample_size, replace=False)
    for idx in sample_indices:
        data = x_test_scaled[idx]
        prediction = model.predict([data])[0]
        actual = y_test.iloc[idx]
        feature_values = df.iloc[idx]
        print(f"\nSample #{idx}:")
        print(f"Feature Values: {feature_values.to_dict()}")
        print(f"Predicted: {prediction}, Actual: {actual}")

# Example usage with standard RandomForest or AdaBoost
# TODO : Replicate with fine-tuned models
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(x_train_scaled, y_train)

analyze_correlations(df, df_values, random_forest_model)
feature_importance_analysis(random_forest_model, x_train_scaled, df)
manual_inference(random_forest_model, x_test_scaled, df)
