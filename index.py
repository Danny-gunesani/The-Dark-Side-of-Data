import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")

file_path = "titanic.csv"
titanic_df = pd.read_csv(file_path)

print("Initial Data Overview:")
print(titanic_df.head())
print(titanic_df.info())

print("\nMissing Values Before Cleaning:")
print(titanic_df.isnull().sum())

titanic_df["Age"].fillna(titanic_df["Age"].median(), inplace=True)
titanic_df["Embarked"].fillna(titanic_df["Embarked"].mode()[0], inplace=True)
titanic_df["Fare"].fillna(titanic_df["Fare"].median(), inplace=True)
titanic_df.drop(columns=["Cabin"], inplace=True)

print("\nMissing Values After Cleaning:")
print(titanic_df.isnull().sum())

titanic_df.to_csv("titanic_cleaned.csv", index=False)

plt.figure(figsize=(6, 4))
sns.countplot(x="Survived", data=titanic_df, palette="coolwarm")
plt.xlabel("Survival Status (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.title("Survival Count")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(titanic_df["Age"], bins=30, kde=True, color="blue")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Distribution of Titanic Passengers")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x="Pclass", data=titanic_df, palette="viridis")
plt.xlabel("Passenger Class (1 = First, 2 = Second, 3 = Third)")
plt.ylabel("Count")
plt.title("Passenger Class Distribution")
plt.show()

plt.figure(figsize=(6, 4))
sns.barplot(x="Sex", y="Survived", data=titanic_df, palette="coolwarm")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")
plt.title("Survival Rate by Gender")
plt.show()
