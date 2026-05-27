import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

# loading dataset
df = pd.read_csv("data/heart.csv")

print(df.head())

# checking null values
print(df.isnull().sum())

# correlation matrix
plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")

plt.show()

# selecting input and output columns
X = df.drop("target", axis=1)

y = df["target"]

# splitting dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# hyperparameter tuning
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3,5,7,9],
    "min_samples_split": [2,5,10]
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(),
    param_grid,
    cv=5
)

grid_search.fit(X_train, y_train)

# best model
model = grid_search.best_estimator_

# predictions
y_pred = model.predict(X_test)

# accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy :", accuracy)

print("Best Parameters :")

print(grid_search.best_params_)

# saving model
pickle.dump(
    model,
    open("models/dt_model.pkl", "wb")
)

print("Model Saved Successfully")