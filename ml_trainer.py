# ml_trainer.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_classification_model(df: pd.DataFrame, target_column: str):
    """
    Trains a RandomForestClassifier model and returns performance metrics 
    and a confusion matrix figure.
    """
    # Simple data preparation: one-hot encode categorical features
    X = pd.get_dummies(df.drop(target_column, axis=1), drop_first=True)
    y = df[target_column]

    # Split into training and testing sets with stratification to handle potential class imbalance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Generate the classification report and confusion matrix
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Create the confusion matrix visualization
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=model.classes_, yticklabels=model.classes_)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('Actual Labels')

    return report, fig