from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def classify_cleaned(X, y, test_size=0.2, random_state=42, verbose=True):
    """
    Trains and evaluates multiple classifiers using pre-cleaned X and y.

    Parameters:
    - X: cleaned feature DataFrame
    - y: target Series
    - test_size: float, test split ratio
    - random_state: int, for reproducibility
    - verbose: bool, print metrics to console

    Returns:
    - results: dict of metrics for each model
    - best_model_name: name of the best model by accuracy
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(),
        "SVC": SVC()
    }

    results = {}
    best_model_name = None
    best_accuracy = 0

    if verbose:
        print("\n📊 Evaluating Models...")

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='weighted', zero_division=0)
        rec = recall_score(y_test, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        }

        if verbose:
            print(f"{name:20s} - Acc: {acc:.2f} | Prec: {prec:.2f} | Rec: {rec:.2f} | F1: {f1:.2f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name

    if verbose:
        print(f"\n🏆 Best Model: {best_model_name} with Accuracy = {best_accuracy:.2f}")

    return results, best_model_name
