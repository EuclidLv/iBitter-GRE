import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import warnings

np.random.seed(45)
warnings.filterwarnings('ignore')

def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path).sample(frac=1, random_state=45).reset_index(drop=True)
    test_data = pd.read_csv(test_path).sample(frac=1, random_state=45).reset_index(drop=True)
    X_train = train_data.iloc[:, 1:].values
    y_train = train_data.iloc[:, 0].values
    X_test = test_data.iloc[:, 1:].values
    y_test = test_data.iloc[:, 0].values
    return X_train, y_train, X_test, y_test

def evaluate_model(clf, X_test, y_test, name):
    y_test_pred = clf.predict(X_test)
    y_test_pred_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

    cm = confusion_matrix(y_test, y_test_pred)
    acc = accuracy_score(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    mcc = matthews_corrcoef(y_test, y_test_pred)

    print(f"{name} Test set metrics:\n"
          f"Accuracy: {acc:.4f}\n"
          f"Sensitivity (Recall): {sn:.4f}\n"
          f"Specificity: {sp:.4f}\n"
          f"MCC: {mcc:.4f}\n")

    if y_test_pred_proba is not None:
        auc_score = roc_auc_score(y_test, y_test_pred_proba)
        print(f"AUC: {auc_score:.4f}\n")

        fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba, drop_intermediate=False)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

def main():
    X_train, y_train, X_test, y_test = load_data('', '')                     #you should use ur own path

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    base_classifiers = [
        ('extra_trees', ExtraTreesClassifier(n_jobs=-1, bootstrap=False, max_features='sqrt', n_estimators=150, verbose=1, random_state=45)),
        ('random_forest', RandomForestClassifier(n_jobs=-1, n_estimators=200, random_state=45, bootstrap=False, max_features='sqrt')),
        ('Gradient Boosting', GradientBoostingClassifier(learning_rate=0.05, loss='exponential', max_features='sqrt',
                                                         n_estimators=300, subsample=0.5, verbose=1, random_state=37)),
    ]

    meta_classifier = LogisticRegression()

    sclf = StackingClassifier(estimators=base_classifiers, final_estimator=meta_classifier, n_jobs=-1)
    sclf.fit(X_train_scaled, y_train)

    evaluate_model(sclf, X_test_scaled, y_test, "StackingClassifier")

if __name__ == "__main__":
    main()




