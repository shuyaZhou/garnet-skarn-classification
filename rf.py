import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import os
import joblib
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

# The calling path must match
data_path = '.csv'
scaler_path = ".pkl"
output_model_path = '.pkl'
output_training_process_path = '.pkl'

data = np.genfromtxt(data_path, delimiter=',', skip_header=1)
features = data[:, 1:9]  # Corresponding features
labels = data[:, 10].astype(int)  # Corresponding deposit types
feature_names = ["SIO2", "TIO2", "AL2O3", "FEOT", "MNO", "MGO", "CAO","CR2O3"]
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler saved to {scaler_path}")

os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
if os.path.exists(output_model_path):
    print(f"Model found at {output_model_path}, loading the model...")
    best_model = joblib.load(output_model_path)
    if os.path.exists(output_training_process_path):
        print(f"Training process found at {output_training_process_path}, loading training details...")
        with open(output_training_process_path, 'rb') as f:
            training_details = pickle.load(f)
    else:
        training_details = None
else:
    rf_model = RandomForestClassifier(random_state=42)

    parameters = {
        'n_estimators': [150,200,250,300,400],
        'max_depth': [9,11,13,15,17],
        'min_samples_split': [5,10]
    }
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=parameters,
        scoring=['accuracy','f1', 'roc_auc'],
        cv=5,
        refit='roc_auc',
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    joblib.dump(best_model, output_model_path)
    training_details = {
        'cv_results': grid_search.cv_results_,
        'best_params': grid_search.best_params_
    }
    with open(output_training_process_path, 'wb') as f:
        pickle.dump(training_details, f)
    print(f"Best model saved to {output_model_path}")
    print(f"Training process saved to {output_training_process_path}")

    best_params = grid_search.best_params_
    cv_results = grid_search.cv_results_
    best_param_index = cv_results['params'].index(best_params)
    
    for fold_idx in range(5):
        test_score_accuracy = cv_results[f'split{fold_idx}_test_accuracy'][best_param_index]
        test_score_f1 = cv_results[f'split{fold_idx}_test_f1'][best_param_index]
        test_score_auc = cv_results[f'split{fold_idx}_test_roc_auc'][best_param_index]
        
        print(f"Fold {fold_idx + 1} - Accuracy: {test_score_accuracy:.3f},"
            f"F1-score: {test_score_f1:.3f}"
            f"Auc: {test_score_auc:.3f},")

y_pred_test = best_model.predict(X_test)

accuracy_test = accuracy_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)
y_test_proba = best_model.predict_proba(X_test)[:, 1] 
auc_test = roc_auc_score(y_test, y_test_proba)

print("Test Set Metrics:")
print(f"Accuracy: {accuracy_test:.3f}")
print(f"F1-score: {f1_test:.3f}")
print(f"AUC: {auc_test:.3f}")
