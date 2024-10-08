### Bayesian Optimization of Hyperparameter Tuning for XGBoost ###

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, log_loss, brier_score_loss,
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, max_error, mean_absolute_percentage_error,
    matthews_corrcoef, cohen_kappa_score, jaccard_score, hamming_loss
)
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from sklearn.preprocessing import OneHotEncoder

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

# Create preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine data transformers into single preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ], remainder='drop'  # Drop columns not specified, such as IDs
)

# Create XGBoost model
seed = np.random.seed(123)
xgb_model = XGBClassifier(random_state=seed, eval_metric='logloss')

# Create pipeline
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', xgb_model)
])

# Define parameter space for Bayesian optimization
param_space = {
    'classifier__n_estimators': Integer(100, 4000),
    'classifier__max_depth': Integer(1, 40),
    'classifier__min_child_weight': Integer(1, 20),
    'classifier__learning_rate': Real(0.01, 1.0, prior='log-uniform')
}

# Print the parameter space
print("Parameter Space:")
for param, space in param_space.items():
    print(f"{param}: {space}")

# Perform Bayesian optimization
bayes_search = BayesSearchCV(
    xgb_pipeline,
    param_space,
    n_iter=50,  # number of parameter settings that are sampled
    cv=kfold,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=seed,
    verbose=1
)

# Fit to training dat
bayes_search.fit(X_train, y_train)

# Print results
print("\nBest parameters:", bayes_search.best_params_)
print("Best ROC AUC score:", round(bayes_search.best_score_, 3))

# Results
results = pd.DataFrame(bayes_search.cv_results_)
results = results.sort_values('rank_test_score')
print("\nTop 10 results:")
print(results[['params', 'mean_test_score', 'std_test_score']].head(10).round(3))

# Best Model
best_model = bayes_search.best_estimator_

# Test Predictions
y_pred = best_model.predict(X_test)
test_roc_auc = roc_auc_score(y_test, y_pred)
print(f"\nTest set ROC AUC: {test_roc_auc:.3f}")

# Feature Importance
feature_importance = best_model.named_steps['classifier'].feature_importances_
feature_names = (numeric_features.tolist() + 
                 best_model.named_steps['preprocessor']
                 .named_transformers_['cat']
                 .named_steps['onehot']
                 .get_feature_names_out(categorical_features).tolist())

importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
importance_df = importance_df.sort_values('importance', ascending=False)
print("\nTop 10 important features:")
print(importance_df.head(10))



## Evaluate Model ##
def evaluate_xgboost_model(best_model, X_test, y_test):
    """
    Evaluate the XGBoost model using various metrics.
    
    :param best_model: Trained XGBoost model (from BayesSearchCV)
    :param X_test: Test features
    :param y_test: True labels
    :return: Dictionary of all metrics
    """
    # Make predictions
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]  # Assuming binary classification

    metrics = {}

    # Classification metrics
    metrics['Accuracy'] = accuracy_score(y_test, y_pred)
    metrics['Precision'] = precision_score(y_test, y_pred, average='weighted')
    metrics['Recall'] = recall_score(y_test, y_pred, average='weighted')
    metrics['F1 Score'] = f1_score(y_test, y_pred, average='weighted')
    metrics['ROC AUC'] = roc_auc_score(y_test, y_prob)
    metrics['Log Loss'] = log_loss(y_test, y_prob)
    metrics['Brier Score'] = brier_score_loss(y_test, y_prob)
    metrics['Matthews Correlation Coefficient'] = matthews_corrcoef(y_test, y_pred)
    metrics['Cohen\'s Kappa'] = cohen_kappa_score(y_test, y_pred)
    metrics['Jaccard Score'] = jaccard_score(y_test, y_pred, average='weighted')
    metrics['Hamming Loss'] = hamming_loss(y_test, y_pred)

    # Regression-like metrics (can be used for probabilistic outputs)
    metrics['Mean Squared Error'] = mean_squared_error(y_test, y_prob)
    metrics['Root Mean Squared Error'] = np.sqrt(metrics['Mean Squared Error'])
    metrics['Mean Absolute Error'] = mean_absolute_error(y_test, y_prob)
    metrics['R-squared'] = r2_score(y_test, y_prob)
    metrics['Explained Variance Score'] = explained_variance_score(y_test, y_prob)
    metrics['Max Error'] = max_error(y_test, y_prob)
    
    # Additional custom metrics
    metrics['Mean Bias Error'] = np.mean(y_prob - y_test)
    metrics['Median Absolute Error'] = np.median(np.abs(y_prob - y_test))

    # Confusion matrix derived metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics['True Positives'] = tp
    metrics['True Negatives'] = tn
    metrics['False Positives'] = fp
    metrics['False Negatives'] = fn
    metrics['Sensitivity (True Positive Rate)'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['Specificity (True Negative Rate)'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['Fall-out (False Positive Rate)'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['False Negative Rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return metrics

# Pull eval results
best_model = bayes_search.best_estimator_
metrics = evaluate_xgboost_model(best_model, X_test, y_test)
for metric, value in metrics.items():
     print(f"{metric}: {value:.4f}")

# Get results from bayes search
def create_cv_results_chart(bayes_search, top_n=10):
    # Get the results DataFrame
    results = pd.DataFrame(bayes_search.cv_results_)
    
    results = results.sort_values('rank_test_score')
    
    top_results = results.head(top_n)
    
    param_names = list(bayes_search.best_params_.keys())
    
    chart_data = pd.DataFrame()
    chart_data['iter'] = top_results['rank_test_score']
    chart_data['target'] = top_results['mean_test_score'].round(4)
    
    for param in param_names:
        chart_data[param] = top_results['params'].apply(lambda x: x[param])
    
    # Convert to a list of lists for tabulate
    table_data = chart_data.values.tolist()
    
    headers = ['iter', 'mean_roc-auc'] + param_names
    table = tabulate(table_data, headers=headers, tablefmt='pipe', floatfmt='.4f')
    
    # Add the best result information
    best_params_str = '; '.join([f"{k}={v:.4f}" for k, v in bayes_search.best_params_.items()])
    best_score = bayes_search.best_score_
    table += f"\nBest result: {{{best_params_str}}}; f(x) = {best_score:.4f}."
    
    return table

# Usage:
chart = create_cv_results_chart(bayes_search)
print(chart)
