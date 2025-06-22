import os
import sys
import json
import time
import click
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, log_loss, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_heart_disease_data():
    """
    Load heart disease dataset from preprocessed files
    """
    print("📊 Loading heart disease dataset...")
    
    # Try different possible locations for the data
    data_files = [
        'heart_processed.csv',
        'heart_preprocessing/heart_processed.csv',
        '../heart_preprocessing/heart_processed.csv',
        'data/heart_processed.csv'
    ]
    
    df = None
    for data_file in data_files:
        if os.path.exists(data_file):
            df = pd.read_csv(data_file)
            print(f"✅ Data loaded from: {data_file}")
            break
    
    if df is None:
        print("❌ Preprocessed data not found. Generating sample data...")
        # Generate sample data for demonstration
        np.random.seed(42)
        n_samples = 1000
        
        # Generate features similar to heart disease dataset
        data = {
            'age': np.random.normal(54, 9, n_samples),
            'sex': np.random.binomial(1, 0.7, n_samples),
            'cp': np.random.randint(0, 4, n_samples),
            'trestbps': np.random.normal(131, 17, n_samples),
            'chol': np.random.normal(246, 51, n_samples),
            'fbs': np.random.binomial(1, 0.15, n_samples),
            'restecg': np.random.randint(0, 3, n_samples),
            'thalach': np.random.normal(149, 22, n_samples),
            'exang': np.random.binomial(1, 0.33, n_samples),
            'oldpeak': np.random.exponential(1, n_samples),
            'slope': np.random.randint(0, 3, n_samples),
            'ca': np.random.randint(0, 4, n_samples),
            'thal': np.random.randint(0, 4, n_samples),
        }
        
        # Generate target with some correlation
        features_sum = (data['age'] > 55).astype(int) + \
                      data['sex'] + \
                      (data['cp'] > 2).astype(int) + \
                      (data['thalach'] < 140).astype(int)
        
        target_prob = 1 / (1 + np.exp(-(features_sum - 2)))
        data['target'] = np.random.binomial(1, target_prob, n_samples)
        
        df = pd.DataFrame(data)
        print("✅ Sample dataset generated")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")
    
    return df

def calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate comprehensive metrics for model evaluation
    """
    metrics = {}
    
    # Standard classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    
    # Matthews Correlation Coefficient
    metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
    
    # Confusion matrix based metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Specificity (True Negative Rate)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Negative Predictive Value
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # False Positive Rate
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # False Discovery Rate
    metrics['fdr'] = fp / (fp + tp) if (fp + tp) > 0 else 0
    
    # Probability-based metrics (if available)
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba[:, 1])
        metrics['log_loss'] = log_loss(y_true, y_pred_proba)
    
    return metrics

def create_model_visualizations(model, model_name, X_test, y_test, y_pred, y_pred_proba=None, save_artifacts=True):
    """
    Create and save model visualizations
    """
    if not save_artifacts:
        return
        
    try:
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        cm_filename = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(cm_filename)
        plt.close()
        
        # 2. ROC Curve (if probabilities available)
        if y_pred_proba is not None:
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
            
            plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            roc_filename = f'roc_curve_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(roc_filename, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(roc_filename)
            plt.close()
        
        # 3. Feature Importance (if available)
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            feature_names = [f'Feature_{i}' for i in range(len(model.feature_importances_))]
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True).tail(15)
            
            plt.barh(importance_df['feature'], importance_df['importance'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top 15 Feature Importances - {model_name}')
            plt.tight_layout()
            
            fi_filename = f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(fi_filename, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(fi_filename)
            plt.close()
            
            # Save feature importance as CSV
            fi_csv_filename = f'feature_importance_{model_name.lower().replace(" ", "_")}.csv'
            importance_df.to_csv(fi_csv_filename, index=False)
            mlflow.log_artifact(fi_csv_filename)
            
    except Exception as e:
        print(f"⚠️  Error creating visualizations for {model_name}: {str(e)}")

def train_model(model, model_name, X_train, X_test, y_train, y_test, save_artifacts=True):
    """
    Train individual model and log comprehensive metrics
    """
    print(f"\n🔄 Training {model_name}...")
    
    start_time = time.time()
    
    # Train model
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Get probabilities if available
    y_pred_proba = None
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)
    
    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)
    metrics['training_time'] = training_time
    
    # Cross-validation scores
    try:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        metrics['cv_mean_accuracy'] = cv_scores.mean()
        metrics['cv_std_accuracy'] = cv_scores.std()
    except Exception as e:
        print(f"⚠️  CV error for {model_name}: {str(e)}")
        metrics['cv_mean_accuracy'] = 0
        metrics['cv_std_accuracy'] = 0
    
    # Log all metrics to MLflow
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    
    # Log model parameters
    if hasattr(model, 'get_params'):
        for param_name, param_value in model.get_params().items():
            mlflow.log_param(param_name, param_value)
    
    # Create visualizations
    create_model_visualizations(model, model_name, X_test, y_test, y_pred, y_pred_proba, save_artifacts)
    
    # Save model
    mlflow.sklearn.log_model(model, f"{model_name.lower().replace(' ', '_')}_model")
    
    print(f"✅ {model_name} - Accuracy: {metrics['accuracy']:.4f}, ROC AUC: {metrics.get('roc_auc', 'N/A')}")
    
    return model, metrics

@click.command()
@click.option('--test_size', default=0.2, type=float, help='Test set size ratio')
@click.option('--random_state', default=42, type=int, help='Random state for reproducibility')
@click.option('--max_iter', default=1000, type=int, help='Maximum iterations for LogisticRegression')
@click.option('--n_estimators', default=100, type=int, help='Number of estimators for ensemble methods')
@click.option('--experiment_name', default='Heart_Disease_CI', type=str, help='MLflow experiment name')
@click.option('--save_artifacts', default=True, type=bool, help='Save visualization artifacts')
@click.option('--run_id', default='local_run', type=str, help='GitHub Actions run ID')
@click.option('--commit_sha', default='unknown', type=str, help='Git commit SHA')
@click.option('--hyperparameter_tuning', default=False, type=bool, help='Enable hyperparameter tuning')
@click.option('--cv_folds', default=5, type=int, help='Cross-validation folds for tuning')
def main(test_size, random_state, max_iter, n_estimators, experiment_name, save_artifacts, run_id, commit_sha, hyperparameter_tuning, cv_folds):
    """
    Heart Disease ML Training Pipeline for MLflow Project
    """
    print("="*60)
    print("HEART DISEASE ML TRAINING - MLFLOW PROJECT")
    print("="*60)
    
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run() as run:
        # Log run metadata
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("github_run_id", run_id)
        mlflow.log_param("commit_sha", commit_sha)
        mlflow.log_param("hyperparameter_tuning", hyperparameter_tuning)
        
        # Load data
        df = load_heart_disease_data()
        
        # Prepare features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Log dataset info
        mlflow.log_param("dataset_shape", f"{df.shape[0]}x{df.shape[1]}")
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_positive", int(y.sum()))
        mlflow.log_param("n_negative", int(len(y) - y.sum()))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=random_state, 
                max_iter=max_iter,
                solver='liblinear'
            ),
            'Random Forest': RandomForestClassifier(
                random_state=random_state,
                n_estimators=n_estimators
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=random_state,
                n_estimators=n_estimators
            ),
            'Support Vector Machine': SVC(
                random_state=random_state,
                probability=True,
                kernel='rbf'
            )
        }
        
        # Train models
        results = {}
        best_model = None
        best_accuracy = 0
        
        for model_name, model in models.items():
            try:
                # Create child run for each model
                with mlflow.start_run(run_name=f"{experiment_name}_{model_name}", nested=True):
                    trained_model, metrics = train_model(
                        model, model_name, X_train, X_test, y_train, y_test, save_artifacts
                    )
                    results[model_name] = metrics
                    
                    # Track best model
                    if metrics['accuracy'] > best_accuracy:
                        best_accuracy = metrics['accuracy']
                        best_model = model_name
                        
            except Exception as e:
                print(f"❌ Error training {model_name}: {str(e)}")
                continue
        
        # Log best model info
        if best_model:
            mlflow.log_param("best_model", best_model)
            mlflow.log_metric("best_accuracy", best_accuracy)
        
        # Create results summary
        summary = {
            'experiment_name': experiment_name,
            'run_id': run.info.run_id,
            'github_run_id': run_id,
            'commit_sha': commit_sha,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_shape': f"{df.shape[0]}x{df.shape[1]}",
            'test_size': test_size,
            'best_model': best_model,
            'best_accuracy': best_accuracy,
            'models_trained': len(results),
            'results': results
        }
        
        # Save summary
        with open('training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        mlflow.log_artifact('training_summary.json')
        
        # Create performance comparison
        if results:
            comparison_df = pd.DataFrame(results).T
            comparison_df = comparison_df.round(4)
            comparison_df.to_csv('model_comparison.csv')
            mlflow.log_artifact('model_comparison.csv')
            
            print("\n📊 Model Performance Summary:")
            print(comparison_df[['accuracy', 'roc_auc', 'f1_score', 'matthews_corrcoef']].to_string())
        
        print(f"\n🎉 Training completed successfully!")
        print(f"Best Model: {best_model} (Accuracy: {best_accuracy:.4f})")
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"Artifacts saved: {save_artifacts}")

if __name__ == "__main__":
    main()