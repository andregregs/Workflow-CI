"""
MLflow Project Compatible Heart Disease Model Training
Author: Andre
Date: June 2025
Description: CI/CD compatible model training for MLflow Project (Advance Level - 4 pts)
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           classification_report, confusion_matrix, roc_auc_score,
                           roc_curve, precision_recall_curve, auc, log_loss,
                           matthews_corrcoef, balanced_accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime
import warnings
warnings.filterwarnings('ignore')

def parse_arguments():
    """Parse command line arguments for MLflow Project"""
    parser = argparse.ArgumentParser(description='Heart Disease ML Training')
    
    # Data parameters
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size ratio (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    
    # Model parameters
    parser.add_argument('--max_iter', type=int, default=1000,
                       help='Maximum iterations for LogisticRegression (default: 1000)')
    parser.add_argument('--n_estimators', type=int, default=100,
                       help='Number of estimators for ensemble methods (default: 100)')
    
    # MLflow parameters
    parser.add_argument('--experiment_name', type=str, default='Heart_Disease_CI',
                       help='MLflow experiment name (default: Heart_Disease_CI)')
    parser.add_argument('--save_artifacts', type=bool, default=True,
                       help='Save artifacts to MLflow (default: True)')
    
    # CI/CD parameters
    parser.add_argument('--run_id', type=str, default=None,
                       help='GitHub run ID for tracking (default: None)')
    parser.add_argument('--commit_sha', type=str, default=None,
                       help='Git commit SHA for tracking (default: None)')
    
    return parser.parse_args()

def setup_mlflow(experiment_name, run_id=None, commit_sha=None):
    """Setup MLflow tracking"""
    print(f"🔧 Setting up MLflow...")
    
    # End any active runs first to avoid conflicts
    try:
        mlflow.end_run()
    except:
        pass  # No active run to end
    
    # Set tracking URI (local for CI/CD)
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Create experiment with timestamp to avoid conflicts
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    full_experiment_name = f"{experiment_name}_{timestamp}"
    
    try:
        mlflow.set_experiment(full_experiment_name)
        print(f"✅ Experiment: {full_experiment_name}")
    except Exception as e:
        print(f"⚠️  Using default experiment: {e}")
        mlflow.set_experiment("Default")
    
    # Log CI/CD metadata if provided
    if run_id or commit_sha:
        with mlflow.start_run(run_name="metadata_setup"):
            if run_id:
                mlflow.set_tag("github_run_id", run_id)
            if commit_sha:
                mlflow.set_tag("commit_sha", commit_sha)
            
            mlflow.set_tag("ci_cd_run", "true")
            mlflow.set_tag("automated_training", "github_actions")
        # End the metadata run immediately
        mlflow.end_run()
    
    return full_experiment_name

def load_data():
    """Load heart disease dataset"""
    print("📊 Loading heart disease dataset...")
    
    try:
        # Try multiple possible locations for the dataset
        data_paths = [
            'heart.csv',
            'data/heart.csv', 
            'data/processed/heart_processed.csv',
            '../heart.csv'
        ]
        
        df = None
        for path in data_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                print(f"✅ Data loaded from: {path}")
                break
        
        if df is None:
            raise FileNotFoundError("Dataset not found in any expected location")
        
        print(f"   Shape: {df.shape}")
        print(f"   Target distribution: {df['target'].value_counts().to_dict()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)

def preprocess_data(df, test_size=0.2, random_state=42):
    """Preprocess heart disease data"""
    print("🔄 Preprocessing data...")
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Define feature types
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Fit and transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"✅ Preprocessing completed")
    print(f"   Training set: {X_train_processed.shape}")
    print(f"   Test set: {X_test_processed.shape}")
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

def calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba):
    """Calculate comprehensive metrics including additional ones beyond autolog"""
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    # Advanced metrics
    roc_auc = 0.0
    pr_auc = 0.0
    logloss = 0.0
    
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
        pr_auc = auc(recall_curve, precision_curve)
        logloss = log_loss(y_true, y_pred_proba)
    
    # Additional metrics (beyond autolog)
    mcc = matthews_corrcoef(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Confusion matrix derived metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fdr = fp / (fp + tp) if (fp + tp) > 0 else 0.0
    
    return {
        # Standard metrics
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        
        # Additional metrics (beyond autolog)
        'matthews_corrcoef': float(mcc),
        'balanced_accuracy': float(balanced_acc),
        'log_loss': float(logloss),
        'pr_auc': float(pr_auc),
        'specificity': float(specificity),
        'npv': float(npv),
        'fpr': float(fpr),
        'fdr': float(fdr)
    }

def create_artifacts(y_true, y_pred, y_pred_proba, model_name, feature_importance=None):
    """Create and save artifacts"""
    artifacts = []
    
    # 1. Confusion Matrix
    try:
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        cm_path = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        artifacts.append(cm_path)
    except Exception as e:
        print(f"   ⚠️  Confusion matrix error: {e}")
        plt.close('all')
    
    # 2. ROC Curve
    if y_pred_proba is not None:
        try:
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            roc_auc_val = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc_val:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            roc_path = f'roc_curve_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(roc_path, dpi=150, bbox_inches='tight')
            plt.close()
            artifacts.append(roc_path)
        except Exception as e:
            print(f"   ⚠️  ROC curve error: {e}")
            plt.close('all')
    
    # 3. Feature Importance
    if feature_importance is not None:
        try:
            plt.figure(figsize=(10, 6))
            top_n = min(15, len(feature_importance))
            indices = np.argsort(feature_importance)[::-1][:top_n]
            
            plt.bar(range(top_n), feature_importance[indices])
            plt.xticks(range(top_n), [f'F{i}' for i in indices], rotation=45)
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.title(f'Feature Importance - {model_name}')
            plt.tight_layout()
            
            fi_path = f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(fi_path, dpi=150, bbox_inches='tight')
            plt.close()
            artifacts.append(fi_path)
        except Exception as e:
            print(f"   ⚠️  Feature importance error: {e}")
            plt.close('all')
    
    return artifacts

def train_single_model(model, model_name, X_train, X_test, y_train, y_test, args):
    """Train a single model with MLflow logging"""
    
    print(f"\n🤖 Training {model_name}...")
    
    # Ensure no active runs before starting
    try:
        mlflow.end_run()
    except:
        pass
    
    # Start new run with unique name
    run_name = f"CI_{model_name.replace(' ', '_')}_{datetime.datetime.now().strftime('%H%M%S')}"
    
    try:
        with mlflow.start_run(run_name=run_name):
            # Disable autolog for manual control
            mlflow.sklearn.autolog(disable=True)
            
            start_time = time.time()
            
            # Train model
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state),
                scoring='accuracy'
            )
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            metrics = calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)
            metrics.update({
                'cv_mean_accuracy': float(np.mean(cv_scores)),
                'cv_std_accuracy': float(np.std(cv_scores)),
                'training_time_seconds': float(training_time)
            })
            
            # Log parameters
            try:
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("test_size", args.test_size)
                mlflow.log_param("random_state", args.random_state)
                for key, value in model.get_params().items():
                    mlflow.log_param(f"model_{key}", str(value)[:250])
                print(f"   ✅ Parameters logged")
            except Exception as e:
                print(f"   ⚠️  Parameter logging: {e}")
            
            # Log metrics
            try:
                for metric_name, metric_value in metrics.items():
                    if np.isfinite(float(metric_value)):
                        mlflow.log_metric(metric_name, float(metric_value))
                print(f"   ✅ {len(metrics)} metrics logged")
            except Exception as e:
                print(f"   ⚠️  Metrics logging: {e}")
            
            # Create and log artifacts
            if args.save_artifacts:
                feature_importance = getattr(model, 'feature_importances_', None)
                artifacts = create_artifacts(y_test, y_pred, y_pred_proba, model_name, feature_importance)
                
                for artifact_path in artifacts:
                    try:
                        mlflow.log_artifact(artifact_path)
                    except Exception as e:
                        print(f"   ⚠️  Artifact logging {artifact_path}: {e}")
                
                print(f"   ✅ {len(artifacts)} artifacts logged")
            
            # Log model
            try:
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=f"model_{model_name.lower().replace(' ', '_')}"
                )
                print(f"   ✅ Model logged")
            except Exception as e:
                print(f"   ⚠️  Model logging: {e}")
                # Fallback
                try:
                    model_path = f'model_{model_name.lower().replace(" ", "_")}.pkl'
                    joblib.dump(model, model_path)
                    mlflow.log_artifact(model_path)
                    print(f"   ✅ Model saved as artifact")
                except Exception as e2:
                    print(f"   ❌ Model fallback: {e2}")
            
            print(f"   📊 Accuracy: {metrics['accuracy']:.4f}")
            print(f"   📊 MCC: {metrics['matthews_corrcoef']:.4f}")
            
            return {
                'model': model,
                'metrics': metrics,
                'success': True
            }
            
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        return {
            'model': None,
            'metrics': {},
            'success': False,
            'error': str(e)
        }
    finally:
        # Ensure run is ended
        try:
            mlflow.end_run()
        except:
            pass

def main():
    """Main training function for MLflow Project"""
    print("🚀 HEART DISEASE ML TRAINING - MLFLOW PROJECT")
    print("="*60)
    
    try:
        # Parse arguments
        args = parse_arguments()
        print(f"📋 Parameters: test_size={args.test_size}, random_state={args.random_state}")
        
        # Ensure no active runs at start
        try:
            mlflow.end_run()
        except:
            pass
        
        # Setup MLflow
        experiment_name = setup_mlflow(args.experiment_name, args.run_id, args.commit_sha)
        
        # Load and preprocess data
        df = load_data()
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
            df, args.test_size, args.random_state
        )
        
        # Define models with parameters from args
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=args.random_state, 
                max_iter=args.max_iter, 
                C=1.0, 
                solver='liblinear'
            ),
            'Random Forest': RandomForestClassifier(
                random_state=args.random_state,
                n_estimators=args.n_estimators,
                max_depth=10,
                min_samples_split=5
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=args.random_state,
                n_estimators=args.n_estimators,
                learning_rate=0.1,
                max_depth=3
            ),
            'SVM': SVC(
                random_state=args.random_state,
                probability=True,
                C=1.0,
                kernel='rbf'
            )
        }
        
        # Train all models
        results = {}
        successful_models = 0
        failed_models = []
        
        for model_name, model in models.items():
            try:
                result = train_single_model(model, model_name, X_train, X_test, y_train, y_test, args)
                results[model_name] = result
                if result['success']:
                    successful_models += 1
                else:
                    failed_models.append(model_name)
            except Exception as e:
                print(f"   ❌ {model_name} failed: {e}")
                failed_models.append(model_name)
                results[model_name] = {'success': False, 'error': str(e)}
        
        # Save summary
        summary = {
            'experiment_name': experiment_name,
            'total_models': len(models),
            'successful_models': successful_models,
            'failed_models': failed_models,
            'timestamp': datetime.datetime.now().isoformat(),
            'parameters': vars(args),
            'results': {k: {'success': v['success'], 'accuracy': v.get('metrics', {}).get('accuracy', 0)} for k, v in results.items()}
        }
        
        # Save training summary
        summary_path = 'training_summary.json'
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Final results
        print("\n" + "="*60)
        print("🎉 MLFLOW PROJECT TRAINING COMPLETED!")
        print("="*60)
        print(f"📊 Successful models: {successful_models}/{len(models)}")
        print(f"🧪 Experiment: {experiment_name}")
        print(f"📁 MLruns directory: ./mlruns")
        print(f"📄 Summary saved: {summary_path}")
        
        if successful_models > 0:
            print(f"\n✅ {successful_models} models trained successfully:")
            for name, result in results.items():
                if result.get('success'):
                    acc = result.get('metrics', {}).get('accuracy', 0)
                    print(f"   🔸 {name}: {acc:.4f}")
        
        if failed_models:
            print(f"\n⚠️  {len(failed_models)} models failed:")
            for name in failed_models:
                print(f"   ❌ {name}")
        
        # Determine exit code
        if successful_models == len(models):
            print("\n🎉 All models trained successfully!")
            sys.exit(0)
        elif successful_models > 0:
            print(f"\n✅ Partial success: {successful_models}/{len(models)} models trained")
            sys.exit(0)  # Allow partial success for CI/CD
        else:
            print("\n❌ All models failed - check logs")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Critical error in main function: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Ensure all runs are closed
        try:
            mlflow.end_run()
        except:
            pass

if __name__ == "__main__":
    main()