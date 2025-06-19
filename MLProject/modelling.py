"""
Simple MLflow Project Compatible Heart Disease Model Training (FIXED)
Author: Andre
Date: June 2025
Description: Fixed version that handles MLflow run conflicts properly
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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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

def setup_mlflow_clean(experiment_name):
    """Setup MLflow with clean state"""
    print(f"🔧 Setting up MLflow...")
    
    # Force end any existing runs
    try:
        while mlflow.active_run():
            mlflow.end_run()
    except:
        pass
    
    # Set tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Create unique experiment name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    full_experiment_name = f"{experiment_name}_{timestamp}"
    
    try:
        mlflow.set_experiment(full_experiment_name)
        print(f"✅ Experiment: {full_experiment_name}")
    except Exception as e:
        print(f"⚠️  Using default experiment: {e}")
        mlflow.set_experiment("Default")
        full_experiment_name = "Default"
    
    return full_experiment_name

def load_data():
    """Load heart disease dataset"""
    print("📊 Loading heart disease dataset...")
    
    # Try multiple possible locations
    data_paths = [
        'heart.csv',
        'data/heart.csv', 
        'data/processed/heart_processed.csv',
        '../heart.csv',
        '../Membangun_model/heart.csv'
    ]
    
    df = None
    for path in data_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"✅ Data loaded from: {path}")
            break
    
    if df is None:
        print("❌ Dataset not found. Creating dummy data for testing...")
        # Create dummy data for testing
        np.random.seed(42)
        n_samples = 1000
        data = {
            'age': np.random.randint(29, 77, n_samples),
            'sex': np.random.randint(0, 2, n_samples),
            'cp': np.random.randint(0, 4, n_samples),
            'trestbps': np.random.randint(94, 200, n_samples),
            'chol': np.random.randint(126, 564, n_samples),
            'fbs': np.random.randint(0, 2, n_samples),
            'restecg': np.random.randint(0, 3, n_samples),
            'thalach': np.random.randint(71, 202, n_samples),
            'exang': np.random.randint(0, 2, n_samples),
            'oldpeak': np.random.uniform(0, 6.2, n_samples),
            'slope': np.random.randint(0, 3, n_samples),
            'ca': np.random.randint(0, 4, n_samples),
            'thal': np.random.randint(0, 4, n_samples),
            'target': np.random.randint(0, 2, n_samples)
        }
        df = pd.DataFrame(data)
        print("✅ Dummy data created for testing")
    
    print(f"   Shape: {df.shape}")
    if 'target' in df.columns:
        print(f"   Target distribution: {df['target'].value_counts().to_dict()}")
    
    return df

def preprocess_data(df, test_size=0.2, random_state=42):
    """Preprocess heart disease data"""
    print("🔄 Preprocessing data...")
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Define feature types (basic approach for compatibility)
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    # Keep only features that exist in the dataset
    numeric_features = [f for f in numeric_features if f in X.columns]
    categorical_features = [f for f in categorical_features if f in X.columns]
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
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

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate comprehensive metrics"""
    
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
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
            pr_auc = auc(recall_curve, precision_curve)
            logloss = log_loss(y_true, y_pred_proba)
        except:
            pass
    
    # Additional metrics
    mcc = matthews_corrcoef(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Confusion matrix derived metrics
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fdr = fp / (fp + tp) if (fp + tp) > 0 else 0.0
    except:
        specificity = npv = fpr = fdr = 0.0
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'matthews_corrcoef': float(mcc),
        'balanced_accuracy': float(balanced_acc),
        'log_loss': float(logloss),
        'pr_auc': float(pr_auc),
        'specificity': float(specificity),
        'npv': float(npv),
        'fpr': float(fpr),
        'fdr': float(fdr)
    }

def create_simple_artifacts(y_true, y_pred, model_name):
    """Create simple artifacts without complex plotting"""
    artifacts = []
    
    try:
        # Simple confusion matrix
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_true, y_pred)
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.colorbar()
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center')
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        cm_path = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(cm_path, dpi=100, bbox_inches='tight')
        plt.close()
        artifacts.append(cm_path)
        
    except Exception as e:
        print(f"   ⚠️  Artifact creation error: {e}")
        plt.close('all')
    
    return artifacts

def train_model_simple(model, model_name, X_train, X_test, y_train, y_test, args):
    """Train a single model with simple MLflow logging"""
    
    print(f"\n🤖 Training {model_name}...")
    
    # Ensure clean state
    try:
        while mlflow.active_run():
            mlflow.end_run()
    except:
        pass
    
    run_name = f"Simple_{model_name.replace(' ', '_')}"
    
    try:
        mlflow.start_run(run_name=run_name)
        
        start_time = time.time()
        
        # Train model
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Cross-validation
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))
        except:
            cv_mean = cv_std = 0.0
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                y_pred_proba = model.predict_proba(X_test)
            except:
                pass
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        metrics.update({
            'cv_mean_accuracy': cv_mean,
            'cv_std_accuracy': cv_std,
            'training_time_seconds': float(training_time)
        })
        
        # Log basic parameters
        try:
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("test_size", args.test_size)
            mlflow.log_param("random_state", args.random_state)
        except Exception as e:
            print(f"   ⚠️  Parameter logging: {e}")
        
        # Log metrics
        try:
            for metric_name, metric_value in metrics.items():
                if np.isfinite(float(metric_value)):
                    mlflow.log_metric(metric_name, float(metric_value))
        except Exception as e:
            print(f"   ⚠️  Metrics logging: {e}")
        
        # Create and log simple artifacts
        if args.save_artifacts:
            try:
                artifacts = create_simple_artifacts(y_test, y_pred, model_name)
                for artifact_path in artifacts:
                    mlflow.log_artifact(artifact_path)
            except Exception as e:
                print(f"   ⚠️  Artifacts: {e}")
        
        # Log model (simple approach)
        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=f"model"
            )
        except Exception as e:
            print(f"   ⚠️  Model logging: {e}")
            # Simple fallback
            try:
                model_path = f'model_{model_name.lower().replace(" ", "_")}.pkl'
                joblib.dump(model, model_path)
                mlflow.log_artifact(model_path)
            except:
                pass
        
        mlflow.end_run()
        
        print(f"   ✅ Accuracy: {metrics['accuracy']:.4f}")
        print(f"   ✅ MCC: {metrics['matthews_corrcoef']:.4f}")
        
        return {
            'model': model,
            'metrics': metrics,
            'success': True
        }
        
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        try:
            mlflow.end_run()
        except:
            pass
        return {
            'model': None,
            'metrics': {},
            'success': False,
            'error': str(e)
        }

def main():
    """Main training function - simplified and robust"""
    print("🚀 HEART DISEASE ML TRAINING - SIMPLE & ROBUST")
    print("="*60)
    
    try:
        # Parse arguments
        args = parse_arguments()
        print(f"📋 Parameters: test_size={args.test_size}, random_state={args.random_state}")
        
        # Setup MLflow
        experiment_name = setup_mlflow_clean(args.experiment_name)
        
        # Load and preprocess data
        df = load_data()
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
            df, args.test_size, args.random_state
        )
        
        # Define simple models
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=args.random_state, 
                max_iter=args.max_iter
            ),
            'Random Forest': RandomForestClassifier(
                random_state=args.random_state,
                n_estimators=min(args.n_estimators, 50)  # Limit for speed
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=args.random_state,
                n_estimators=min(args.n_estimators, 50)
            ),
            'SVM': SVC(
                random_state=args.random_state,
                probability=True
            )
        }
        
        # Train models
        results = {}
        successful_models = 0
        
        for model_name, model in models.items():
            result = train_model_simple(model, model_name, X_train, X_test, y_train, y_test, args)
            results[model_name] = result
            if result['success']:
                successful_models += 1
        
        # Save summary
        summary = {
            'experiment_name': experiment_name,
            'successful_models': successful_models,
            'total_models': len(models),
            'timestamp': datetime.datetime.now().isoformat(),
            'results': {k: v['success'] for k, v in results.items()}
        }
        
        import json
        with open('training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Final results
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED!")
        print("="*60)
        print(f"📊 Successful models: {successful_models}/{len(models)}")
        print(f"🧪 Experiment: {experiment_name}")
        
        if successful_models > 0:
            print("\n✅ Successfully trained models:")
            for name, result in results.items():
                if result['success']:
                    acc = result.get('metrics', {}).get('accuracy', 0)
                    print(f"   🔸 {name}: {acc:.4f}")
            
            print("\n🎉 Training successful!")
            sys.exit(0)
        else:
            print("\n❌ No models trained successfully")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Final cleanup
        try:
            while mlflow.active_run():
                mlflow.end_run()
        except:
            pass

if __name__ == "__main__":
    main()