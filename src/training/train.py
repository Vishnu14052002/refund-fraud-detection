import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
import mlflow
import mlflow.xgboost
from imblearn.over_sampling import SMOTE
import pickle
import json
from datetime import datetime
from feature_engineering import FeatureEngineer

class FraudDetectionTrainer:
    def __init__(self, experiment_name="refund-fraud-detection"):
        mlflow.set_experiment(experiment_name)
        self.feature_engineer = FeatureEngineer()
        self.model = None
        self.threshold = 0.5
        
    def load_data(self, train_path, val_path):
        """Load training and validation data"""
        print("📂 Loading data...")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        
        # Separate features and target
        self.X_train = train_df.drop(['is_fraud', 'transaction_id'], axis=1)
        self.y_train = train_df['is_fraud']
        self.X_val = val_df.drop(['is_fraud', 'transaction_id'], axis=1)
        self.y_val = val_df['is_fraud']
        
        print(f"✅ Training set: {len(self.X_train)} samples, Fraud rate: {self.y_train.mean():.2%}")
        print(f"✅ Validation set: {len(self.X_val)} samples, Fraud rate: {self.y_val.mean():.2%}")
        
    def preprocess(self):
        """Apply feature engineering"""
        print("\n🔧 Applying feature engineering...")
        
        # Create features
        self.X_train_processed = self.feature_engineer.create_features(
            pd.concat([self.X_train, self.y_train], axis=1),
            is_training=True
        )
        self.X_val_processed = self.feature_engineer.create_features(
            pd.concat([self.X_val, self.y_val], axis=1),
            is_training=False
        )
        
        print(f"✅ Features created: {self.X_train_processed.shape[1]} features")
        
        # Handle class imbalance with SMOTE
        print("\n⚖️  Applying SMOTE to balance classes...")
        smote = SMOTE(random_state=42, k_neighbors=5)
        self.X_train_balanced, self.y_train_balanced = smote.fit_resample(
            self.X_train_processed, self.y_train
        )
        
        print(f"✅ After SMOTE: {len(self.X_train_balanced)} samples")
        print(f"✅ Balanced fraud rate: {self.y_train_balanced.mean():.2%}")
        
    def train(self, params=None):
        """Train XGBoost model"""
        if params is None:
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 5,
                'scale_pos_weight': 1,
                'random_state': 42,
                'tree_method': 'hist'
            }
        
        print("\n🤖 Training XGBoost model...")
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("features_count", self.X_train_processed.shape[1])
            mlflow.log_param("training_samples", len(self.X_train_balanced))
            
            # Train model
            self.model = xgb.XGBClassifier(**params)
            
            self.model.fit(
                self.X_train_balanced,
                self.y_train_balanced,
                eval_set=[(self.X_val_processed, self.y_val)],
                verbose=False
            )
            
            # Get predictions
            y_pred_proba = self.model.predict_proba(self.X_val_processed)[:, 1]
            self.threshold = 0.5
            y_pred = (y_pred_proba >= self.threshold).astype(int)
            
            # Calculate metrics
            metrics = {
                'accuracy': float((y_pred == self.y_val).mean()),
                'precision': float(precision_score(self.y_val, y_pred)),
                'recall': float(recall_score(self.y_val, y_pred)),
                'f1_score': float(f1_score(self.y_val, y_pred)),
                'roc_auc': float(roc_auc_score(self.y_val, y_pred_proba)),
                'threshold': float(self.threshold)
            }
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log confusion matrix
            cm = confusion_matrix(self.y_val, y_pred)
            
            # Log model
            mlflow.xgboost.log_model(self.model, "model")
            
            print("\n" + "="*50)
            print("✅ Training Complete!")
            print("="*50)
            print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1-Score:  {metrics['f1_score']:.4f}")
            print(f"\nConfusion Matrix:")
            print(f"  TN: {cm[0,0]:5d}  |  FP: {cm[0,1]:5d}")
            print(f"  FN: {cm[1,0]:5d}  |  TP: {cm[1,1]:5d}")
            
            return metrics, mlflow.active_run().info.run_id
    
    def save_model(self, model_path='models/fraud_detector.pkl'):
        """Save model and artifacts"""
        print(f"\n💾 Saving model to {model_path}...")
        
        artifacts = {
            'model': self.model,
            'feature_engineer': self.feature_engineer,
            'threshold': self.threshold,
            'feature_names': self.feature_engineer.feature_names,
            'metadata': {
                'trained_at': datetime.now().isoformat(),
                'training_samples': len(self.X_train_balanced)
            }
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(artifacts, f)
        
        print(f"✅ Model saved successfully!")

if __name__ == "__main__":
    print("🚀 Starting Fraud Detection Model Training")
    print("="*50)
    
    # Initialize trainer
    trainer = FraudDetectionTrainer()
    
    # Load data
    trainer.load_data('data/raw/train_data.csv', 'data/raw/val_data.csv')
    
    # Preprocess
    trainer.preprocess()
    
    # Train
    metrics, run_id = trainer.train()
    
    # Save
    trainer.save_model()
    
    print(f"\n🎉 All done!")
    print(f"📊 MLflow Run ID: {run_id}")
    print(f"💡 View results: mlflow ui --port 5000")