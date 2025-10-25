from prefect import flow, task
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'training'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.training.train import FraudDetectionTrainer

@task(name="load_data", retries=2)
def load_data_task(trainer):
    """Load training data"""
    print("📂 Task: Loading data...")
    trainer.load_data('data/raw/train_data.csv', 'data/raw/val_data.csv')
    print("✅ Task complete: Data loaded")
    return "Data loaded"

@task(name="preprocess_data")
def preprocess_data_task(trainer):
    """Preprocess and engineer features"""
    print("🔧 Task: Preprocessing data...")
    trainer.preprocess()
    print("✅ Task complete: Preprocessing done")
    return "Preprocessing complete"

@task(name="train_model")
def train_model_task(trainer):
    """Train the fraud detection model"""
    print("🤖 Task: Training model...")
    metrics, run_id = trainer.train()
    print("✅ Task complete: Model trained")
    return metrics, run_id

@task(name="save_model")
def save_model_task(trainer):
    """Save trained model"""
    print("💾 Task: Saving model...")
    trainer.save_model('models/fraud_detector.pkl')
    print("✅ Task complete: Model saved")
    return "Model saved"

@task(name="validate_model")
def validate_model_task(metrics):
    """Validate model performance"""
    print("✅ Task: Validating model...")
    
    min_auc = 0.75
    min_precision = 0.60
    
    if metrics['roc_auc'] < min_auc:
        raise Exception(f"❌ Model AUC {metrics['roc_auc']:.4f} below threshold {min_auc}")
    
    if metrics['precision'] < min_precision:
        raise Exception(f"❌ Model precision {metrics['precision']:.4f} below threshold {min_precision}")
    
    print(f"✅ Task complete: Model validation passed!")
    print(f"   - ROC-AUC: {metrics['roc_auc']:.4f} (threshold: {min_auc})")
    print(f"   - Precision: {metrics['precision']:.4f} (threshold: {min_precision})")
    
    return "Model validation passed"

@flow(name="fraud-detection-training-pipeline", log_prints=True)
def training_pipeline():
    """Complete end-to-end training pipeline"""
    
    print("=" * 70)
    print("🚀 STARTING FRAUD DETECTION TRAINING PIPELINE")
    print("=" * 70)
    print()
    
    # Initialize trainer
    print("📦 Initializing trainer...")
    trainer = FraudDetectionTrainer()
    print("✅ Trainer initialized")
    print()
    
    # Pipeline steps
    load_data_task(trainer)
    print()
    
    preprocess_data_task(trainer)
    print()
    
    metrics, run_id = train_model_task(trainer)
    print()
    
    save_model_task(trainer)
    print()
    
    validate_model_task(metrics)
    print()
    
    print("=" * 70)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"📊 Final Metrics:")
    print(f"   - ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"   - Precision: {metrics['precision']:.4f}")
    print(f"   - Recall: {metrics['recall']:.4f}")
    print(f"   - F1-Score: {metrics['f1_score']:.4f}")
    print(f"\n📝 MLflow Run ID: {run_id}")
    print()
    
    return {
        'status': 'success',
        'metrics': metrics,
        'mlflow_run_id': run_id
    }

if __name__ == "__main__":
    # Run the pipeline
    result = training_pipeline()
    print(f"\n✅ Pipeline result: {result['status']}")