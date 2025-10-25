# 🚨 Refund Fraud Detection System with MLOps

A production-ready fraud detection system for e-commerce refund requests with complete MLOps infrastructure.

## 🎯 Project Overview

This system detects fraudulent refund requests in real-time using Machine Learning, helping e-commerce businesses prevent refund abuse and friendly fraud.

**Key Features:**
- ✅ Real-time fraud detection API (< 25ms latency)
- ✅ XGBoost ML model with 84% ROC-AUC
- ✅ MLflow experiment tracking & model versioning
- ✅ FastAPI REST API with automatic documentation
- ✅ Prometheus metrics for monitoring
- ✅ Complete training pipeline
- ✅ Synthetic realistic dataset (50,000 transactions)

## 📊 Model Performance

- **ROC-AUC**: 0.84
- **Precision**: 71%
- **Recall**: 42%
- **Inference Time**: < 25ms (p95)
- **Dataset**: 50,000 transactions (6.8% fraud rate)

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd refund-fraud-detection

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Data & Train Model
```bash
# Generate synthetic dataset
python data/synthetic_generator.py

# Train the model
python src/training/train.py
```

### 3. Start the API
```bash
# Start the fraud detection API
uvicorn src.api.main:app --reload --port 8000
```

### 4. Test the System
```bash
# Run demo tests
python demo/test_api.py

# Or open interactive API docs
open http://localhost:8000/docs
```

### 5. View MLflow Experiments
```bash
# Start MLflow UI
mlflow ui --port 5000

# Open in browser
open http://localhost:5000
```

## 📡 API Endpoints

### POST `/predict`
Predict if a refund request is fraudulent

**Request Example:**
```json
{
  "transaction_id": "TXN00012345",
  "order_amount": 299.99,
  "refund_amount": 289.99,
  "days_since_purchase": 2,
  "customer_age_days": 45,
  "total_orders": 3,
  "previous_refunds": 1,
  ...
}
```

**Response Example:**
```json
{
  "transaction_id": "TXN00012345",
  "is_fraud": false,
  "fraud_probability": 0.0341,
  "risk_level": "LOW",
  "confidence": 0.9318,
  "inference_time_ms": 14.37
}
```

### Other Endpoints
- `GET /` - API information
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `GET /model/info` - Model metadata

## 🏗️ Project Structure
```
refund-fraud-detection/
├── data/
│   ├── raw/                    # Train/val/test datasets
│   └── synthetic_generator.py  # Data generation script
├── models/
│   └── fraud_detector.pkl      # Trained model + artifacts
├── src/
│   ├── training/
│   │   ├── feature_engineering.py
│   │   └── train.py
│   └── api/
│       ├── main.py             # FastAPI application
│       └── schemas.py          # API models
├── demo/
│   └── test_api.py             # Demo script
├── mlruns/                     # MLflow experiments
├── requirements.txt
└── README.md
```

## 🔧 Technology Stack

- **ML Framework**: XGBoost, scikit-learn
- **MLOps**: MLflow (experiment tracking, model registry)
- **API**: FastAPI, Uvicorn
- **Monitoring**: Prometheus
- **Data**: Pandas, NumPy
- **Class Imbalance**: SMOTE (imbalanced-learn)

## 🎯 How It Works

### Training Pipeline
1. Generate/load refund transaction data
2. Feature engineering (30+ features)
3. Handle class imbalance with SMOTE
4. Train XGBoost classifier
5. Track experiments with MLflow
6. Validate model performance
7. Save model artifacts

### Inference Pipeline
1. Receive refund request via API
2. Apply same feature transformations
3. Predict fraud probability
4. Calculate risk level (HIGH/MEDIUM/LOW)
5. Return prediction in < 25ms
6. Log metrics to Prometheus

## 📈 Key Features

### Feature Engineering
- **Time-based**: refund_velocity, purchase_to_refund_ratio
- **Amount-based**: refund_percentage, avg_order_value
- **Behavioral**: engagement_score, cs_contact_ratio
- **Risk flags**: new_customer, suspicious_timing

### MLOps Components
- ✅ Experiment tracking (MLflow)
- ✅ Model versioning
- ✅ Automated training pipeline
- ✅ Model validation gates
- ✅ Performance monitoring
- ✅ Fast model serving

## 🚀 Production Deployment

This system is production-ready and can be deployed using:

- **Docker**: Containerized API service
- **Kubernetes**: Horizontal scaling
- **AWS/GCP/Azure**: Cloud deployment
- **Load Balancer**: For high availability

## 📊 Use Cases

1. **E-commerce Platforms**: Detect refund abuse
2. **Marketplaces**: Prevent friendly fraud
3. **Subscription Services**: Identify chargeback fraud
4. **Retail**: Flag suspicious return patterns

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ End-to-end ML pipeline development
- ✅ MLOps best practices
- ✅ Production API development
- ✅ Model monitoring and versioning
- ✅ Real-time inference optimization
- ✅ Handling imbalanced datasets

## 📝 Future Enhancements

- [ ] Prefect workflow orchestration
- [ ] Docker containerization
- [ ] Grafana dashboards
- [ ] A/B testing framework
- [ ] Feature drift detection
- [ ] SHAP explainability

## 👨‍💻 Author

**Your Name**
- LinkedIn: [your-profile]
- GitHub: [your-github]
- Email: your-email@example.com

## 📄 License

MIT License

---

**Built with ❤️ for production MLOps**