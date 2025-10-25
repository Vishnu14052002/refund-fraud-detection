import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

np.random.seed(42)

def generate_refund_dataset(n_samples=50000):
    """Generate realistic refund transaction data"""
    
    print(f"Generating {n_samples} refund transactions...")
    
    # Base features
    data = {
        # Transaction features
        'transaction_id': [f'TXN{i:08d}' for i in range(n_samples)],
        'order_amount': np.random.lognormal(4, 1, n_samples).clip(10, 5000),
        'refund_amount': None,
        'days_since_purchase': np.random.exponential(15, n_samples).clip(0, 90),
        
        # Customer features
        'customer_age_days': np.random.exponential(365, n_samples).clip(1, 3650),
        'total_orders': np.random.poisson(5, n_samples).clip(1, 100),
        'previous_refunds': np.random.poisson(0.5, n_samples).clip(0, 20),
        'account_age_days': np.random.exponential(500, n_samples).clip(30, 3650),
        
        # Behavioral features
        'customer_service_contacts': np.random.poisson(1, n_samples).clip(0, 10),
        'login_frequency_30d': np.random.poisson(15, n_samples).clip(0, 100),
        'products_viewed': np.random.poisson(20, n_samples).clip(1, 200),
        
        # Item features
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Beauty', 'Sports'], n_samples),
        'product_price': None,
        'is_high_value': None,
        
        # Refund request features
        'refund_reason': np.random.choice([
            'defective', 'not_as_described', 'arrived_late', 
            'changed_mind', 'accidental_order', 'other'
        ], n_samples),
        'has_images': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'shipping_address_changes': np.random.poisson(0.3, n_samples).clip(0, 5),
        
        # Time features
        'is_weekend': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'hour_of_day': np.random.randint(0, 24, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate derived features
    df['product_price'] = df['order_amount']
    df['refund_amount'] = df['order_amount'] * np.random.uniform(0.8, 1.0, n_samples)
    df['is_high_value'] = (df['order_amount'] > 500).astype(int)
    df['refund_ratio'] = df['previous_refunds'] / (df['total_orders'] + 1)
    df['refund_speed'] = (df['days_since_purchase'] < 2).astype(int)
    df['late_night'] = ((df['hour_of_day'] >= 23) | (df['hour_of_day'] <= 5)).astype(int)
    
    # Generate fraud labels with realistic patterns
    fraud_score = (
        (df['refund_speed'] * 2) +
        (df['refund_ratio'] * 3) +
        (df['shipping_address_changes'] * 2) +
        ((df['days_since_purchase'] < 1).astype(int) * 1.5) +
        ((df['refund_amount'] / df['order_amount'] > 0.95).astype(int) * 1) +
        ((df['previous_refunds'] > 3).astype(int) * 2) +
        ((df['customer_age_days'] < 30).astype(int) * 1.5) +
        (df['late_night'] * 0.5) +
        ((df['has_images'] == 0).astype(int) * 0.5) +
        np.random.normal(0, 1, n_samples)
    )
    
    # Create imbalanced dataset (5% fraud rate)
    threshold = np.percentile(fraud_score, 95)
    df['is_fraud'] = (fraud_score > threshold).astype(int)
    
    # Add some noise
    flip_indices = np.random.choice(len(df), size=int(0.02 * len(df)), replace=False)
    df.loc[flip_indices, 'is_fraud'] = 1 - df.loc[flip_indices, 'is_fraud']
    
    print(f"\n✅ Dataset generated: {len(df)} samples")
    print(f"📊 Fraud rate: {df['is_fraud'].mean():.2%}")
    print(f"\nClass distribution:")
    print(df['is_fraud'].value_counts())
    
    return df

if __name__ == "__main__":
    # Generate dataset
    df = generate_refund_dataset(50000)
    
    # Split and save
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['is_fraud'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['is_fraud'], random_state=42)
    
    # Save datasets
    train_df.to_csv('data/raw/train_data.csv', index=False)
    val_df.to_csv('data/raw/val_data.csv', index=False)
    test_df.to_csv('data/raw/test_data.csv', index=False)
    
    print("\n📁 Datasets saved:")
    print(f"   Train: {len(train_df)} samples")
    print(f"   Val: {len(val_df)} samples")
    print(f"   Test: {len(test_df)} samples")
    print("\n🎉 Data generation complete!")