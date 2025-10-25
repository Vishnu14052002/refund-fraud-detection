import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

class FeatureEngineer:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = None
        
    def create_features(self, df, is_training=True):
        """Create and transform features"""
        df = df.copy()
        
        # Time-based features
        df['purchase_to_refund_ratio'] = df['days_since_purchase'] / (df['account_age_days'] + 1)
        df['refund_velocity'] = df['previous_refunds'] / (df['account_age_days'] / 30 + 1)
        df['order_frequency'] = df['total_orders'] / (df['account_age_days'] / 30 + 1)
        
        # Amount-based features
        df['refund_percentage'] = (df['refund_amount'] / df['order_amount']) * 100
        df['avg_order_value'] = df['order_amount'] / (df['total_orders'] + 1)
        
        # Behavioral features
        df['cs_contact_ratio'] = df['customer_service_contacts'] / (df['total_orders'] + 1)
        df['engagement_score'] = df['login_frequency_30d'] * df['products_viewed']
        
        # Risk flags
        df['new_customer'] = (df['account_age_days'] < 30).astype(int)
        df['suspicious_timing'] = ((df['days_since_purchase'] < 1) | (df['late_night'] == 1)).astype(int)
        df['high_refund_customer'] = (df['refund_ratio'] > 0.3).astype(int)
        
        # Encode categorical features
        categorical_cols = ['product_category', 'refund_reason']
        
        for col in categorical_cols:
            if is_training:
                self.encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col].astype(str))
            else:
                df[f'{col}_encoded'] = self.encoders[col].transform(df[col].astype(str))
        
        # Select features for modeling
        feature_cols = [
            # Original features
            'order_amount', 'refund_amount', 'days_since_purchase',
            'customer_age_days', 'total_orders', 'previous_refunds',
            'account_age_days', 'customer_service_contacts', 'login_frequency_30d',
            'products_viewed', 'is_high_value', 'has_images',
            'shipping_address_changes', 'is_weekend', 'hour_of_day',
            'refund_ratio', 'refund_speed', 'late_night',
            
            # Engineered features
            'purchase_to_refund_ratio', 'refund_velocity', 'order_frequency',
            'refund_percentage', 'avg_order_value', 'cs_contact_ratio',
            'engagement_score', 'new_customer', 'suspicious_timing',
            'high_refund_customer',
            
            # Encoded features
            'product_category_encoded', 'refund_reason_encoded'
        ]
        
        X = df[feature_cols].copy()
        
        # Handle any missing values
        X = X.fillna(0)
        
        # Scale features
        if is_training:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self.feature_names = feature_cols
        else:
            X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)
    
    def save(self, path='models/feature_engineer.pkl'):
        """Save feature engineer"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path='models/feature_engineer.pkl'):
        """Load feature engineer"""
        with open(path, 'rb') as f:
            return pickle.load(f)