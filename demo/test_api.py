import requests
import json

API_URL = "http://localhost:8000/predict"

# Test cases
test_cases = [
    {
        "name": "🚨 HIGH RISK - Suspicious Fraud",
        "data": {
            "transaction_id": "TXN_FRAUD_001",
            "order_amount": 1200.00,
            "refund_amount": 1200.00,
            "days_since_purchase": 0,
            "customer_age_days": 5,
            "total_orders": 1,
            "previous_refunds": 0,
            "account_age_days": 7,
            "customer_service_contacts": 5,
            "login_frequency_30d": 1,
            "products_viewed": 3,
            "product_category": "Electronics",
            "product_price": 1200.00,
            "is_high_value": 1,
            "refund_reason": "changed_mind",
            "has_images": 0,
            "shipping_address_changes": 3,
            "is_weekend": 1,
            "hour_of_day": 2,
            "refund_ratio": 0.0,
            "refund_speed": 1,
            "late_night": 1
        }
    },
    {
        "name": "✅ LOW RISK - Legitimate Customer",
        "data": {
            "transaction_id": "TXN_LEGIT_001",
            "order_amount": 150.00,
            "refund_amount": 150.00,
            "days_since_purchase": 10,
            "customer_age_days": 500,
            "total_orders": 15,
            "previous_refunds": 1,
            "account_age_days": 730,
            "customer_service_contacts": 1,
            "login_frequency_30d": 20,
            "products_viewed": 50,
            "product_category": "Clothing",
            "product_price": 150.00,
            "is_high_value": 0,
            "refund_reason": "not_as_described",
            "has_images": 1,
            "shipping_address_changes": 0,
            "is_weekend": 0,
            "hour_of_day": 14,
            "refund_ratio": 0.067,
            "refund_speed": 0,
            "late_night": 0
        }
    },
    {
        "name": "⚠️ MEDIUM RISK - Needs Review",
        "data": {
            "transaction_id": "TXN_MEDIUM_001",
            "order_amount": 450.00,
            "refund_amount": 450.00,
            "days_since_purchase": 3,
            "customer_age_days": 90,
            "total_orders": 5,
            "previous_refunds": 2,
            "account_age_days": 180,
            "customer_service_contacts": 3,
            "login_frequency_30d": 10,
            "products_viewed": 30,
            "product_category": "Home",
            "product_price": 450.00,
            "is_high_value": 0,
            "refund_reason": "defective",
            "has_images": 0,
            "shipping_address_changes": 1,
            "is_weekend": 0,
            "hour_of_day": 18,
            "refund_ratio": 0.4,
            "refund_speed": 1,
            "late_night": 0
        }
    }
]

def test_fraud_detection():
    print("=" * 70)
    print("🚨 FRAUD DETECTION SYSTEM - DEMO")
    print("=" * 70)
    print()
    
    for test in test_cases:
        print(f"📝 Testing: {test['name']}")
        print("-" * 70)
        
        try:
            response = requests.post(API_URL, json=test['data'])
            
            if response.status_code == 200:
                result = response.json()
                
                # Display results
                print(f"Transaction ID:      {result['transaction_id']}")
                print(f"Fraud Detected:      {'🚨 YES' if result['is_fraud'] else '✅ NO'}")
                print(f"Fraud Probability:   {result['fraud_probability']*100:.2f}%")
                print(f"Risk Level:          {result['risk_level']}")
                print(f"Confidence:          {result['confidence']*100:.2f}%")
                print(f"Inference Time:      {result['inference_time_ms']:.2f}ms")
                
                # Business decision
                print(f"\n💼 Recommended Action:")
                if result['risk_level'] == 'HIGH':
                    print("   → ⛔ BLOCK refund - Send to manual review")
                    print("   → 🔔 Alert fraud team immediately")
                elif result['risk_level'] == 'MEDIUM':
                    print("   → ⚠️  REQUEST additional verification")
                    print("   → 📸 Ask customer to upload photos")
                else:
                    print("   → ✅ AUTO-APPROVE refund")
                    print("   → 📧 Send confirmation email to customer")
            else:
                print(f"❌ Error: {response.status_code}")
                print(response.text)
        
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print()
        print("=" * 70)
        print()

if __name__ == "__main__":
    print("⚡ Starting Fraud Detection Demo...\n")
    test_fraud_detection()
    print("✅ Demo complete!")