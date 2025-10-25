from pydantic import BaseModel, Field

class RefundRequest(BaseModel):
    transaction_id: str
    order_amount: float = Field(gt=0)
    refund_amount: float = Field(gt=0)
    days_since_purchase: float = Field(ge=0)
    customer_age_days: int = Field(gt=0)
    total_orders: int = Field(ge=1)
    previous_refunds: int = Field(ge=0)
    account_age_days: int = Field(gt=0)
    customer_service_contacts: int = Field(ge=0)
    login_frequency_30d: int = Field(ge=0)
    products_viewed: int = Field(ge=0)
    product_category: str
    product_price: float
    is_high_value: int
    refund_reason: str
    has_images: int
    shipping_address_changes: int = Field(ge=0)
    is_weekend: int
    hour_of_day: int = Field(ge=0, le=23)
    refund_ratio: float = Field(ge=0, le=1)
    refund_speed: int
    late_night: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TXN00012345",
                "order_amount": 299.99,
                "refund_amount": 289.99,
                "days_since_purchase": 2,
                "customer_age_days": 45,
                "total_orders": 3,
                "previous_refunds": 1,
                "account_age_days": 180,
                "customer_service_contacts": 2,
                "login_frequency_30d": 5,
                "products_viewed": 25,
                "product_category": "Electronics",
                "product_price": 299.99,
                "is_high_value": 0,
                "refund_reason": "defective",
                "has_images": 0,
                "shipping_address_changes": 1,
                "is_weekend": 0,
                "hour_of_day": 14,
                "refund_ratio": 0.33,
                "refund_speed": 1,
                "late_night": 0
            }
        }

class PredictionResponse(BaseModel):
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    confidence: float
    inference_time_ms: float