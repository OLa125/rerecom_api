
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
from Recommendation import ContentBasedRecommender

app = FastAPI()

recommender = ContentBasedRecommender()

class Product(BaseModel):
    id: int
    name: str
    category: str
    description: str

class UserActivity(BaseModel):
    user_id: str
    product_id: int
    event_type: str

class ProductRequest(BaseModel):
    product_id: int
    top_n: int = 5

class UserRequest(BaseModel):
    user_id: str
    top_n: int = 5

@app.post("/train")
def train_model(products: List[Product]):
    df = pd.DataFrame([p.dict() for p in products])
    recommender.fit(df)
    return {"message": f"Trained on {len(products)} products."}

@app.post("/update-user-logs")
def update_user_logs(logs: List[UserActivity]):
    df = pd.DataFrame([l.dict() for l in logs])
    recommender.update_user_profiles(df)
    return {"message": f"Updated profiles for users."}

@app.post("/recommend-product")
def recommend_for_product(req: ProductRequest):
    try:
        recommendations = recommender.recommend_similar_products(req.product_id, req.top_n)
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/recommend-user")
def recommend_for_user(req: UserRequest):
    try:
        recommendations = recommender.recommend_for_user(req.user_id, req.top_n)
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
