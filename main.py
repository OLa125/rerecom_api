from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
from Recommendation import ContentBasedRecommender

# إنشاء تطبيق FastAPI
app = FastAPI()

# ✅ تمكين CORS لكل origins (أي بورت أو دومين)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ← يسمح بأي دومين/بورت
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# إنشاء نموذج التوصية
recommender = ContentBasedRecommender()

# تعريف نماذج البيانات
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

# نقطة تدريب الموديل
@app.post("/train")
def train_model(products: List[Product]):
    try:
        df = pd.DataFrame([p.dict() for p in products])
        recommender.fit(df)
        recommender.user_profiles = {}  # ← مسح ملفات المستخدمين القديمة
        return {"message": f"Trained on {len(products)} products."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# نقطة تحديث نشاطات المستخدم
@app.post("/update-user-logs")
def update_user_logs(logs: List[UserActivity]):
    try:
        df = pd.DataFrame([l.dict() for l in logs])
        recommender.update_user_profiles(df)
        return {"message": "Updated profiles for users."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# نقطة التوصية بمنتجات مشابهة
@app.post("/recommend-product")
def recommend_for_product(req: ProductRequest):
    try:
        recommendations = recommender.recommend_similar_products(req.product_id, req.top_n)
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# نقطة التوصية لمستخدم معين
@app.post("/recommend-user")
def recommend_for_user(req: UserRequest):
    try:
        recommendations = recommender.recommend_for_user(req.user_id, req.top_n)
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# نقطة اختبار بسيطة
@app.get("/")
def root():
    return {"message": "✅ API is running with CORS enabled!"}
