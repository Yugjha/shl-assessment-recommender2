
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import requests
from bs4 import BeautifulSoup

app = FastAPI(title="SHL Assessment Recommendation API")

# Load model and data
model = SentenceTransformer('all-MiniLM-L6-v2')
df_assessments = pd.read_csv('shl_assessments.csv')
embeddings = np.load('embeddings.npy')
index = faiss.read_index('assessments.index')

class RecommendRequest(BaseModel):
    query: str

class Assessment(BaseModel):
    name: str
    url: str
    description: Optional[str] = ""
    test_type: Optional[str] = ""
    score: float

class RecommendResponse(BaseModel):
    success: bool
    recommendations: List[Assessment]

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    try:
        query = request.query
        
        # Check if query is URL
        if query.startswith('http'):
            # Fetch content from URL
            response = requests.get(query, timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')
            query = soup.get_text(separator=' ', strip=True)[:2000]
        
        # Encode query
        query_embedding = model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = index.search(query_embedding.astype('float32'), 10)
        
        # Build results
        recommendations = []
        for i, idx in enumerate(indices[0]):
            if idx < len(df_assessments):
                row = df_assessments.iloc[idx]
                recommendations.append(Assessment(
                    name=row.get('name', 'Unknown'),
                    url=row.get('url', ''),
                    description=str(row.get('description', ''))[:200],
                    test_type=str(row.get('test_type', '')),
                    score=float(scores[0][i])
                ))
        
        return RecommendResponse(success=True, recommendations=recommendations)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
