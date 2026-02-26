from fastapi import FastAPI, HTTPException
from huggingface_hub import snapshot_download
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pyvi import ViTokenizer
import os
from pydantic import BaseModel, Field
import asyncio

app = FastAPI()

model_path = os.path.dirname(os.path.abspath(__file__)) + "/../hate-speech-model"
def checking_download(model_path):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        print("Đã tạo thư mục model rỗng")

    if not os.listdir(model_path):
        print("Thư mục model rỗng, bắt đầu tải về ...")
        snapshot_download(
            repo_id="VietAnh-1027/hate-speech-model",
            repo_type="model",
            local_dir=model_path
        )
    else: print("Thư mục model đã tồn tại")

checking_download(model_path)

classes = ["normal", "rude", "toxic"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)
model.to(device)
model.eval()

def classify_comment(text):
    text = ViTokenizer.tokenize(text)
    with torch.no_grad():
        inputs = tokenizer(text, padding="max_length", max_length=100, truncation=True, return_tensors="pt").to(device)
        outputs = model(**inputs)
        probabilites = F.softmax(outputs.logits, dim=1)
        confident, cls_id = torch.max(probabilites, dim=1)
    return round(confident.item()*100, 2), cls_id

class RequestComment(BaseModel):
    user: str
    comment: str = Field(..., min_length=1)

class ResponseComment(BaseModel):
    user: str
    comment: str
    type: str
    confident: float

@app.post("/predict", response_model=ResponseComment)
async def predict(request: RequestComment):
    try:
        conf_score, cls_id = await asyncio.to_thread(classify_comment, request.comment)
        return {
            "user": request.user,
            "comment": request.comment,
            "type": classes[cls_id],
            "confident": conf_score
        }
    except Exception as e:
        return HTTPException(500, detail=f"Lỗi server khi dự đoán, chi tiết: {e}")