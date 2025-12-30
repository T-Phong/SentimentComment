
# Thay bằng repo_id bạn đã tạo ở trên
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_name_3sentiment = "phongnt251199/phobert-sentiment-reviews-v5"
model_name_5sentiment = "phongnt251199/phobert-sentiment-reviews-v4"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Device being used: {device}")

# Load tokenizer và model cho 3 nhãn
logger.info(f"Start loading model 3sentiment: {model_name_3sentiment}")
try:
    tokenizer_3sentiment = AutoTokenizer.from_pretrained(model_name_3sentiment)
    model_3sentiment = AutoModelForSequenceClassification.from_pretrained(model_name_3sentiment, num_labels=3).to(device)
    logger.info("Loaded model 3sentiment successfully")
except Exception as e:
    logger.error(f"Error loading model 3sentiment: {e}")
    raise e

# Load tokenizer và model như bình thườngclear
logger.info(f"Start loading model 5sentiment: {model_name_5sentiment}")
try:
    tokenizer_5sentiment = AutoTokenizer.from_pretrained(model_name_5sentiment)
    model_5sentiment = AutoModelForSequenceClassification.from_pretrained(model_name_5sentiment, num_labels=5).to(device)
    logger.info("Loaded model 5sentiment successfully")
except Exception as e:
    logger.error(f"Error loading model 5sentiment: {e}")
    raise e

def predict_sentiment_3sentiment(text):
    inputs = tokenizer_3sentiment(text, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model_3sentiment(**inputs)
        probs = outputs.logits.softmax(dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        # Dòng sau dùng để debug, có thể xóa hoặc comment lại
        # probs_point = F.softmax(outputs.logits, dim=1)
        # print(probs_point[0])
    #labels_map = {0: 'Rất tệ', 1: 'Tệ', 2: 'Bình thường', 3: 'Khá tốt', 4: 'Rất tốt'}
    labels_map = {0: 'Tiêu cực', 1: 'Bình thường', 2: 'Tích cực'}
    return labels_map[pred_label], probs[0][pred_label].item()

def predict_sentiment_5sentiment(text):
    inputs = tokenizer_5sentiment(text, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model_5sentiment(**inputs)
        probs = outputs.logits.softmax(dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        # Dòng sau dùng để debug, có thể xóa hoặc comment lại
        # probs_point = F.softmax(outputs.logits, dim=1)
        # print(probs_point[0])
    labels_map = {0: 'Rất tệ (1 sao)', 1: 'Tệ (2 sao)', 2: 'Bình thường (3 sao)', 3: 'Khá tốt (4 sao)', 4: 'Rất tốt (5 sao)'}
    return labels_map[pred_label], probs[0][pred_label].item()