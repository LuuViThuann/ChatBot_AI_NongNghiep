"""
config.py — Cấu hình toàn bộ dự án Agricultural AI Chatbot.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────
# API KEYS
# ──────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ──────────────────────────────────────────
# MODEL SETTINGS
# ──────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"          # Model chính cho Q&A
GROQ_MODEL_VISION = "llava-v1.5-7b"     # Fallback text-only (Groq không support vision trực tiếp → dùng local CLIP)

# ──────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────
BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
DATA_DIR            = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR       = os.path.join(BASE_DIR, "artifacts")
IMAGES_DIR          = os.path.join(BASE_DIR, "data", "Images")   # Folder ảnh từ HuggingFace
CSV_PATH            = os.path.join(DATA_DIR, "PlantVillageVQA.csv")
JSON_PATH           = os.path.join(DATA_DIR, "PlantVillageVQA.json")

# ──────────────────────────────────────────
# EMBEDDING / RETRIEVAL
# ──────────────────────────────────────────
EMBEDDING_MODEL     = "all-MiniLM-L6-v2"
TFIDF_MATRIX_PATH   = os.path.join(ARTIFACTS_DIR, "tfidf_matrix.pkl")
TFIDF_VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.pkl")
EMBEDDING_CACHE_PATH  = os.path.join(ARTIFACTS_DIR, "embeddings_cache.pkl")
LABEL_ENCODER_PATH    = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")

# ──────────────────────────────────────────
# IMAGE CLASSIFICATION (Local CLIP / torchvision)
# ──────────────────────────────────────────
USE_LOCAL_CLIP      = True              # True → dùng CLIP local để classify ảnh
CLIP_MODEL_NAME     = "ViT-B/32"        # openai CLIP

# ──────────────────────────────────────────
# LANGUAGE
# ──────────────────────────────────────────
DEFAULT_LANG        = "vi"              # "vi" hoặc "en"
SUPPORTED_LANGS     = ["vi", "en"]

# ──────────────────────────────────────────
# GROQ CHAT SETTINGS
# ──────────────────────────────────────────
MAX_TOKENS          = 1500
TEMPERATURE         = 0.4
SYSTEM_PROMPT_VI = """Bạn là trợ lý AI chuyên tư vấn nông nghiệp cho nông dân Việt Nam. 
Bạn có kiến thức sâu về bệnh cây trồng, sâu bệnh, kỹ thuật canh tác, điều kiện khí hậu và các phương pháp phòng chữa bệnh.
Hãy trả lời bằng tiếng Việt, ngôn ngữ đơn giản, dễ hiểu. 
Nếu được cung cấp thông tin phân đoán từ ảnh hoặc hệ thống tra cứu, hãy sử dụng thông tin đó để đưa ra phản hồi chính xác và hữu ích.
Luôn đưa ra lời khuyên thực hành, bao gồm các biện pháp phòng ngừa và điều trị."""

SYSTEM_PROMPT_EN = """You are an AI agricultural advisor specializing in plant disease diagnosis and farming guidance.
You have deep knowledge of crop diseases, pests, cultivation techniques, climate conditions and disease prevention methods.
Answer in English, using clear and simple language.
If you are provided with diagnostic information from an image or a knowledge base retrieval, use that information to give accurate and helpful responses.
Always provide practical advice including prevention and treatment measures."""