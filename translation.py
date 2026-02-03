"""
translation.py — Hỗ trợ đa ngôn ngữ (Vietnamese ↔ English).
Không dùng API ngoài — chỉ dùng dictionaries từ-khái niệm cố định
kết hợp với Groq để translate dynamically khi cần.
"""
from config import GROQ_API_KEY, GROQ_MODEL

# ──────────────────────────────────────────
# Static keyword dictionaries (cả hai chiều)
# ──────────────────────────────────────────
EN_TO_VI = {
    # Diseases
    "late blight":              "bệnh héo lá muộn",
    "early blight":             "bệnh héo lá sớm",
    "powdery mildew":           "bệnh phấn trắng",
    "downy mildew":             "bệnh héo lá mưa",
    "black rot":                "bệnh thối đen",
    "apple scab":               "bệnh ghẻ táo",
    "leaf spot":                "bệnh đốm lá",
    "rust":                     "bệnh rỉ lá",
    "blight":                   "bệnh héo",
    "mosaic virus":             "bệnh virus khảm",
    "bacterial spot":           "bệnh đốm vi khuẩn",
    "septoria leaf spot":       "bệnh đốm lá septoria",
    "target spot":              "bệnh đốm vòng lá",
    "leaf mold":                "bệnh nấm lá",
    "leaf scorch":              "bệnh cháy lá",
    "gray leaf spot":           "bệnh đốm lá xám",
    "northern leaf blight":     "bệnh héo lá phía bắc",
    "citrus greening":          "bệnh vàng lá cam",
    "huanglongbing":            "bệnh vàng lá cam",
    "spider mites":             "bệnh mò nhện",
    "cedar apple rust":         "bệnh rỉ lá tuyết giáp",

    # Plants
    "tomato":                   "cà chua",
    "potato":                   "khoai tây",
    "apple":                    "táo",
    "grape":                    "nho",
    "corn":                     "ngô",
    "maize":                    "ngô",
    "pepper":                   "ớt",
    "bell pepper":              "ớt chuông",
    "strawberry":               "dâu tây",
    "cherry":                   "anh đào",
    "peach":                    "đào",
    "orange":                   "cam",
    "blueberry":                "việt quất",
    "raspberry":                "mâm xôi",
    "citrus":                   "cam quất",

    # General agriculture terms
    "disease":                  "bệnh",
    "healthy":                  "khỏe mạnh",
    "leaf":                     "lá",
    "stem":                     "thân cây",
    "root":                     "rễ",
    "fruit":                    "quả",
    "symptom":                  "triệu chứng",
    "treatment":                "điều trị",
    "prevention":               "phòng ngừa",
    "fertilizer":               "phân bón",
    "pesticide":                "thuốc trừ sâu",
    "fungicide":                "thuốc trừ nấm",
    "irrigation":               "tưới nước",
    "crop":                     "cây trồng",
    "harvest":                  "thu hoạch",
    "soil":                     "đất",
    "water":                    "nước",
    "sunlight":                 "ánh sáng mặt trời",
    "temperature":              "nhiệt độ",
    "humidity":                 "độ ẩm",
    "organic":                  "hữu cơ",
    "diagnosis":                "chẩn đoán",
    "confidence":               "độ chắc chắn",
}

# Reverse mapping
VI_TO_EN = {v: k for k, v in EN_TO_VI.items()}


def detect_language(text: str) -> str:
    """
    Simple heuristic: nếu text chứa nhiều ký tự diacritical marks → Vietnamese.
    Ngược lại → English.
    """
    vi_chars = set("àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợỏỡùúụủũưừứựỳýỵỷỹđ"
                   "ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỎỠÙÚỤỦŨƯỪỨỰỲÝỴỶỸĐ")
    vi_count = sum(1 for c in text if c in vi_chars)
    # If >8% of alpha chars are Vietnamese → Vietnamese
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count == 0:
        return "vi"  # default
    ratio = vi_count / alpha_count
    return "vi" if ratio > 0.04 else "en"


def translate_label_to_vi(label: str) -> str:
    """Translate a disease/plant label like 'Tomato___Late_blight' → tiếng Việt."""
    label_clean = label.replace("___", " - ").replace("_", " ").strip()
    parts = label_clean.split(" - ")

    translated_parts = []
    for part in parts:
        part_lower = part.lower().strip()
        if part_lower in EN_TO_VI:
            translated_parts.append(EN_TO_VI[part_lower])
        else:
            # Try partial match
            found = False
            for en_key, vi_val in EN_TO_VI.items():
                if en_key in part_lower:
                    translated_parts.append(vi_val)
                    found = True
                    break
            if not found:
                translated_parts.append(part)  # keep original

    return " - ".join(translated_parts)


def translate_label_to_en(label: str) -> str:
    """Ensure label is in English format."""
    label_clean = label.replace("___", " - ").replace("_", " ").strip()
    return label_clean


def get_system_prompt(lang: str) -> str:
    """Return system prompt in the selected language."""
    from config import SYSTEM_PROMPT_VI, SYSTEM_PROMPT_EN
    return SYSTEM_PROMPT_VI if lang == "vi" else SYSTEM_PROMPT_EN


def format_disease_info(plant: str, disease: str, confidence: float, lang: str) -> str:
    """Format disease diagnosis info in selected language."""
    if lang == "vi":
        plant_vi   = translate_label_to_vi(plant)
        disease_vi = translate_label_to_vi(disease)
        healthy_text = "khỏe mạnh"
        is_healthy = "healthy" in disease.lower()
        return (
            f"🌱 **Cây trồng:** {plant_vi}\n"
            f"🔍 **Chẩn đoán:** {'✅ ' + healthy_text if is_healthy else '⚠️ ' + disease_vi}\n"
            f"📊 **Độ chắc chắn:** {confidence:.1f}%"
        )
    else:
        is_healthy = "healthy" in disease.lower()
        return (
            f"🌱 **Plant:** {plant}\n"
            f"🔍 **Diagnosis:** {'✅ Healthy' if is_healthy else '⚠️ ' + disease}\n"
            f"📊 **Confidence:** {confidence:.1f}%"
        )