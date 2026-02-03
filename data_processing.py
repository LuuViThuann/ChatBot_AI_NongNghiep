"""
data_processing.py — Xử lý dữ liệu PlantVillageVQA (JSON + CSV).
Tạo TF-IDF index + Sentence Embeddings để tra cứu câu hỏi.

=== FIXES ===
1. [NotFittedError] build_tfidf() giờ validate vectorizer bằng check_is_fitted()
   TRƯỚC khi dump .pkl. Nếu chưa fitted → raise lỗi rõ ràng thay vì dump file hỏng.
2. [Safety] Thêm len-check: nếu DataFrame rỗng hoặc Combined rỗng → skip build + warning.

JSON thực tế có cấu trúc nested-by-filename:
{
  "image_000000.JPG": {
      "image_path": "images/train/image_000000.JPG",
      "split": "train",
      "questions": [
          {"question_type": "...", "question": "...", "answer": "..."},
          ...
      ]
  },
  ...
}
→ Bước 1: Flatten thành 1 row per câu hỏi.
→ Bước 2: Extract Plant + Disease từ nội dung question/answer bằng keyword matching
   (image_id chỉ là "image_000000.JPG" — không chứa thông tin plant/disease).
"""
import os, warnings
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

warnings.filterwarnings("ignore")

from config import (
    CSV_PATH, JSON_PATH, ARTIFACTS_DIR,
    TFIDF_MATRIX_PATH, TFIDF_VECTORIZER_PATH,
    EMBEDDING_CACHE_PATH, LABEL_ENCODER_PATH,
    EMBEDDING_MODEL
)

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════
# KEYWORD TABLES — 14 cây trồng + 38 bệnh PlantVillage
# ══════════════════════════════════════════════════════
# Thứ tự quan trọng: multi-word / dài hơn đặt TRƯỚC single-word.

PLANT_KEYWORDS = [
    ("bell pepper",   "Pepper, Bell"),
    ("cherry",        "Cherry"),
    ("blueberry",     "Blueberry"),
    ("strawberry",    "Strawberry"),
    ("raspberry",     "Raspberry"),
    ("potato",        "Potato"),
    ("tomato",        "Tomato"),
    ("grape",         "Grape"),
    ("apple",         "Apple"),
    ("peach",         "Peach"),
    ("orange",        "Orange"),
    ("corn",          "Corn (Maize)"),
    ("maize",         "Corn (Maize)"),
    ("pepper",        "Pepper, Bell"),
    ("citrus",        "Orange"),
]

DISEASE_KEYWORDS = [
    # Multi-word trước
    ("powdery mildew",        "Powdery mildew"),
    ("downy mildew",          "Downy mildew"),
    ("apple scab",            "Apple scab"),
    ("black rot",             "Black rot"),
    ("cedar apple rust",      "Cedar apple rust"),
    ("early blight",          "Early blight"),
    ("late blight",           "Late blight"),
    ("leaf mold",             "Leaf Mold"),
    ("leaf scorch",           "Leaf scorch"),
    ("septoria leaf spot",    "Septoria leaf spot"),
    ("target spot",           "Target Spot"),
    ("spider mite",           "Spider mites"),
    ("northern leaf blight",  "Northern Leaf Blight"),
    ("cercospora",            "Cercospora leaf blight"),
    ("gray leaf spot",        "Gray leaf spot"),
    ("common rust",           "Common rust"),
    ("bacterial spot",        "Bacterial spot"),
    ("leaf spot",             "Leaf spot"),
    ("mosaic virus",          "Tomato mosaic virus"),
    ("yellow leaf curl",      "Tomato Yellow Leaf Curl Virus"),
    ("huanglongbing",         "Huanglongbing (Citrus greening)"),
    ("citrus greening",       "Huanglongbing (Citrus greening)"),
    # Single-word fallback
    ("rust",                  "Common rust"),
    ("blight",                "Late blight"),
    ("scab",                  "Apple scab"),
    ("mildew",                "Powdery mildew"),
    ("rot",                   "Black rot"),
    ("mold",                  "Leaf Mold"),
    ("virus",                 "Tomato mosaic virus"),
]

HEALTHY_PHRASES = (
    "healthy and free of disease",
    "free of disease",
    "appears to be healthy",
    "is healthy",
    "looks healthy",
    "no disease",
    "no sign of disease",
    "not infected",
    "not diseased",
    "healthy",          # ← single word LAST (catch-all)
)


def _extract_plant(text: str) -> str:
    """Tìm tên cây trồng đầu tiên trong text (case-insensitive)."""
    low = text.lower()
    for pattern, canonical in PLANT_KEYWORDS:
        if pattern in low:
            return canonical
    return "Unknown"


def _extract_disease(text: str) -> str:
    """
    Tìm tên bệnh trong text.
    Quy tắc: tìm disease keyword TRƯỚC, chỉ fallback về 'healthy'
    khi không tìm được bệnh nào cụ thể.
    """
    low = text.lower()

    # Thử tìm disease keyword cụ thể trước
    for pattern, canonical in DISEASE_KEYWORDS:
        if pattern in low:
            return canonical

    # Không có bệnh cụ thể → kiểm tra healthy
    for hp in HEALTHY_PHRASES:
        if hp in low:
            return "healthy"

    return "Unknown"


# ══════════════════════════════════════════════════════
# 1. LOAD JSON — flatten nested-by-filename structure
# ══════════════════════════════════════════════════════
def _load_from_json() -> pd.DataFrame:
    """
    Đọc JSON có cấu trúc:
        { "filename.jpg": { "image_path": "...", "split": "...", "questions": [...] }, ... }

    Flatten → DataFrame với 1 row per câu hỏi:
        image_id | image_path | split | question_type | Question | Answer
    """
    if not os.path.exists(JSON_PATH):
        return None

    import json
    print(f"[DATA] Đang đọc JSON: {JSON_PATH} …")

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        print("[DATA] ⚠️ JSON root không phải dict → fallback CSV.")
        return None

    # ── Detect format: kiểm tra first value ──
    first_key = next(iter(raw), None)
    first_val = raw.get(first_key) if first_key else None

    # CASE A: nested-by-filename  { "image_xxx.jpg": { "questions": [...] } }
    if isinstance(first_val, dict) and "questions" in first_val:
        return _flatten_nested(raw)

    # CASE B: các format khác (flat array, split-keys, dict-of-lists)
    print("[DATA] Thử parse format khác …")
    return _load_json_other(raw)


def _flatten_nested(raw: dict) -> pd.DataFrame:
    """
    Flatten cấu trúc nested-by-filename.
    Mỗi image có thể có nhiều questions → 1 row per question.
    """
    rows = []
    for filename, entry in raw.items():
        if not isinstance(entry, dict):
            continue

        image_path = entry.get("image_path", "")
        split      = entry.get("split", "")
        questions  = entry.get("questions", [])

        for q in questions:
            if not isinstance(q, dict):
                continue
            rows.append({
                "image_id":      filename,
                "image_path":    image_path,
                "split":         split,
                "question_type": q.get("question_type", "General"),
                "Question":      q.get("question", ""),
                "Answer":        q.get("answer", ""),
            })

    if not rows:
        print("[DATA] ⚠️ Flatten → 0 rows.")
        return None

    df = pd.DataFrame(rows)
    print(f"[DATA] ✅ Flatten: {len(df):,} câu hỏi từ {len(raw):,} ảnh.")
    return df


def _load_json_other(raw) -> pd.DataFrame:
    """Fallback cho flat JSON formats."""
    records = None

    if isinstance(raw, list):
        records = raw

    elif isinstance(raw, dict):
        # split-keys: {"train":[...], "test":[...]}
        split_names = {"train", "val", "validation", "test"}
        matching    = {k for k in raw if k in split_names and isinstance(raw[k], list)}
        if matching:
            records = []
            for sn in ("train", "val", "validation", "test"):
                if sn in raw and isinstance(raw[sn], list):
                    for item in raw[sn]:
                        item["split"] = sn
                    records.extend(raw[sn])

        # nested key: {"data":[...]}
        if not records:
            for key in ("data", "samples", "questions", "records"):
                if key in raw and isinstance(raw[key], list):
                    records = raw[key]
                    break

        # dict-of-lists: {"col1":[...], "col2":[...]}
        if not records:
            first_val = next(iter(raw.values()), None)
            if isinstance(first_val, list) and len(first_val) > 0 and not isinstance(first_val[0], dict):
                records = pd.DataFrame(raw).to_dict("records")

    if not records:
        print("[DATA] ⚠️ Không parse được JSON → fallback CSV.")
        return None

    df = pd.DataFrame(records)
    print(f"[DATA] Flat JSON: {len(df):,} records.")
    return df


# ══════════════════════════════════════════════════════
# 2. LOAD CSV (fallback)
# ══════════════════════════════════════════════════════
def _load_from_csv() -> pd.DataFrame:
    if not os.path.exists(CSV_PATH):
        return None
    print(f"[DATA] Đang đọc CSV: {CSV_PATH} …")
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    df.columns = df.columns.str.strip()
    print(f"[DATA] CSV: {len(df):,} records.")
    return df


# ══════════════════════════════════════════════════════
# 3. NORMALIZE — extract Plant/Disease + build Combined
# ══════════════════════════════════════════════════════
def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa DataFrame về schema cuối:
        image_id | image_path | split | question_type
        Question | Answer | Plant | Disease | Combined
    """
    # ── Rename columns (case-insensitive) ──
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if   cl in ("question", "q")            and c != "Question":  col_map[c] = "Question"
        elif cl in ("answer", "a", "ans")       and c != "Answer":    col_map[c] = "Answer"
        elif cl in ("image_path", "image", "image_name", "filename"): col_map[c] = "image_path"
        elif cl in ("image_id", "id", "idx"):                         col_map[c] = "image_id"
        elif cl in ("question_type", "q_type", "category", "type"):   col_map[c] = "question_type"
        elif cl in ("split", "partition", "set"):                     col_map[c] = "split"
        elif cl in ("plant", "plant_name", "crop", "species"):        col_map[c] = "Plant"
        elif cl in ("disease", "disease_name", "label", "class"):     col_map[c] = "Disease"
    df.rename(columns=col_map, inplace=True)

    # ── Validate ──
    for col in ("Question", "Answer"):
        if col not in df.columns:
            raise ValueError(f"Cột '{col}' thiếu. Columns: {list(df.columns)}")

    df.dropna(subset=["Question", "Answer"], inplace=True)
    df["Question"] = df["Question"].astype(str).str.strip()
    df["Answer"]   = df["Answer"].astype(str).str.strip()

    # ── Extract Plant từ Question + Answer ──
    if "Plant" not in df.columns:
        search = df["Question"] + " " + df["Answer"]
        df["Plant"] = search.apply(_extract_plant)
        n = (df["Plant"] != "Unknown").sum()
        print(f"[DATA] Extract Plant: {df['Plant'].nunique()} loại cây ({n}/{len(df)} rows matched).")

    # ── Extract Disease từ Answer trước, fallback sang Question+Answer ──
    if "Disease" not in df.columns:
        from_answer = df["Answer"].apply(_extract_disease)
        from_both   = (df["Question"] + " " + df["Answer"]).apply(_extract_disease)
        df["Disease"] = from_answer.where(from_answer != "Unknown", from_both)
        n = (df["Disease"] != "Unknown").sum()
        print(f"[DATA] Extract Disease: {df['Disease'].nunique()} loại bệnh ({n}/{len(df)} rows matched).")

    # ── Defaults ──
    for col, val in [("question_type", "General"), ("image_path", ""), ("split", "train")]:
        if col not in df.columns:
            df[col] = val
    if "image_id" not in df.columns:
        df["image_id"] = df.index.astype(str)

    # ── Combined text cho TF-IDF / embedding retrieval ──
    df["Combined"] = (
        df["Plant"].astype(str)   + " " +
        df["Disease"].astype(str) + " " +
        df["Question"]            + " " +
        df["Answer"]
    )

    print(f"[DATA] Schema: {list(df.columns)}")
    print(f"[DATA] Plants: {df['Plant'].nunique()} | Diseases: {df['Disease'].nunique()} | Q-types: {df['question_type'].nunique()}")
    return df


# ══════════════════════════════════════════════════════
# 4. MASTER LOADER
# ══════════════════════════════════════════════════════
def load_dataset() -> pd.DataFrame:
    """JSON (primary) → CSV (fallback) → normalize."""
    df = _load_from_json()
    if df is None:
        df = _load_from_csv()
    if df is None:
        raise FileNotFoundError(
            "Không tìm thấy dữ liệu!\n"
            f"  JSON: {JSON_PATH}\n"
            f"  CSV:  {CSV_PATH}\n"
            "Tải về: https://huggingface.co/datasets/SyedNazmusSakib/PlantVillageVQA"
        )
    df = _normalize(df)
    print(f"[DATA] ✅ Tổng: {len(df):,} mục.")
    return df


# ══════════════════════════════════════════════════════
# 5. BUILD ARTIFACTS
# ══════════════════════════════════════════════════════
def build_tfidf(df: pd.DataFrame):
    """
    Build TF-IDF vectorizer + matrix từ DataFrame.

    FIX: Sau fit_transform(), validate bằng check_is_fitted() trước khi dump.
    Nếu validation thất bại → raise lỗi thay vì dump file hỏng ra disk.
    """
    # ── Guard: check DataFrame hợp lệ ──
    if df is None or len(df) == 0:
        print("[TFIDF] ❌ DataFrame rỗng — skip build.")
        return

    if "Combined" not in df.columns:
        print("[TFIDF] ❌ Cột 'Combined' không tìm thấy — skip build.")
        return

    corpus = df["Combined"].tolist()
    # Filter out empty strings
    corpus_clean = [t for t in corpus if t and t.strip()]
    if len(corpus_clean) == 0:
        print("[TFIDF] ❌ Corpus rỗng sau clean — skip build.")
        return

    print(f"[TFIDF] Đang fit TF-IDF trên {len(corpus_clean):,} documents...")

    vectorizer   = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # ── Validate: kiểm tra fitted ──
    try:
        check_is_fitted(vectorizer, attributes=["idf_"])
        print("[TFIDF] ✅ Vectorizer validated — idf_ vector OK.")
    except NotFittedError:
        raise RuntimeError(
            "[TFIDF] ❌ CRITICAL: Vectorizer chưa fitted sau fit_transform()! "
            "Không dump .pkl. Kiểm tra corpus và scikit-learn version."
        )

    # ── Dump artifacts ──
    joblib.dump(vectorizer,  TFIDF_VECTORIZER_PATH)
    joblib.dump(tfidf_matrix, TFIDF_MATRIX_PATH)
    print(f"[TFIDF] ✅ {tfidf_matrix.shape} → artifacts/")


def build_embeddings(df: pd.DataFrame):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(df["Combined"].tolist(), show_progress_bar=True, batch_size=128)
    joblib.dump(embeddings, EMBEDDING_CACHE_PATH)
    print(f"[EMBED] {embeddings.shape} → artifacts/")


def build_label_encoder(df: pd.DataFrame):
    if "Disease" not in df.columns:
        return None
    le = LabelEncoder()
    le.fit(df["Disease"].astype(str).str.strip())
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"[LABEL] {len(le.classes_)} labels → artifacts/")


def build_all():
    df = load_dataset()
    build_tfidf(df)
    build_embeddings(df)
    build_label_encoder(df)
    print("\n✅ Build hoàn thành!")


if __name__ == "__main__":
    build_all()