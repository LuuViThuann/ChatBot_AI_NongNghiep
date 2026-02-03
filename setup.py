"""
setup.py — Script thiết lập ban đầu.
Kiểm tra cấu trúc, download nltk data, và chạy data_processing pipeline.

Chạy: python setup.py
"""
import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


def check_env():
    """Kiểm tra .env file có GROQ_API_KEY."""
    env_path = os.path.join(BASE_DIR, ".env")
    if not os.path.exists(env_path):
        print("⚠️  File .env chưa tồn tại.")
        print("    Tạo file .env và thêm: GROQ_API_KEY=gsk_xxxxxxxxxxxx")
        print("    Lấy API key tại: https://console.groq.com\n")
    else:
        from dotenv import load_dotenv
        load_dotenv(env_path)
        key = os.getenv("GROQ_API_KEY", "")
        if key and key != "your_groq_api_key_here":
            print("✅ GROQ_API_KEY đã được đặt.")
        else:
            print("⚠️  GROQ_API_KEY chưa hợp lệ trong .env")


def check_data():
    """Kiểm tra JSON, CSV và thư mục Images."""
    import sys
    sys.path.insert(0, BASE_DIR)
    from config import JSON_PATH, CSV_PATH

    img_path = os.path.join(BASE_DIR, "data", "Images")

    print("\n📁 Kiểm tra dữ liệu:")

    # JSON (primary)
    if os.path.exists(JSON_PATH):
        size_mb = os.path.getsize(JSON_PATH) / (1024 * 1024)
        print(f"  ✅ JSON tìm thấy: {JSON_PATH} ({size_mb:.1f} MB)")
    else:
        print(f"  ⚠️  JSON không tìm thấy: {JSON_PATH}")

    # CSV (fallback)
    if os.path.exists(CSV_PATH):
        size_mb = os.path.getsize(CSV_PATH) / (1024 * 1024)
        print(f"  ✅ CSV tìm thấy:  {CSV_PATH} ({size_mb:.1f} MB)")
    else:
        print(f"  ⚠️  CSV không tìm thấy:  {CSV_PATH}")

    # Cần ít nhất 1 trong 2
    if not os.path.exists(JSON_PATH) and not os.path.exists(CSV_PATH):
        print("  ❌ Cần ít nhất JSON hoặc CSV!")
        print("      Tải về từ: https://huggingface.co/datasets/SyedNazmusSakib/PlantVillageVQA")
        print("      Đặt vào thư mục: data/")

    if os.path.exists(img_path):
        img_count = sum(1 for f in os.listdir(img_path) if f.lower().endswith(('.jpg', '.png', '.jpeg')))
        print(f"  ✅ Thư mục Images tìm thấy ({img_count:,} ảnh)")
    else:
        print(f"  ⚠️  Thư mục Images không tìm thấy: {img_path}")
        print("      Tải về từ: https://huggingface.co/datasets/SyedNazmusSakib/PlantVillageVQA")


def check_artifacts():
    """Kiểm tra artifacts đã build chưa."""
    artifacts_dir = os.path.join(BASE_DIR, "artifacts")
    files_needed = ["tfidf_matrix.pkl", "tfidf_vectorizer.pkl", "embeddings_cache.pkl"]

    print("\n📦 Kiểm tra artifacts:")
    all_ok = True
    for f in files_needed:
        fpath = os.path.join(artifacts_dir, f)
        if os.path.exists(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  ✅ {f} ({size_kb:.1f} KB)")
        else:
            print(f"  ❌ {f} — chưa tạo")
            all_ok = False

    return all_ok


def build_artifacts():
    """Chạy data_processing để build all artifacts."""
    print("\n🔨 Đang build artifacts...")
    print("=" * 50)
    from data_processing import build_all
    build_all()
    print("=" * 50)


def print_usage():
    """In hướng dẫn sử dụng."""
    print("\n" + "=" * 60)
    print("🌿  AGRIBOT — Trợ lý AI Nông Nghiệp")
    print("=" * 60)
    print("\n📋 Cách chạy ứng dụng:")
    print("   streamlit run main.py")
    print("\n📋 Cấu trúc dự án:")
    print("   ├── main.py              # Streamlit frontend")
    print("   ├── config.py            # Cấu hình hệ thống")
    print("   ├── data_processing.py   # Xử lý JSON/CSV, build index")
    print("   ├── recommendation.py    # Hệ thống tra cứu (Retrieval)")
    print("   ├── image_classifier.py  # Phân loại bệnh từ ảnh (CLIP)")
    print("   ├── groq_client.py       # Tích hợp Groq LLM API")
    print("   ├── translation.py       # Đa ngôn ngữ Vi/En")
    print("   ├── setup.py             # Script thiết lập")
    print("   ├── requirements.txt     # Dependencies")
    print("   ├── .env                 # API Keys (GROQ_API_KEY)")
    print("   ├── data/")
    print("   │   ├── PlantVillageVQA.json   # ← PRIMARY (193,609 QA pairs)")
    print("   │   ├── PlantVillageVQA.csv    # ← FALLBACK (same data)")
    print("   │   └── Images/                # Ảnh lá (55,448 files)")
    print("   └── artifacts/           # Cached models & indexes")
    print("       ├── tfidf_matrix.pkl")
    print("       ├── tfidf_vectorizer.pkl")
    print("       ├── embeddings_cache.pkl")
    print("       └── label_encoder.pkl")
    print("\n📋 Schema PlantVillageVQA (JSON & CSV):")
    print("   image_id | question_type | question | answer | image_path | split")
    print("   • 14 crops · 38 diseases · 9 question categories")
    print("   • Plant + Disease được parse từ image_path/image_id")
    print("\n📋 Bước thiết lập:")
    print("   1. pip install -r requirements.txt")
    print("   2. Tạo .env, đặt GROQ_API_KEY=gsk_xxxx")
    print("   3. Tải PlantVillageVQA.csv + Images/ vào data/")
    print("   4. python setup.py          # Build artifacts")
    print("   5. streamlit run main.py    # Chạy app")
    print("\n" + "=" * 60)


def main():
    print("🌿  AGRIBOT — Setup & Validation")
    print("-" * 40)

    check_env()
    check_data()
    artifacts_ok = check_artifacts()

    if not artifacts_ok:
        print("\n💡 Artifacts chưa đầy đủ. Bạn muốn build ngay bây giờ? (y/n)")
        choice = input("   > ").strip().lower()
        if choice in ("y", "yes", ""):
            try:
                build_artifacts()
            except Exception as e:
                print(f"\n❌ Build thất bại: {e}")
                print("   Kiểm tra CSV file tại data/PlantVillageVQA.csv")
        else:
            print("   Bỏ qua build. Chạy 'python setup.py' sau khi có dữ liệu.")
    else:
        print("\n✅ Tất cả artifacts đã sẵn sàng!")

    print_usage()


if __name__ == "__main__":
    main()