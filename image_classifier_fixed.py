"""
image_classifier.py — Phân loại bệnh cây trồng từ ảnh.
✅ FIXED: Proper DLL path initialization BEFORE any torch import
"""

# ═══════════════════════════════════════════
# CRITICAL: FIX TORCH DLL *BEFORE* ANY IMPORTS
# This must be the FIRST thing that runs
# ═══════════════════════════════════════════
import os
import sys

# Create a global flag to ensure this only runs once across all modules
if not hasattr(sys, '_torch_dll_path_fixed'):
    print("[IMG-INIT] Applying DLL path fix...")
    try:
        for path in sys.path:
            torch_lib = os.path.join(path, 'torch', 'lib')
            if os.path.exists(torch_lib):
                # Method 1: add_dll_directory (Windows 10+)
                if hasattr(os, 'add_dll_directory'):
                    try:
                        os.add_dll_directory(torch_lib)
                        print(f"[IMG-FIX] ✅ Added to DLL directory: {torch_lib}")
                    except Exception as e:
                        print(f"[IMG-FIX] ⚠️ add_dll_directory failed: {e}")
                
                # Method 2: PATH environment (universal, more reliable)
                current_path = os.environ.get('PATH', '')
                if torch_lib not in current_path:
                    # Put torch lib at the BEGINNING of PATH
                    os.environ['PATH'] = torch_lib + os.pathsep + current_path
                    print(f"[IMG-FIX] ✅ Added to PATH: {torch_lib}")
                
                # Mark as fixed
                sys._torch_dll_path_fixed = True
                break
        
        if not hasattr(sys, '_torch_dll_path_fixed'):
            print(f"[IMG-FIX] ⚠️ Could not find torch lib in sys.path")
            sys._torch_dll_path_fixed = False
            
    except Exception as e:
        print(f"[IMG-FIX] ⚠️ Exception during DLL fix: {e}")
        sys._torch_dll_path_fixed = False
else:
    print("[IMG-INIT] DLL path already fixed")

# Now safe to import other modules
import warnings
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")

print("[IMG-INIT] Starting Torch initialization...")

# ═══════════════════════════════════════════
# NOW IMPORT TORCH - After DLL path is fixed
# ═══════════════════════════════════════════
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    print(f"[IMG] ✅ Torch {torch.__version__} imported successfully")
    print(f"[IMG] 📊 Device: {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    print(f"[IMG] 🎯 CLIP classification is ENABLED")
except Exception as e:
    print(f"[IMG] ❌ Torch import failed: {e}")
    print("[IMG] " + "="*60)
    print("[IMG] 🔧 GIẢI PHÁP / SOLUTION:")
    print("[IMG] " + "="*60)
    print("[IMG] ")
    print("[IMG] Torch đã được test và hoạt động tốt khi chạy trực tiếp,")
    print("[IMG] nhưng bị lỗi khi import trong Streamlit.")
    print("[IMG] ")
    print("[IMG] ĐÂY LÀ VẤN ĐỀ VỀ THỨ TỰ IMPORT!")
    print("[IMG] ")
    print("[IMG] GIẢI PHÁP:")
    print("[IMG] 1. Tạo file sitecustomize.py để fix DLL path globally:")
    print("[IMG]    python permanent_fix.py")
    print("[IMG] ")
    print("[IMG] 2. Hoặc luôn chạy app với:")
    print("[IMG]    python -m streamlit run main.py")
    print("[IMG] ")
    print("[IMG] 3. Hoặc dùng launcher:")
    print("[IMG]    START_AGRIBOT.bat")
    print("[IMG] ")
    print("[IMG] " + "="*60)
    print("[IMG] ⚠️ Image classification is DISABLED until fixed")
    print("[IMG] " + "="*60)
    TORCH_AVAILABLE = False

from config import LABEL_ENCODER_PATH, IMAGES_DIR


def _make_prompts(plant: str, disease: str, is_healthy: bool) -> list[str]:
    """
    Tạo list text prompts cho 1 label.
    Multi-prompt → average → confidence cao hơn.
    """
    plant = plant.strip()
    disease = disease.strip()

    if is_healthy:
        return [
            f"a photo of a healthy {plant} leaf",
            f"a {plant} leaf that is healthy and green",
            f"a close up of a healthy {plant} leaf with no disease",
            f"a normal {plant} leaf",
        ]
    else:
        return [
            f"a photo of a {plant} leaf infected with {disease}",
            f"a {plant} leaf showing symptoms of {disease}",
            f"a close up of a {plant} leaf with {disease}",
            f"a diseased {plant} leaf with {disease} infection",
        ]


class ImageClassifier:
    """
    Zero-shot image classifier dùng CLIP.
    Khởi tạo 1 lần, reuse trong session.
    """

    def __init__(self):
        self.labels     = self._load_labels()
        self.model      = None
        self.preprocess = None
        self.device     = None
        self._use_open_clip = False
        self._load_clip()

    # ──────────────────────────────────────
    # Inject labels from live DataFrame
    # ──────────────────────────────────────
    def set_labels_from_df(self, df):
        if "Plant" not in df.columns or "Disease" not in df.columns:
            return
        pairs  = (df["Plant"].astype(str) + "___" + df["Disease"].astype(str)).unique()
        labels = [p for p in pairs if "Unknown" not in p]
        if labels:
            self.labels = sorted(set(labels))
            print(f"[IMG] Labels updated from DataFrame: {len(self.labels)} classes.")

    # ──────────────────────────────────────
    # Load labels từ LabelEncoder
    # ──────────────────────────────────────
    def _load_labels(self) -> list[str]:
        import joblib
        if os.path.exists(LABEL_ENCODER_PATH):
            le = joblib.load(LABEL_ENCODER_PATH)
            labels = list(le.classes_)
            print(f"[IMG] Loaded {len(labels)} disease labels from LabelEncoder.")
            return labels
        else:
            print("[IMG] ⚠️ LabelEncoder not found. Using fallback labels.")
            return [
                "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
                "Blueberry___healthy",
                "Cherry_(including_sour_cherry)___Powdery_mildew", "Cherry_(including_sour_cherry)___healthy",
                "Corn_(Maize)___Cercospora_leaf_blight_Gray_leaf_spot",
                "Corn_(Maize)___Common_rust_", "Corn_(Maize)___Northern_Leaf_Blight", "Corn_(Maize)___healthy",
                "Grape___Black_rot", "Grape___Downy_mildew", "Grape___Leaf_scorch", "Grape___healthy",
                "Orange___Huanglongbing_(Citrus_greening)",
                "Peach___Bacterial_spot", "Peach___healthy",
                "Pepper,_Bell___Bacterial_spot", "Pepper,_Bell___healthy",
                "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
                "Raspberry___healthy",
                "Strawberry___Leaf_scorch", "Strawberry___healthy",
                "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
                "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
                "Tomato___Spider_mites_(Two-spotted_spider_mite)",
                "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
                "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
            ]

    # ──────────────────────────────────────
    # Load CLIP
    # ──────────────────────────────────────
    def _load_clip(self):
        if not TORCH_AVAILABLE:
            print("[IMG] ⚠️ Torch not available. Image classification disabled.")
            print("[IMG] 💡 App will continue working but cannot classify images.")
            self.model = None
            return
        
        try:
            import clip
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[IMG] Loading CLIP model on {self.device}...")
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            self.model.eval()
            print("[IMG] ✅ CLIP model loaded (openai/clip).")
            
        except (ImportError, OSError, RuntimeError) as e:
            print(f"[IMG] ⚠️ OpenAI CLIP failed: {e}")
            
            try:
                import open_clip
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"[IMG] Loading open_clip on {self.device}...")
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="openai", device=self.device
                )
                self.model.eval()
                self._tokenize = open_clip.tokenize
                self._use_open_clip = True
                print("[IMG] ✅ CLIP model loaded (open_clip).")
                
            except (ImportError, OSError, RuntimeError) as e2:
                print(f"[IMG] ⚠️ open_clip also failed: {e2}")
                print("[IMG] ❌ Image classification completely disabled.")
                self.model = None

    # ──────────────────────────────────────
    # Parse label → (plant, disease, is_healthy)
    # ──────────────────────────────────────
    @staticmethod
    def _parse_label(label: str):
        if "___" in label:
            plant, disease = label.split("___", 1)
        else:
            plant, disease = label, label

        plant   = plant.replace("_", " ").replace("(", "").replace(")", "").strip()
        disease = disease.replace("_", " ").replace("(", "").replace(")", "").strip()
        is_healthy = "healthy" in disease.lower()
        return plant, disease, is_healthy

    # ──────────────────────────────────────
    # Classify
    # ──────────────────────────────────────
    def classify(self, image_path_or_pil, top_k: int = 3) -> list[dict]:
        if self.model is None:
            print("[IMG] ⚠️ Classification requested but model is not loaded")
            print("[IMG] 💡 Returning placeholder result")
            return [{"label": "N/A", "confidence": 0.0, "plant": "Unknown", "disease": "Torch not available"}]

        # Load image
        if isinstance(image_path_or_pil, str):
            img = Image.open(image_path_or_pil).convert("RGB")
        else:
            img = image_path_or_pil.convert("RGB")

        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

        # ── Encode image ──
        with torch.no_grad():
            image_features = self.model.encode_image(img_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # ── Build prompts + compute scores per label ──
        all_prompts  = []
        prompt_map   = []

        for idx, label in enumerate(self.labels):
            plant, disease, is_healthy = self._parse_label(label)
            prompts = _make_prompts(plant, disease, is_healthy)
            all_prompts.extend(prompts)
            prompt_map.append((idx, len(prompts)))

        # Tokenize + encode all prompts at once
        try:
            import clip
            text_tokens = clip.tokenize(all_prompts).to(self.device)
        except ImportError:
            text_tokens = self._tokenize(all_prompts).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarities = (image_features @ text_features.T).squeeze(0)

        # ── Average per label ──
        sims_np = similarities.cpu().numpy()
        offset  = 0
        per_label_scores = np.zeros(len(self.labels))

        for label_idx, n_prompts in prompt_map:
            per_label_scores[label_idx] = sims_np[offset:offset + n_prompts].mean()
            offset += n_prompts

        # ── Softmax → probabilities ──
        scores_tensor = torch.tensor(per_label_scores, dtype=torch.float32)
        probs         = F.softmax(scores_tensor * 100.0, dim=0).numpy()

        # ── Top-k ──
        top_indices = np.argsort(probs)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            label = self.labels[idx]
            plant, disease, _ = self._parse_label(label)
            results.append({
                "label":      label,
                "plant":      plant,
                "disease":    disease,
                "confidence": float(probs[idx]) * 100
            })

        return results

    # ──────────────────────────────────────
    # Get unique plants
    # ──────────────────────────────────────
    def get_plants(self) -> list[str]:
        plants = set()
        for label in self.labels:
            plant, _, _ = self._parse_label(label)
            plants.add(plant)
        return sorted(plants)