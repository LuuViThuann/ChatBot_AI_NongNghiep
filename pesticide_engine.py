"""
pesticide_engine.py — Hệ thống tra cứu thuốc điều trị bệnh cây trồng.
Sử dụng dữ liệu từ Canada Pesticide Product Information Database (PPID):
  - ingredient_extract.csv  (hoạt chất thuốc)
  - product_extract.csv     (sản phẩm thuốc thương mại)

Mapping logic:
  Disease name → active ingredient(s) → commercial product(s)

Chạy: python pesticide_engine.py   (để test standalone)
"""

import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
INGREDIENT_CSV = os.path.join(DATA_DIR, "ingredient_extract.csv")
PRODUCT_CSV    = os.path.join(DATA_DIR, "product_extract.csv")


# ─── Disease → Active Ingredient keyword mapping ─────────────────────────────
# Ánh xạ tên bệnh cây → danh sách từ khóa hoạt chất liên quan.
# Dựa trên kiến thức nông nghiệp + PPID database.
DISEASE_TO_INGREDIENTS = {
    # Fungal diseases (Nấm)
    "late blight":             ["chlorothalonil", "mancozeb", "metalaxyl", "cymoxanil", "copper"],
    "early blight":            ["chlorothalonil", "mancozeb", "azoxystrobin", "difenoconazole"],
    "powdery mildew":          ["sulfur", "trifloxystrobin", "myclobutanil", "azoxystrobin", "tebuconazole"],
    "downy mildew":            ["mancozeb", "metalaxyl", "fosetyl", "copper", "cymoxanil"],
    "apple scab":              ["captan", "myclobutanil", "mancozeb", "ziram", "copper"],
    "black rot":               ["captan", "myclobutanil", "mancozeb", "copper"],
    "cedar apple rust":        ["myclobutanil", "trifloxystrobin", "mancozeb"],
    "leaf mold":               ["chlorothalonil", "mancozeb", "azoxystrobin"],
    "septoria leaf spot":      ["chlorothalonil", "mancozeb", "azoxystrobin", "copper"],
    "target spot":             ["azoxystrobin", "chlorothalonil", "mancozeb"],
    "gray leaf spot":          ["azoxystrobin", "pyraclostrobin", "propiconazole"],
    "northern leaf blight":    ["azoxystrobin", "pyraclostrobin", "propiconazole", "mancozeb"],
    "cercospora leaf blight":  ["azoxystrobin", "pyraclostrobin", "mancozeb"],
    "leaf scorch":             ["captan", "myclobutanil", "mancozeb"],
    "leaf spot":               ["chlorothalonil", "mancozeb", "copper"],

    # Bacterial diseases (Vi khuẩn)
    "bacterial spot":          ["copper", "streptomycin", "oxytetracycline"],

    # Viral diseases (Virus)
    "tomato mosaic virus":     ["imidacloprid", "thiamethoxam"],    # target aphid vectors
    "tomato yellow leaf curl": ["imidacloprid", "thiamethoxam", "acetamiprid"],

    # Pest / mite (Sâu / nhện)
    "spider mites":            ["abamectin", "bifenazate", "hexythiazox", "spiromesifen"],

    # Citrus
    "huanglongbing":           ["imidacloprid", "thiamethoxam"],    # psyllid vector control
    "citrus greening":         ["imidacloprid", "thiamethoxam"],

    # Common rust
    "common rust":             ["azoxystrobin", "propiconazole", "pyraclostrobin", "mancozeb"],

    # Healthy (không cần thuốc)
    "healthy":                 [],
}

# ─── Plant → Crop type mapping (for filtering PPID) ──────────────────────────
PLANT_TO_CROP_KEYWORDS = {
    "Tomato":          ["tomato", "vegetable", "solanaceae"],
    "Potato":          ["potato", "vegetable", "solanaceae"],
    "Apple":           ["apple", "fruit", "pome"],
    "Grape":           ["grape", "vine", "viticulture"],
    "Corn (Maize)":    ["corn", "maize", "cereal", "grain"],
    "Pepper, Bell":    ["pepper", "vegetable"],
    "Strawberry":      ["strawberry", "berry", "fruit"],
    "Cherry":          ["cherry", "fruit"],
    "Peach":           ["peach", "stone fruit", "fruit"],
    "Orange":          ["orange", "citrus", "fruit"],
    "Blueberry":       ["blueberry", "berry"],
    "Raspberry":       ["raspberry", "berry"],
}


class PesticideEngine:
    """
    Engine tra cứu thuốc trừ sâu/nấm/khuẩn phù hợp với bệnh cây trồng.
    Kết hợp:
        1. Keyword-based mapping (disease → active ingredients)
        2. Product lookup từ PPID CSVs
        3. Fallback built-in recommendations khi CSV thiếu
    """

    def __init__(self):
        self.ingredient_df = None
        self.product_df    = None
        self._load_data()

    # ─── Safe CSV reader with encoding fallback ─────────────────────────────
    @staticmethod
    def _read_csv_safe(filepath: str, filename: str) -> pd.DataFrame | None:
        """
        Thử đọc CSV với nhiều encoding khác nhau.
        Canada PPID thường dùng latin-1 hoặc cp1252.
        """
        encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1", "utf-8-sig"]
        for enc in encodings:
            try:
                df = pd.read_csv(filepath, encoding=enc, low_memory=False)
                df.columns = df.columns.str.strip().str.lower()
                print(f"[PPID] ✅ {filename}: {len(df):,} rows | encoding={enc} | cols: {list(df.columns[:8])}")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"[PPID] ⚠️ Lỗi đọc {filename} (enc={enc}): {e}")
                break
        print(f"[PPID] ❌ Không đọc được {filename} với mọi encoding.")
        return None

    # ─── Load CSVs ──────────────────────────────────────────────────────────
    def _load_data(self):
        """Load ingredient và product CSVs từ thư mục data/."""

        # Load ingredient
        if os.path.exists(INGREDIENT_CSV):
            self.ingredient_df = self._read_csv_safe(INGREDIENT_CSV, "ingredient_extract.csv")
        else:
            print(f"[PPID] ⚠️ Không tìm thấy: {INGREDIENT_CSV}")

        # Load product
        if os.path.exists(PRODUCT_CSV):
            self.product_df = self._read_csv_safe(PRODUCT_CSV, "product_extract.csv")
        else:
            print(f"[PPID] ⚠️ Không tìm thấy: {PRODUCT_CSV}")

    # ─── Core lookup ────────────────────────────────────────────────────────
    def get_treatment_recommendations(
        self,
        disease: str,
        plant: str = "",
        lang: str = "vi",
        top_products: int = 5
    ) -> dict:
        """
        Tra cứu gợi ý thuốc điều trị.

        Returns dict:
        {
            "disease": str,
            "plant": str,
            "is_healthy": bool,
            "active_ingredients": list[str],
            "products": list[dict],   # từ PPID CSV
            "fallback_advice": str,   # built-in advice nếu CSV thiếu data
            "treatment_notes": str,   # hướng dẫn sử dụng
        }
        """
        is_healthy = "healthy" in disease.lower()

        if is_healthy:
            return {
                "disease":            disease,
                "plant":              plant,
                "is_healthy":         True,
                "active_ingredients": [],
                "products":           [],
                "fallback_advice":    self._healthy_advice(plant, lang),
                "treatment_notes":    "",
            }

        # ── Step 1: Find active ingredients for this disease ──
        active_ingredients = self._get_ingredients(disease)

        # ── Step 2: Lookup products from CSV ──
        products = []
        if self.product_df is not None and active_ingredients:
            products = self._lookup_products(active_ingredients, top_k=top_products)

        # ── Step 3: Fallback built-in advice ──
        fallback_advice = self._get_fallback_advice(disease, plant, lang)
        treatment_notes = self._get_treatment_notes(disease, lang)

        return {
            "disease":            disease,
            "plant":              plant,
            "is_healthy":         False,
            "active_ingredients": active_ingredients,
            "products":           products,
            "fallback_advice":    fallback_advice,
            "treatment_notes":    treatment_notes,
        }

    # ─── Find active ingredients ─────────────────────────────────────────────
    def _get_ingredients(self, disease: str) -> list[str]:
        """Match disease name → active ingredient keywords."""
        disease_lower = disease.lower().strip()
        for key, ingredients in DISEASE_TO_INGREDIENTS.items():
            if key in disease_lower or disease_lower in key:
                return ingredients

        # Partial match
        for key, ingredients in DISEASE_TO_INGREDIENTS.items():
            words = key.split()
            if any(w in disease_lower for w in words if len(w) > 4):
                return ingredients

        return []

    # ─── Lookup products from CSV ────────────────────────────────────────────
    def _lookup_products(self, active_ingredients: list[str], top_k: int = 5) -> list[dict]:
        """
        Tìm sản phẩm thương mại nông nghiệp từ product_df.
        Filter phi nông nghiệp + làm sạch tên hiển thị.
        """
        if self.product_df is None or not active_ingredients:
            return []

        df   = self.product_df
        cols = list(df.columns)

        name_col   = self._find_col(cols, ["product_name", "productname", "name", "trade_name"])
        reg_col    = self._find_col(cols, ["registration_number", "reg_number", "regno", "registration"])
        status_col = self._find_col(cols, ["status", "product_status", "registration_status"])
        type_col   = self._find_col(cols, ["product_type", "type", "pest_type", "pest_control_type"])
        ingr_col   = self._find_col(cols, ["active_ingredient", "ingredient", "active", "chemical_name"])

        results    = []
        seen_names = set()

        for ingredient in active_ingredients[:4]:
            mask = pd.Series([False] * len(df))

            if ingr_col:
                mask = mask | df[ingr_col].astype(str).str.lower().str.contains(
                    ingredient.lower(), na=False
                )
            if name_col:
                mask = mask | df[name_col].astype(str).str.lower().str.contains(
                    ingredient.lower(), na=False
                )

            subset = df[mask].head(max(2, top_k // len(active_ingredients) + 2))

            for _, row in subset.iterrows():
                row_dict = row.to_dict()

                # Filter non-agricultural
                if not self._is_agricultural_product(row_dict):
                    continue

                product_name = self._clean_product_name(
                    str(row[name_col]) if name_col else "N/A"
                )
                active_ingr  = self._clean_ingredient_name(
                    str(row[ingr_col]) if ingr_col else ingredient
                )
                reg_num = str(row[reg_col]).strip() if reg_col else "N/A"
                reg_num = reg_num if reg_num not in ("nan", "NaN", "") else "N/A"

                if product_name in ("N/A", "") or product_name in seen_names:
                    continue

                seen_names.add(product_name)
                results.append({
                    "ingredient":        ingredient,
                    "product_name":      product_name,
                    "registration":      reg_num,
                    "status":            str(row[status_col]).strip() if status_col else "N/A",
                    "type":              str(row[type_col]).strip() if type_col else "N/A",
                    "active_ingredient": active_ingr,
                })

            if len(results) >= top_k:
                break

        return results[:top_k]

    @staticmethod
    def _find_col(cols: list[str], candidates: list[str]) -> str | None:
        """Find first matching column name from candidates list."""
        for c in candidates:
            matches = [col for col in cols if c in col.lower()]
            if matches:
                return matches[0]
        return None

    # ─── Fallback built-in advice ────────────────────────────────────────────
    FALLBACK_ADVICE_VI = {
        "late blight": (
            "**Thuốc điều trị bệnh sương mai (Late Blight):**\n"
            "• **Chlorothalonil** (Daconil): Phun 7-10 ngày/lần, nồng độ 0.2%\n"
            "• **Mancozeb + Metalaxyl** (Ridomil Gold): Phun khi thấy triệu chứng đầu tiên\n"
            "• **Cymoxanil** (Curzate): Hiệu quả cao trong điều kiện ẩm ướt\n"
            "• **Đồng oxychloride**: Phun phòng ngừa định kỳ\n\n"
            "⚠️ *Luân phiên loại thuốc để tránh kháng thuốc*"
        ),
        "early blight": (
            "**Thuốc điều trị bệnh đốm vòng (Early Blight):**\n"
            "• **Chlorothalonil**: Phun 5-7 ngày/lần\n"
            "• **Azoxystrobin** (Amistar): Diệt nấm phổ rộng\n"
            "• **Difenoconazole** (Score): Nồng độ 0.1%, phun lên lá\n"
            "• **Mancozeb** (Dithane): Phun phòng định kỳ\n\n"
            "⚠️ *Phun buổi sáng, tránh phun khi nhiệt độ > 35°C*"
        ),
        "powdery mildew": (
            "**Thuốc điều trị bệnh phấn trắng (Powdery Mildew):**\n"
            "• **Lưu huỳnh ướt** (Sulfur): An toàn, hiệu quả cao\n"
            "• **Azoxystrobin** (Amistar 25SC): 0.1-0.2%, phun 2 lần cách 7 ngày\n"
            "• **Tebuconazole** (Folicur): Phun khi bệnh mới xuất hiện\n"
            "• **Myclobutanil** (Nova): Nồng độ 0.1%\n\n"
            "⚠️ *Không dùng lưu huỳnh khi nhiệt độ > 32°C*"
        ),
        "bacterial spot": (
            "**Thuốc điều trị bệnh đốm vi khuẩn (Bacterial Spot):**\n"
            "• **Đồng hydroxide** (Kocide 2000): Phun 5-7 ngày/lần\n"
            "• **Streptomycin + Đồng**: Kết hợp tăng hiệu quả\n"
            "• **Oxytetracycline**: Phun khi điều kiện dễ lây\n\n"
            "⚠️ *Vi khuẩn lây qua nước, tránh tưới overhead*"
        ),
        "spider mites": (
            "**Thuốc điều trị nhện đỏ (Spider Mites):**\n"
            "• **Abamectin** (Vertimec 1.8EC): Diệt nhện đặc hiệu\n"
            "• **Bifenazate** (Floramite): Không độc với thiên địch\n"
            "• **Spiromesifen** (Oberon): Hiệu quả lâu dài\n"
            "• **Dầu neem**: Lựa chọn hữu cơ, an toàn\n\n"
            "⚠️ *Phun cả mặt dưới lá nơi nhện sinh sống*"
        ),
        "apple scab": (
            "**Thuốc điều trị bệnh ghẻ táo (Apple Scab):**\n"
            "• **Captan**: Phun phòng ngừa từ khi nở hoa\n"
            "• **Myclobutanil** (Rally): Điều trị khi bệnh xuất hiện\n"
            "• **Mancozeb** (Dithane): Bảo vệ lá non\n"
            "• **Ziram**: Phun 7-10 ngày/lần\n\n"
            "⚠️ *Phun ngay sau mưa để phòng bào tử nảy mầm*"
        ),
    }

    FALLBACK_ADVICE_EN = {
        "late blight": (
            "**Treatment for Late Blight:**\n"
            "• **Chlorothalonil** (Daconil): Apply every 7-10 days, 0.2% concentration\n"
            "• **Mancozeb + Metalaxyl** (Ridomil Gold): Apply at first symptoms\n"
            "• **Cymoxanil** (Curzate): Highly effective in wet conditions\n"
            "• **Copper oxychloride**: Preventive spray program\n\n"
            "⚠️ *Rotate fungicide classes to prevent resistance*"
        ),
        "powdery mildew": (
            "**Treatment for Powdery Mildew:**\n"
            "• **Wettable sulfur**: Safe and effective\n"
            "• **Azoxystrobin** (Amistar): Systemic protection\n"
            "• **Tebuconazole** (Folicur): Apply at first signs\n"
            "• **Myclobutanil** (Nova): 0.1% solution\n\n"
            "⚠️ *Do not apply sulfur when temps > 32°C*"
        ),
        "bacterial spot": (
            "**Treatment for Bacterial Spot:**\n"
            "• **Copper hydroxide** (Kocide 2000): Every 5-7 days\n"
            "• **Streptomycin + Copper**: Combined for better efficacy\n"
            "• **Oxytetracycline**: Under high disease pressure\n\n"
            "⚠️ *Bacteria spread via water — avoid overhead irrigation*"
        ),
    }

    def _get_fallback_advice(self, disease: str, plant: str, lang: str) -> str:
        """Return built-in treatment advice for common diseases."""
        disease_lower = disease.lower()
        advice_map = self.FALLBACK_ADVICE_VI if lang == "vi" else self.FALLBACK_ADVICE_EN

        for key, advice in advice_map.items():
            if key in disease_lower:
                return advice

        # Generic fallback
        if lang == "vi":
            return (
                f"**Hướng dẫn điều trị bệnh {disease} trên {plant}:**\n"
                "• Xác định chính xác tác nhân gây bệnh (nấm/vi khuẩn/virus)\n"
                "• Với bệnh nấm: Dùng thuốc nhóm triazole hoặc strobilurin\n"
                "• Với bệnh vi khuẩn: Dùng chế phẩm đồng hoặc kháng sinh nông nghiệp\n"
                "• Với bệnh virus: Kiểm soát côn trùng môi giới (rệp, bọ phấn)\n"
                "• Luôn tham khảo nhân viên khuyến nông tại địa phương trước khi sử dụng thuốc"
            )
        else:
            return (
                f"**Treatment guide for {disease} on {plant}:**\n"
                "• Identify the exact pathogen (fungal/bacterial/viral)\n"
                "• For fungal: Use triazole or strobilurin fungicides\n"
                "• For bacterial: Use copper-based or agricultural antibiotics\n"
                "• For viral: Control insect vectors (aphids, whiteflies)\n"
                "• Always consult local agricultural extension before applying pesticides"
            )

    def _get_treatment_notes(self, disease: str, lang: str) -> str:
        """Return general treatment safety notes."""
        if lang == "vi":
            return (
                "📋 **Lưu ý khi sử dụng thuốc:**\n"
                "• Đọc kỹ nhãn thuốc và tuân thủ liều lượng khuyến cáo\n"
                "• Mặc đồ bảo hộ (găng tay, khẩu trang) khi phun thuốc\n"
                "• Phun vào buổi sáng sớm hoặc chiều mát, tránh gió to\n"
                "• Không phun khi sắp thu hoạch (tuân thủ thời gian cách ly)\n"
                "• Luân phiên loại thuốc để tránh kháng thuốc\n"
                "• Bảo quản thuốc nơi khô ráo, thoáng mát, xa trẻ em"
            )
        else:
            return (
                "📋 **Pesticide Application Safety Notes:**\n"
                "• Read label carefully and follow recommended dosage\n"
                "• Wear protective equipment (gloves, mask) when spraying\n"
                "• Apply early morning or late afternoon, avoid windy conditions\n"
                "• Observe pre-harvest intervals (PHI) before harvest\n"
                "• Rotate pesticide classes to prevent resistance development\n"
                "• Store pesticides in cool, dry place away from children"
            )

    def _healthy_advice(self, plant: str, lang: str) -> str:
        if lang == "vi":
            return (
                f"✅ **Cây {plant} đang khỏe mạnh!**\n\n"
                "**Biện pháp duy trì sức khỏe cây:**\n"
                "• Bón phân cân đối NPK theo giai đoạn sinh trưởng\n"
                "• Tưới nước đúng lượng, tránh úng ngập hay khô hạn\n"
                "• Phun thuốc phòng ngừa định kỳ (đặc biệt mùa mưa)\n"
                "• Kiểm tra vườn thường xuyên để phát hiện bệnh sớm\n"
                "• Loại bỏ lá/cành bệnh kịp thời"
            )
        else:
            return (
                f"✅ **{plant} appears healthy!**\n\n"
                "**Preventive care recommendations:**\n"
                "• Apply balanced NPK fertilizer according to growth stage\n"
                "• Water appropriately — avoid waterlogging or drought stress\n"
                "• Apply preventive fungicide sprays (especially during wet season)\n"
                "• Inspect field regularly for early disease detection\n"
                "• Remove infected leaves/branches promptly"
            )

    # ─── Format output for Groq context ─────────────────────────────────────
    def format_for_groq(self, rec: dict, lang: str) -> str:
        """
        Format treatment recommendation thành chuỗi để inject vào Groq context.
        """
        if rec["is_healthy"]:
            return rec["fallback_advice"]

        disease = rec["disease"]
        plant   = rec["plant"]
        ingr    = rec["active_ingredients"]
        prods   = rec["products"]

        if lang == "vi":
            lines = [f"💊 **Gợi ý thuốc điều trị: {disease} trên cây {plant}**\n"]

            if ingr:
                lines.append("**Hoạt chất (Active Ingredients) khuyến nghị:**")
                lines.append(", ".join(ingr[:5]))
                lines.append("")

            if prods:
                lines.append("**Sản phẩm thuốc từ cơ sở dữ liệu PPID:**")
                for i, p in enumerate(prods[:5], 1):
                    name = p.get("product_name", "N/A")
                    reg  = p.get("registration", "N/A")
                    ing  = p.get("active_ingredient", p.get("ingredient", "N/A"))
                    lines.append(f"  {i}. **{name}** (Hoạt chất: {ing}, Đăng ký: {reg})")
                lines.append("")

            lines.append(rec["fallback_advice"])
            lines.append("")
            lines.append(rec["treatment_notes"])

        else:
            lines = [f"💊 **Treatment Recommendations: {disease} on {plant}**\n"]

            if ingr:
                lines.append("**Recommended Active Ingredients:**")
                lines.append(", ".join(ingr[:5]))
                lines.append("")

            if prods:
                lines.append("**Products from PPID Database:**")
                for i, p in enumerate(prods[:5], 1):
                    name = p.get("product_name", "N/A")
                    reg  = p.get("registration", "N/A")
                    ing  = p.get("active_ingredient", p.get("ingredient", "N/A"))
                    lines.append(f"  {i}. **{name}** (Ingredient: {ing}, Reg: {reg})")
                lines.append("")

            lines.append(rec["fallback_advice"])
            lines.append("")
            lines.append(rec["treatment_notes"])

        return "\n".join(lines)

    # ─── Quick search (for sidebar / direct query) ──────────────────────────
    # ── Keywords để loại bỏ sản phẩm không liên quan nông nghiệp ──
    _NON_AGRI_KEYWORDS = [
        "antifouling", "paint", "marine", "boat", "ship", "vinyl",
        "wood preserv", "timber", "impregnated", "ddt", "fisherman",
        "swimming pool", "disinfect", "sanitiz", "household",
        "rat", "rodent", "cockroach", "mosquito repel",
    ]

    # ── Keywords ưu tiên — sản phẩm nông nghiệp ──
    _AGRI_KEYWORDS = [
        "fungicide", "insecticide", "herbicide", "bactericide",
        "agricultural", "crop", "plant", "foliar", "spray",
        "granule", "wettable powder", "emulsifiable", "suspension",
        "technical", "concentrate",
    ]

    def _is_agricultural_product(self, row: dict) -> bool:
        """Kiểm tra sản phẩm có phải thuốc nông nghiệp không."""
        # Gộp tất cả text của row để kiểm tra
        all_text = " ".join(str(v) for v in row.values()).lower()

        # Loại bỏ nếu chứa keyword phi nông nghiệp
        for kw in self._NON_AGRI_KEYWORDS:
            if kw in all_text:
                return False

        return True

    def _clean_ingredient_name(self, raw: str) -> str:
        """
        Rút gọn tên hoạt chất dài để hiển thị gọn hơn.
        VD: 'NOT AVAILABLE (THE CODE N/A WAS APPLIED...)' → 'N/A'
        """
        if not raw or raw.strip() in ("", "nan", "NaN"):
            return "N/A"
        raw = raw.strip()
        # Cắt chuỗi giải thích dài
        if "NOT AVAILABLE" in raw.upper() or len(raw) > 80:
            # Lấy phần trong ngoặc đầu tiên nếu có
            import re
            paren = re.search(r'\(([^)]{3,40})\)', raw)
            if paren:
                inner = paren.group(1).strip()
                # Bỏ qua nếu là giải thích dài
                if len(inner) < 50 and "CODE" not in inner.upper():
                    return inner
            return "N/A"
        return raw

    def _clean_product_name(self, raw: str) -> str:
        """Chuẩn hóa tên sản phẩm: UPPER → Title Case, bỏ ký tự thừa."""
        if not raw or raw.strip() in ("", "nan", "NaN"):
            return "N/A"
        raw = raw.strip()
        # Nếu toàn chữ hoa → chuyển Title Case
        if raw.isupper():
            return raw.title()
        return raw

    def search_by_ingredient(self, keyword: str, top_k: int = 8) -> list[dict]:
        """
        Tìm sản phẩm NÔNG NGHIỆP theo từ khóa hoạt chất.
        - Loại bỏ sản phẩm phi nông nghiệp (sơn, DDT, thuốc gia dụng...)
        - Ưu tiên sản phẩm có tên hoạt chất khớp chính xác
        - Làm sạch tên hiển thị
        """
        if self.product_df is None:
            return []

        df   = self.product_df
        cols = list(df.columns)

        ingr_col = self._find_col(cols, ["active_ingredient", "ingredient", "active", "chemical_name"])
        name_col = self._find_col(cols, ["product_name", "productname", "name", "trade_name"])
        reg_col  = self._find_col(cols, ["registration_number", "reg_number", "regno", "registration"])
        type_col = self._find_col(cols, ["product_type", "pest_type", "type", "pest_control_type"])

        if not ingr_col and not name_col:
            return []

        kw_lower = keyword.lower().strip()

        # ── Search: ưu tiên match trong ingredient column ──
        mask_ingr = pd.Series([False] * len(df))
        mask_name = pd.Series([False] * len(df))

        if ingr_col:
            mask_ingr = df[ingr_col].astype(str).str.lower().str.contains(kw_lower, na=False)
        if name_col:
            mask_name = df[name_col].astype(str).str.lower().str.contains(kw_lower, na=False)

        # Ưu tiên: ingredient match trước, rồi name match
        subset_ingr = df[mask_ingr]
        subset_name = df[mask_name & ~mask_ingr]  # name match nhưng không trùng

        subset = pd.concat([subset_ingr, subset_name]).head(top_k * 4)  # Lấy dư để filter

        results = []
        seen_names = set()

        for _, row in subset.iterrows():
            row_dict = row.to_dict()

            # Lọc sản phẩm phi nông nghiệp
            if not self._is_agricultural_product(row_dict):
                continue

            product_name = self._clean_product_name(
                str(row[name_col]) if name_col else "N/A"
            )
            active_ingr  = self._clean_ingredient_name(
                str(row[ingr_col]) if ingr_col else keyword
            )
            reg_num      = str(row[reg_col]).strip() if reg_col else "N/A"
            prod_type    = str(row[type_col]).strip().title() if type_col else ""

            # Bỏ qua nếu tên sản phẩm N/A hoặc trùng
            if product_name in ("N/A", "") or product_name in seen_names:
                continue
            if active_ingr == "N/A" and product_name == "N/A":
                continue

            seen_names.add(product_name)
            results.append({
                "product_name":      product_name,
                "active_ingredient": active_ingr,
                "registration":      reg_num if reg_num not in ("nan", "NaN", "") else "N/A",
                "product_type":      prod_type if prod_type not in ("Nan", "NaN", "") else "",
            })

            if len(results) >= top_k:
                break

        return results

    def get_stats(self) -> dict:
        """Trả về thống kê dữ liệu PPID."""
        return {
            "n_products":    len(self.product_df)    if self.product_df    is not None else 0,
            "n_ingredients": len(self.ingredient_df) if self.ingredient_df is not None else 0,
            "diseases_mapped": len([k for k, v in DISEASE_TO_INGREDIENTS.items() if v]),
        }


# ─── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = PesticideEngine()
    stats  = engine.get_stats()
    print(f"\n📊 PPID Stats: {stats}")

    test_cases = [
        ("Late blight", "Tomato", "vi"),
        ("Powdery mildew", "Apple", "en"),
        ("Bacterial spot", "Pepper, Bell", "vi"),
        ("healthy", "Corn (Maize)", "vi"),
        ("Spider mites", "Strawberry", "en"),
    ]

    for disease, plant, lang in test_cases:
        print(f"\n{'='*60}")
        print(f"Disease: {disease} | Plant: {plant} | Lang: {lang}")
        rec = engine.get_treatment_recommendations(disease, plant, lang)
        print(engine.format_for_groq(rec, lang))