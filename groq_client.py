"""
groq_client.py — Tích hợp Groq API cho LLM responses.
Nhận context từ retrieval + image classification + PPID pesticide data,
rồi gửi lên Groq để tạo response chính xác + hữu ích.

=== UPDATES ===
- Tích hợp PesticideEngine: tự động inject context thuốc điều trị
  vào mỗi request liên quan đến bệnh cây (khi image_classifications có disease).
- Groq nhận context đầy đủ: ảnh + retrieval + gợi ý thuốc từ PPID.

Fix: Groq SDK không có parameter 'system' trong .create().
     System prompt phải đặt vào messages list với role="system".
"""
import warnings
warnings.filterwarnings("ignore")

from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL, MAX_TOKENS, TEMPERATURE
from translation import get_system_prompt, translate_label_to_vi


class GroqClient:
    """Wrapper cho Groq API."""

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY chưa được đặt.\n"
                "1. Tạo API key tại: https://console.groq.com\n"
                "2. Copy key vào file .env: GROQ_API_KEY=gsk_xxxx"
            )
        self.client = Groq(api_key=GROQ_API_KEY)

        # ── Lazy-load PesticideEngine ──
        self._pesticide_engine = None
        print("[GROQ] Client initialized.")

    # ──────────────────────────────────────
    # Lazy-load pesticide engine
    # ──────────────────────────────────────
    def _get_pesticide_engine(self):
        if self._pesticide_engine is None:
            try:
                from pesticide_engine import PesticideEngine
                self._pesticide_engine = PesticideEngine()
                print("[GROQ] ✅ PesticideEngine loaded.")
            except Exception as e:
                print(f"[GROQ] ⚠️ PesticideEngine failed to load: {e}")
                self._pesticide_engine = False  # Mark as failed
        return self._pesticide_engine if self._pesticide_engine else None

    # ──────────────────────────────────────
    # Build context from retrieval results
    # ──────────────────────────────────────
    @staticmethod
    def _build_retrieval_context(retrieval_results: list[dict], lang: str) -> str:
        if not retrieval_results:
            return ""

        if lang == "vi":
            header = "📚 **Thông tin tham chiếu từ cơ sở dữ liệu:**\n"
        else:
            header = "📚 **Reference information from knowledge base:**\n"

        items = []
        for i, r in enumerate(retrieval_results, 1):
            plant   = translate_label_to_vi(r.get("Plant", "")) if lang == "vi" else r.get("Plant", "")
            disease = translate_label_to_vi(r.get("Disease", "")) if lang == "vi" else r.get("Disease", "")
            items.append(
                f"{i}. Cây: {plant} | Bệnh: {disease}\n"
                f"   Q: {r['Question']}\n"
                f"   A: {r['Answer']}\n"
            )

        return header + "\n".join(items)

    # ──────────────────────────────────────
    # Build context from image classification
    # ──────────────────────────────────────
    @staticmethod
    def _build_image_context(classifications: list[dict], lang: str) -> str:
        if not classifications:
            return ""

        if lang == "vi":
            header = "🖼️ **Kết quả phân loại ảnh:**\n"
        else:
            header = "🖼️ **Image classification results:**\n"

        items = []
        for i, c in enumerate(classifications, 1):
            plant   = translate_label_to_vi(c["plant"]) if lang == "vi" else c["plant"]
            disease = translate_label_to_vi(c["disease"]) if lang == "vi" else c["disease"]
            is_healthy = "healthy" in c["disease"].lower()
            status = "✅ Khỏe mạnh" if (is_healthy and lang == "vi") else ("✅ Healthy" if is_healthy else f"⚠️ {disease}")
            items.append(f"  {i}. {plant} → {status} (Độ chắc chắn: {c['confidence']:.1f}%)")

        return header + "\n".join(items)

    # ──────────────────────────────────────
    # Build pesticide treatment context
    # ──────────────────────────────────────
    def _build_pesticide_context(
        self,
        disease: str,
        plant: str,
        lang: str
    ) -> str:
        """
        Tra cứu thuốc điều trị từ PesticideEngine + format thành context.
        Trả về chuỗi rỗng nếu engine không available hoặc cây khỏe.
        """
        engine = self._get_pesticide_engine()
        if not engine:
            return ""

        try:
            rec = engine.get_treatment_recommendations(
                disease=disease,
                plant=plant,
                lang=lang,
                top_products=5
            )
            if rec["is_healthy"]:
                return ""  # Không cần gợi ý thuốc cho cây khỏe
            return engine.format_for_groq(rec, lang)
        except Exception as e:
            print(f"[GROQ] ⚠️ Pesticide lookup error: {e}")
            return ""

    # ──────────────────────────────────────
    # Main chat method
    # ──────────────────────────────────────
    def chat(
        self,
        user_message: str,
        lang: str = "vi",
        retrieval_results: list[dict] | None = None,
        image_classifications: list[dict] | None = None,
        conversation_history: list[dict] | None = None
    ) -> str:
        """
        Gửi message lên Groq với full context.

        user_message đã được main.py enrichen với:
        - instruction nghiệp vụ (từ QTYPE_NGHIEP_VU mapping)
        - context ảnh (plant/disease)
        → Ở đây chỉ cần append retrieval context + image context + pesticide context vào.
        """
        system_prompt = get_system_prompt(lang)

        # ── Build enriched user message with context ──
        context_parts = []

        if image_classifications:
            context_parts.append(self._build_image_context(image_classifications, lang))

        if retrieval_results:
            context_parts.append(self._build_retrieval_context(retrieval_results, lang))

        # ── Add pesticide context when disease detected ──
        if image_classifications:
            top = image_classifications[0]
            detected_disease = top.get("disease", "")
            detected_plant   = top.get("plant", "")
            is_healthy       = "healthy" in detected_disease.lower()

            # Only add pesticide recommendations if disease detected (not healthy)
            if not is_healthy and detected_disease:
                pesticide_ctx = self._build_pesticide_context(
                    disease=detected_disease,
                    plant=detected_plant,
                    lang=lang
                )
                if pesticide_ctx:
                    context_parts.append(pesticide_ctx)

        # Combine context + user_message (đã có instruction từ main.py)
        if context_parts:
            enriched_message = "\n\n".join(context_parts) + "\n\n"
            if lang == "vi":
                enriched_message += (
                    f"💬 **Yêu cầu của bạn:** {user_message}\n\n"
                    "Dựa trên thông tin trên (kết quả chẩn đoán ảnh, tài liệu tham chiếu, "
                    "và gợi ý thuốc điều trị từ cơ sở dữ liệu thuốc bảo vệ thực vật), "
                    "hãy trả lời chi tiết và hữu ích. "
                    "Nếu có thông tin về thuốc điều trị, hãy tổng hợp và đưa ra khuyến nghị thực hành cụ thể."
                )
            else:
                enriched_message += (
                    f"💬 **Your request:** {user_message}\n\n"
                    "Based on the above information (image diagnosis, reference documents, "
                    "and pesticide treatment suggestions from the PPID database), "
                    "provide a detailed and helpful response. "
                    "If pesticide information is available, synthesize it into specific practical recommendations."
                )
        else:
            enriched_message = user_message

        # ── Build messages list ──
        messages = []

        # 1. System prompt
        messages.append({
            "role": "system",
            "content": system_prompt
        })

        # 2. Conversation history (giữ 6 exchanges gần nhất)
        if conversation_history:
            messages.extend(conversation_history[-12:])

        # 3. Current user message
        messages.append({
            "role": "user",
            "content": enriched_message
        })

        # ── Call Groq API ──
        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=0.9,
            )
            answer = response.choices[0].message.content
            return answer

        except Exception as e:
            if lang == "vi":
                return (
                    f"⚠️ Đã xảy ra lỗi khi gọi API Groq: `{str(e)}`\n\n"
                    "Vui lòng kiểm tra:\n"
                    "1. GROQ_API_KEY có hợp lệ không?\n"
                    "2. Kết nối internet có ổn định không?\n"
                    "3. Thử lại sau một lúc."
                )
            else:
                return (
                    f"⚠️ Error calling Groq API: `{str(e)}`\n\n"
                    "Please check:\n"
                    "1. Is your GROQ_API_KEY valid?\n"
                    "2. Is your internet connection stable?\n"
                    "3. Try again in a moment."
                )