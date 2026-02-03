"""
recommendation.py — Hệ thống tra cứu (Retrieval) tìm Q&A liên quan nhất.
Kết hợp TF-IDF cosine similarity + Sentence Embedding cosine similarity.

=== FIXES ===
1. [NotFittedError] Thêm _validate_tfidf() kiểm tra idf_ attribute sau khi load.
   Nếu vectorizer chưa fitted → tự gọi rebuild TF-IDF từ DataFrame.
2. [Robustness] Nếu rebuild cũng thất bại → set vectorizer = None,
   _tfidf_scores() trả về zeros → hệ thống fallback sang embedding-only.
3. [Logging] In rõ trạng thái: loaded OK / rebuild / disabled.
"""
import warnings, os
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

warnings.filterwarnings("ignore")

from config import (
    TFIDF_MATRIX_PATH, TFIDF_VECTORIZER_PATH,
    EMBEDDING_CACHE_PATH, EMBEDDING_MODEL
)


class RetrievalEngine:
    """
    Chứa toàn bộ logic tra cứu.
    Khởi tạo 1 lần, reuse trong suốt session Streamlit.
    """

    def __init__(self, df):
        self.df = df
        self.tfidf_vectorizer = None
        self.tfidf_matrix     = None
        self.embeddings       = None
        self.embed_model      = None
        self._load_artifacts()

    # ──────────────────────────────────────
    # Load pre-built artifacts
    # ──────────────────────────────────────
    def _load_artifacts(self):
        self._load_tfidf()
        self._load_embeddings()

    # ──────────────────────────────────────
    # Load + Validate TF-IDF
    # ──────────────────────────────────────
    def _load_tfidf(self):
        """
        Load TF-IDF vectorizer + matrix từ .pkl.
        Sau khi load, validate bằng check_is_fitted().
        Nếu chưa fitted hoặc file không tồn tại → auto-rebuild từ self.df.
        """
        loaded_vectorizer = None
        loaded_matrix     = None

        # ── Step 1: Thử load từ file ──
        if os.path.exists(TFIDF_VECTORIZER_PATH) and os.path.exists(TFIDF_MATRIX_PATH):
            try:
                loaded_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
                loaded_matrix     = joblib.load(TFIDF_MATRIX_PATH)
                print("[RETRIEVAL] TF-IDF .pkl files loaded from disk.")
            except Exception as e:
                print(f"[RETRIEVAL] ⚠️ Lỗi load TF-IDF .pkl: {e}")
                loaded_vectorizer = None
                loaded_matrix     = None
        else:
            print("[RETRIEVAL] ⚠️ TF-IDF .pkl files không tìm thấy.")

        # ── Step 2: Validate fitted ──
        if loaded_vectorizer is not None:
            if self._validate_tfidf(loaded_vectorizer):
                # ✅ Vectorizer hợp lệ + đã fitted
                self.tfidf_vectorizer = loaded_vectorizer
                self.tfidf_matrix     = loaded_matrix
                print("[RETRIEVAL] ✅ TF-IDF vectorizer validated — fitted OK.")
                return
            else:
                # ❌ Vectorizer load được nhưng chưa fitted → rebuild
                print("[RETRIEVAL] ⚠️ TF-IDF vectorizer chưa fitted. Đang rebuild...")

        # ── Step 3: Rebuild nếu cần ──
        self._rebuild_tfidf()

    # ──────────────────────────────────────
    # Validate: check idf_ attribute
    # ──────────────────────────────────────
    @staticmethod
    def _validate_tfidf(vectorizer) -> bool:
        """
        Kiểm tra vectorizer đã được fit chưa.
        TfidfVectorizer sau fit sẽ có attribute 'idf_'.
        """
        try:
            check_is_fitted(vectorizer, attributes=["idf_"])
            return True
        except NotFittedError:
            return False

    # ──────────────────────────────────────
    # Rebuild TF-IDF từ DataFrame
    # ──────────────────────────────────────
    def _rebuild_tfidf(self):
        """
        Rebuild TF-IDF vectorizer + matrix từ self.df.
        Dump lại .pkl để reuse cho lần chạy sau.
        """
        if self.df is None or "Combined" not in self.df.columns:
            print("[RETRIEVAL] ❌ Không thể rebuild TF-IDF: DataFrame thiếu cột 'Combined'.")
            print("[RETRIEVAL] ⚠️ TF-IDF disabled — hệ thống sẽ chỉ dùng Sentence Embeddings.")
            self.tfidf_vectorizer = None
            self.tfidf_matrix     = None
            return

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            print("[RETRIEVAL] 🔨 Đang rebuild TF-IDF vectorizer từ DataFrame...")
            vectorizer  = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
            tfidf_matrix = vectorizer.fit_transform(self.df["Combined"].tolist())

            # Validate lại sau fit
            if not self._validate_tfidf(vectorizer):
                raise RuntimeError("Vectorizer vẫn chưa fitted sau fit_transform().")

            # ── Save artifacts ──
            os.makedirs(os.path.dirname(TFIDF_VECTORIZER_PATH), exist_ok=True)
            joblib.dump(vectorizer,  TFIDF_VECTORIZER_PATH)
            joblib.dump(tfidf_matrix, TFIDF_MATRIX_PATH)

            self.tfidf_vectorizer = vectorizer
            self.tfidf_matrix     = tfidf_matrix
            print(f"[RETRIEVAL] ✅ TF-IDF rebuild hoàn thành: {tfidf_matrix.shape} → artifacts/")

        except Exception as e:
            print(f"[RETRIEVAL] ❌ Rebuild TF-IDF thất bại: {e}")
            print("[RETRIEVAL] ⚠️ TF-IDF disabled — hệ thống sẽ chỉ dùng Sentence Embeddings.")
            self.tfidf_vectorizer = None
            self.tfidf_matrix     = None

    # ──────────────────────────────────────
    # Load Sentence Embeddings
    # ──────────────────────────────────────
    def _load_embeddings(self):
        if os.path.exists(EMBEDDING_CACHE_PATH):
            try:
                self.embeddings = joblib.load(EMBEDDING_CACHE_PATH)
                from sentence_transformers import SentenceTransformer
                self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
                print("[RETRIEVAL] ✅ Embedding cache loaded.")
            except Exception as e:
                print(f"[RETRIEVAL] ⚠️ Lỗi load Embeddings: {e}")
                self.embeddings  = None
                self.embed_model = None
        else:
            print("[RETRIEVAL] ⚠️ Embedding cache not found. Run data_processing.py first.")
            self.embeddings  = None
            self.embed_model = None

    # ──────────────────────────────────────
    # TF-IDF Similarity
    # ──────────────────────────────────────
    def _tfidf_scores(self, query: str) -> np.ndarray:
        """
        Tính TF-IDF cosine similarity.
        Nếu vectorizer None (chưa fitted / disabled) → trả về zeros.
        """
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            return np.zeros(len(self.df))

        # Double-check fitted trước khi transform (safety net)
        if not self._validate_tfidf(self.tfidf_vectorizer):
            print("[RETRIEVAL] ⚠️ TF-IDF vectorizer mất fitted state. Thử rebuild...")
            self._rebuild_tfidf()
            if self.tfidf_vectorizer is None:
                return np.zeros(len(self.df))

        try:
            q_vec  = self.tfidf_vectorizer.transform([query])
            scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
            return scores
        except NotFittedError:
            print("[RETRIEVAL] ⚠️ NotFittedError tại transform — fallback zeros.")
            return np.zeros(len(self.df))

    # ──────────────────────────────────────
    # Sentence Embedding Similarity
    # ──────────────────────────────────────
    def _embed_scores(self, query: str) -> np.ndarray:
        if self.embed_model is None or self.embeddings is None:
            return np.zeros(len(self.df))
        q_emb  = self.embed_model.encode([query])
        scores = cosine_similarity(q_emb, self.embeddings).flatten()
        return scores

    # ──────────────────────────────────────
    # Combined retrieval (weighted ensemble)
    # ──────────────────────────────────────
    def retrieve(self, query: str, top_k: int = 3, alpha: float = 0.45) -> list[dict]:
        """
        Trả về top_k kết quả liên quan nhất.
        alpha: weight cho TF-IDF (1-alpha cho embedding).

        Adaptive alpha:
        - Nếu TF-IDF disabled (vectorizer = None) → alpha = 0 (chỉ dùng embedding).
        - Nếu Embedding disabled → alpha = 1 (chỉ dùng TF-IDF).
        - Nếu cả hai disabled → trả về top_k rows đầu tiên (fallback).
        """
        tfidf_s  = self._tfidf_scores(query)
        embed_s  = self._embed_scores(query)

        # ── Adaptive: nếu một trong hai bị disabled ──
        tfidf_active  = self.tfidf_vectorizer is not None and np.any(tfidf_s != 0)
        embed_active  = self.embed_model is not None and np.any(embed_s != 0)

        if not tfidf_active and not embed_active:
            # ── Cả hai disabled → fallback: trả về top_k rows đầu ──
            print("[RETRIEVAL] ⚠️ Cả TF-IDF và Embedding đều inactive. Fallback top rows.")
            results = []
            for i in range(min(top_k, len(self.df))):
                row = self.df.iloc[i]
                results.append({
                    "Plant":         row.get("Plant", "Unknown"),
                    "Disease":       row.get("Disease", "Unknown"),
                    "Question":      row["Question"],
                    "Answer":        row["Answer"],
                    "question_type": row.get("question_type", "General"),
                    "image_path":    row.get("image_path", ""),
                    "score":         0.0
                })
            return results

        if not tfidf_active:
            effective_alpha = 0.0   # chỉ embedding
        elif not embed_active:
            effective_alpha = 1.0   # chỉ TF-IDF
        else:
            effective_alpha = alpha # default ensemble

        # Normalize to [0,1]
        def norm(arr):
            mn, mx = arr.min(), arr.max()
            return (arr - mn) / (mx - mn + 1e-9)

        combined = effective_alpha * norm(tfidf_s) + (1 - effective_alpha) * norm(embed_s)
        top_idx  = np.argsort(combined)[-top_k:][::-1]

        results = []
        for i in top_idx:
            row = self.df.iloc[i]
            results.append({
                "Plant":         row.get("Plant", "Unknown"),
                "Disease":       row.get("Disease", "Unknown"),
                "Question":      row["Question"],
                "Answer":        row["Answer"],
                "question_type": row.get("question_type", "General"),
                "image_path":    row.get("image_path", ""),
                "score":         float(combined[i])
            })
        return results

    # ──────────────────────────────────────
    # Filter by plant / disease name
    # ──────────────────────────────────────
    def retrieve_by_disease(self, disease_name: str, top_k: int = 3) -> list[dict]:
        """Tìm các Q&A có Disease chứa disease_name (case-insensitive)."""
        if "Disease" not in self.df.columns:
            return []
        mask = self.df["Disease"].astype(str).str.lower().str.contains(
            disease_name.lower(), na=False
        )
        subset = self.df[mask]
        if subset.empty:
            return []

        results = []
        for _, row in subset.head(top_k).iterrows():
            results.append({
                "Plant":         row.get("Plant", "Unknown"),
                "Disease":       row.get("Disease", "Unknown"),
                "Question":      row["Question"],
                "Answer":        row["Answer"],
                "question_type": row.get("question_type", "General"),
                "image_path":    row.get("image_path", ""),
                "score":         1.0
            })
        return results

    # ──────────────────────────────────────
    # Filter by question_type (9 PlantVillageVQA categories)
    # ──────────────────────────────────────
    def retrieve_by_question_type(self, qtype: str, plant: str = "", top_k: int = 5) -> list[dict]:
        """
        Tìm Q&A theo question_type.
        Các loại hỏi của PlantVillageVQA:
            - Existence & Sanity Check
            - Plant Species Identification
            - General Health Assessment
            - Visual Attribute Grounding
            - Detailed Verification
            - Specific Disease Identification
            - Comprehensive Description
            - Causal Reasoning
            - Counterfactual Reasoning
        """
        if "question_type" not in self.df.columns:
            return []

        mask = self.df["question_type"].astype(str).str.lower().str.contains(
            qtype.lower(), na=False
        )
        if plant:
            mask = mask & self.df["Plant"].astype(str).str.lower().str.contains(
                plant.lower(), na=False
            )

        subset = self.df[mask]
        if subset.empty:
            return []

        results = []
        for _, row in subset.head(top_k).iterrows():
            results.append({
                "Plant":         row.get("Plant", "Unknown"),
                "Disease":       row.get("Disease", "Unknown"),
                "Question":      row["Question"],
                "Answer":        row["Answer"],
                "question_type": row.get("question_type", "General"),
                "image_path":    row.get("image_path", ""),
                "score":         1.0
            })
        return results

    # ──────────────────────────────────────
    # Get unique plants / diseases / question_types
    # ──────────────────────────────────────
    def get_plants(self) -> list[str]:
        if "Plant" not in self.df.columns:
            return []
        return sorted(self.df["Plant"].astype(str).unique().tolist())

    def get_diseases(self) -> list[str]:
        if "Disease" not in self.df.columns:
            return []
        return sorted(self.df["Disease"].astype(str).unique().tolist())

    def get_question_types(self) -> list[str]:
        if "question_type" not in self.df.columns:
            return []
        return sorted(self.df["question_type"].astype(str).unique().tolist())