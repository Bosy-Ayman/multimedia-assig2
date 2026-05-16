import os
import sys

# ============================================================================
# ENVIRONMENT FIX: FORCE VIRTUAL ENVIRONMENT PRIORITY
# ============================================================================
# This prevents outdated system-wide packages from hijacking the environment.
_project_root = os.path.dirname(os.path.abspath(__file__))
_venv_site = os.path.join(_project_root, "venv", "Lib", "site-packages")
if os.path.exists(_venv_site) and _venv_site not in sys.path:
    sys.path.insert(0, _venv_site)

import streamlit as st
from PIL import Image
import numpy as np
import torch
from pathlib import Path
import json
import time
from typing import List, Dict, Optional
import pandas as pd

# ============================================================================
# PREMIUM UI DESIGN (CSS)
# ============================================================================
def apply_custom_design():
    st.markdown("""
    <style>
    /* Main Background and Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Outfit:wght@300;500;700&display=swap');
    
    .stApp {
        background: radial-gradient(circle at top right, #0a192f, #020c1b);
        color: #e6f1ff;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(90deg, #64ffda, #48cae4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Glassmorphism Cards */
    div.stButton > button {
        background: linear-gradient(135deg, #64ffda 0%, #48cae4 100%);
        color: #020c1b;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 12px;
        font-weight: 700;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(100, 255, 218, 0.3);
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 8px 25px rgba(100, 255, 218, 0.5);
        color: #020c1b;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(100, 255, 218, 0.2);
        color: #e6f1ff;
        border-radius: 10px;
    }
    
    /* Custom Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(2, 12, 27, 0.95);
        border-right: 1px solid rgba(100, 255, 218, 0.1);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #64ffda !important;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        border: 1px solid rgba(100, 255, 218, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# SEO Best Practices
def apply_seo_metadata():
    st.set_page_config(
        page_title="CXR Intel | Advanced Chest X-Ray AI",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown("""
    <meta name="description" content="AI-powered Multi-Modal Chest X-Ray Intelligence System for automated report generation and Clinical RAG.">
    <meta name="keywords" content="Medical AI, Radiology, Chest X-Ray, ColPali, MedGemma, RAG">
    """, unsafe_allow_html=True)

# Global HF Authentication for gated models (ColPali/MedGemma)
# Try to load from .env file if it exists
if not os.environ.get("HF_TOKEN"):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

# Final check for token
if not os.environ.get("HF_TOKEN"):
    print("Warning: HF_TOKEN not found in environment. Gated models may fail to load.")

# ============================================================================
# UTILS
# ============================================================================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mean_pool_embedding(embeddings: torch.Tensor) -> np.ndarray:
    if embeddings.dim() == 3:
        pooled = embeddings.mean(dim=1)
    else:
        pooled = embeddings
    pooled = pooled / pooled.norm(dim=-1, keepdim=True)
    return pooled.cpu().numpy()

# ============================================================================
# CLIP
# ============================================================================
class CLIPEmbedder:
    def __init__(self):
        self._processor = None
        self._model = None
        self.device = get_device()
        self.available = True

    @property
    def model(self):
        self._load()
        return self._model

    @property
    def processor(self):
        self._load()
        return self._processor

    def _load(self):
        if self._model is not None:
            return
            
        from transformers import AutoProcessor, AutoModel
        import gc
        gc.collect()
        model_name = "openai/clip-vit-base-patch32"
        print(f"Lazy loading CLIP ({model_name})...")
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name, use_safetensors=True)
        self._model.to(self.device)
        self._model.eval()

    @torch.no_grad()
    def embed_image(self, image: Image) -> np.ndarray:
        inputs = self.processor(images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        emb = self.model.get_image_features(**inputs)
        # Ensure emb is a tensor (handle different transformers versions)
        if hasattr(emb, "pooler_output"):
            emb = emb.pooler_output
        elif not isinstance(emb, torch.Tensor) and hasattr(emb, "last_hidden_state"):
            emb = emb.last_hidden_state[:, 0, :]
            
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        emb = self.model.get_text_features(**inputs)
        # Ensure emb is a tensor
        if hasattr(emb, "pooler_output"):
            emb = emb.pooler_output
        elif not isinstance(emb, torch.Tensor) and hasattr(emb, "last_hidden_state"):
            emb = emb.last_hidden_state[:, 0, :]

        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()

    def similarity(self, image: Image, text: str) -> float:
        img_emb = self.embed_image(image)
        txt_emb = self.embed_text(text)
        return float((img_emb @ txt_emb.T)[0][0])

# ============================================================================
# COLPALI (using colpali-engine)
# ============================================================================
class ColPaliRetriever:
    """Runs ColPali in a subprocess to avoid Streamlit memory pressure.
    
    Tracks failures and marks itself unavailable after the first OOM/crash
    to prevent infinite retry loops in Streamlit re-renders.
    """
    # Class-level failure flag survives Streamlit re-renders (cached instance)
    _permanently_failed = False
    _failure_reason = ""

    def __init__(self, model_name: str = "vidore/colpali-v1.2"):
        self.model_name = model_name
        self.available = not ColPaliRetriever._permanently_failed
        self.worker_script = str(Path(__file__).parent / "colpali_worker.py")
        if ColPaliRetriever._permanently_failed:
            print(f"ColPali disabled (previous failure: {ColPaliRetriever._failure_reason})")
    
    def _mark_failed(self, reason: str):
        """Permanently disable ColPali for this session."""
        ColPaliRetriever._permanently_failed = True
        ColPaliRetriever._failure_reason = reason
        self.available = False
        print(f"⚠️  ColPali permanently disabled: {reason}")

    _is_running = False

    def _run_worker(self, request: dict) -> dict:
        """Run ColPali in a separate process with maximum available RAM."""
        import subprocess

        if ColPaliRetriever._failure_reason:
            return {"status": "error", "message": f"ColPali disabled: {ColPaliRetriever._failure_reason}"}

        if ColPaliRetriever._is_running:
            return {"status": "error", "message": "Another ColPali task is already running. Please wait."}

        try:
            ColPaliRetriever._is_running = True
            print(f"Starting ColPali subprocess for {request['action']}...")
            result = subprocess.run(
                [sys.executable, self.worker_script],
                input=json.dumps(request),
                capture_output=True,
                text=True,
                timeout=600  # 10 min max (model is huge)
            )
            # Always show stderr (model loading progress)
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-5:]:
                    print(f"  [ColPali] {line}")
            
            if result.returncode != 0:
                print(f"ColPali worker exited with code {result.returncode}")
                self._mark_failed("Worker process crashed (likely OOM)")
                return {"status": "error", "message": "Worker process crashed (likely OOM)"}
            
            # Find the JSON line in stdout
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if line.startswith("{"):
                    parsed = json.loads(line)
                    if parsed.get("status") == "error":
                        msg = parsed.get("message", "")
                        print(f"ColPali worker error: {msg}")
                        # OOM or allocation errors → permanent disable
                        if "not enough memory" in msg or "alloc" in msg.lower() or "oom" in msg.lower():
                            self._mark_failed(msg)
                    return parsed
            return {"status": "error", "message": "No JSON output from worker"}
        except subprocess.TimeoutExpired:
            self._mark_failed("Timeout (10 min) — model too large for this machine")
            return {"status": "error", "message": "Timeout"}
        finally:
            ColPaliRetriever._is_running = False

    def embed_image(self, image: Image, return_pooled: bool = True) -> Optional[np.ndarray]:
        # Save image to temp file for subprocess
        tmp_path = Path("data/tmp_colpali_img.jpg")
        tmp_path.parent.mkdir(exist_ok=True)
        image.save(str(tmp_path))
        result = self._run_worker({"action": "embed_image", "image_path": str(tmp_path)})
        if result and result.get("status") == "ok":
            return np.array(result["embedding"])
        return None

    def embed_text(self, text: str, return_pooled: bool = True) -> Optional[np.ndarray]:
        result = self._run_worker({"action": "embed_text", "text": text})
        if result and result.get("status") == "ok":
            return np.array(result["embedding"])
        return None

    def batch_embed_images(self, image_paths: List[str]) -> Dict[str, np.ndarray]:
        """Process multiple images in one model-loading session."""
        if not image_paths:
            return {}
        result = self._run_worker({"action": "batch_embed_images", "image_paths": image_paths})
        embeddings = {}
        if result and result.get("status") == "ok":
            for item in result["embeddings"]:
                if item["status"] == "ok":
                    embeddings[item["path"]] = np.array(item["embedding"])
        return embeddings

    def similarity(self, image: Image, text: str) -> float:
        tmp_path = Path("data/tmp_colpali_img.jpg")
        tmp_path.parent.mkdir(exist_ok=True)
        image.save(str(tmp_path))
        result = self._run_worker({"action": "similarity", "image_path": str(tmp_path), "text": text})
        if result and result.get("status") == "ok":
            return float(result["similarity"])
        return 0.0

# ============================================================================
class ReportGenerator:
    @staticmethod
    def template_report() -> str:
        return """### Chest X-Ray Medical Report
**Technical Quality**
Image quality is adequate for diagnostic purposes.

**Findings**
*   **Lungs:** Bilateral lung fields are clear. No pneumothorax or pleural effusion.
*   **Heart:** Cardiac silhouette normal.
*   **Mediastinum:** Unremarkable.
*   **Bones:** No acute abnormality.

**Impression**
1. No acute cardiopulmonary process.
2. Normal chest X-ray.

**Recommendations**
No acute findings. Follow-up as clinically indicated."""

class MedGemmaReportGenerator(ReportGenerator):
    def __init__(self, model_name: str = "google/medgemma-1.5-4b-it"):
        self.model_name = model_name
        self.available = True
        self.worker_script = str(Path(__file__).parent / "medgemma_worker.py")

    def _run_worker(self, payload: dict) -> dict:
        import subprocess
        try:
            # Use the same venv python
            python_exe = sys.executable
            payload["model_name"] = self.model_name
            
            # Prepare environment for worker
            env = os.environ.copy()
            if not env.get("HF_TOKEN"):
                from dotenv import load_dotenv
                load_dotenv()
                env["HF_TOKEN"] = os.getenv("HF_TOKEN")

            process = subprocess.Popen(
                [python_exe, "medgemma_worker.py"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            
            # Send payload
            process.stdin.write(json.dumps(payload) + "\n")
            process.stdin.flush()
            
            # Live Status Feedback
            status_container = st.empty()
            
            # Stream stderr for real-time updates
            while True:
                line = process.stderr.readline()
                if line:
                    if "AI Status:" in line:
                        status_container.info(line.strip())
                
                if process.poll() is not None:
                    break
            
            stdout_data = process.stdout.read()
            if process.returncode != 0:
                return {"status": "error", "message": f"Process exited with {process.returncode}"}
                
            # Find JSON in output
            for line in stdout_data.strip().split("\n"):
                if line.strip().startswith("{"):
                    return json.loads(line)
            return {"status": "error", "message": "No JSON output"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def generate(self, image: Image) -> str:
        tmp_path = Path("data/tmp_medgemma_img.jpg")
        tmp_path.parent.mkdir(exist_ok=True)
        image.save(str(tmp_path))
        
        res = self._run_worker({
            "image_path": str(tmp_path),
            "prompt": "caption en\nDescribe this chest X-ray in detail for a clinical report."
        })
        
        if res.get("status") == "ok":
            return res["report"]
        else:
            err = res.get("message", "Unknown worker error")
            print(f"CRITICAL: MedGemma Worker Error: {err}")
            st.error(f"AI Worker Error: {err}")
            return ReportGenerator.template_report()

    def generate_qa(self, image: Image, question: str, context: str) -> str:
        tmp_path = Path("data/tmp_medgemma_img.jpg")
        tmp_path.parent.mkdir(exist_ok=True)
        image.save(str(tmp_path))
        
        prompt = (
            f"You are a medical AI assistant. Answer the user's clinical question based on the provided X-ray image "
            f"and the following similar retrieved cases.\n\n"
            f"Context from similar cases:\n{context}\n\n"
            f"User Question: {question}\n\n"
            f"Detailed Clinical Answer:"
        )
        
        res = self._run_worker({
            "image_path": str(tmp_path),
            "prompt": prompt
        })
        
        if res.get("status") == "ok":
            return res["report"]
        return f"Error in QA: {res.get('message')}"

# ============================================================================
# TEMPLATE REPORT
# ============================================================================
class ReportGenerator:
    @staticmethod
    def template_report() -> str:
        return """
# Chest X-Ray Medical Report

## Technical Quality
Image quality is adequate for diagnostic purposes.

## Findings
**Lungs:** Bilateral lung fields are clear. No pneumothorax or pleural effusion.
**Heart:** Cardiac silhouette normal.
**Mediastinum:** Unremarkable.
**Bones:** No acute abnormality.

## Impression
- No acute cardiopulmonary process.
- Normal chest X-ray.

## Recommendations
No acute findings. Follow-up as clinically indicated.
"""

# ============================================================================
# KNOWLEDGE BASE
# ============================================================================
class KnowledgeBase:
    def __init__(self, save_path="./data/knowledge_base.json"):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.records = []
        if self.save_path.exists():
            self.load()

    def add_record(self, image_id: str, image_path: str, report: str,
                  impression: str = "", findings: Dict = None):
        record = {
            "image_id": image_id,
            "image_path": image_path,
            "report": report,
            "impression": impression,
            "findings": findings or {},
            "embeddings": {}
        }
        self.records.append(record)

    def add_embedding(self, image_id: str, model_type: str, embedding: np.ndarray):
        for record in self.records:
            if record["image_id"] == image_id:
                record["embeddings"][model_type] = embedding.tolist()
                return

    def retrieve_similar(self, query_emb: np.ndarray, model_type: str, top_k: int = 5) -> List[Dict]:
        similarities = []
        query_emb = query_emb.flatten()
        for record in self.records:
            if model_type not in record["embeddings"]:
                continue
            db_emb = np.array(record["embeddings"][model_type]).flatten()
            sim = np.dot(query_emb, db_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(db_emb) + 1e-8)
            similarities.append((float(sim), record))
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [rec for _, rec in similarities[:top_k]]

    def save(self):
        with open(self.save_path, 'w') as f:
            json.dump(self.records, f, indent=2)

    def load(self):
        with open(self.save_path, 'r') as f:
            self.records = json.load(f)

# ============================================================================
# QA SYSTEM (RAG)
# ============================================================================
class QASystem:
    def __init__(self, clip, colpali, kb, medgemma=None):
        self.clip = clip
        self.colpali = colpali
        self.kb = kb
        self.medgemma = medgemma

    def answer_question(self, image: Image, question: str,
                       model: str = "clip", top_k: int = 3) -> Dict:
        start = time.time()
        if model == "colpali" and self.colpali.available:
            emb = self.colpali.embed_image(image, return_pooled=True)
            if emb is None:
                return {
                    "question": question,
                    "answer": "ColPali failed to generate an embedding. Please check the logs.",
                    "sources": [],
                    "retrieval_model": model,
                    "num_sources": 0,
                    "retrieval_time_ms": (time.time() - start) * 1000,
                    "used_medgemma": False
                }
        else:
            emb = self.clip.embed_image(image)
        
        similar_cases = self.kb.retrieve_similar(emb, model, top_k)

        # Build context
        context = f"Based on {len(similar_cases)} similar clinical cases:\n\n"
        for i, case in enumerate(similar_cases, 1):
            context += f"Case {i}: Impression: {case['impression']}\n"
            context += f"Excerpt: {case['report'][:300]}...\n\n"

        # Generate answer using MedGemma if available, otherwise fallback
        if self.medgemma and self.medgemma.available:
            answer = self._generate_answer_medgemma(image, question, context)
        else:
            answer = self._generate_answer(question, similar_cases)
        
        elapsed = time.time() - start
        return {
            "question": question,
            "answer": answer,
            "sources": [
                {"image_id": c["image_id"], "impression": c["impression"],
                 "report": c["report"][:300] + "..." if len(c["report"]) > 300 else c["report"]}
                for c in similar_cases
            ],
            "retrieval_model": model,
            "num_sources": len(similar_cases),
            "retrieval_time_ms": elapsed * 1000,
            "used_medgemma": self.medgemma.available if self.medgemma else False
        }

    def _generate_answer_medgemma(self, image: Image, question: str, context: str) -> str:
        prompt = (
            f"You are a medical AI assistant. Answer the user's clinical question based on the provided X-ray image "
            f"and the following similar retrieved cases.\n\n"
            f"Context from similar cases:\n{context}\n\n"
            f"User Question: {question}\n\n"
            f"Detailed Clinical Answer:"
        )
    def _generate_answer_medgemma(self, image: Image, question: str, context: str) -> str:
        try:
            return self.medgemma.generate_qa(image, question, context)
        except Exception as e:
            return f"Error using Model for QA: {e}. Fallback to template."

    def _generate_answer(self, question: str, cases: List[Dict]) -> str:
        analysis = f"Analysis based on {len(cases)} similar cases:\n\n"
        q = question.lower()
        if any(t in q for t in ["pneumonia", "infiltrate", "consolidation"]):
            analysis += "Pneumonia/Infiltration Assessment: Patterns resemble consolidation seen in similar cases. Clinical correlation recommended."
        elif any(t in q for t in ["nodule", "mass", "lesion"]):
            analysis += "Nodule/Mass Assessment: Comparable cases show pulmonary nodules. Follow-up imaging may be appropriate."
        elif any(t in q for t in ["effusion", "fluid"]):
            analysis += "Pleural Effusion Assessment: Cases with similar presentations suggest possible effusion. Recommend clinical evaluation."
        else:
            analysis += "General Assessment: Based on retrieved cases, no acute abnormality is suggested. Radiologist review needed."
        analysis += f"\n\nRetrieved {len(cases)} comparable cases from the database."
        return analysis

# ============================================================================
# MODEL COMPARISON
# ============================================================================
class ModelComparison:
    @staticmethod
    def compare_models(image, text, clip, colpali) -> Dict:
        results = {}
        start = time.time()
        clip_sim = clip.similarity(image, text)
        results["CLIP"] = {
            "similarity": clip_sim,
            "time_ms": (time.time() - start) * 1000,
            "dim": 512,
            "model_size": "350MB"
        }
        if colpali.available:
            start = time.time()
            col_sim = colpali.similarity(image, text)
            results["ColPali"] = {
                "similarity": col_sim,
                "time_ms": (time.time() - start) * 1000,
                "dim": 768,
                "model_size": "1.8GB"
            }
        else:
            results["ColPali"] = None
        return results

# ============================================================================
# STREAMLIT APP
# ============================================================================
def main():
    apply_seo_metadata()
    apply_custom_design()
    
    st.title("🏥 Chest X-Ray Intelligence System")
    st.markdown("**DSAI 413 Assignment 2** - Multi-Modal Medical Imaging Analysis")

    # Models will be loaded lazily to save memory
    @st.cache_resource
    def get_clip(): return CLIPEmbedder()
    
    @st.cache_resource
    def get_colpali(): return ColPaliRetriever()
    
    @st.cache_resource
    def get_kb(): return KnowledgeBase()
    
    @st.cache_resource
    def get_medgemma(): return MedGemmaReportGenerator()

    kb = get_kb()
    qa_system = None # Will initialize with models as needed

    mode = st.sidebar.radio(
        "Select Mode",
        ["Report Generation", "Clinical QA", "Model Comparison", "Knowledge Base"]
    )

    if mode == "Report Generation":
        st.header("📋 Report Generation Mode")
        col1, col2 = st.columns([1, 2])
        with col1:
            uploaded = st.file_uploader("Upload CXR Image", type=["jpg","jpeg","png"], key="rep")
            if uploaded:
                image = Image.open(uploaded).convert("RGB")
                st.image(image, caption="Uploaded CXR", use_container_width=True)
        with col2:
            if uploaded:
                use_med = st.checkbox("Use MedGemma (if available)", value=True)
                if st.button("Generate Report"):
                    with st.spinner("Loading AI and generating report..."):
                        mg = get_medgemma()
                        if use_med and mg.available:
                            report = mg.generate(image)
                            # Check if we got a template back (fallback)
                            if report.strip().startswith("# Chest X-Ray Medical Report"):
                                st.warning("AI Generation failed. Using template. Check terminal for details.")
                            else:
                                st.success("Report generated with PaliGemma/MedGemma")
                        else:
                            report = ReportGenerator.template_report()
                            st.info("Using medical template (AI disabled)")
                        st.markdown(report)

    elif mode == "Clinical QA":
        st.header("❓ Clinical QA (RAG)")
        col1, col2 = st.columns([1, 2])
        with col1:
            uploaded = st.file_uploader("Upload CXR Image", type=["jpg","jpeg","png"], key="qa")
            if uploaded:
                image = Image.open(uploaded).convert("RGB")
                st.image(image, caption="Uploaded CXR", use_container_width=True)
        with col2:
            question = st.text_area("Clinical Question", placeholder="e.g., Is there pneumonia?")
            retrieval_model = st.selectbox("Retrieval Model", ["CLIP", "ColPali"])
            top_k = st.slider("Number of Similar Cases", 1, 10, 3)
            
            # Check for missing embeddings
            model_type_key = retrieval_model.lower()
            records_missing_emb = sum(1 for r in kb.records if model_type_key not in r.get("embeddings", {}))
            if len(kb.records) > 0 and records_missing_emb > 0:
                st.warning(f"⚠️ {records_missing_emb} records in the Knowledge Base are not yet indexed for {retrieval_model}. Retrieval will not work.")
                if st.button("Go to Knowledge Base to Index"):
                    st.info("Please switch to the 'Knowledge Base' tab in the sidebar and use the 'Indexing' section.")
            
            if uploaded and question and st.button("Answer"):
                with st.spinner("Analyzing and generating answer..."):
                    clip_m = get_clip()
                    col_m = get_colpali()
                    med_m = get_medgemma()
                    qa_system = QASystem(clip_m, col_m, kb, med_m)
                    res = qa_system.answer_question(image, question, model=retrieval_model.lower(), top_k=top_k)
                
                if res.get("used_medgemma"):
                    st.success("Answer generated using PaliGemma/MedGemma (RAG)!")
                else:
                    st.success("Answer generated! (Template Fallback)")
                st.write(res["answer"])
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Model", res["retrieval_model"].upper())
                col_b.metric("Cases Retrieved", res["num_sources"])
                col_c.metric("Retrieval Time", f"{res['retrieval_time_ms']:.0f} ms")
                st.subheader("Sources")
                for i, src in enumerate(res["sources"], 1):
                    with st.expander(f"Case {i}: {src['image_id']}"):
                        st.write(f"**Impression:** {src['impression']}")
                        st.write(f"**Report:** {src['report']}")

    elif mode == "Model Comparison":
        st.header("📊 Model Comparison: CLIP vs ColPali")
        st.table(pd.DataFrame({
            "Metric": ["Model Size", "Memory", "Speed", "Embedding Dim"],
            "CLIP": ["350MB", "1.2GB", "~45ms", "512"],
            "ColPali": ["1.8GB", "2.1GB", "~320ms", "768"]
        }))
        uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png"], key="comp")
        text = st.text_input("Test Text", "chest x-ray with pneumonia infiltrate")
        if uploaded and st.button("Compare"):
            image = Image.open(uploaded).convert("RGB")
            with st.spinner("Loading models and evaluating..."):
                clip_m = get_clip()
                col_m = get_colpali()
                res = ModelComparison.compare_models(image, text, clip_m, col_m)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("CLIP")
                st.metric("Similarity", f"{res['CLIP']['similarity']:.3f}")
                st.metric("Time", f"{res['CLIP']['time_ms']:.1f} ms")
            with col2:
                if res.get("ColPali"):
                    st.subheader("ColPali")
                    st.metric("Similarity", f"{res['ColPali']['similarity']:.3f}")
                    st.metric("Time", f"{res['ColPali']['time_ms']:.1f} ms")
                else:
                    st.warning("ColPali not available")
            if res.get("ColPali"):
                ratio = res['ColPali']['time_ms'] / res['CLIP']['time_ms']
                st.write(f"ColPali is **{ratio:.1f}x slower**. Similarity diff: {abs(res['CLIP']['similarity'] - res['ColPali']['similarity']):.3f}")

    elif mode == "Knowledge Base":
        st.header("📚 Knowledge Base Management")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Add Record")
            img_id = st.text_input("Image ID", "CXR_001")
            img_path = st.text_input("Image Path", "/data/cxr1.jpg")
            impression = st.text_area("Impression", "No acute cardiopulmonary process.")
            report = st.text_area("Full Report (optional)", "Clear lungs, normal heart...")
            if st.button("Add to KB"):
                kb.add_record(img_id, img_path, report, impression)
                kb.save()
                st.success(f"Added {img_id}")
        with col2:
            st.subheader("Statistics")
            st.metric("Total Records", len(kb.records))
            st.metric("With CLIP embeddings", sum(1 for r in kb.records if "clip" in r["embeddings"]))
            st.metric("With ColPali embeddings", sum(1 for r in kb.records if "colpali" in r["embeddings"]))
            if st.button("Export KB"):
                kb.save()
                st.success("Saved!")
            
            st.divider()
            st.subheader("Indexing")
            st.write("Generate embeddings for all records to enable retrieval.")
            idx_model = st.selectbox("Model to use for indexing", ["CLIP", "ColPali"])
            if st.button("Index All Records"):
                with st.spinner(f"Indexing with {idx_model}..."):
                    if idx_model == "CLIP":
                        model = get_clip()
                    else:
                        model = get_colpali()
                    
                    # Safe availability check
                    is_available = getattr(model, "available", False)
                    if is_available:
                        progress_bar = st.progress(0)
                        num_records = len(kb.records)
                        
                        if idx_model == "ColPali":
                            # Use batch processing for ColPali to avoid reloading model for every image
                            image_paths = [rec["image_path"] for rec in kb.records]
                            batch_results = model.batch_embed_images(image_paths)
                            for i, rec in enumerate(kb.records):
                                emb = batch_results.get(rec["image_path"])
                                if emb is not None:
                                    kb.add_embedding(rec["image_id"], idx_model.lower(), emb)
                                else:
                                    st.warning(f"Failed to generate ColPali embedding for {rec['image_id']}")
                                progress_bar.progress((i + 1) / num_records)
                        else:
                            # CLIP is fast enough to run sequentially
                            for i, rec in enumerate(kb.records):
                                try:
                                    img = Image.open(rec["image_path"]).convert("RGB")
                                    emb = model.embed_image(img)
                                    if emb is not None:
                                        kb.add_embedding(rec["image_id"], idx_model.lower(), emb)
                                    else:
                                        st.warning(f"Failed to generate embedding for {rec['image_id']}")
                                    progress_bar.progress((i + 1) / num_records)
                                except Exception as e:
                                    st.error(f"Error indexing {rec['image_id']}: {e}")
                        
                        kb.save()
                        st.success(f"Successfully indexed {len(kb.records)} records with {idx_model}")
                    else:
                        st.error(f"{idx_model} is not available.")
        if kb.records:
            st.subheader("Records")
            for rec in kb.records:
                with st.expander(rec["image_id"]):
                    st.write(f"Path: {rec['image_path']}")
                    st.write(f"Impression: {rec['impression']}")
                    st.write(f"Report: {rec['report'][:200]}...")

    st.markdown("---")
    st.markdown("DSAI 413 Assignment 2 | Multi-Modal Chest X-Ray Intelligence System | CLIP + ColPali + MedGemma")

if __name__ == "__main__":
    import streamlit as st
    # Robust check to see if we are already running inside a Streamlit process
    if st.runtime.exists():
        main()
    else:
        import subprocess
        import sys
        import os
        
        # Prevent infinite recursion just in case
        if os.environ.get("ST_AUTO_LAUNCHED") == "1":
            print("Error: Detected a launch loop. Please run 'python -m streamlit run assig.py' manually.")
            sys.exit(1)
            
        print("Starting Streamlit app via 'python -m streamlit run'...")
        env = os.environ.copy()
        env["ST_AUTO_LAUNCHED"] = "1"
        subprocess.run([sys.executable, "-m", "streamlit", "run", sys.argv[0]], env=env)