# Multi-Modal Chest X-Ray Intelligence System
**DSAI 413 - Assignment 2**

## 1. Architecture Overview
The system is built as a dual-mode application using Streamlit to provide an intuitive interface for clinicians. It supports two main functionalities:
1. **Report Generation Mode:** An end-to-end multimodal pipeline that takes a chest X-ray image as input and produces a structured medical report (Technical Quality, Findings, Impression, Recommendations).
2. **Clinical QA Mode (RAG):** A Retrieval-Augmented Generation pipeline where users can upload an X-ray and ask a clinical question. The system embeds the uploaded image to retrieve similar historical cases from a predefined Knowledge Base. These retrieved cases act as context for a generative model to answer the specific question.

## 2. Model Choices
The implementation utilizes three primary models, each serving a distinct purpose in the multimodal pipeline:

* **MedGemma (`google/medgemma-1.5-4b-it`):** 
  * *Role:* Acts as the core generative engine for both report generation and grounded RAG answer generation. 
  * *Why:* MedGemma is specifically fine-tuned on medical texts and imagery (vision-to-sequence). We implemented **4-bit quantization** (NF4) and **GPU acceleration** to allow this 4B-parameter model to run efficiently on consumer-grade hardware.

* **ColPali (`vidore/colpali-v1.2`):**
  * *Role:* Document/Image retrieval for the RAG pipeline.
  * *Why:* ColPali leverages vision-language models for direct image patch embeddings. It captures the rich visual layout and localized features of the X-rays, making it extremely effective at retrieving semantically and visually similar cases.

* **CLIP (`openai/clip-vit-base-patch32`):**
  * *Role:* Baseline retrieval model for comparison against ColPali.
  * *Why:* CLIP is a standard, lightweight contrastive vision-language model. While excellent for general image-text similarity, it serves as a strong baseline to highlight the specialized retrieval capabilities of ColPali in the medical domain.

## 3. Dataset and Knowledge Base Creation
Since a pre-existing QA dataset was not explicitly provided in the codebase, a custom data preparation script (`prepare_data.py`) was developed. 
- **Data Source:** Sample chest X-ray images (e.g., normal and pneumonia presentations) and synthetic medical reports were created.
- **QA Generation:** For each case, clinical question-answer pairs were generated based strictly on the report findings (e.g., questions regarding opacities, heart size, and general impressions). 
- **Storage:** The data is serialized into `data/knowledge_base.json`, which serves as the vector database for our RAG system. At runtime, embeddings are calculated dynamically or loaded from this base.

## 4. Comparison Results: ColPali vs CLIP
A direct comparison module within the application evaluates the retrieval components based on speed, memory footprint, and similarity scoring.

| Metric | CLIP (openai/clip-vit-base-patch32) | ColPali (vidore/colpali-v1.2) |
| :--- | :--- | :--- |
| **Model Size** | ~350 MB | ~1.8 GB |
| **Embedding Dimension** | 512 | 768 (pooled from patches) |
| **Inference Speed** | ~45 ms (Fast) | ~320 ms (Slower, requires more compute) |
| **Retrieval Accuracy**| Good for general semantic alignment. | Excellent for localized, specific radiological features. |

**Insights & Hardware Optimizations:**
* **CLIP** is highly efficient and lightweight but struggles with fine-grained medical details because it compresses the entire image into a single dense vector.
* **ColPali Architecture (Late Interaction vs Mean Pooling):** During testing, we observed that forcing ColPali into a traditional "mean-pooling" setup (squashing its 1,000+ image patch vectors into a single vector) results in cosine similarity scores near `0.000`. This perfectly demonstrates that ColPali requires a specialized "Late Interaction" vector database (like Vespa or Qdrant) that can mathematically compare all multi-vector patches simultaneously, unlike CLIP's single-vector approach.
* **Hardware & Memory Triage (6GB VRAM Limit):** Running a 3B-parameter ColPali model alongside a 4B-parameter MedGemma model on an RTX 4050 (6GB VRAM) presented severe Out-Of-Memory (OOM) challenges. To overcome this:
  1. **Subprocess Isolation:** Models were isolated into dedicated worker processes (`medgemma_worker.py`, `colpali_worker.py`) that spin up and completely release VRAM back to the OS upon exiting.
  2. **Aggressive Quantization:** Both models run in **4-bit NF4 quantization**, compressing 6GB models down to ~2.5GB.
  3. **SDPA (Scaled Dot-Product Attention):** For ColPali, processing 1,000+ high-res image patches typically requires ~3GB of activation VRAM. By switching from `eager` attention to `sdpa` (Flash Attention), we drastically reduced the memory spike, allowing ColPali to successfully run on the 6GB GPU in under 30 seconds (down from 15 minutes on CPU).
 
### Report Generation Flow
```text
Chest X-Ray Image
        ↓
MedGemma Vision Encoder
        ↓
Medical Language Generation
        ↓
Structured Clinical Report

## 5. End-to-End System Integration

```
## Rag flow
```text
Query X-Ray + Clinical Question
                ↓
Image Embedding (CLIP / ColPali)
                ↓
Similarity Search in Knowledge Base
                ↓
Retrieve Top-K Similar Cases
                ↓
Build Context
                ↓
MedGemma Generation
                ↓
Grounded Clinical Answer
```

## Indexing Phase
```text
Knowledge Base Images
        ↓
CLIP / ColPali Encoder
        ↓
Embedding Vectors
        ↓
Saved inside knowledge_base.json
```
