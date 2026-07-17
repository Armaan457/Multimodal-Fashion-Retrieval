# Multimodal Fashion Retrieval

A multimodal fashion retrieval system implementing two complementary retrieval pipelines designed for high-quality image search from natural language descriptions.

**Project Report:** [Report](https://docs.google.com/document/d/1k5635CgrfiRUiMr3FztmOzXu-P4K0-D2BxSNKV4BA0g/edit?usp=sharing)

---

## Overview

This project explores two different approaches to fashion retrieval:

### 1. Multi-Head Retrieval Pipeline

**FashionCLIP + Semantic Heads**

A hierarchical retrieval framework that combines a shared FashionCLIP embedding space with specialized semantic heads.

**Pipeline**

```
Text Query
     │
     ▼
 FashionCLIP
     │
     ▼
 Vector Retrieval (HNSW)
     │
     ▼
 Multi-Head Semantic Scoring
 ├── Category Head
 ├── Attribute Head
 └── Style Head
     │
     ▼
 Weighted Reciprocal Rank Fusion (RRF)
     │
     ▼
 Ranked Results
```

### 2. R2 Retrieval Pipeline

**Fashion SigLIP + BLIP Re-ranking**

A two-stage retrieval architecture focused on maximizing retrieval quality.

**Pipeline**

```
Text Query
     │
     ▼
 Fashion SigLIP
     │
     ▼
Vector Retrieval (HNSW)
     │
     ▼
 Top-K Candidates
     │
     ▼
 BLIP Cross-Modal Re-ranking
     │
     ▼
 Final Ranked Results
```

---

## Features

* Natural language fashion search
* HNSW-based approximate nearest-neighbor retrieval
* Fashion-specific embedding models
* Multi-head semantic ranking
* Cross-modal BLIP re-ranking
* Reciprocal Rank Fusion (RRF)
* Modular retrieval pipeline
* Fast candidate generation with high-quality final ranking
* Easily extensible to new semantic heads or rerankers

---

## Components

### Multi-Head Pipeline

* **Backbone:** FashionCLIP
* **Semantic Heads:**
  * Category
  * Attribute
  * Style
* **Fusion:** Weighted Reciprocal Rank Fusion (RRF)

### R2 Pipeline

* **Backbone:** FashionSigLIP
* **Re-ranker:** BLIP Cross-Modal Model

---

## Retrieval Flow

1. Encode the text query.
2. Retrieve the nearest image candidates using HNSW.
3. Apply semantic scoring (Multi-Head) or cross-modal re-ranking (BLIP).
4. Produce the final ranked fashion images.

---

## Tech Stack

* Python
* PyTorch
* FashionCLIP
* FashionSigLIP
* BLIP
* HNSW
* NumPy

---

Both pipelines are implemented independently in their respective folders, enabling direct comparison between semantic multi-head retrieval and cross-modal re-ranking approaches for fashion image search. You can test them out using `testing.ipynb` notebook.
