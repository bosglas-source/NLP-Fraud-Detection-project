# NLP-Fraud-Detection-project
Public Procurement Collusion & Fraud Detector

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![NLP Framework](https://img.shields.io/badge/NLP-HuggingFace_Transformers-orange)
![UI](https://img.shields.io/badge/UI-Streamlit-red)

👥 The Team

[Malcolm Palmer] - Role ()

[Nourhene Ben Juddou] - Role ()

[Giulia Lorelli] - Role ()

[Brian O'Sullivan] - Role ()

Developed for Bologna Buisness School - Master's in AI & Innovation Management

## 📌 Project Overview
This repository contains an end-to-end Natural Language Processing application designed to detect bid-rigging and semantic collusion in public procurement processes. 

Public procurement accounts for a massive portion of government spending. A common tactic in procurement fraud is the submission of artificially high bids by supposed competitors using identical boilerplate text, allowing a pre-chosen winner to secure the contract. This tool empowers public auditors by automating the detection of these hidden linguistic links across thousands of tender submissions.

## ⚖️ Socio-Political & Ethical Context
Built as an intersection of Computational Social Science and AI, this project addresses systemic corruption and algorithmic accountability. 
* **The Problem:** Collusion drains public tax euros, creates monopolies, and fundamentally undermines fair democratic market access. 
* **The Solution:** By moving beyond standard financial audits to analyze the *semantic architecture* of bids, we expose cartel behaviors that traditional accounting misses.
* **Ethical AI Design:** The tool acts as an 'auditor's co-pilot,' explicitly avoiding black-box decision-making. All flagged collusion risks are explainable, tracing back to the exact syntactic overlap that triggered the anomaly detection.

## ⚙️ Technical Architecture
Our pipeline processes unstructured text from public tender portals (e.g., OpenTender/TED) into an interactive collusion-risk dashboard.

1. **Data Ingestion & Preprocessing:** Cleaning and normalizing messy procurement text, isolating the core descriptions and boilerplate language.
2. **Document Embeddings:** Utilizing deep learning models (e.g., Longformer / LegalBERT) to generate high-dimensional vector representations of each bid.
3. **Similarity Matrix & Anomaly Detection:** Calculating cosine similarity across bids for the same contract and applying unsupervised anomaly detection (Isolation Forest / DBSCAN) to flag statistically improbable textual overlaps.
4. **Interactive Dashboard:** A Streamlit-based user interface that visualizes hidden bid-rigging networks using `NetworkX` graph visualizations.

## 📂 Repository Structure
```text
├── data/               # Raw and processed EU tender data (JSON/CSV)
├── notebooks/          # Exploratory Data Analysis (EDA) and model experimentation
├── src/                # Core pipeline modules
│   ├── data_loader.py  # Scripts to fetch and clean tender data
│   ├── embedder.py     # Embedding generation using Hugging Face
│   ├── anomaly.py      # Similarity calculations and Isolation Forest logic
│   └── visualizer.py   # NetworkX graph generation
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # You are here
