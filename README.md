# NLP-Fraud-Detection-project
Public Procurement Collusion & Fraud Detector

👥 The Team

[Malcolm Palmer]

[Nourhene Ben Juddou]

[Giulia Lorelli]

[Brian O'Sullivan]

Developed for Bologna Buisness School - Master's in AI & Innovation Management
---

## 1. Project Overview
Public procurement accounts for roughly 14% of EU GDP. Irregularities—ranging from bid rigging and single-tendering abuse to inflated contract values and winner concentration—cost European taxpayers an estimated **EUR 5-25 billion per year**. 

This project tackles this systemic problem by building a multi-layered NLP framework that ingests, cleans, analyzes, and explains potential fraud patterns within public sector data.

* **Primary Data Source:** [Tenders Electronic Daily (TED)](https://ted.europa.eu) — the official EU public procurement journal. It publishes approximately 700,000 notices per year across all 24 EU official languages. 
* **Data Access:** Machine-readable subsets are ingested via OpenTender.eu and the DIGIWHIST dataset on Harvard Dataverse.

### 🚩 Typology of Targeted Irregularities
Our pipeline focuses on identifying specific "red flags" defined by procurement and anti-fraud frameworks:
* **Single-Bidding / Lack of Competition:** Only one tender is received for a given competitive contract.
* **Winner Concentration:** The exact same supplier repeatedly wins contracts from the same public buyer.
* **Abnormally Short Tender Period:** Insufficient or artificially narrow time windows allocated for competitors to submit bids.
* **Contract Splitting:** Artificially dividing a large contract into smaller pieces to stay below mandatory publication thresholds.
* **Copy-Paste Descriptions:** Identical or near-identical notices issued by different buyers, indicating potential collusion or pre-written specs.
* **Value Discrepancy:** Significant variation between the initial estimated contract value and final awarded value.
* **Entity-Linked Red Flags:** Shared addresses, contact information, or registration details across multiple distinct bidders (detected via NER).

---

## 📅 2. Project Timeline
The roadmap covers a **16 working day sprint** starting from 26 May, broken down into 6 distinct phases leading up to the final evaluation.

| Phase | Focus | Dates | Duration |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Problem Framing & Data Definition | May 26–28 | 3 Days |
| **Phase 2** | Text Preprocessing & EDA | May 28–31 | 3 Days |
| **Phase 3** | Feature Engineering & Embeddings | Jun 01–04 | 4 Days |
| **Phase 4** | NLP Model & Detection Pipeline | Jun 04–07 | 4 Days |
| **Phase 5** | RAG + LLM Explainability Layer | Jun 07–09 | 2 Days |
| **Phase 6** | Evaluation, Repository Cleanup & Presentation | Jun 09–11 | 2 Days |

---

## 🔧 3. Phase Details

### Phase 1 — Problem Framing & Data (May 26-28)
Establishing ground truth definitions, data parameters, and the core tasks.

* **Choose Dataset:** * *Option A (Recommended):* OpenTender.eu export for 1-2 target countries (e.g., Romania + Bulgaria — representing high historical irregularity rates). Extracted via REST API or packaged CSV.
  * *Option B:* DIGIWHIST dataset (Harvard Dataverse) — pre-cleaned data inclusive of calculated corruption risk scores per contract.
* **Define Labels:** Construct proxy indicators to act as our ground truth targets: `single_bid` (0/1), `winner_concentration` (0/1), and `short_tender_period` (0/1). These forms a multi-label setup for final evaluation.
* **Define NLP Tasks:** 1. *Binary/Multi-label Classification:* Identify clean vs. suspicious entries.
  2. *Semantic Similarity:* Scan for copy-paste notice duplication.
  3. *Named Entity Recognition (NER):* Extract and cross-reference companies and buyers.
  4. *LLM Explanation:* Generate structured risk reports for flag vectors.
* **Deliverable:** Initial repository scaffolding with draft README containing dataset cards, label specifications, task breakdowns, and an annotated sample comparison (clean vs. suspicious).

### Phase 2 — Text Preprocessing & EDA (May 28-31)
Building the ingestion and preprocessing backbone to handle multilingual, noisy HTML-laden procurement texts.

* **Language Detection:** Run `langdetect` or `langid.py` over the target text field. Isolate and flag cross-lingual mismatches (e.g., a Spanish text layout filed by a Romanian buyer).
* **Text Cleaning:** Strip HTML tags, boilerplate legal headers, and CPV code strings using custom regex pipelines compiled into a unified `spaCy` text cleaner.
* **Tokenization:** Tokenize using `spaCy` (`xx_core_web_sm`). Compare token counts dynamically against `tiktoken` to benchmark context limits for the downstream LLM modules.
* **Exploratory Data Analysis:** Profile contract distributions on log scales, map label imbalances (historically around 10-20% anomalies), check CPV distribution, and construct tender period histograms.
* **Deliverable:** Notebook `01_preprocessing_eda.ipynb` and the fully processed `contracts_clean.csv`.

### Phase 3 — Feature Engineering & Embeddings (June 1-4)
Transitioning from classical matrix representations to deep multilingual space configurations.

* **TF-IDF Baseline:** Build a sparse text matrix (`TfidfVectorizer(sublinear_tf=True, max_features=50000)`) to act as our benchmark text layer.
* **Structured Features:** Extract numerical indicators—`contract_value_ratio` (awarded vs estimated), `n_bidders`, `tender_duration_days`, and `buyer_winner_pair_frequency`—and stack them horizontally with our text features.
* **Multilingual Embeddings:** Generate sentence vectors using the pre-trained `paraphrase-multilingual-mpnet-base-v2` transformer model (768 dimensions, supports 50+ languages).
* **Similarity Detection:** Apply batch cosine similarity to detect parallel descriptions across separate buying authorities. An unsupervised similarity threshold of $>0.92$ tags potential copy-paste collusions.
* **NER for Entity Linking:** Deploy `dslim/bert-base-NER` or a multilingual XLM-RoBERTa equivalent to parse and structure `ORG` identifiers to chart hidden shell company profiles.
* **Deliverable:** Notebook `02_features_embeddings.ipynb` along with stored vectors `embeddings.npy`, `contract_ids.csv`, and extracted entities in `ner_entities.csv`.

### Phase 4 — NLP Model & Detection Pipeline (June 4-7)
Constructing models that span classical estimators to fine-tuned transformer networks.

* **Baseline Estimators:** Fit `LogisticRegression` and `LinearSVC` classifiers over the text TF-IDF space. Employ `class_weight='balanced'` handling to balance skewed label footprints. Evaluation is anchored on Precision, Recall, F1-Score, and PR-AUC.
* **Embedding Classifiers:** Train linear models directly over the concatenated transformer embeddings and structured feature vectors, measuring improvements over the baseline.
* **Anomaly Detection:** Execute `IsolationForest` and `DBSCAN` over target embedding spaces to identify outliers completely unsupervised. Project clusters down using UMAP or t-SNE vectors.
* **Transformer Fine-Tuning:** Perform parameter updates on `mBERT` or `XLM-RoBERTa` models using the HuggingFace `Trainer` loop. Use LoRA/QLoRA if processing under constrained GPU environments.
* **Error Analysis:** Conduct a deep dive into false positives and false negatives to diagnose structural model vulnerabilities across specific contract domains.
* **Deliverable:** Notebook `03_detection_models.ipynb` and serialized model files alongside a unified model metrics benchmark panel.

### Phase 5 — RAG + LLM Explainability Layer (June 7-9)
Injecting a Retrieval-Augmented Generation layout to ensure all generated audit summaries remain fully grounded in legal realities.

> 💡 **Why RAG?** Plain LLM prompting introduces unacceptable hallucination risks when evaluating legal compliance. A local RAG framework guarantees that final text explanations are explicitly cross-referenced with actual statutory text.

* **Document Corpus:** Chunk EU Directive 2014/24/EU (Public Procurement), OLAF anti-fraud guidelines, and OpenTender fraud typologies into 400-token blocks featuring a 50-token window overlap.
* **Vector Store:** Ingest structural blocks into a local `FAISS` index using the primary `sentence-transformer` embedding model. Retrieve the top $k=5$ most relevant legal criteria context vectors.
* **RAG Pipeline:** Develop a `LangChain` execution frame connecting: `Contract Notice` $\rightarrow$ `Dense Retrieval Search` $\rightarrow$ `Context-Augmented Prompt Construction` $\rightarrow$ `LLM Generation`.
* **Structured Output:** Enforce strict JSON object output templates (`risk_score`, `risk_factors`, `regulation_references`, `explanation`) leveraging Pydantic AI or custom engine function-calling parameters.
* **RAGAS Evaluation:** Programmatically grade pipeline output metrics checking for *faithfulness*, *answer relevance*, and *context precision* across 20-30 curated testing indices.
* **Deliverable:** Notebook `04_rag_explainability.ipynb`, `ragas_results.json`, and an operational terminal CLI utility (`analyze_contract.py --id XYZ`).

### Phase 6 — Evaluation, Repository & Presentation (June 9-11)
Consolidating final repository standards, pipeline testing routines, and presentation assets.

* **Final Evaluation:** Compile metrics across all experiments into a clean summary table containing classification logs, PR-AUC charts, and target RAGAS scores.
* **Repository Cleanup:** Standardize configurations for `requirements.txt`, `.gitignore`, and download utilities (`data/download_data.sh`). Ensure raw source text fields are not committed.
* **Reproducibility Check:** Run full end-to-end sandbox execution audits of all 4 notebooks in empty Google Colab spaces to isolate path or dependency collisions.
* **Presentation Build:** Map presentation flow into a tight 15-minute sequence (Problem/Data $\rightarrow$ Pipe Architecture $\rightarrow$ Performance Metrics $\rightarrow$ RAG Live Demo $\rightarrow$ Technical Frontiers).
* **Oral Exam Prep:** Complete group synchronization to defend architecture selections independently (e.g., explaining transformer embeddings selection, cross-lingual choices, or model quantization metrics).

---

## 🛠️ 4. Full NLP Technology Stack

| Layer | Tool / Model | Course Module Alignment |
| :--- | :--- | :--- |
| **Preprocessing** | `spaCy` (multilingual), `langdetect`, `NLTK` | Module 1: Foundational Text Processing |
| **Vectorization** | `scikit-learn` (`TfidfVectorizer`) | Module 2: Text Representations |
| **Word Embeddings** | `FastText` (multilingual) | Module 2: Vector Space Configurations |
| **Sentence Embeddings** | `paraphrase-multilingual-mpnet-base-v2` | Module 2: Contextual Vectors |
| **Classification Baseline**| `LogisticRegression` / `LinearSVC` + TF-IDF | Module 2: Classical Classification |
| **Embedding Classifier** | `LogisticRegression` on Sentence Vectors | Module 2: Neural Token Embeddings |
| **Anomaly Detection** | `IsolationForest` / `DBSCAN` (on Embeddings) | Module 2 & 3: Unsupervised Landscapes |
| **NER Extraction** | `XLM-RoBERTa NER` | Module 3: Sequence Labeling |
| **Deep Classifier** | Fine-Tuned `mBERT` / `XLM-RoBERTa` (HF) | Module 3: Transformer Networks |
| **RAG Pipeline** | `LangChain` + `FAISS` + `sentence-transformers` | Module 4: Generative Frameworks |
| **Structured Output** | Function Calling / `Pydantic AI` (JSON Format) | Module 4: Production LLM Design |
| **Evaluation Framework** | `scikit-learn metrics`, `RAGAS`, LLM-as-a-Judge | Module 4 & 5: Validation & LLMOps |

---

## 🏗️ 5. Recommended System Architecture

The analytical application relies on three distinct decoupled steps configured to optimize computation footprint and pipeline costs:

### 🔹 Layer 1 — Classical Detection Baseline
$$\text{Contract Description Text + Structured Features} \longrightarrow \text{TF-IDF Vectorization} \longrightarrow \text{Logistic Regression / LinearSVC} \longrightarrow \text{Binary Label + Confidence Score}$$

### 🔹 Layer 2 — Neural Detection
$$\text{Contract Description Text} \longrightarrow \text{Multilingual Sentence-Transformer Encoder} \longrightarrow \text{Dense Vector Space + Metadata Features} \longrightarrow \text{Classifier Head} \longrightarrow \text{Anomalous Vector Flag + Copy-Paste Cosine Verification}$$

### 🔹 Layer 3 — RAG Explanation
$$\text{Flagged Suspicious Notice} \longrightarrow \text{FAISS Legal Document Index Query} \longrightarrow \text{Context-Injected System Prompt} \longrightarrow \text{LLM Processing} \longrightarrow \text{Structured JSON Risk Audit Report}$$

> 📌 **Architectural Note:** Layers 1 and 2 run completely local and offline in batch processing configurations at negligible cost. Layer 3 (External LLM API invocation) triggers dynamically *only* when a record successfully breaches the anomaly thresholds set by Layer 2. This structure maintains tight latency boundaries and manages production API budgets efficiently.

---

## 📊 6. Grading Rubric Alignment
The team-based software repository contributes **40% of the course weight** (scaled across 30 total points).

| Evaluation Criterion | Targets & Minimum Implementations |
| :--- | :--- |
| **Problem Framing & Design (6 Pts)** | Coherent definition of targeted procurement fraud typologies, proper dataset ingestion rationale, and clear cross-lingual text mapping boundaries. |
| **Code Quality & Reproducibility (8 Pts)** | Fully normalized Git code layouts, strict `requirements.txt` environment locks, and pristine execution passes across all notebooks in clean virtual environments. |
| **Technical Soundness (8 Pts)** | Rigorous technical validation of modeling pathways (e.g., transformer embedding setups, fine-tuning choices, and architectural reasoning behind the local vector indexes). |
| **Evaluation Methodology (5 Pts)** | Proper handling of severe target label class imbalances via F1 and PR-AUC calculations, complemented by formal RAGAS evaluation arrays. |
| **Presentation Clarity (3 Pts)** | Strict coordination across the 15-minute demo block, concluding with a functional live pipeline run across real procurement samples. |

---

## 🚀 7. Recommended Data Ingestion Repositories
* **[OpenTender.eu](https://opentender.eu):** Provides access to structured versions of historical TED data. Features internal risk index parameters (`single_bid`, `tender_period`). Excellent for generating clean target labels.
* **[TED CSV Bulk Export](https://data.europa.eu):** The raw, official publishing baseline of all 24 official EU language fields. Highly complete, but requires specialized token cleaning loops.
* **[DIGIWHIST Dataset (Harvard Dataverse)](https://dataverse.harvard.edu):** Curated repository of academic-grade public tenders. Includes computed corruption proxy indicators per entry, making it an ideal choice for supervised workflows.

---

## 📂 8. Repository Structure Diagram

```text
├── data/               # Local subdirectory for raw or processed inputs
│   └── download_data.sh# Script utility to hit target OpenTender API nodes
├── notebooks/          # Step-by-step modular pipeline notebooks
│   ├── 01_preprocessing_eda.ipynb      # Phase 2: Ingestion, cleaning, language flags
│   ├── 02_features_embeddings.ipynb    # Phase 3: TF-IDF, embeddings, NER extraction
│   ├── 03_detection_models.ipynb      # Phase 4: Scikit-learn estimators & Transformers
│   └── 04_rag_explainability.ipynb     # Phase 5: FAISS vector pipeline & RAGAS scores
├── src/                # Modular library scripts
│   ├── pipeline.py     # Global utility cleaning and data transforms
│   └── rag.py          # Class definitions for the LangChain structure
├── models/             # Serialized joblib configurations and target weights
├── results/            # Performance score logging matrices, JSONs, and PR-AUC plots
├── analyze_contract.py # Deployment CLI hook: `python analyze_contract.py --id XYZ`
├── requirements.txt    # Frozen virtual environment package specifications
├── .gitignore          # Repository block filter rules
└── README.md           # Main documentation hub
