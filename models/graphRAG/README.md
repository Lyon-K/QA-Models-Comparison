# GraphRAG Build Pipeline (Heavy Dependency Module)

This submodule contains the **graph construction pipeline** for the GraphRAG system.

⚠️ **Important:** This module is **not used during inference** and is intentionally separated due to heavier dependencies and setup requirements.

---

## 🚧 Why This Exists

The root project is designed for **lightweight inference**.

This module handles:

* Knowledge graph construction
* Triplet extraction via local LLM
* Neo4j population

These steps require:

* Local model serving
* Database setup
* Additional setup time and resources

To avoid slowing down or complicating inference, this is **excluded from the root `requirements.txt`**.

---

## 📦 Installation

Install dependencies using the **requirements file in this folder**, not the root directory:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Setup Instructions

### 1. Install and pull the required model

Ensure Ollama is installed, then pull the model required:

```bash
ollama pull sciphi/triplex
```

---

### 2. start Ollama

Then start the server:

```bash
ollama serve
```

---

### 3. Start Neo4j

Ensure your Neo4j database is running and accessible.

---

### 4. Enable Neo4j GDS (Graph Data Science)

Ensure the Neo4j Graph Data Science (GDS) plugin is installed and enabled.

This is required for operations such as **cosine similarity** used during retrieval.

---

### 5. Configure environment variables

Update the `.env` file with your Neo4j credentials:

```env
NEO4J_URI="ENTER_YOUR_NEO4J_URI"
NEO4J_USER="ENTER_YOUR_NEO4J_USER"
NEO4J_PASSWORD="ENTER_YOUR_NEO4J_PASSWORD"
NEO4J_DATABASE="ENTER_YOUR_NEO4J_DATABASE"
```

---

## 🧠 Usage

### Step 1 — Initialize GraphRAG

```python
from graphRAG import GraphRAG

graph_rag = GraphRAG()
```

---

### Step 2 — Clear existing graph

```python
from your_module import clear_db

clear_db(graph_rag)
```

---

### Step 3 — Load dataset

```python
import pandas as pd

mock_dataset = pd.read_parquet("data/sample_data.parquet").drop("Dataset", axis=1)
```

---

### Step 4 — Build the graph

```python
from your_module import build_graph

build_graph(graph_rag, mock_dataset, logging=False)
```

---

## 🔄 When to Rebuild the Graph

Rebuilding is required when:

* The **dataset changes**
* The **embedding model changes**
* The **graph construction logic is modified**
* The **GraphRAG structure or schema is updated**

---

## 🧩 Key Functions

* `build_graph(...)`
  → Main pipeline for constructing the knowledge graph

* `triplextract_ollama(...)`
  → Calls the Ollama model for triplet extraction

* `insert_neo4j(...)`
  → Writes graph data into Neo4j

* `clear_db(...)`
  → Clears all existing nodes and relationships

---

## ⚠️ Notes

* **Do NOT import this module in inference code**
* Requires both:

  * Neo4j database running
  * Ollama server running (`ollama serve`)
* Graph construction can take time depending on dataset size

---

## 🧪 Example Run

```bash
python your_script.py
```

This will:

1. Clear the database
2. Load the dataset
3. Build the knowledge graph

---

## 🧼 Recommended Workflow

* Run this module only when the graph needs to be updated
* Keep it separate from inference workflows
* Treat this as a **build step**, not part of runtime

---

## 📌 Summary

This module is a **graph builder**, not part of the inference pipeline.

It is separated to:

* Keep the main project lightweight
* Avoid unnecessary setup for inference
* Isolate heavier components and dependencies

---
