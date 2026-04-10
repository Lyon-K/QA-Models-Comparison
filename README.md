# **Q&A Retrieval Benchmarking**

This repo benchmarks various **Q&A retrieval models**, comparing **Word2Vec-based** methods with **RAG-based** approaches.

---

## **Work Distribution**
- {name} - **Dataset**: Set of trivia and multi-hop Q&A tasks, enabling the evaluation of various retrieval models across different question complexities
- {name} - **Website**: Interactive web app to display benchmarks, query the models and view responses.
- {name} - **Word2Vec**: Embedding-based model that retrieves answers by measuring word similarity, serving as a baseline for comparison with other retrieval methods.
- {name} - **Vanilla RAG**: Standard retrieval-augmented generation approach.
- Lyon - **GraphRAG**: Graph-based Q&A retrieval using entity relationships and multi-hop reasoning.
- {name} - **[Undecided RAG]**: An additional RAG method for comparison.
- {name} - **Evaluation**: Benchmarks model performance using accuracy, precision, recall, and response time across Q&A tasks to assess retrieval effectiveness

---

## **Dataset**
- **Natural Questions, HotpotQA, and TriviaQA**: A collection of trivia and multi-hop Q&A tasks, including factual questions (Natural Questions), multi-hop reasoning tasks (HotpotQA), and complex trivia questions (TriviaQA).
- **Format**: Format: JSON with the following structure:
```JSON
{
  "Dataset": "dataset_name",
  "Question": "question_text",
  "Context": "context_information",
  "True_Answer": "correct_answer"
}
```

## **Installation**
1. **Clone the repo:**
   ```bash
   git clone https://github.com/Lyon-K/QA-Models-Comparison
   cd QA-Models-Comparison
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment:**
   
   Windows:
   ```bash
   .\.venv\Scripts\activate
   ```
   macOS/Linux:
   ```bash
   source .venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. [**Install Neo4j**](https://neo4j.com/docs/desktop/current/installation/)
   
   Host an instance and change the .env file

6. **Enable Neo4j GDS (Graph Data Science):**
   
   Ensure the Neo4j Graph Data Science (GDS) plugin is installed and enabled. (*This is required for operations such as **cosine similarity** used during retrieval.*)
