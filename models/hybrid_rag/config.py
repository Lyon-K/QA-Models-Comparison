from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# Output directories
OUTPUT_DIR = BASE_DIR / "outputs"
INDEX_DIR = OUTPUT_DIR / "indexes"
RETRIEVAL_RESULTS_DIR = OUTPUT_DIR / "retrieval_results"

# Input dataset file
DATA_FILE = RAW_DATA_DIR / "test_dataset.csv"

# Embedding model name
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking configuration
CHUNK_SIZE_WORDS = 120
CHUNK_OVERLAP_WORDS = 30

# Retrieval configuration
SPARSE_TOP_K = 10
DENSE_TOP_K = 10
FINAL_TOP_K = 5

# Fusion configuration
RRF_K = 60

# Deduplication configuration
DEDUP_SIMILARITY_THRESHOLD = 0.90
MAX_CHUNKS_PER_DOCUMENT = 2