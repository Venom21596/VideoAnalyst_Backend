from sentence_transformers import SentenceTransformer
from typing import Dict, Any

# Shared state
video_memory: Dict[str, Any] = {}

# Shared model instance
embed_model = SentenceTransformer('all-MiniLM-L6-v2')