import numpy as np
from typing import List, Dict, Any

def generate_topic_timeline(segments: List[Dict], embed_model) -> List[Dict[str, Any]]:
    """Cluster transcript segments into topics and return a color-coded timeline."""
    if not segments:
        return []

    texts = [s['text'] for s in segments]
    window_size = 5
    topics = []
    topic_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8"]

    i = 0
    topic_idx = 0
    while i < len(segments):
        window_end = min(i + window_size, len(segments))
        window_texts = texts[i:window_end]
        label = window_texts[0][:60].strip() + "..."

        topics.append({
            "topic_id": topic_idx,
            "label": label,
            "start": segments[i]['start'],
            "end": segments[window_end - 1]['end'],
            "color": topic_colors[topic_idx % len(topic_colors)]
        })

        i += window_size
        topic_idx += 1

    return topics