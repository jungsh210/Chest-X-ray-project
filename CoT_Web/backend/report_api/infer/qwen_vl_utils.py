from typing import Any, Dict, List, Tuple

def process_vision_info(messages: List[Dict[str, Any]]) -> Tuple[List[Any], List[Any]]:
    images = []
    videos = None
    for m in messages:
        for c in m.get("content", []):
            if c.get("type") == "image":
                images.append(c.get("image"))
    return images, videos
