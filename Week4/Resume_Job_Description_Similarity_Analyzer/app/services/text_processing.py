"""Text normalization helpers used by the similarity pipeline."""
import re

# Canonical tokens keep meaningful technical punctuation from disappearing
# during TF-IDF preprocessing. Skill extraction still uses the original text.
TECH_TERM_ALIASES = {
    "c++": "cplusplus",
    "c#": "csharp",
    ".net": "dotnet",
    "asp.net": "aspdotnet",
    "node.js": "nodejs",
    "react.js": "reactjs",
    "next.js": "nextjs",
    "vue.js": "vuejs",
    "express.js": "expressjs",
    "three.js": "threejs",
    "f#": "fsharp",
    "objective-c": "objectivec",
    "ci/cd": "cicd",
    "ui/ux": "uiux",
    "socket.io": "socketio",
}


def normalize_for_similarity(text: str) -> str:
    """Lowercase text and preserve selected technical terms as single tokens."""
    normalized = text.lower()

    # Longer terms first prevents a shorter term from consuming part of a longer one.
    for term, canonical in sorted(
        TECH_TERM_ALIASES.items(), key=lambda item: len(item[0]), reverse=True
    ):
        pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
        normalized = re.sub(pattern, f" {canonical} ", normalized)

    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized
