"""Dictionary-based skill extraction with regex word boundaries."""
import re
from typing import Dict, List

SKILLS: Dict[str, List[str]] = {
    "programming_languages": [
        "python", "c++", "c#", "java", "javascript", "typescript", "c", "go", "ruby", "php"
    ],
    "web_technologies": [
        "html", "css", "node.js", "react.js", "next.js", "vue.js", "express.js",
        "fastapi", "django", "flask", "rest api", "restful api"
    ],
    "databases": ["sql", "mysql", "postgresql", "mongodb", "sqlite", "redis", "nosql"],
    "cloud_and_devops": [
        "aws", "azure", "google cloud", "amazon s3", "aws lambda", "docker",
        "kubernetes", "jenkins", "github actions", "ci/cd", "linux"
    ],
    "data_science_and_ml": [
        "machine learning", "deep learning", "pandas", "numpy", "scikit-learn",
        "tensorflow", "pytorch", "opencv", "natural language processing", "nlp"
    ],
    "tools_and_concepts": [
        "git", "github", "jira", "agile", "object oriented programming", "oop",
        "data structures", "algorithms", "unit testing", "pytest", "ui/ux", "socket.io"
    ],
}


def _contains_skill(text: str, skill: str) -> bool:
    # re.escape ensures symbols such as +, #, and . are treated literally.
    if skill.lower() == "c":
        # Do not classify the C inside C++ or C# as the standalone C language.
        pattern = r"(?<![\w+#])c(?![\w+#])"
    else:
        pattern = r"(?<!\w)" + re.escape(skill.lower()) + r"(?!\w)"
    return re.search(pattern, text.lower()) is not None


def extract_skills(text: str) -> Dict[str, List[str]]:
    """Return only categories that contain at least one detected skill."""
    found: Dict[str, List[str]] = {}
    for category, skill_list in SKILLS.items():
        matches = [skill for skill in skill_list if _contains_skill(text, skill)]
        if matches:
            found[category] = matches
    return found


def flatten_skills(skills_by_category: Dict[str, List[str]]) -> List[str]:
    """Flatten categorized skills into a sorted, duplicate-free list."""
    return sorted({skill for values in skills_by_category.values() for skill in values})
