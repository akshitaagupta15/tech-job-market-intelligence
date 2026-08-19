"""
Phase 2: Data Cleaning + NLP Skill Extraction + Frequency Mapping
-------------------------------------------------------------------
Input : internshala_internships_raw.csv (from Phase 1)
Output: skill_frequency.csv  -> Skill vs Frequency count across all postings
        internships_with_skills.csv -> original data + extracted skills column

Steps:
  1. Clean raw job description / skills text (strip HTML, special chars)
  2. Extract tech skills using a curated dictionary + spaCy PhraseMatcher
  3. Count skill frequency across the whole dataset
"""

import re
import pandas as pd
from bs4 import BeautifulSoup
from collections import Counter
import spacy
from spacy.matcher import PhraseMatcher

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------
INPUT_CSV = "internshala_internships_raw.csv"

# IMPORTANT: change this to whatever column in YOUR csv holds the free-text
# job description / skills requirement string.
TEXT_COLUMN = "skills"   # e.g. could be "Job Description" in your version

# -----------------------------------------------------------------------
# STEP 1: TEXT CLEANING
# -----------------------------------------------------------------------
def clean_text(raw_text: str) -> str:
    """
    Clean a single raw text field:
      - Strip any HTML tags (job descriptions scraped from web pages
        often carry <li>, <p>, <br> etc.)
      - Remove characters that aren't useful for skill matching, but
        KEEP characters like '+', '#', '.' because they matter for
        skill names such as 'C++', 'C#', 'Node.js'
      - Lowercase everything, since we'll match case-insensitively anyway
    """
    if not isinstance(raw_text, str) or raw_text.strip() == "":
        return ""

    # Remove HTML tags using BeautifulSoup (safer than regex for HTML)
    text = BeautifulSoup(raw_text, "html.parser").get_text(separator=" ")

    # Remove anything that isn't a letter, digit, space, '+', '#', or '.'
    text = re.sub(r"[^a-zA-Z0-9\+\#\.\s]", " ", text)

    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text).strip()

    return text.lower()


# -----------------------------------------------------------------------
# STEP 2: SKILL DICTIONARY
# -----------------------------------------------------------------------
# A curated, extensible list of tech skills relevant to Data Analyst
# and Web Developer roles. In a real project, this list grows over time
# as you discover more skills appearing in postings (Phase 2.5 idea:
# log unmatched frequent words to spot skills you're missing).
TECH_SKILLS = [
    # --- Data Analyst skills ---
    "python", "sql", "excel", "power bi", "tableau", "r programming",
    "machine learning", "statistics", "data visualization", "pandas",
    "numpy", "google sheets", "data cleaning", "mysql", "postgresql",
    "big query", "looker", "data analysis", "predictive modeling",

    # --- Web Developer skills ---
    "html", "css", "javascript", "react", "react.js", "node.js", "angular",
    "vue.js", "django", "flask", "mongodb", "express.js", "rest api",
    "typescript", "bootstrap", "tailwind css", "php", "wordpress",
    "git", "github", "jquery", "next.js", "graphql",

    # --- Common cross-cutting tools ---
    "java", "c++", "c#", "aws", "docker", "linux", "api", "agile",
    "communication skills", "problem solving",
]


def build_matcher(nlp) -> PhraseMatcher:
    """
    Build a spaCy PhraseMatcher pre-loaded with our skill dictionary.
    attr="LOWER" tells spaCy to match case-insensitively (so "Python",
    "PYTHON", and "python" in the text all match our "python" pattern).
    """
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    # Convert each skill string into a spaCy Doc pattern
    patterns = [nlp.make_doc(skill) for skill in TECH_SKILLS]
    matcher.add("TECH_SKILLS", patterns)

    return matcher


def extract_skills(text: str, nlp, matcher: PhraseMatcher) -> list[str]:
    """
    Run the PhraseMatcher over a cleaned text string and return the
    unique list of skills found (in their canonical dictionary form).
    """
    if not text:
        return []

    doc = nlp(text)
    matches = matcher(doc)

    found_skills = set()
    for match_id, start, end in matches:
        span = doc[start:end]
        found_skills.add(span.text.lower())

    return sorted(found_skills)


# -----------------------------------------------------------------------
# STEP 3: MAIN PIPELINE
# -----------------------------------------------------------------------
def main():
    # Load a lightweight English model (no need for the larger models —
    # we're only using tokenization here, not spaCy's own NER/parser)
    print("[INFO] Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    matcher = build_matcher(nlp)

    print(f"[INFO] Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    if TEXT_COLUMN not in df.columns:
        raise ValueError(
            f"Column '{TEXT_COLUMN}' not found. "
            f"Available columns: {list(df.columns)} "
            f"-> update TEXT_COLUMN at the top of this script."
        )

    print("[INFO] Cleaning text column...")
    df["cleaned_text"] = df[TEXT_COLUMN].apply(clean_text)

    print("[INFO] Extracting skills (this may take a moment)...")
    df["extracted_skills"] = df["cleaned_text"].apply(
        lambda text: extract_skills(text, nlp, matcher)
    )

    # Save enriched dataframe (original data + skills list per row)
    df.to_csv("internships_with_skills.csv", index=False)
    print("[DONE] Saved internships_with_skills.csv")

    # ---------------------------------------------------------------
    # SKILL FREQUENCY MAPPING
    # ---------------------------------------------------------------
    # Flatten the list-of-lists into one long list of individual skills,
    # then count occurrences using Counter (a dict subclass built for
    # exactly this: counting hashable items).
    all_skills = [skill for skill_list in df["extracted_skills"] for skill in skill_list]
    skill_counts = Counter(all_skills)

    skill_freq_df = (
        pd.DataFrame(skill_counts.items(), columns=["Skill", "Frequency"])
        .sort_values(by="Frequency", ascending=False)
        .reset_index(drop=True)
    )

    skill_freq_df.to_csv("skill_frequency.csv", index=False)
    print("[DONE] Saved skill_frequency.csv")

    print("\nTop 10 most in-demand skills in your dataset:")
    print(skill_freq_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
