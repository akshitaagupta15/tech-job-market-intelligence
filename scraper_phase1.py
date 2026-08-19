"""
Phase 1: Internshala Internship Scraper
-----------------------------------------
Scrapes internship title, company, location, stipend, and required skills
for a given role category (e.g. 'data-analyst', 'web-development') from
Internshala, and saves the results to a CSV file.

Author: <your name> | Tech Job Market Intelligence Project
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------
# Internshala uses predictable URL slugs per category, e.g.:
#   https://internshala.com/internships/data-analyst-internship
#   https://internshala.com/internships/web-development-internship
BASE_URL = "https://internshala.com/internships/{category}-internship/page-{page}"

HEADERS = {
    # Pretend to be a normal browser so we don't get blocked outright.
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# How many listing pages to scrape per category (start small while testing!)
PAGES_PER_CATEGORY = 2

# Be a polite scraper — random delay between requests so we don't hammer
# the server (and don't look like a bot flooding requests).
MIN_DELAY = 1.5
MAX_DELAY = 3.0


def get_soup(url: str) -> BeautifulSoup | None:
    """
    Fetch a URL and return a parsed BeautifulSoup object.
    Returns None if the request fails, so callers can skip gracefully
    instead of crashing the whole script on one bad page.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()  # raises an error for 4xx/5xx status codes
        return BeautifulSoup(response.text, "html.parser")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to fetch {url}: {e}")
        return None


def parse_listing_page(soup: BeautifulSoup) -> list[dict]:
    """
    Extract basic internship info (title, company, location, stipend, link)
    from a single search-results page.

    NOTE FOR DEBUGGING:
    If this returns an empty list, the first thing to check is whether
    the class name below ('internship_meta' / 'container-fluid individual_internship')
    still matches what you see in your browser's DevTools. Site markup
    changes over time — update the selector here, not the logic.
    """
    internships = []

    # Each internship listing is wrapped in a container div.
    cards = soup.find_all("div", class_="individual_internship")

    for card in cards:
        try:
            title_tag = card.find("h2", class_="job-internship-name")
            company_tag = card.find("p", class_="company-name")
            location_tag = card.find("a", class_="location_link")
            stipend_tag = card.find("span", class_="stipend")
            link_tag = title_tag.find("a") if title_tag else None

            internships.append({
                "title": title_tag.get_text(strip=True) if title_tag else None,
                "company": company_tag.get_text(strip=True) if company_tag else None,
                "location": location_tag.get_text(strip=True) if location_tag else None,
                "stipend": stipend_tag.get_text(strip=True) if stipend_tag else None,
                "detail_url": (
                    "https://internshala.com" + link_tag["href"]
                    if link_tag and link_tag.has_attr("href")
                    else None
                ),
            })
        except AttributeError as e:
            # One malformed card shouldn't kill the whole scrape.
            print(f"[WARN] Skipped a card due to parsing issue: {e}")
            continue

    return internships


def get_skills_for_internship(detail_url: str) -> list[str]:
    """
    Visit an individual internship's detail page and extract the
    'Skills(s) required' tag list.
    """
    if not detail_url:
        return []

    soup = get_soup(detail_url)
    if soup is None:
        return []

    # Skills are usually rendered as a list of <span class="round_tabs">
    # inside a section labelled "Skill(s) required".
    skill_tags = soup.find_all("span", class_="round_tabs")
    skills = [tag.get_text(strip=True) for tag in skill_tags]

    return skills


def scrape_category(category: str, role_label: str) -> list[dict]:
    """
    Scrape multiple pages for one category (e.g. 'data-analyst')
    and enrich each listing with its skills from the detail page.
    """
    all_internships = []

    for page in range(1, PAGES_PER_CATEGORY + 1):
        url = BASE_URL.format(category=category, page=page)
        print(f"[INFO] Fetching listing page: {url}")

        soup = get_soup(url)
        if soup is None:
            continue

        listings = parse_listing_page(soup)
        if not listings:
            print(f"[INFO] No listings found on page {page} — stopping pagination for '{category}'.")
            break

        for internship in listings:
            internship["role_category"] = role_label
            print(f"  -> Fetching skills for: {internship['title']}")
            internship["skills"] = ", ".join(get_skills_for_internship(internship["detail_url"]))

            # Politeness delay between detail-page hits
            time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

        all_internships.extend(listings)

        # Politeness delay between listing pages too
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

    return all_internships


def main():
    # Map: (Internshala URL slug) -> (human-readable label we store in CSV)
    categories = {
        "data-analyst": "Data Analyst",
        "web-development": "Web Developer",
    }

    combined_data = []
    for slug, label in categories.items():
        print(f"\n=== Scraping category: {label} ===")
        combined_data.extend(scrape_category(slug, label))

    df = pd.DataFrame(combined_data)
    output_file = "internshala_internships_raw.csv"
    df.to_csv(output_file, index=False)
    print(f"\n[DONE] Saved {len(df)} records to {output_file}")


if __name__ == "__main__":
    main()
