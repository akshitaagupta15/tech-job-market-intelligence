# 📊 Tech Job Market Intelligence & Skill Forecasting System

> Stop guessing what to learn next. Start investing in skills backed by real market data.

An end-to-end data pipeline and interactive dashboard that scrapes live internship postings, extracts required technical skills using NLP, forecasts future skill demand with machine learning, and generates personalized "what to learn next" recommendations.

**[🚀 Live Demo](#) | [📽️ Video Walkthrough](#)**

---

## 🎯 The Problem: Blind Upskilling

Most students pick what to learn next based on gut feeling, YouTube trends, or what a senior told them two years ago. There's no easy way to answer a simple question: *"Which skill, right now, gives me the best return on my learning time?"*

This project answers that question with data — by scraping real internship postings, quantifying which skills are actually in demand, and forecasting where that demand is heading, rather than where it's already been.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🕷️ **Live Web Scraping** | Automated scraper (`requests` + `BeautifulSoup`) collects real internship postings across Data Analyst and Web Developer roles |
| 🧠 **NLP Skill Extraction** | A spaCy `PhraseMatcher` pipeline parses raw job descriptions and extracts structured technical skills from unstructured text |
| 📈 **Skill Demand Forecasting** | A Linear Regression model (`scikit-learn`) forecasts future demand for top skills, quantifying **skill growth velocity** |
| 🎯 **Personalized Recommendation Engine** | Users select their current skills; the system computes market-skill gaps via set theory and ranks recommendations by real demand |
| 📊 **Interactive Dashboard** | A Streamlit + Plotly dashboard presents KPIs, demand charts, and forecasts in a clean, recruiter-friendly UI |

---

## 🛠️ Tech Stack

**Data Collection:** Python, Requests, BeautifulSoup4
**Data Storage & Processing:** Pandas, SQLite
**NLP:** spaCy (PhraseMatcher-based skill extraction)
**Machine Learning:** scikit-learn (Linear Regression)
**Frontend / Dashboard:** Streamlit, Plotly
**Deployment:** Streamlit Community Cloud, GitHub

---

## 🏗️ System Architecture

```
Internshala Postings
        │
        ▼
 [1] Web Scraper (BeautifulSoup)
        │  raw HTML → structured CSV
        ▼
 [2] NLP Cleaning + Skill Extraction (spaCy PhraseMatcher)
        │  raw text → skill frequency table
        ▼
 [3] Streamlit Dashboard
        ├── KPI Cards & Demand Charts
        ├── [4] Recommendation Engine (set-difference logic)
        └── [5] Forecasting Engine (Linear Regression)
```

---

## 📸 Screenshots

*(Add 2-3 screenshots of your dashboard here — the KPI row, the skill chart, and the recommendation cards make the strongest first impression)*

```
![Dashboard Overview](screenshots/dashboard_overview.png)
![Recommendation Engine](screenshots/recommendations.png)
![Skill Forecast](screenshots/forecast_chart.png)
```

---

## 🚀 Getting Started (Run Locally)

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/tech-job-market-intelligence.git
cd tech-job-market-intelligence

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate     # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the scraper to generate fresh data (optional — sample CSVs included)
python scraper_phase1.py
python phase2_skill_extraction.py

# 5. Launch the dashboard
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## 📁 Project Structure

```
tech-job-market-intelligence/
├── app.py                        # Streamlit dashboard (main entry point)
├── scraper_phase1.py             # Internshala web scraper
├── phase2_skill_extraction.py    # Text cleaning + NLP skill extraction
├── requirements.txt              # Python dependencies
├── skill_frequency.csv           # Extracted skill frequency dataset
├── internships_with_skills.csv   # Enriched internship dataset
└── README.md
```

---

## 🧪 How the ML/NLP Actually Works

- **Skill extraction** uses spaCy's `PhraseMatcher` against a curated tech-skill dictionary rather than generic Named Entity Recognition — because the target vocabulary (tech skills) is closed and known in advance, rule-based phrase matching gives higher precision than training a custom NER model from scratch.
- **Forecasting** fits a Linear Regression model per skill on (Month → Frequency) data, where the model's learned coefficient (slope) is directly interpretable as that skill's **growth velocity** — the core metric this project is built around.
- **Recommendations** are computed via a set difference (`market_skills - user_skills`), then ranked by real posting frequency — a simple, explainable, and fully deterministic algorithm.

---

## 🗺️ Roadmap

- [ ] Replace simulated forecasting history with real time-series data via scheduled scraping (cron / GitHub Actions)
- [ ] Expand scraping to additional job roles and platforms (LinkedIn, Naukri)
- [ ] Migrate skill dictionary to a self-expanding list based on unmatched frequent terms
- [ ] Add user accounts to save skill profiles across sessions

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙋 Author

**[Your Name]**
2nd Year B.Tech CS Student
[LinkedIn](#) • [GitHub](#) • [Email](#)

---

*Built as a portfolio project to demonstrate end-to-end skills in web scraping, NLP, machine learning, and full-stack dashboard development.*
