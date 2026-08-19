"""
Phase 3: Streamlit Dashboard - Tech Job Market Intelligence
--------------------------------------------------------------
Reads skill_frequency.csv (from Phase 2) and renders an interactive
dashboard with KPI cards and a horizontal bar chart of in-demand skills.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------------------------
# PAGE CONFIG
# Must be the very first Streamlit command in the script.
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Tech Job Market Intelligence",
    page_icon="📊",
    layout="wide",  # use full browser width instead of a narrow centered column
)


# -----------------------------------------------------------------------
# DATA LOADING (cached so we don't re-read the CSV on every interaction)
# -----------------------------------------------------------------------
@st.cache_data
def load_skill_data(csv_path: str) -> pd.DataFrame:
    """
    Load the skill frequency CSV produced in Phase 2.
    Returns an empty, correctly-shaped DataFrame if the file is missing,
    so the rest of the app can render a friendly message instead of crashing.
    """
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=["Skill", "Frequency"])


# -----------------------------------------------------------------------
# UI SECTIONS (each function renders one piece of the page —
# keeping these separate makes it easy to reorder or reuse sections later)
# -----------------------------------------------------------------------
def render_header():
    st.title("📊 Tech Job Market Intelligence")
    st.caption("Skill demand insights extracted from live internship postings")
    st.divider()


def render_sidebar(df: pd.DataFrame) -> int:
    """
    Sidebar controls. Currently just a 'Top N skills to display' slider.
    Returns the selected value so the main page can use it.
    Add more filters here later (e.g. role category) as your dataset grows.
    """
    st.sidebar.header("⚙️ Dashboard Controls")

    top_n = st.sidebar.slider(
        "Number of top skills to display",
        min_value=5,
        max_value=min(30, max(len(df), 5)),
        value=min(10, max(len(df), 5)),
    )

    st.sidebar.divider()
    st.sidebar.markdown(
        "**About this project**\n\n"
        "Data is scraped from live internship postings and processed "
        "through an NLP skill-extraction pipeline (spaCy PhraseMatcher)."
    )

    return top_n


def render_kpis(df: pd.DataFrame):
    """
    Top-row KPI cards using st.columns() + st.metric().
    st.columns(3) returns 3 placeholder containers we can 'with' into,
    laying widgets out side-by-side instead of stacked vertically.
    """
    if df.empty:
        st.warning("No data found. Make sure skill_frequency.csv is in this folder.")
        return

    top_skill_row = df.loc[df["Frequency"].idxmax()]
    total_skills_tracked = df["Skill"].nunique()
    total_mentions = int(df["Frequency"].sum())

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="🏆 Most In-Demand Skill",
            value=top_skill_row["Skill"].title(),
            delta=f"{int(top_skill_row['Frequency'])} mentions",
        )

    with col2:
        st.metric(
            label="🧠 Unique Skills Tracked",
            value=total_skills_tracked,
        )

    with col3:
        st.metric(
            label="📄 Total Skill Mentions",
            value=total_mentions,
        )

    st.divider()


def render_skill_chart(df: pd.DataFrame, top_n: int):
    """
    Horizontal bar chart of the top N skills by frequency, using Plotly
    for interactivity (hover tooltips, zoom, etc.).
    """
    if df.empty:
        return

    st.subheader(f"Top {top_n} Skills by Demand")

    # Sort descending, then take the top N rows for a focused chart
    chart_df = df.sort_values(by="Frequency", ascending=False).head(top_n)

    fig = px.bar(
        chart_df.sort_values(by="Frequency", ascending=True),  # ascending so highest bar lands on top visually
        x="Frequency",
        y="Skill",
        orientation="h",
        text="Frequency",
        color="Frequency",
        color_continuous_scale="Blues",
    )

    fig.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        xaxis_title="Number of Postings Mentioning Skill",
        yaxis_title="",
        height=500,
        margin=dict(l=10, r=10, t=10, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)


def calculate_investment_score(frequency: int, max_frequency: int) -> int:
    """
    Normalize a skill's raw frequency into a 0-100 'Investment Score',
    scaled against the single most in-demand skill in the dataset.
    Makes the number intuitive without needing to know the raw scale.
    """
    if max_frequency == 0:
        return 0
    return round((frequency / max_frequency) * 100)


def render_recommendation_engine(df: pd.DataFrame, top_n_recommendations: int = 3):
    """
    Lets the user pick their current skills, then recommends the
    highest-demand skills missing from that set, using set difference.
    """
    st.divider()
    st.subheader("🎯 What Should You Learn Next?")
    st.caption(
        "Select the skills you already know — we'll find the highest-demand "
        "gaps in your skillset based on live market data."
    )

    if df.empty:
        st.warning("No skill data available yet.")
        return

    # Title-case for display, but we'll compare in lowercase to avoid
    # case-mismatch bugs (e.g. "Python" vs "python" being treated as different)
    all_skills_display = sorted(df["Skill"].str.title().unique().tolist())

    user_selection = st.multiselect(
        "✅ Select your current skills:",
        options=all_skills_display,
        placeholder="e.g. HTML, CSS, Python...",
    )

    if not user_selection:
        st.info("👆 Select a few skills above to unlock your personalized recommendations.")
        return

    # --- Core set logic ---
    user_skills_set = {s.lower() for s in user_selection}
    market_skills_set = {s.lower() for s in all_skills_display}
    missing_skills = market_skills_set - user_skills_set   # set difference

    if not missing_skills:
        st.success("🎉 You already cover every skill tracked in this market — nice work!")
        return

    # Filter the frequency table down to only the missing skills,
    # then rank by demand and take the top N.
    recommend_df = (
        df[df["Skill"].str.lower().isin(missing_skills)]
        .sort_values(by="Frequency", ascending=False)
        .head(top_n_recommendations)
    )

    max_frequency = df["Frequency"].max()
    medals = ["🥇", "🥈", "🥉"]

    st.markdown(f"### 🚀 Your Top {len(recommend_df)} Recommended Skills")
    cols = st.columns(len(recommend_df))

    for i, (_, row) in enumerate(recommend_df.iterrows()):
        score = calculate_investment_score(row["Frequency"], max_frequency)
        with cols[i]:
            badge = medals[i] if i < len(medals) else "⭐"
            st.success(f"{badge} **{row['Skill'].title()}**")
            st.write(f"Mentioned in **{int(row['Frequency'])}** postings")
            st.progress(score / 100, text=f"Investment Score: {score}/100")


def simulate_historical_data(skill: str, current_frequency: int, months: int = 6, seed: int = None) -> pd.DataFrame:
    """
    Generate a mock 6-month historical trend that plausibly LEADS UP TO
    the skill's current real frequency (treated as the most recent month).

    IMPORTANT: This is simulated data for prototyping the ML pipeline
    only. Each skill gets a randomly-assigned growth rate + noise so
    trends look distinct and realistic rather than identical curves.

    `seed` is passed per-skill so results are reproducible on re-runs
    (same skill always generates the same mock history) rather than
    reshuffling randomly every time the Streamlit app re-executes.
    """
    rng = np.random.default_rng(seed)

    # Assume each skill has been growing at somewhere between 5%-18%
    # month-over-month on average, landing at its current real frequency.
    monthly_growth_rate = rng.uniform(0.05, 0.18)

    month_numbers = np.arange(1, months + 1)  # e.g. [1, 2, 3, 4, 5, 6]

    # Work BACKWARDS from current_frequency (assumed = month 6) using
    # the growth rate, so earlier months are smaller.
    base_values = current_frequency / ((1 + monthly_growth_rate) ** (months - month_numbers))

    # Add random noise so it's not a perfectly smooth exponential curve
    # (real-world data is never this clean).
    noise = rng.normal(loc=0, scale=max(current_frequency * 0.07, 1), size=months)

    values = np.clip(base_values + noise, a_min=1, a_max=None).round().astype(int)

    return pd.DataFrame({
        "Skill": skill,
        "Month": month_numbers,
        "Frequency": values,
    })


def train_and_forecast(skill_history_df: pd.DataFrame, future_months: list[int] = [7, 8]) -> pd.DataFrame:
    """
    Fit a Linear Regression model on one skill's historical (Month, Frequency)
    data and predict frequency for future months.

    X shape: (n_months, 1)  -> each row is one month, 1 feature (the month number)
    y shape: (n_months,)    -> each row is that month's posting frequency

    Returns a small DataFrame with the forecasted months + predicted values.
    """
    X = skill_history_df[["Month"]].values   # scikit-learn requires 2D input for X
    y = skill_history_df["Frequency"].values  # 1D target array

    model = LinearRegression()
    model.fit(X, y)  # finds the best-fit line: frequency = m * month + b

    X_future = np.array(future_months).reshape(-1, 1)  # reshape to (n, 1) to match X's shape
    predictions = model.predict(X_future)
    predictions = np.clip(predictions, a_min=0, a_max=None)  # frequency can't be negative

    return pd.DataFrame({
        "Month": future_months,
        "Frequency": predictions.round().astype(int),
    })


def render_forecast_chart(history_df: pd.DataFrame, forecast_df: pd.DataFrame, skill_name: str):
    """
    Plot historical (solid line) + forecasted (dashed line) frequency
    for one skill using Plotly, with a clear visual + label distinction
    between real trend and predicted trend.
    """
    fig = go.Figure()

    # Historical trend - solid line
    fig.add_trace(go.Scatter(
        x=history_df["Month"], y=history_df["Frequency"],
        mode="lines+markers", name="Historical (simulated)",
        line=dict(color="#1f77b4", width=3),
    ))

    # Bridge point: connect last historical point to first forecast point
    # so the dashed line visually continues from where history ends.
    bridge_x = [history_df["Month"].iloc[-1]] + forecast_df["Month"].tolist()
    bridge_y = [history_df["Frequency"].iloc[-1]] + forecast_df["Frequency"].tolist()

    fig.add_trace(go.Scatter(
        x=bridge_x, y=bridge_y,
        mode="lines+markers", name="Forecasted (Linear Regression)",
        line=dict(color="#ff7f0e", width=3, dash="dash"),
    ))

    fig.update_layout(
        title=f"{skill_name.title()} — Demand Trend & Forecast",
        xaxis_title="Month",
        yaxis_title="Job Postings Mentioning Skill",
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_forecasting_section(df: pd.DataFrame, top_n: int = 5):
    """
    Main Phase 5 section: simulates history, trains a Linear Regression
    model, and renders a forecast chart for each of the top N skills,
    laid out in tabs so the page stays clean.
    """
    st.divider()
    st.subheader("📈 Skill Demand Forecasting")
    st.info(
        "⚠️ **Note:** Historical monthly data below is **simulated** for demonstration, "
        "since only one real snapshot has been scraped so far. The forecasting "
        "pipeline (Linear Regression) itself is fully functional and will use real "
        "historical data automatically once the scraper has run across multiple months.",
        icon="ℹ️",
    )

    if df.empty:
        st.warning("No skill data available yet.")
        return

    top_skills_df = df.sort_values(by="Frequency", ascending=False).head(top_n)

    tabs = st.tabs([skill.title() for skill in top_skills_df["Skill"]])

    for tab, (_, row) in zip(tabs, top_skills_df.iterrows()):
        with tab:
            # Fixed seed per skill (based on its name) so the simulated
            # history doesn't change every time Streamlit re-runs the script.
            seed = abs(hash(row["Skill"])) % (2**32)

            history_df = simulate_historical_data(row["Skill"], int(row["Frequency"]), seed=seed)
            forecast_df = train_and_forecast(history_df)

            render_forecast_chart(history_df, forecast_df, row["Skill"])

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Current (Month 6) Demand", int(history_df["Frequency"].iloc[-1]))
            with col2:
                growth = forecast_df["Frequency"].iloc[-1] - history_df["Frequency"].iloc[-1]
                st.metric(
                    "Predicted Demand (Month 8)",
                    int(forecast_df["Frequency"].iloc[-1]),
                    delta=int(growth),
                )


def render_raw_data(df: pd.DataFrame):
    """An expandable section to view the underlying data table on demand."""
    with st.expander("🔍 View raw skill frequency data"):
        st.dataframe(
            df.sort_values(by="Frequency", ascending=False),
            use_container_width=True,
            hide_index=True,
        )


# -----------------------------------------------------------------------
# MAIN APP ENTRY POINT
# -----------------------------------------------------------------------
def main():
    df = load_skill_data("skill_frequency.csv")

    render_header()
    top_n = render_sidebar(df)
    render_kpis(df)
    render_skill_chart(df, top_n)
    render_recommendation_engine(df)
    render_forecasting_section(df)
    render_raw_data(df)


if __name__ == "__main__":
    main()
