import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Netflix Content Analytics",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_netflix.csv")
    return df


df = load_data()


# ============================================================
# DATA PREPARATION
# ============================================================

# Make sure important columns exist
for column in [
    "title",
    "type",
    "director",
    "cast",
    "country",
    "rating",
    "duration",
    "listed_in",
    "description"
]:
    if column in df.columns:
        df[column] = df[column].fillna("")


# Convert release year if possible
if "release_year" in df.columns:
    df["release_year"] = pd.to_numeric(
        df["release_year"],
        errors="coerce"
    )


# ============================================================
# HEADER
# ============================================================

st.title("🎬 Netflix Content Analytics and Recommendation System")

st.markdown(
    """
    **Exploratory Data Analysis • Interactive Analytics •
    Content-Based Recommendation**
    
    Explore Netflix content and generate recommendations using
    TF-IDF and cosine similarity.
    """
)

st.divider()


# ============================================================
# KPI SECTION
# ============================================================

total_titles = len(df)

movies = len(
    df[df["type"].astype(str).str.strip().str.lower() == "movie"]
)

tv_shows = len(
    df[df["type"].astype(str).str.strip().str.lower() == "tv show"]
)

movie_percentage = (
    (movies / total_titles) * 100
    if total_titles > 0 else 0
)

tv_percentage = (
    (tv_shows / total_titles) * 100
    if total_titles > 0 else 0
)


st.subheader("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Titles",
        f"{total_titles:,}"
    )

with col2:
    st.metric(
        "Movies",
        f"{movies:,}",
        f"{movie_percentage:.1f}%"
    )

with col3:
    st.metric(
        "TV Shows",
        f"{tv_shows:,}",
        f"{tv_percentage:.1f}%"
    )

with col4:
    if "release_year" in df.columns:
        avg_year = df["release_year"].mean()
        if pd.notna(avg_year):
            st.metric(
                "Average Release Year",
                f"{avg_year:.0f}"
            )
        else:
            st.metric("Average Release Year", "N/A")
    else:
        st.metric("Average Release Year", "N/A")


st.divider()


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("🔎 Content Filters")

filtered_df = df.copy()


# Type filter
if "type" in df.columns:
    type_options = ["All"] + sorted(
        df["type"].dropna().astype(str).unique().tolist()
    )

    selected_type = st.sidebar.selectbox(
        "Content Type",
        type_options
    )

    if selected_type != "All":
        filtered_df = filtered_df[
            filtered_df["type"] == selected_type
        ]


# Rating filter
if "rating" in df.columns:
    rating_options = ["All"] + sorted(
        [
            str(x)
            for x in df["rating"].dropna().unique()
            if str(x).strip() != ""
        ]
    )

    selected_rating = st.sidebar.selectbox(
        "Rating",
        rating_options
    )

    if selected_rating != "All":
        filtered_df = filtered_df[
            filtered_df["rating"].astype(str) == selected_rating
        ]


# Release year filter
if "release_year" in df.columns:

    valid_years = df["release_year"].dropna()

    if not valid_years.empty:

        min_year = int(valid_years.min())
        max_year = int(valid_years.max())

        selected_years = st.sidebar.slider(
            "Release Year",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )

        filtered_df = filtered_df[
            filtered_df["release_year"].between(
                selected_years[0],
                selected_years[1]
            )
        ]


st.sidebar.markdown("---")

st.sidebar.write(
    f"**Filtered titles:** {len(filtered_df):,}"
)


# ============================================================
# CONTENT TYPE CHART
# ============================================================

st.subheader("🎞️ Movies vs TV Shows")

type_counts = (
    filtered_df["type"]
    .value_counts()
    .reset_index()
)

type_counts.columns = ["type", "count"]

st.bar_chart(
    type_counts.set_index("type")
)


# ============================================================
# TWO-COLUMN ANALYTICS
# ============================================================

left_col, right_col = st.columns(2)


# ------------------------------------------------------------
# RATINGS
# ------------------------------------------------------------

with left_col:

    st.subheader("⭐ Content Ratings")

    rating_counts = (
        filtered_df["rating"]
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .head(10)
    )

    if not rating_counts.empty:
        st.bar_chart(rating_counts)
    else:
        st.info("No rating data available.")


# ------------------------------------------------------------
# RELEASE YEAR
# ------------------------------------------------------------

with right_col:

    st.subheader("📅 Releases by Year")

    if "release_year" in filtered_df.columns:

        year_counts = (
            filtered_df["release_year"]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
        )

        if not year_counts.empty:
            st.line_chart(year_counts)
        else:
            st.info("No release-year data available.")


st.divider()


# ============================================================
# TOP GENRES
# ============================================================

st.subheader("🎭 Top Genres")

genre_series = (
    filtered_df["listed_in"]
    .replace("", pd.NA)
    .dropna()
    .astype(str)
    .str.split(",")
    .explode()
    .str.strip()
)

genre_counts = genre_series.value_counts().head(10)

if not genre_counts.empty:
    st.bar_chart(genre_counts)
else:
    st.info("No genre information available.")


st.divider()


# ============================================================
# CONTENT EXPLORER
# ============================================================

st.subheader("🔎 Netflix Content Explorer")

st.markdown(
    "Use the table below to explore individual Netflix titles."
)

search_text = st.text_input(
    "Search by title",
    placeholder="Enter a Netflix title..."
)

explorer_df = filtered_df.copy()

if search_text.strip():

    explorer_df = explorer_df[
        explorer_df["title"]
        .astype(str)
        .str.contains(
            search_text,
            case=False,
            na=False
        )
    ]


explorer_columns = [
    "title",
    "type",
    "release_year",
    "rating",
    "duration",
    "country",
    "listed_in",
    "director"
]

available_explorer_columns = [
    column
    for column in explorer_columns
    if column in explorer_df.columns
]


st.dataframe(
    explorer_df[available_explorer_columns],
    use_container_width=True,
    hide_index=True
)


st.divider()


# ============================================================
# RECOMMENDATION SYSTEM
# ============================================================

st.header("🤖 Netflix Recommendation System")

st.markdown(
    """
    Select a Netflix title and generate similar titles using a
    **content-based recommendation system**.

    The model uses metadata such as title, director, cast,
    country, genre and description.
    """
)


# ============================================================
# CREATE RECOMMENDATION TEXT
# ============================================================

@st.cache_resource
def build_recommendation_model(data):

    recommendation_text = (
        data["title"].fillna("").astype(str) + " " +
        data["director"].fillna("").astype(str) + " " +
        data["cast"].fillna("").astype(str) + " " +
        data["country"].fillna("").astype(str) + " " +
        data["listed_in"].fillna("").astype(str) + " " +
        data["description"].fillna("").astype(str)
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50000
    )

    tfidf_matrix = vectorizer.fit_transform(
        recommendation_text
    )

    similarity_matrix = cosine_similarity(
        tfidf_matrix
    )

    title_indices = pd.Series(
        data.index,
        index=data["title"].str.lower()
    ).drop_duplicates()

    return (
        vectorizer,
        similarity_matrix,
        title_indices
    )


vectorizer, cosine_sim, indices = build_recommendation_model(df)


# ============================================================
# RECOMMENDATION CONTROLS
# ============================================================

available_titles = sorted(
    df["title"]
    .dropna()
    .astype(str)
    .unique()
)


col1, col2 = st.columns([3, 1])

with col1:

    selected_title = st.selectbox(
        "Select a Netflix title",
        available_titles
    )

with col2:

    number_of_recommendations = st.slider(
        "Recommendations",
        min_value=5,
        max_value=15,
        value=10
    )


recommend_button = st.button(
    "🎯 Recommend Similar Titles",
    type="primary"
)


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

if recommend_button:

    title_key = selected_title.lower()

    if title_key not in indices:

        st.error(
            "The selected title could not be found."
        )

    else:

        selected_index = indices[title_key]

        similarity_scores = list(
            enumerate(
                cosine_sim[selected_index]
            )
        )

        similarity_scores = sorted(
            similarity_scores,
            key=lambda x: x[1],
            reverse=True
        )

        # Remove the selected title itself
        similarity_scores = similarity_scores[1:]

        # Select top N
        similarity_scores = similarity_scores[
            :number_of_recommendations
        ]

        recommended_indices = [
            item[0]
            for item in similarity_scores
        ]

        result = df.iloc[
            recommended_indices
        ].copy()

        result["similarity_score"] = [
            round(item[1], 3)
            for item in similarity_scores
        ]


        # ----------------------------------------------------
        # SELECT RESULT COLUMNS
        # ----------------------------------------------------

        recommendation_columns = [
            "title",
            "type",
            "release_year",
            "rating",
            "duration",
            "listed_in",
            "country",
            "director",
            "description",
            "similarity_score"
        ]

        available_recommendation_columns = [
            column
            for column in recommendation_columns
            if column in result.columns
        ]


        st.subheader(
            f"🎯 Recommendations for: {selected_title}"
        )

        st.dataframe(
            result[
                available_recommendation_columns
            ],
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # SIMILARITY SCORE
        # ----------------------------------------------------

        st.subheader("📈 Similarity Scores")

        score_chart = result[
            ["title", "similarity_score"]
        ].set_index("title")

        st.bar_chart(
            score_chart
        )


        st.info(
            "Recommendations are generated using content "
            "similarity based on Netflix metadata and "
            "descriptions. Higher similarity scores indicate "
            "greater content similarity."
        )


# ============================================================
# ABOUT PROJECT
# ============================================================

st.divider()

with st.expander("ℹ️ About this Project"):

    st.markdown(
        """
        ### Netflix Content Analytics and Recommendation System

        This project combines exploratory data analysis,
        interactive business intelligence and a content-based
        recommendation system.

        **Technologies used:**

        - Python
        - Pandas
        - Scikit-learn
        - TF-IDF
        - Cosine Similarity
        - Streamlit
        - Power BI
        - JupyterLab

        **Recommendation methodology:**

        Netflix metadata and descriptions are converted into
        TF-IDF vectors. Cosine similarity is then used to
        identify titles with similar content characteristics.

        **Limitation:**

        The Netflix Titles dataset does not contain individual
        user ratings, viewing history or user preferences.
        Therefore, this project uses content-based rather than
        collaborative filtering.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Netflix Content Analytics and Recommendation System | "
    "Internship Mini Project"
)