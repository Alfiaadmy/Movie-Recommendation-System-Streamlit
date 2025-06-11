import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64
from io import BytesIO


# Load dataset
df = pd.read_csv('dataset-movie.csv')
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['normalized_title'] = df['title'].str.strip().str.lower()

# Gabungkan overview + genre + aktor
df['combined_features'] = df['overview'].fillna('') + ' ' + \
                          df['name_genres'].fillna('') + ' ' + \
                          df['actor_name'].fillna('')

# Inisialisasi session state
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "delete_history" not in st.session_state:
    st.session_state.delete_history = False

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])
vectors = tfidf_matrix
judul_list = df['title'].dropna().tolist()

def fuzzy_match(query):
    result = process.extractOne(query, judul_list, score_cutoff=60)
    return result[0] if result else None

def recommend(movie_title, num_recommendations=10,
              year_range=None, rating_range=None,
              selected_genres=None, selected_actors=None,
              ascending=False):

    movie_title = fuzzy_match(movie_title.strip())
    if not movie_title:
        return pd.DataFrame({'title': ['Movie title not found.']})

    matches = df[df['title'] == movie_title]
    idx = matches.index[0]
    movie_vector = vectors[idx]
    similarity_scores = cosine_similarity(movie_vector, vectors).flatten()

    input_film = df.loc[[idx]]
    df_copy = df.copy()
    df_copy['similarity'] = similarity_scores

    if year_range:
        df_copy = df_copy[(df_copy['release_date'].dt.year >= year_range[0]) &
                          (df_copy['release_date'].dt.year <= year_range[1])]
    if rating_range:
        df_copy = df_copy[(df_copy['vote_average'] >= rating_range[0]) &
                          (df_copy['vote_average'] <= rating_range[1])]
    if selected_genres:
        pattern = '|'.join(selected_genres)
        df_copy = df_copy[df_copy['name_genres'].str.contains(pattern, case=False, na=False)]
    if selected_actors:
        pattern = '|'.join(selected_actors)
        df_copy = df_copy[df_copy['actor_name'].str.contains(pattern, case=False, na=False)]

    df_copy = df_copy[df_copy.index != idx]
    df_copy = df_copy.sort_values(by='vote_average', ascending=ascending).head(num_recommendations)

    final_df = pd.concat([input_film, df_copy])
    return final_df[['title', 'name_genres', 'vote_average', 'actor_name', 'release_date', 'overview']]

# Streamlit UI
# Baca gambar dari file lokal
img_path = 'system-logo.jpg'
image = Image.open(img_path)

# Convert gambar ke base64 agar bisa dimasukkan ke HTML img tag
buffered = BytesIO()
image.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode()

# Buat markdown dengan base64 image
st.markdown(f"""
<div style='text-align:center; padding: 15px; background-color: #f9f1f2; border-radius:15px; margin-bottom:15px;'>
    <img src="data:image/png;base64,{img_str}" width='300' style='margin-bottom: 8px;' />
    <p style='color:#8CCDEB; font-size:30px; margin: 0; font-weight: bold; font-family: "Lora", sans-serif;'>
        <b>Movie Recommendation System</b>
    </p>
</div>
""", unsafe_allow_html=True)

# CSS untuk ubah warna slider
custom_css = """
<style>
/* Ganti warna utama slider */
div[data-testid="stSlider"] input[type=range] {
    accent-color: #27548A;
}

/* Ganti warna angka di atas thumb slider */
div[data-testid="stSlider"] .css-1cpxqw2,  /* angka atas */
div[data-testid="stSlider"] .css-qrbaxs,  /* angka bawah */
div[data-testid="stSlider"] .css-14gy7wr, /* label slider */
div[data-testid="stSlider"] .css-1n76uvr {
    color: #27548A !important;
}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)


st.markdown("---")
tab1, tab2 = st.tabs(["🏠 Home", "🎬 Movie Recommendations"])

# Track active tab via session_state
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Home"
# Track active tab via session_state
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Home"

with tab1:
   st.session_state.active_tab = "Home"
   st.title("🍿 Hey there, movie lover!\nCan’t decide what to watch? Chill, we’ll hook you up with the best picks — trending hits, fire ratings, and recs that match your vibe.")
   
   year_min = int(df['release_date'].dt.year.min())
   year_max = int(df['release_date'].dt.year.max())
   year_range_filter = st.slider("Release Year Filter", year_min, year_max, (year_min, year_max))
   
   filtered_df = df[(df['release_date'].dt.year >= year_range_filter[0]) &
                     (df['release_date'].dt.year <= year_range_filter[1])]
   
   top_rated = filtered_df.sort_values(by='vote_average', ascending=False).dropna(subset=['title']).head(5)
   titles = top_rated['title'].tolist()
   ratings = top_rated['vote_average'].tolist()
   ranked = sorted(zip(ratings, titles), reverse=True)
   podium_indices = [3, 1, 0, 2, 4]
   ordered = [ranked[i] for i in podium_indices]
   ordered_ratings, ordered_titles = zip(*ordered)
   wrapped_titles = [t if len(t) <= 15 else '<br>'.join(t[i:i+15] for i in range(0, len(t), 15)) for t in ordered_titles]
   x_labels = ['4th place', '🥈 2nd place', '🥇 1st place', '🥉 3rd place', '5th place']
   colors = ['#9FB3DF', '#C0C0C0', '#FFD700', '#CD7F32', '#9FB3DF']
   
   fig = go.Figure()
   fig.add_trace(go.Bar(
        x=x_labels,
        y=ordered_ratings,
        text=[f"{wrapped_titles[i]}<br>Rating: {ordered_ratings[i]}" for i in range(5)],
        textposition='auto',
        marker_color=colors,
        hovertemplate='<b>%{x}</b><br>%{text}<extra></extra>'
    ))
   
   fig.update_layout(
        title="🏆 Top 5 Highest Rated Movies",
        yaxis_title="Rating",
        xaxis_title="Ranking",
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(t=40, b=40, l=40, r=40)
    )
   
   st.plotly_chart(fig, use_container_width=True)
   st.subheader("\U0001F4F0 🆕 Movies with Recent Releases")
   
   newest_films = filtered_df.dropna(subset=['release_date']).sort_values(by='release_date', ascending=False).head(10)
   newest_films = newest_films.sort_values(by='release_date', ascending=True)
   
   fig2 = px.bar(
        newest_films,
        x='release_date',
        y='title',
        orientation='h',
        labels={'release_date': 'Release Date', 'title': ''},
        color_discrete_sequence=['#BDDDE4'],
        hover_data=None,
        custom_data=['vote_average', 'name_genres', 'actor_name']
    )
   
   fig2.update_layout(
        height=600,
        title='Top 10 Newest Movies',
        xaxis_title='Release Date',
        yaxis=dict(showticklabels=True),
        margin=dict(l=40, r=40, t=50, b=50),
        plot_bgcolor='rgba(0,0,0,0)'
    )
   
   fig2.update_traces(
        hovertemplate="<b>%{y}</b><br>" +
                      "Tanggal Rilis: %{x|%Y-%m-%d}<br>" +
                      "Rating: %{customdata[0]:.1f}<br>" +
                      "Genre: %{customdata[1]}<br>" +
                      "Aktor: %{customdata[2]}<extra></extra>"
    )
   
   st.plotly_chart(fig2, use_container_width=True)
   st.subheader("\U0001F3AD 🎞 Movie Genre Distribution (Top 5)")
   genre_series = filtered_df['name_genres'].dropna().str.split(', ')
   all_genres = genre_series.explode()
   genre_counts = all_genres.value_counts().head(5)
   
   fig3 = px.pie(
        names=genre_counts.index,
        values=genre_counts.values,
        title="Who’s Taking the Spotlight?",
        color_discrete_sequence=['#B1F0F7', '#81BFDA', '#F5F0CD', '#FADA7A', '#0A97B0'],
        hole=0.3
    )
   
   fig3.update_traces(
        textinfo='label+percent+value',
        textfont_size=14,
        hovertemplate='<b>%{label}</b><br>%{value} film (%{percent})<extra></extra>'
    )
   st.plotly_chart(fig3, use_container_width=True)


with tab2:
    st.session_state.active_tab = "Movie Recommendations"

    st.caption("🎯 Get Movie Recommendations that You Like!")
    movie_input = st.text_input("Enter the movie title you want to search for:")

    if 'selected_title' in st.session_state:
        movie_input = st.session_state.selected_title

    if movie_input:
        # Show sidebar only if active tab is Recommendations
        if st.session_state.active_tab == "Movie Recommendations":
            with st.sidebar:
                st.markdown("## 🎮 Search Filters")
                st.header("🔍 Filters")

                year_range = st.slider("Release Year", year_min, year_max, (year_min, year_max))
                rating_min = float(df['vote_average'].min())
                rating_max = float(df['vote_average'].max())
                rating_range = st.slider("Rating", rating_min, rating_max, (rating_min, rating_max))

                st.markdown("🎭 Genres")
                genres = sorted(set(g for sublist in df['name_genres'].dropna().str.split(', ') for g in sublist))
                selected_genres = st.multiselect("", genres)

                st.markdown("🎬 Actors")
                actors = sorted(df['actor_name'].dropna().unique())
                selected_actors = st.multiselect("", actors)

                ascending = st.radio("Sort", options=[False, True],
                                     format_func=lambda x: "Descending" if not x else "Ascending")

                st.markdown("### 🕘 Search History")
                if st.button("Clear All History"):
                    st.session_state.search_history.clear()
                    st.rerun()

                if st.session_state.search_history:
                    for i, title in enumerate(reversed(st.session_state.search_history)):
                        col1, col2 = st.columns([0.8, 0.2])
                        if col1.button(title, key=f"history_{i}"):
                            st.session_state.selected_title = title
                            st.rerun()
                        if col2.button("🗑", key=f"delete_{i}"):
                            st.session_state.search_history.remove(title)
                            st.rerun()
                else:
                    st.markdown("No history yet.")

        with st.spinner("🔍 Finding the best movies for you..."):
            results = recommend(
                movie_input,
                num_recommendations=10,
                year_range=year_range,
                rating_range=rating_range,
                selected_genres=selected_genres,
                selected_actors=selected_actors,
                ascending=ascending
            )

        normalized_input = movie_input.strip().title()
        if normalized_input not in st.session_state.search_history:
            st.session_state.search_history.append(normalized_input)

        if results.iloc[0]['title'] == 'Movie title not found.':
            st.warning("❌ Movie not found. Please check the title or choose another.")
        else:
            st.markdown("""
                <div style='background-color:#f0f4f8; padding:15px; border-radius:12px; margin-bottom:15px;'>
                <h3 style='color:#333;'>🎥 Recommended Movies for You</h3>
                <p>Here are some great picks based on your favorite movie.</p>
                </div>
                """, unsafe_allow_html=True)

            for _, row in results.iterrows():
                with st.container():
                    st.markdown(f"### 🎬 {row['title']}")
                    st.markdown(f"Genre: {row['name_genres']}  |  Rating: {row['vote_average']:.1f}")
                    st.markdown(f"Actor: {row['actor_name']}")
                    st.markdown(f"Release Date: {row['release_date'].date() if pd.notna(row['release_date']) else '-'}")
                    st.markdown(f"Synopsis: {row['overview']}")
                    st.divider()
    else:
        st.info("Enter a movie title above to start getting recommendations.")