# 🎬 Movie-Recommendation-System-Streamlit

An interactive web app built with Streamlit that recommends movies based on genre, rating, and popularity using TF-IDF, cosine similarity, and RapidFuzz. The system is designed to help users discover relevant films even with typo-prone queries and offers rich visualizations to enhance user experience.

## Features

🔍 Movie Recommendations using content-based filtering  
🤖 Fuzzy Matching with RapidFuzz for typo-tolerant search  
📊 Visualizations with Plotly, Seaborn, and Matplotlib  
🎛️ Filtering Options by genre and rating  
💡 Intelligent Suggestions powered by TF-IDF & cosine similarity  
🧼 Clean and Interactive UI via Streamlit  

## 🛠️ Tech Stack

- Python  
- Pandas  
- Scikit-learn  
- Streamlit  
- RapidFuzz  
- Matplotlib & Seaborn  
- Plotly  
- Pillow (PIL)  

## 📁 Project Structure

```
Movie-Recommendation-System-Streamlit/
├── app.py               # Main Streamlit app
├── dataset-movie/       # Movie dataset
├── system-logo/         # Logo assets
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## 🧠 How It Works

- **Text Representation**: Uses `TfidfVectorizer` to convert movie metadata (e.g., genre, keywords) into numerical vectors.  
- **Similarity Calculation**: Computes cosine similarity between vectors to find similar movies.  
- **Fuzzy Matching**: RapidFuzz allows search inputs to tolerate typos or partial matches.  
- **Visualization**: Plots show genre distribution, top-rated movies, and more.  
