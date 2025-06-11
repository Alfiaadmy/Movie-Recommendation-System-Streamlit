# 🎬 Movie-Recommendation-System-Streamlit

An interactive web app built with Streamlit that recommends movies based on genre, rating, and popularity using TF-IDF, cosine similarity, and RapidFuzz. The system is designed to help users discover relevant films even with typo-prone queries and offers rich visualizations to enhance user experience.

---

## ✨ Features

- 🔍 **Movie Recommendations** using content-based filtering  
- 🤖 **Fuzzy Matching** with RapidFuzz for typo-tolerant search  
- 📊 **Visualizations** with Plotly, Seaborn, and Matplotlib  
- 🎛️ **Filtering Options** by genre and rating  
- 💡 **Intelligent Suggestions** powered by TF-IDF & cosine similarity  
- 🧼 **Clean and Interactive UI** via Streamlit  

---

## 🛠️ Tech Stack

- Python  
- Pandas  
- Scikit-learn  
- Streamlit  
- RapidFuzz  
- Matplotlib & Seaborn  
- Plotly  
- Pillow (PIL)  

---

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

1. **Text Representation**  
   Uses `TfidfVectorizer` to convert movie metadata (e.g., genre, keywords) into numerical vectors.

2. **Similarity Calculation**  
   Computes **cosine similarity** between vectors to find similar movies.

3. **Fuzzy Matching**  
   Utilizes **RapidFuzz** to allow typo-tolerant and partial match searches for movie titles.

4. **Visualization**  
   Displays data insights like:
   - Genre distribution  
   - Top-rated movies  
   - Popular genres over time  
   - User preferences

---

