import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
import re
from datetime import datetime
import requests
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from collections import Counter

# Set page configuration
st.set_page_config(
    page_title="Google Reviews Analyzer",
    page_icon="📊",
    layout="wide",
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stat-box {
        background-color: white;
        border-radius: 5px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    .chart-container {
        background-color: white;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'reviews_df' not in st.session_state:
    st.session_state.reviews_df = None
if 'themes' not in st.session_state:
    st.session_state.themes = None
if 'sentiment_over_time' not in st.session_state:
    st.session_state.sentiment_over_time = None
if 'ollama_available' not in st.session_state:
    st.session_state.ollama_available = None

# Helper functions for Ollama
def check_ollama_availability():
    """Check if Ollama is available"""
    url = "http://localhost:11434/api/tags"
    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            return True
        return False
    except Exception:
        return False

def query_llama(prompt, system_prompt=None, temperature=0.5):
    """Query the Llama 3 model via Ollama API with fallback"""
    if not st.session_state.ollama_available:
        return None
    
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False,
        "temperature": temperature
    }
    
    if system_prompt:
        payload["system"] = system_prompt
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            st.warning(f"Error querying Ollama: {response.status_code}")
            st.session_state.ollama_available = False
            return None
    except Exception as e:
        st.warning(f"Error connecting to Ollama: {e}")
        st.session_state.ollama_available = False
        return None

def extract_json_from_text(text):
    """Extract JSON from the generated text"""
    if text is None:
        return None
        
    # Find JSON pattern between triple backticks
    pattern = r"```json\s*([\s\S]*?)\s*```"
    match = re.search(pattern, text)
    
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Second attempt: look for anything that looks like JSON
    pattern = r"\{[\s\S]*\}"
    match = re.search(pattern, text)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    return None

def preprocess_date(date_str):
    """Convert date string to datetime object"""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except:
        try:
            return datetime.strptime(date_str, "%m/%d/%Y")
        except:
            return None

def extract_topics_nmf(reviews_df, n_topics=5):
    """Extract topics from reviews using NMF"""
    # Combine preprocessing and NMF topic extraction
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.95, 
        min_df=2,
        stop_words='english'
    )
    
    # Create document-term matrix
    try:
        tfidf = tfidf_vectorizer.fit_transform(reviews_df['text'])
        
        # If we have fewer documents than requested topics, adjust
        n_topics = min(n_topics, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
        if n_topics < 1:
            n_topics = 1
        
        # Apply NMF
        nmf = NMF(n_components=n_topics, random_state=42)
        nmf.fit(tfidf)
        
        # Get feature names
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        # Extract top words for each topic
        topics = []
        for topic_idx, topic in enumerate(nmf.components_):
            top_words_idx = topic.argsort()[:-11:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append({
                "id": topic_idx,
                "words": top_words
            })
        
        return topics
    except Exception as e:
        st.error(f"Error in topic extraction: {e}")
        return []

def extract_themes_from_topics(reviews_df, topics, use_llm=True):
    """Convert NMF topics to themes, optionally enhanced by LLM"""
    themes = []
    
    if not topics:
        # Create basic themes based on rating if no topics
        themes = [
            {
                "title": "Positive Experiences",
                "sample_phrases": ["Great experience", "Excellent service", "Very satisfied"],
                "sentiment": 0.8,
                "recommendation": "Continue providing excellent service"
            },
            {
                "title": "Negative Experiences",
                "sample_phrases": ["Poor quality", "Bad service", "Disappointed"],
                "sentiment": -0.8,
                "recommendation": "Address customer complaints promptly"
            }
        ]
        return {"themes": themes}
    
    # Get positive and negative reviews
    positive_reviews = reviews_df[reviews_df['rating'] >= 4]['text'].tolist()
    negative_reviews = reviews_df[reviews_df['rating'] <= 2]['text'].tolist()
    
    for topic in topics:
        keywords = topic["words"]
        topic_title = " & ".join(keywords[:2]).title()
        
        # Find reviews that contain these keywords
        sample_phrases = []
        
        # Look for phrases in positive reviews
        pos_phrases = []
        for review in positive_reviews:
            if any(keyword in review.lower() for keyword in keywords):
                # Find sentences containing keywords
                sentences = re.split(r'[.!?]+', review)
                for sentence in sentences:
                    if any(keyword in sentence.lower() for keyword in keywords):
                        pos_phrases.append(sentence.strip())
                        break
                if len(pos_phrases) >= 2:
                    break
        
        # Look for phrases in negative reviews
        neg_phrases = []
        for review in negative_reviews:
            if any(keyword in review.lower() for keyword in keywords):
                sentences = re.split(r'[.!?]+', review)
                for sentence in sentences:
                    if any(keyword in sentence.lower() for keyword in keywords):
                        neg_phrases.append(sentence.strip())
                        break
                if len(neg_phrases) >= 2:
                    break
        
        # Combine positive and negative phrases
        sample_phrases = (pos_phrases + neg_phrases)[:3]
        if not sample_phrases and len(keywords) > 0:
            sample_phrases = [f"Reviews mentioning {keywords[0]}"]
        
        # Calculate sentiment based on ratings of reviews containing keywords
        sentiment_sum = 0
        sentiment_count = 0
        
        for _, review in reviews_df.iterrows():
            if any(keyword in review['text'].lower() for keyword in keywords):
                sentiment_sum += (review['rating'] - 3) / 2  # Convert 1-5 to -1 to 1
                sentiment_count += 1
        
        sentiment = sentiment_sum / max(1, sentiment_count)
        
        # Generate recommendation based on sentiment
        if sentiment > 0.3:
            recommendation = f"Continue emphasizing {keywords[0]} in your business"
        elif sentiment < -0.3:
            recommendation = f"Improve {keywords[0]} to address customer concerns"
        else:
            recommendation = f"Monitor customer feedback about {keywords[0]}"
        
        themes.append({
            "title": topic_title,
            "sample_phrases": sample_phrases,
            "sentiment": sentiment,
            "recommendation": recommendation
        })
    
    # If we have LLM available and user wants to use it, enhance themes
    if use_llm and st.session_state.ollama_available:
        themes = enhance_themes_with_llm(themes, reviews_df)
    
    return {"themes": themes}

def enhance_themes_with_llm(themes, reviews_df):
    """Use LLM to improve theme titles and recommendations"""
    enhanced_themes = []
    
    for theme in themes:
        system_prompt = """
        You are an expert at analyzing business reviews.
        Given a theme extracted from customer reviews, improve the theme title to be more business-relevant and provide a specific recommendation based on the sentiment.
        
        Format your response as JSON with the following structure:
        ```json
        {
            "improved_title": "Better theme title",
            "improved_recommendation": "More specific recommendation"
        }
        ```
        Keep your response concise and focused.
        """
        
        user_prompt = f"""
        Theme information:
        - Current title: {theme['title']}
        - Sample phrases: {', '.join(theme['sample_phrases'])}
        - Sentiment score: {theme['sentiment']} (on a scale from -1 to +1)
        - Current recommendation: {theme['recommendation']}
        
        Please improve the title to be more business-relevant and provide a more specific recommendation.
        """
        
        result = query_llama(user_prompt, system_prompt)
        
        if result:
            json_result = extract_json_from_text(result)
            if json_result and 'improved_title' in json_result:
                theme['title'] = json_result['improved_title']
            if json_result and 'improved_recommendation' in json_result:
                theme['recommendation'] = json_result['improved_recommendation']
        
        enhanced_themes.append(theme)
    
    return enhanced_themes

def analyze_sentiment_basic(reviews_df):
    """Analyze sentiment using a basic approach based on ratings"""
    # Group reviews by month
    reviews_df['month'] = pd.to_datetime(reviews_df['time']).dt.strftime('%Y-%m')
    monthly_groups = reviews_df.groupby('month')
    
    monthly_sentiment = []
    
    for month, group in monthly_groups:
        # Calculate sentiment based on ratings
        avg_rating = group['rating'].mean()
        normalized_sentiment = (avg_rating - 3) / 2  # Convert 1-5 to -1 to 1
        
        # Extract common terms
        all_text = ' '.join(group['text'].tolist()).lower()
        word_counts = Counter(re.findall(r'\b\w+\b', all_text))
        common_words = [word for word, count in word_counts.most_common(5) 
                        if len(word) > 3 and word not in ('this', 'that', 'with', 'they', 'their', 'have', 'very')]
        
        explanation = f"Common terms: {', '.join(common_words)}"
        
        monthly_sentiment.append({
            'month': month,
            'sentiment': normalized_sentiment,
            'explanation': explanation
        })
    
    return pd.DataFrame(monthly_sentiment)

def analyze_sentiment_over_time(reviews_df):
    """Analyze sentiment trends over time using Llama 3 with fallback"""
    # Check if LLM is available, otherwise use basic analysis 
    if not st.session_state.ollama_available:
        return analyze_sentiment_basic(reviews_df)
    
    with st.spinner("Analyzing sentiment trends..."):
        # Group reviews by month
        reviews_df['month'] = pd.to_datetime(reviews_df['time']).dt.strftime('%Y-%m')
        monthly_groups = reviews_df.groupby('month')
        
        monthly_sentiment = []
        
        for month, group in monthly_groups:
            # Sample reviews from this month (up to 5)
            sample_size = min(5, len(group))
            sample_reviews = group.sample(sample_size)
            
            review_samples = "\n".join([
                f"Review {i+1}: {row['text'][:200]}..." 
                for i, (_, row) in enumerate(sample_reviews.iterrows())
            ])
            
            system_prompt = """
            You are an expert at analyzing sentiment in customer reviews.
            Analyze the provided reviews and determine the overall sentiment score from -1 (very negative) to +1 (very positive).
            Provide only a JSON response with the sentiment score and a brief explanation.
            
            Format:
            ```json
            {
                "sentiment_score": 0.5,
                "explanation": "Brief explanation"
            }
            ```
            """
            
            user_prompt = f"""
            Here are some customer reviews from {month}:
            
            {review_samples}
            
            Analyze the overall sentiment of these reviews.
            """
            
            result = query_llama(user_prompt, system_prompt)
            
            if result:
                json_result = extract_json_from_text(result)
                if json_result and 'sentiment_score' in json_result:
                    monthly_sentiment.append({
                        'month': month,
                        'sentiment': json_result['sentiment_score'],
                        'explanation': json_result.get('explanation', '')
                    })
                    continue
            
            # Fallback: use average rating as sentiment
            avg_rating = group['rating'].mean()
            normalized_sentiment = (avg_rating - 3) / 2  # Convert 1-5 to -1 to 1
            
            # Extract common terms
            all_text = ' '.join(group['text'].tolist()).lower()
            word_counts = Counter(re.findall(r'\b\w+\b', all_text))
            common_words = [word for word, count in word_counts.most_common(5) 
                            if len(word) > 3 and word not in ('this', 'that', 'with', 'they', 'their', 'have', 'very')]
            
            explanation = f"Based on average rating. Common terms: {', '.join(common_words)}"
            
            monthly_sentiment.append({
                'month': month,
                'sentiment': normalized_sentiment,
                'explanation': explanation
            })
        
        return pd.DataFrame(monthly_sentiment)

# Check Ollama availability at startup
st.session_state.ollama_available = check_ollama_availability()

# Streamlit UI
st.title("📊 Google Reviews Analyzer")
st.markdown("Extract themes and insights from your Google Reviews")

# Notification about Ollama status
if st.session_state.ollama_available:
    st.success("✅ LLM (Llama 3) is available via Ollama")
else:
    st.warning("⚠️ Ollama connection not available. Running in fallback mode with basic NLP.")

# Sidebar for data upload and settings
with st.sidebar:
    st.header("Settings")
    
    upload_option = st.radio(
        "Choose how to load data",
        ["Upload CSV", "Use Sample Data"]
    )
    
    if upload_option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload Google reviews CSV", type=["csv"])
        
        if uploaded_file is not None:
            try:
                # Read the CSV file into a pandas DataFrame
                reviews_df = pd.read_csv(uploaded_file)
                
                # Ensure required columns are present
                required_cols = ['rating', 'text', 'time']
                if all(col in reviews_df.columns for col in required_cols):
                    st.session_state.reviews_df = reviews_df
                    # Reset analysis results
                    st.session_state.themes = None
                    st.session_state.sentiment_over_time = None
                else:
                    st.error("CSV must contain 'rating', 'text', and 'time' columns")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
    else:
        # Use sample data
        if st.button("Load Sample Data"):
            # Create sample data
            sample_data = {
                'rating': np.random.choice([1, 2, 3, 4, 5], size=50, p=[0.05, 0.1, 0.15, 0.3, 0.4]),
                'text': [
                    "Great service, very friendly staff!",
                    "The product quality was excellent, will definitely come back.",
                    "Waited too long for my order, but the food was good.",
                    "Not impressed with customer service but prices are reasonable.",
                    "Love the atmosphere and the staff is super helpful.",
                    "The location is convenient but parking is a nightmare.",
                    "Amazing experience overall, highly recommend!",
                    "Product broke after a week, very disappointed.",
                    "Best service I've had in years, thank you!",
                    "Too expensive for the quality you get."
                ] * 5,
                'time': pd.date_range(start='2023-01-01', periods=50, freq='3D').strftime('%Y-%m-%d').tolist()
            }
            
            # Add more variety to sample reviews
            for i in range(len(sample_data['text'])):
                if i % 3 == 0:
                    sample_data['text'][i] += " The staff went above and beyond."
                if i % 5 == 0:
                    sample_data['text'][i] += " Prices are very competitive."
                if i % 7 == 0:
                    sample_data['text'][i] += " The place was very clean and well-maintained."
            
            st.session_state.reviews_df = pd.DataFrame(sample_data)
            # Reset analysis results
            st.session_state.themes = None
            st.session_state.sentiment_over_time = None
    
    # Analysis options
    st.header("Analysis")
    
    if st.session_state.reviews_df is not None:
        use_llm = st.checkbox("Use LLM enhancement", value=st.session_state.ollama_available, 
                              disabled=not st.session_state.ollama_available)
        
        if st.button("Run Analysis"):
            # Extract topics using NMF first
            topics = extract_topics_nmf(st.session_state.reviews_df)
            
            # Convert topics to themes
            st.session_state.themes = extract_themes_from_topics(
                st.session_state.reviews_df,
                topics,
                use_llm
            )
            
            # Analyze sentiment over time
            st.session_state.sentiment_over_time = analyze_sentiment_over_time(
                st.session_state.reviews_df
            )

# Main content
if st.session_state.reviews_df is not None:
    # Display data summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
        st.metric("Total Reviews", len(st.session_state.reviews_df))
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
        avg_rating = round(st.session_state.reviews_df['rating'].mean(), 2)
        st.metric("Average Rating", f"{avg_rating}/5")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
        date_range = f"{st.session_state.reviews_df['time'].min()} to {st.session_state.reviews_df['time'].max()}"
        st.metric("Date Range", date_range)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Display rating distribution
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.subheader("Rating Distribution")
    
    rating_counts = st.session_state.reviews_df['rating'].value_counts().sort_index()
    
    fig = px.bar(
        x=rating_counts.index,
        y=rating_counts.values,
        labels={'x': 'Rating', 'y': 'Count'},
        color=rating_counts.index,
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    fig.update_layout(
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        showlegend=False,
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display themes if available
    if st.session_state.themes is not None:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("Top Themes")
        
        for i, theme in enumerate(st.session_state.themes['themes']):
            sentiment_color = "green" if theme['sentiment'] > 0.3 else "red" if theme['sentiment'] < -0.3 else "orange"
            
            with st.expander(f"{i+1}. {theme['title']} (Sentiment: {theme['sentiment']:.2f})", expanded=True):
                st.markdown(f"**Sample phrases:**")
                for phrase in theme['sample_phrases']:
                    st.markdown(f"• _{phrase}_")
                
                st.markdown(f"**Recommendation:** {theme['recommendation']}")
                
                # Sentiment gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=theme['sentiment'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Sentiment"},
                    gauge={
                        'axis': {'range': [-1, 1]},
                        'bar': {'color': sentiment_color},
                        'steps': [
                            {'range': [-1, -0.3], 'color': "rgba(255, 0, 0, 0.3)"},
                            {'range': [-0.3, 0.3], 'color': "rgba(255, 165, 0, 0.3)"},
                            {'range': [0.3, 1], 'color': "rgba(0, 128, 0, 0.3)"}
                        ]
                    }
                ))
                
                fig.update_layout(height=200)
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Display sentiment over time if available
    if st.session_state.sentiment_over_time is not None:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("Sentiment Trends Over Time")
        
        df = st.session_state.sentiment_over_time
        
        fig = px.line(
            df,
            x='month',
            y='sentiment',
            markers=True,
            labels={'month': 'Month', 'sentiment': 'Sentiment Score'},
            title='Sentiment Score Over Time (-1 to +1)'
        )
        
        fig.update_layout(
            xaxis=dict(tickangle=45),
            yaxis=dict(range=[-1, 1]),
            height=400
        )
        
        # Add a reference line at 0
        fig.add_shape(
            type="line",
            x0=df['month'].min(),
            y0=0,
            x1=df['month'].max(),
            y1=0,
            line=dict(color="gray", width=1, dash="dash")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Monthly Sentiment Analysis"):
            for _, row in df.iterrows():
                sentiment_color = "green" if row['sentiment'] > 0.3 else "red" if row['sentiment'] < -0.3 else "orange"
                st.markdown(f"**{row['month']}**: <span style='color:{sentiment_color}'>{row['sentiment']:.2f}</span> - {row['explanation']}", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Word cloud and key terms
    if st.session_state.reviews_df is not None:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("Most Frequent Terms")
        
        # Create a simple frequency-based visualization
        all_text = ' '.join(st.session_state.reviews_df['text'].tolist()).lower()
        word_counts = Counter(re.findall(r'\b\w+\b', all_text))
        
        # Filter out common stopwords
        stopwords = ['the', 'and', 'is', 'in', 'it', 'to', 'was', 'for', 'with', 
                     'that', 'this', 'of', 'i', 'a', 'my', 'we', 'our', 'they', 
                     'their', 'very', 'are', 'have', 'had', 'not', 'but']
        
        filtered_words = [(word, count) for word, count in word_counts.items() 
                          if word not in stopwords and len(word) > 2]
        
        # Get top words
        top_words = sorted(filtered_words, key=lambda x: x[1], reverse=True)[:20]
        words = [word for word, _ in top_words]
        counts = [count for _, count in top_words]
        
        fig = px.bar(
            x=words,
            y=counts,
            labels={'x': 'Term', 'y': 'Frequency'},
            title='Most Common Terms in Reviews',
            color=counts,
            color_continuous_scale=px.colors.sequential.Viridis
        )
        
        fig.update_layout(
            xaxis=dict(tickangle=45),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Raw data view
    with st.expander("View Raw Data"):
        st.dataframe(st.session_state.reviews_df)

else:
    # Display instructions
    st.info("Please upload a CSV file or load sample data from the sidebar to get started.")
    
    st.markdown("""
    ### Expected CSV Format:
    
    Your CSV should contain at least these columns:
    - `rating`: Numerical rating (typically 1-5)
    - `text`: Review content
    - `time`: Review date (YYYY-MM-DD format)
    
    ### How this tool works:
    
    1. **Upload Data**: Load your Google Reviews data
    2. **Run Analysis**: Process reviews using topic modeling and LLM enhancement (when available)
    3. **View Insights**: Explore themes, sentiments, and trends
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, Topic Modeling, and LLM enhancement")