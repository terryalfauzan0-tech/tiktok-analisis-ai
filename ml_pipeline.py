import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

def generate_dummy_data(n_samples=500):
    np.random.seed(42)
    
    # Generate random features
    likes = np.random.randint(50, 10000, n_samples)
    comments = np.random.randint(5, 1000, n_samples)
    shares = np.random.randint(0, 500, n_samples)
    watch_duration = np.random.uniform(10.0, 120.0, n_samples) # in seconds
    
    # Sentiment (0 = Negative, 1 = Neutral, 2 = Positive)
    sentiment_score = np.random.choice([0, 1, 2], n_samples, p=[0.2, 0.4, 0.4])
    
    # Define underlying logic for target 'purchase_intent'
    # Intent increases with higher engagement and positive sentiment
    engagement_factor = (likes / 10000) + (comments / 1000) + (shares / 500) + (watch_duration / 120)
    sentiment_factor = sentiment_score * 0.5
    
    overall_score = engagement_factor + sentiment_factor + np.random.normal(0, 0.5, n_samples)
    
    # Binarize target
    threshold = np.percentile(overall_score, 60) # Top 40% have high intent
    purchase_intent = (overall_score >= threshold).astype(int)
    
    df = pd.DataFrame({
        'likes': likes,
        'comments': comments,
        'shares': shares,
        'watch_duration': watch_duration,
        'sentiment': sentiment_score,
        'purchase_intent': purchase_intent
    })
    
    return df

class PipelineML:
    def __init__(self):
        self.model = LogisticRegression(random_state=42)
        self.scaler = StandardScaler()
        self.metrics = {}
        
    def train_and_eval(self, df):
        X = df[['likes', 'comments', 'shares', 'watch_duration', 'sentiment']]
        y = df['purchase_intent']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        self.metrics['accuracy'] = accuracy_score(y_test, y_pred)
        self.metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
        self.metrics['recall'] = recall_score(y_test, y_pred, zero_division=0)
        self.metrics['report'] = classification_report(y_test, y_pred)
        
        return self.metrics
        
    def predict_intent(self, likes, comments, shares, watch_duration, sentiment_val):
        """
        predict new data point
        sentiment_val: 0 (neg), 1 (neutral), 2 (pos)
        """
        input_data = pd.DataFrame({
            'likes': [likes],
            'comments': [comments],
            'shares': [shares],
            'watch_duration': [watch_duration],
            'sentiment': [sentiment_val]
        })
        
        scaled_input = self.scaler.transform(input_data)
        prediction = self.model.predict(scaled_input)[0]
        prob = self.model.predict_proba(scaled_input)[0][1]
        
        return prediction, prob
