import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import re
import pickle
import json

# Download English stopwords
nltk.download('stopwords', quiet=True)
english_stopwords = set(stopwords.words('english'))

class ContentBasedRecommender:
    def __init__(self):
        """
        Initialize content-based recommendation model
        """
        self.product_features = None
        self.product_ids = None
        self.tfidf_matrix = None
        self.vectorizer = None
        self.similarity_matrix = None
        self.product_df = None
        # Update event type weights to match the normalized format
        self.user_activity_weights = {
            'view': 1,
            'add_to_cart': 5,
            'add_to_wishlist': 3,
            'remove_from_cart': -2,
            'remove_from_wishlist': -1
        }
        self.user_profiles = {}
    
    def preprocess_text(self, text):
        """
        Preprocess text for analysis
        """
        if not isinstance(text, str):
            return ""
            
        # Convert text to lowercase
        text = text.lower()
        
        # Remove numbers and symbols
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Remove stopwords
        words = text.split()
        filtered_words = [word for word in words if word not in english_stopwords]
        
        return ' '.join(filtered_words)
    
    def fit(self, product_data):
        """
        Train model on product data
        
        Parameters:
        product_data: DataFrame or list of dictionaries containing
                     'id', 'name', 'category', 'description'
        """
        # Convert to DataFrame if list of dictionaries is provided
        if isinstance(product_data, list):
            self.product_df = pd.DataFrame(product_data)
        else:
            self.product_df = product_data.copy()
        
        # Ensure all required columns exist
        required_columns = ['id', 'name', 'category', 'description']
        for col in required_columns:
            if col not in self.product_df.columns:
                raise ValueError(f"Column {col} not found in the data")
        
        # Ensure all text columns are strings
        for col in ['name', 'category', 'description']:
            self.product_df[col] = self.product_df[col].astype(str)
        
        # Process text data
        self.product_df['name_processed'] = self.product_df['name'].apply(self.preprocess_text)
        self.product_df['description_processed'] = self.product_df['description'].apply(self.preprocess_text)
        self.product_df['category_processed'] = self.product_df['category'].apply(self.preprocess_text)
        
        # Combine text features with different weights for fields
        self.product_df['combined_features'] = (
            self.product_df['name_processed'] + ' ' + self.product_df['name_processed'] + ' ' + self.product_df['name_processed'] + ' ' +  # Repeat name to increase importance
            self.product_df['category_processed'] + ' ' + self.product_df['category_processed'] + ' ' +  # Repeat category to increase importance
            self.product_df['description_processed']
        )
        
        # Convert texts to vectors using TF-IDF
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.product_df['combined_features'])
        
        # Calculate similarity matrix between products
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
        # Store product IDs
        self.product_ids = list(self.product_df['id'])
        
        print(f"Analyzed {len(self.product_df)} products and created similarity matrix {self.similarity_matrix.shape}")
        
        return self
    
    def update_user_profiles(self, user_activity_logs):
        """
        Update user profiles based on activity logs
        
        Parameters:
        user_activity_logs: DataFrame or list of dictionaries containing
                           'user_id', 'product_id', 'event_type'
        """
        # Convert to DataFrame if list of dictionaries is provided
        if isinstance(user_activity_logs, list):
            user_activity_logs = pd.DataFrame(user_activity_logs)
        
        required_columns = ['user_id', 'product_id', 'event_type']
        for col in required_columns:
            if col not in user_activity_logs.columns:
                raise ValueError(f"Column {col} not found in activity logs")
        
        # Ensure product_id is of the same type as in our product data
        if user_activity_logs['product_id'].dtype != pd.Series(self.product_ids).dtype:
            user_activity_logs['product_id'] = user_activity_logs['product_id'].astype(pd.Series(self.product_ids).dtype)
        
        # Normalize event types to lowercase for consistency
        user_activity_logs['event_type'] = user_activity_logs['event_type'].str.lower()
        
        # Map common event types to our standard format
        event_type_mapping = {
            'view': 'view',
            'add to cart': 'add_to_cart',
            'add_to_cart': 'add_to_cart',
            'add to wishlist': 'add_to_wishlist',
            'add_to_wishlist': 'add_to_wishlist',
            'remove from cart': 'remove_from_cart',
            'remove_from_cart': 'remove_from_cart',
            'remove from wishlist': 'remove_from_wishlist',
            'remove_from_wishlist': 'remove_from_wishlist'
        }
        
        user_activity_logs['normalized_event'] = user_activity_logs['event_type'].apply(
            lambda x: event_type_mapping.get(x.lower(), x.lower())
        )
        
        # Create profile for each user
        for user_id, user_logs in user_activity_logs.groupby('user_id'):
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = np.zeros(self.tfidf_matrix.shape[1])
            
            for _, row in user_logs.iterrows():
                product_id = row['product_id']
                event_type = row['normalized_event']
                
                if product_id in self.product_ids:
                    product_idx = self.product_ids.index(product_id)
                    
                    # Get product feature vector
                    product_vector = self.tfidf_matrix[product_idx].toarray().flatten()
                    
                    # Update user profile based on event type
                    weight = self.user_activity_weights.get(event_type, 0)
                    self.user_profiles[user_id] += weight * product_vector
        
        print(f"Updated profiles for {len(self.user_profiles)} users")
        return self
    
    def recommend_similar_products(self, product_id, top_n=5):
        """
        Recommend products similar to a given product
        
        Parameters:
        product_id: ID of the product to find similar items for
        top_n: Number of recommendations to return
        
        Returns:
        List of tuples (product_id, similarity_score) ordered by similarity
        """
        if product_id not in self.product_ids:
            raise ValueError(f"Product with ID {product_id} not found in the data")
            
        product_idx = self.product_ids.index(product_id)
        
        # Get similarity scores with the specified product
        similarity_scores = list(enumerate(self.similarity_matrix[product_idx]))
        
        # Sort products by similarity score
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Remove the product itself from the list and select top_n products
        similarity_scores = similarity_scores[1:top_n+1]
        
        # Convert list to tuples (product_id, similarity_score)
        recommendations = [(self.product_ids[i], score) for i, score in similarity_scores]
        
        return recommendations
    
    def recommend_for_user(self, user_id, top_n=5):
        """
        Recommend products for a specific user based on their profile
        
        Parameters:
        user_id: User identifier
        top_n: Number of recommendations to return
        
        Returns:
        List of tuples (product_id, similarity_score) ordered by similarity
        """
        if user_id not in self.user_profiles:
            raise ValueError(f"User with ID {user_id} not found")
            
        user_profile = self.user_profiles[user_id]
        
        # Calculate similarity between user profile and all products
        similarity_scores = cosine_similarity([user_profile], self.tfidf_matrix)[0]
        
        # Sort products by similarity score
        product_indices = list(range(len(self.product_ids)))
        product_scores = list(zip(product_indices, similarity_scores))
        product_scores = sorted(product_scores, key=lambda x: x[1], reverse=True)
        
        # Select top_n products
        product_scores = product_scores[:top_n]
        
        # Convert list to tuples (product_id, similarity_score)
        recommendations = [(self.product_ids[i], score) for i, score in product_scores]
        
        return recommendations
    
    def save_model(self, filepath):
        """
        Save model to file
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """
        Load model from file
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model


# Example of how to use the model
if __name__ == "__main__":
    # Sample product data in the requested format
    products_data = [
        {"id": 101, "name": "iPhone 13", "category": "Smartphones", "description": "Apple smartphone with A15 chip"},
        {"id": 102, "name": "Samsung Galaxy S21", "category": "Smartphones", "description": "Android smartphone with high-resolution camera"},
        {"id": 103, "name": "Dell XPS 15", "category": "Laptops", "description": "Premium laptop with Intel Core i7 processor"},
        {"id": 104, "name": "Sony WH-1000XM4", "category": "Headphones", "description": "Wireless noise-cancelling headphones"},
        {"id": 105, "name": "Apple Watch Series 7", "category": "Smartwatches", "description": "Smartwatch with health tracking and fitness features"}
    ]
    
    # Sample user activity logs in the requested format
    user_logs_data = [
        {"user_id": "u123", "product_id": 101, "event_type": "view"},
        {"user_id": "u123", "product_id": 103, "event_type": "add_to_cart"},
        {"user_id": "u456", "product_id": 102, "event_type": "add_to_wishlist"},
        {"user_id": "u456", "product_id": 104, "event_type": "add_to_cart"},
        {"user_id": "u456", "product_id": 105, "event_type": "view"}
    ]
    
    # Create and train the model
    recommender = ContentBasedRecommender()
    recommender.fit(products_data)
    recommender.update_user_profiles(user_logs_data)
    
    # Get recommendations for a specific product
    product_id = 101
    similar_products = recommender.recommend_similar_products(product_id, top_n=2)
    print(f"\nProducts similar to product {product_id}:")
    for prod_id, score in similar_products:
        product_info = next((p for p in products_data if p["id"] == prod_id), None)
        if product_info:
            print(f"Product: {product_info['name']}, Similarity: {score:.4f}")
    
    # Get recommendations for a specific user
    user_id = "u123"
    user_recommendations = recommender.recommend_for_user(user_id, top_n=2)
    print(f"\nRecommendations for user {user_id}:")
    for prod_id, score in user_recommendations:
        product_info = next((p for p in products_data if p["id"] == prod_id), None)
        if product_info:
            print(f"Product: {product_info['name']}, Similarity: {score:.4f}")
    
    # Save the model
    recommender.save_model('content_based_recommender.pkl')
    
    # Load the model
    loaded_recommender = ContentBasedRecommender.load_model('content_based_recommender.pkl')