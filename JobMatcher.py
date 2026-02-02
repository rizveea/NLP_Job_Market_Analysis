"""
Job Matcher Algorithm using NLP Techniques
Combines TF-IDF, Word Embeddings, and LDA for optimal job matching
"""

import json
import numpy as np
import re
from typing import List, Dict, Tuple
from collections import defaultdict

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Scikit-learn for TF-IDF and LDA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation

# Sentence transformers for semantic embeddings
from sentence_transformers import SentenceTransformer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class JobMatcher:
    """
    A comprehensive job matching system using multiple NLP techniques:
    - TF-IDF for keyword-based similarity
    - Sentence embeddings for semantic similarity
    - LDA for topic modeling
    """
    
    def __init__(self, jobs_file_path: str, n_topics: int = 20):
        """
        Initialize the job matcher with job data
        
        Args:
            jobs_file_path: Path to JSON file containing job data
            n_topics: Number of topics for LDA model
        """
        self.jobs = self._load_jobs(jobs_file_path)
        self.n_topics = n_topics
        
        # Initialize NLP components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add custom stop words for job descriptions
        self.stop_words.update([
            'job', 'work', 'position', 'role', 'candidate', 'experience',
            'team', 'company', 'opportunity', 'responsibilities', 'requirements'
        ])
        
        # Initialize models (will be trained on first use)
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.lda_model = None
        self.lda_vectorizer = None
        self.job_topics = None
        self.embedding_model = None
        self.job_embeddings = None
        
        # Preprocess job data
        self.processed_jobs = self._preprocess_jobs()
        
        print(f"Loaded {len(self.jobs)} jobs")
        
    def _load_jobs(self, file_path: str) -> List[Dict]:
        """Load job data from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize and lemmatize text"""
        tokens = word_tokenize(text)
        lemmatized = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        return lemmatized
    
    def _preprocess_jobs(self) -> List[Dict]:
        """Preprocess all jobs for analysis"""
        processed = []
        
        for job in self.jobs:
            # Combine relevant fields for matching
            full_text = f"{job.get('title', '')} {job.get('description', '')} {job.get('company', '')} {job.get('location', '')}"
            
            # Clean text
            cleaned = self._clean_text(full_text)
            
            # Tokenize and lemmatize
            tokens = self._tokenize_and_lemmatize(cleaned)
            
            processed.append({
                'original': job,
                'cleaned_text': cleaned,
                'tokens': tokens,
                'token_string': ' '.join(tokens)
            })
        
        return processed
    
    def _train_tfidf(self):
        """Train TF-IDF vectorizer on job corpus"""
        print("Training TF-IDF model...")
        
        corpus = [job['token_string'] for job in self.processed_jobs]
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # unigrams and bigrams
            min_df=2,
            max_df=0.8
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
    
    def _train_lda(self):
        """Train LDA model for topic modeling"""
        print("Training LDA model...")
        
        corpus = [job['token_string'] for job in self.processed_jobs]
        
        # Use CountVectorizer for LDA (it requires count data, not TF-IDF)
        self.lda_vectorizer = CountVectorizer(
            max_features=3000,
            min_df=5,
            max_df=0.7
        )
        
        doc_term_matrix = self.lda_vectorizer.fit_transform(corpus)
        
        # Train LDA model
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            n_jobs=-1,
            max_iter=20
        )
        
        self.job_topics = self.lda_model.fit_transform(doc_term_matrix)
        print(f"LDA model trained with {self.n_topics} topics")
        
    def _train_embeddings(self):
        """Generate embeddings for all jobs using sentence transformers"""
        print("Generating job embeddings...")
        
        # Use a pre-trained sentence transformer model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create embeddings for job descriptions
        job_texts = [job['cleaned_text'] for job in self.processed_jobs]
        self.job_embeddings = self.embedding_model.encode(
            job_texts,
            show_progress_bar=True,
            batch_size=32
        )
        
        print(f"Generated embeddings with shape: {self.job_embeddings.shape}")
    
    def train_models(self):
        """Train all NLP models"""
        self._train_tfidf()
        self._train_lda()
        self._train_embeddings()
        print("All models trained successfully!")
    
    def _preprocess_resume(self, resume_text: str) -> Dict:
        """Preprocess resume text"""
        cleaned = self._clean_text(resume_text)
        tokens = self._tokenize_and_lemmatize(cleaned)
        
        return {
            'cleaned_text': cleaned,
            'tokens': tokens,
            'token_string': ' '.join(tokens)
        }
    
    def _calculate_tfidf_similarity(self, resume_processed: Dict) -> np.ndarray:
        """Calculate TF-IDF cosine similarity scores"""
        resume_vector = self.tfidf_vectorizer.transform([resume_processed['token_string']])
        similarities = cosine_similarity(resume_vector, self.tfidf_matrix).flatten()
        return similarities
    
    def _calculate_lda_similarity(self, resume_processed: Dict) -> np.ndarray:
        """Calculate LDA topic similarity scores"""
        # Transform resume to topic distribution
        resume_vector = self.lda_vectorizer.transform([resume_processed['token_string']])
        resume_topics = self.lda_model.transform(resume_vector)
        
        # Calculate cosine similarity between topic distributions
        similarities = cosine_similarity(resume_topics, self.job_topics).flatten()
        return similarities
    
    def _calculate_embedding_similarity(self, resume_processed: Dict) -> np.ndarray:
        """Calculate semantic similarity using embeddings"""
        # Generate embedding for resume
        resume_embedding = self.embedding_model.encode([resume_processed['cleaned_text']])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(resume_embedding, self.job_embeddings).flatten()
        return similarities
    
    def match_jobs(
        self,
        resume_text: str,
        top_k: int = 20,
        weights: Dict[str, float] = None
    ) -> List[Dict]:
        """
        Match jobs to a resume using hybrid approach
        
        Args:
            resume_text: Text content of the resume
            top_k: Number of top matches to return
            weights: Dictionary of weights for each method
                    {'tfidf': 0.3, 'lda': 0.2, 'embedding': 0.5}
        
        Returns:
            List of top matching jobs with scores
        """
        # Ensure models are trained
        if self.tfidf_matrix is None:
            print("Models not trained. Training now...")
            self.train_models()
        
        # Default weights if not provided
        if weights is None:
            weights = {
                'tfidf': 0.25,      # Keyword matching
                'lda': 0.25,        # Topic similarity
                'embedding': 0.50   # Semantic similarity (most important)
            }
        
        print("Preprocessing resume...")
        resume_processed = self._preprocess_resume(resume_text)
        
        print("Calculating similarities...")
        # Calculate similarities using each method
        tfidf_scores = self._calculate_tfidf_similarity(resume_processed)
        lda_scores = self._calculate_lda_similarity(resume_processed)
        embedding_scores = self._calculate_embedding_similarity(resume_processed)
        
        # Normalize scores to 0-1 range
        def normalize(scores):
            min_score = scores.min()
            max_score = scores.max()
            if max_score - min_score > 0:
                return (scores - min_score) / (max_score - min_score)
            return scores
        
        tfidf_scores_norm = normalize(tfidf_scores)
        lda_scores_norm = normalize(lda_scores)
        embedding_scores_norm = normalize(embedding_scores)
        
        # Calculate weighted combined score
        combined_scores = (
            weights['tfidf'] * tfidf_scores_norm +
            weights['lda'] * lda_scores_norm +
            weights['embedding'] * embedding_scores_norm
        )
        
        # Get top k indices
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        # Prepare results
        results = []
        for idx in top_indices:
            job = self.processed_jobs[idx]['original']
            results.append({
                'job': job,
                'overall_score': float(combined_scores[idx]),
                'tfidf_score': float(tfidf_scores_norm[idx]),
                'lda_score': float(lda_scores_norm[idx]),
                'embedding_score': float(embedding_scores_norm[idx])
            })
        
        return results
    
    def get_top_lda_topics(self, n_words: int = 10) -> List[List[Tuple[str, float]]]:
        """
        Get top words for each LDA topic
        
        Args:
            n_words: Number of top words per topic
            
        Returns:
            List of topics, each containing top words and their weights
        """
        if self.lda_model is None:
            raise ValueError("LDA model not trained yet")
        
        feature_names = self.lda_vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_indices = topic.argsort()[-n_words:][::-1]
            top_words = [(feature_names[i], topic[i]) for i in top_indices]
            topics.append(top_words)
        
        return topics
    
    def print_topic_summary(self, n_words: int = 10):
        """Print a summary of discovered topics"""
        topics = self.get_top_lda_topics(n_words)
        
        print(f"\n{'='*60}")
        print(f"LDA TOPICS DISCOVERED ({self.n_topics} topics)")
        print(f"{'='*60}\n")
        
        for idx, topic_words in enumerate(topics):
            words = [word for word, _ in topic_words]
            print(f"Topic {idx + 1}: {', '.join(words)}")
        
        print(f"\n{'='*60}\n")


def save_results_to_json(results: List[Dict], output_file: str):
    """Save matching results to a JSON file"""
    # Prepare data for JSON serialization
    output_data = []
    for result in results:
        output_data.append({
            'job_id': result['job']['id'],
            'title': result['job']['title'],
            'company': result['job']['company'],
            'location': result['job']['location'],
            'salary': result['job']['salary'],
            'url': result['job']['url'],
            'scores': {
                'overall': result['overall_score'],
                'tfidf': result['tfidf_score'],
                'lda': result['lda_score'],
                'embedding': result['embedding_score']
            },
            'description': result['job']['description'][:500] + "..."  # Truncate for readability
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")


def print_results(results: List[Dict], top_n: int = 10):
    """Print matching results in a formatted way"""
    print(f"\n{'='*80}")
    print(f"TOP {top_n} MATCHING JOBS")
    print(f"{'='*80}\n")
    
    for i, result in enumerate(results[:top_n], 1):
        job = result['job']
        print(f"{i}. {job['title']} at {job['company']}")
        print(f"   Location: {job['location']}")
        print(f"   Salary: {job.get('salary', 'Not specified')}")
        print(f"   Overall Score: {result['overall_score']:.4f}")
        print(f"   - TF-IDF: {result['tfidf_score']:.4f}")
        print(f"   - LDA: {result['lda_score']:.4f}")
        print(f"   - Embedding: {result['embedding_score']:.4f}")
        print(f"   URL: {job['url']}")
        print(f"   {'-'*76}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    # Example usage
    print("Job Matcher Example")
    print("="*60)
    
    # Initialize matcher
    matcher = JobMatcher('job_results.json', n_topics=20)
    
    # Train models
    matcher.train_models()
    
    # Show discovered topics
    matcher.print_topic_summary(n_words=8)
    
    # Example resume (you would load this from a file)
    example_resume = """
    Senior Data Scientist with 7 years of experience in machine learning, 
    deep learning, and statistical analysis. Expert in Python, R, TensorFlow, 
    PyTorch, and scikit-learn. Experience with large-scale data processing 
    using Spark and cloud platforms (AWS, GCP). Strong background in NLP, 
    computer vision, and predictive modeling. PhD in Computer Science with 
    focus on artificial intelligence. Published researcher with multiple 
    papers in top-tier conferences.
    """
    
    # Match jobs
    results = matcher.match_jobs(example_resume, top_k=20)
    
    # Print results
    print_results(results, top_n=10)
    
    # Save results
    save_results_to_json(results, 'matched_jobs.json')