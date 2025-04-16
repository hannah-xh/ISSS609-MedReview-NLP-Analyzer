import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
from datetime import timedelta

# Download necessary NLTK resources (required for first run)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    nltk.download('stopwords')


# Text preprocessing function
def preprocess_text(text):
    """Preprocess medication review text"""
    if not isinstance(text, str):
        return ""

    # Preserve original case for later extraction of uppercase word features
    original_text = text

    # Convert to lowercase (for main processing)
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Preserve numbers, periods, exclamation marks, etc. (important for medication dosages and emphasis)
    text = re.sub(r'[^\w\s.,!?;:%\-+]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text, original_text


# Enhanced VADER feature extraction function
def extract_enhanced_vader_features(texts):
    """
    Extract an extended set of sentiment features using VADER, adding medication review-specific features
    """
    sid = SentimentIntensityAnalyzer()
    features = []
    feature_names = []

    # Define medication effect keyword lists
    high_effect_words = ['great', 'excellent', 'amazing', 'completely', 'perfect',
                         'effective', 'cured', 'miracle', 'fantastic', 'wonderful']

    medium_effect_words = ['some', 'better', 'helped', 'improved', 'moderate',
                           'decent', 'noticeable', 'somewhat', 'partial', 'ok', 'okay']

    low_effect_words = ['little', "didn't help", "doesn't work", 'useless', 'waste', 'ineffective',
                        'no effect', 'bad', 'terrible', 'awful', 'horrible', 'worst']

    side_effect_words = ['side effect', 'reaction', 'nausea', 'headache', 'dizziness',
                         'vomiting', 'pain', 'rash', 'tired', 'fatigue']

    # Define time-related vocabulary (medication effectiveness time may be relevant to efficacy)
    time_words = ['day', 'week', 'month', 'hour', 'minute', 'year', 'instantly',
                  'quickly', 'immediately', 'fast', 'rapid', 'slow']

    # Build feature name list
    base_feature_names = [
        'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound',
        'sentiment_diff', 'certainty_ratio', 'sentiment_strength',
        'exclamation_count', 'question_count', 'uppercase_words_count',
        'high_effect_words', 'medium_effect_words', 'low_effect_words',
        'side_effect_words', 'time_words_count',
        'sentence_count', 'word_count', 'avg_sentence_length',
        'numeric_count'  # Frequency of numbers, may indicate dosage information
    ]

    feature_names.extend(base_feature_names)

    for text_idx, text in enumerate(tqdm(texts, desc="Extracting VADER features")):
        if not isinstance(text, str) or not text.strip():
            # Handle empty text
            features.append([0] * len(base_feature_names))
            continue

        # Get original and preprocessed text
        processed_text, original_text = text, text  # Assuming input is already preprocessed

        # Basic VADER sentiment scores
        scores = sid.polarity_scores(processed_text)

        # Calculate derived features
        sentiment_diff = scores['pos'] - scores['neg']
        certainty_ratio = scores['neu'] / (scores['pos'] + scores['neg'] + 0.01)
        sentiment_strength = abs(scores['compound'])

        # Text statistical features
        exclamation_count = processed_text.count('!')
        question_count = processed_text.count('?')
        uppercase_words_count = len(re.findall(r'\b[A-Z]{2,}\b', original_text))

        # Calculate keyword occurrence counts
        high_effect_count = sum(word in processed_text for word in high_effect_words)
        medium_effect_count = sum(word in processed_text for word in medium_effect_words)
        low_effect_count = sum(word in processed_text for word in low_effect_words)
        side_effect_count = sum(word in processed_text for word in side_effect_words)
        time_words_count = sum(word in processed_text for word in time_words)

        # Sentence and word statistics
        sentences = re.split(r'[.!?]+', processed_text)
        sentence_count = len([s for s in sentences if s.strip()])
        words = processed_text.split()
        word_count = len(words)
        avg_sentence_length = word_count / max(1, sentence_count)

        # Number occurrence count (may indicate dosage, etc.)
        numeric_count = len(re.findall(r'\d+', processed_text))

        # Combine all features
        feature_vector = [
            scores['neg'], scores['neu'], scores['pos'], scores['compound'],
            sentiment_diff, certainty_ratio, sentiment_strength,
            exclamation_count, question_count, uppercase_words_count,
            high_effect_count, medium_effect_count, low_effect_count,
            side_effect_count, time_words_count,
            sentence_count, word_count, avg_sentence_length,
            numeric_count
        ]

        features.append(feature_vector)

    return np.array(features), feature_names


# Add TF-IDF features (optional)
def add_tfidf_features(train_texts, val_texts, test_texts, max_features=100):
    """Add TF-IDF features to capture important words in reviews"""
    print("Extracting TF-IDF features...")

    # Initialize TF-IDF vectorizer, limit feature count to avoid dimension explosion
    tfidf_vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=5,  # Must appear in at least 5 documents
        max_df=0.7,  # Must appear in at most 70% of documents
        ngram_range=(1, 2)  # Include single words and two-word combinations
    )

    # Fit and transform on training set
    train_tfidf = tfidf_vectorizer.fit_transform(train_texts).toarray()

    # Transform validation and test sets
    val_tfidf = tfidf_vectorizer.transform(val_texts).toarray()
    test_tfidf = tfidf_vectorizer.transform(test_texts).toarray()

    # Get feature names
    tfidf_feature_names = [f'tfidf_{word}' for word in tfidf_vectorizer.get_feature_names_out()]

    return train_tfidf, val_tfidf, test_tfidf, tfidf_feature_names, tfidf_vectorizer


# Train VADER + Random Forest model
def train_enhanced_vader_model(train_df, val_df, use_tfidf=True, tfidf_features=100, tune_hyperparams=False):
    """
    Train a Random Forest classification model based on enhanced VADER features
    """
    start_time = time.time()

    # Preprocess text data
    print("Preprocessing text data...")
    train_texts = train_df['review'].values
    val_texts = val_df['review'].values

    # Extract VADER features
    print("Extracting enhanced VADER features...")
    train_vader_features, feature_names = extract_enhanced_vader_features(train_texts)
    val_vader_features, _ = extract_enhanced_vader_features(val_texts)

    # Combine with TF-IDF features (optional)
    if use_tfidf:
        train_tfidf, val_tfidf, _, tfidf_feature_names, tfidf_vectorizer = add_tfidf_features(
            train_texts, val_texts, val_texts, max_features=tfidf_features
        )

        # Merge features
        print(f"Combining VADER features({train_vader_features.shape[1]}) and TF-IDF features({train_tfidf.shape[1]})...")
        train_features = np.hstack((train_vader_features, train_tfidf))
        val_features = np.hstack((val_vader_features, val_tfidf))

        # Merge feature names
        feature_names.extend(tfidf_feature_names)

        # Save TF-IDF vectorizer
        joblib.dump(tfidf_vectorizer, 'vader_rf_results/enhanced_vader_tfidf_vectorizer.pkl')
    else:
        train_features = train_vader_features
        val_features = val_vader_features

    # Get labels
    train_labels = train_df['effectiveness']
    val_labels = val_df['effectiveness']

    # Print data dimensions
    print(f"Training features dimension: {train_features.shape}")
    print(f"Validation features dimension: {val_features.shape}")

    # Calculate class weights (handle class imbalance)
    class_counts = train_df['effectiveness'].value_counts()
    total_samples = len(train_df)
    class_weights = {
        label: total_samples / (len(class_counts) * count)
        for label, count in class_counts.items()
    }
    print(f"Class weights: {class_weights}")

    # Train Random Forest classifier
    if tune_hyperparams:
        print("Using grid search to optimize hyperparameters...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        clf = GridSearchCV(
            RandomForestClassifier(random_state=42, class_weight='balanced'),
            param_grid,
            cv=3,
            scoring='accuracy',
            verbose=1,
            n_jobs=-1
        )

        clf.fit(train_features, train_labels)
        best_params = clf.best_params_
        print(f"Best hyperparameters: {best_params}")

        # Retrain with best parameters
        clf = clf.best_estimator_
    else:
        print("Training Random Forest classifier...")
        clf = RandomForestClassifier(
            n_estimators=200,  # More trees for better stability
            max_depth=None,  # Allow trees to fully grow
            min_samples_split=5,  # Prevent overfitting
            min_samples_leaf=2,  # Prevent overfitting
            max_features='sqrt',  # Use feature subset at each node
            class_weight='balanced',  # Handle class imbalance
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )

        clf.fit(train_features, train_labels)

    # Evaluate on validation set
    val_predictions = clf.predict(val_features)
    val_accuracy = accuracy_score(val_labels, val_predictions)
    print(f"Validation accuracy: {val_accuracy:.4f}")

    # Feature importance visualization
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': clf.feature_importances_
    }).sort_values('Importance', ascending=False)

    # Show top 20 most important features
    top_n = min(20, len(feature_importance))
    print(f"Top {top_n} feature importance:")
    print(feature_importance.head(top_n))

    # Visualize feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(top_n))
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    plt.savefig('vader_rf_results/enhanced_vader_feature_importance.png')
    plt.close()

    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(val_labels, val_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not effective(0)', 'Moderately effective(1)', 'Effective(2)'],
                yticklabels=['Not effective(0)', 'Moderately effective(1)', 'Effective(2)'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Validation Confusion Matrix')
    plt.tight_layout()
    plt.savefig('vader_rf_results/enhanced_vader_validation_confusion_matrix.png')
    plt.close()

    # Save model and feature names
    model_data = {
        'model': clf,
        'feature_names': feature_names,
        'use_tfidf': use_tfidf
    }
    joblib.dump(model_data, 'vader_rf_results/enhanced_vader_rf_model.pkl')

    # Calculate training time
    training_time = time.time() - start_time
    print(f"Model training completed, time taken: {timedelta(seconds=int(training_time))}")

    return clf, feature_names, val_accuracy


# Evaluate model on test set
def evaluate_enhanced_vader_model(model_data, test_df):
    """Evaluate model on test set and generate detailed report"""
    start_time = time.time()

    # Unpack model data
    clf = model_data['model']
    feature_names = model_data['feature_names']
    use_tfidf = model_data['use_tfidf']

    # Preprocess text
    print("Preprocessing test set text...")
    test_texts = test_df['review'].values

    # Extract VADER features
    print("Extracting test set VADER features...")
    test_vader_features, _ = extract_enhanced_vader_features(test_texts)

    # If using TF-IDF, need to load vectorizer and extract features
    if use_tfidf:
        print("Extracting test set TF-IDF features...")
        tfidf_vectorizer = joblib.load('vader_rf_results/enhanced_vader_tfidf_vectorizer.pkl')
        test_tfidf = tfidf_vectorizer.transform(test_texts).toarray()

        # Merge features
        test_features = np.hstack((test_vader_features, test_tfidf))
    else:
        test_features = test_vader_features

    # Get labels
    test_labels = test_df['effectiveness']

    # Generate predictions
    test_predictions = clf.predict(test_features)

    # Get prediction probabilities for analysis
    test_proba = clf.predict_proba(test_features)

    # Evaluation
    test_accuracy = accuracy_score(test_labels, test_predictions)
    print(f"Test set accuracy: {test_accuracy:.4f}")

    # Generate classification report
    print("\nClassification report:")
    target_names = ['Not effective(0)', 'Moderately effective(1)', 'Effective(2)']
    report = classification_report(test_labels, test_predictions,
                                   target_names=target_names,
                                   digits=3)
    print(report)

    # More detailed classification report (dictionary form)
    report_dict = classification_report(test_labels, test_predictions,
                                        target_names=target_names,
                                        output_dict=True)

    # Convert to DataFrame for better visualization
    report_df = pd.DataFrame(report_dict).transpose()

    # Confusion matrix visualization
    cm = confusion_matrix(test_labels, test_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Enhanced VADER+RF Model Confusion Matrix')
    plt.tight_layout()
    plt.savefig('vader_rf_results/enhanced_vader_test_confusion_matrix.png')
    plt.close()

    # Analyze middle class ("Moderately effective") predictions
    middle_class_idx = np.where(test_labels == 1)[0]
    middle_class_correct = np.sum(test_predictions[middle_class_idx] == 1)
    middle_class_accuracy = middle_class_correct / len(middle_class_idx) if len(middle_class_idx) > 0 else 0

    print(f"\nMiddle class ('Moderately effective') prediction analysis:")
    print(f"Total samples: {len(middle_class_idx)}")
    print(f"Correctly predicted: {middle_class_correct}")
    print(f"Accuracy: {middle_class_accuracy:.4f}")

    # Analyze incorrectly predicted cases
    error_indices = np.where(test_predictions != test_labels)[0]

    if len(error_indices) > 0:
        print(f"\nError prediction analysis (first 10):")
        for i, idx in enumerate(error_indices[:10]):
            print(f"Sample {idx}:")
            print(f"  Text: {test_texts[idx][:100]}...")
            print(f"  True label: {test_labels.iloc[idx]} ('{target_names[test_labels.iloc[idx]]}')")
            print(f"  Predicted label: {test_predictions[idx]} ('{target_names[test_predictions[idx]]}')")
            print(f"  Prediction probabilities: {test_proba[idx]}")
            print()

    # Calculate evaluation time
    eval_time = time.time() - start_time
    print(f"Evaluation completed, time taken: {timedelta(seconds=int(eval_time))}")

    # Save prediction results
    results_df = pd.DataFrame({
        'review': test_texts,
        'true_label': test_labels,
        'predicted_label': test_predictions,
        'prob_class_0': test_proba[:, 0],
        'prob_class_1': test_proba[:, 1],
        'prob_class_2': test_proba[:, 2]
    })

    results_df.to_csv('vader_rf_results/enhanced_vader_test_predictions.csv', index=False)

    return test_accuracy, report_df


# Main function
def main():
    """Main function: load data, train model, evaluate model"""

    # Load datasets
    def load_datasets(base_dir='dataset'):
        print("Loading datasets...")
        train_path = os.path.join(base_dir, "train_drug_reviews2.csv")
        val_path = os.path.join(base_dir, "val_drug_reviews2.csv")
        test_path = os.path.join(base_dir, "test_drug_reviews2.csv")

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        # Only select needed columns
        train_df = train_df[['review', 'effectiveness']]
        val_df = val_df[['review', 'effectiveness']]
        test_df = test_df[['review', 'effectiveness']]

        # Check data distribution
        print(f"Training set samples: {len(train_df)}")
        print(f"Validation set samples: {len(val_df)}")
        print(f"Test set samples: {len(test_df)}")

        print("\nLabel distribution:")
        print("Training set:", train_df['effectiveness'].value_counts().sort_index())
        print("Validation set:", val_df['effectiveness'].value_counts().sort_index())
        print("Test set:", test_df['effectiveness'].value_counts().sort_index())

        return train_df, val_df, test_df

    # Load data
    train_df, val_df, test_df = load_datasets()

    # Set parameters
    use_tfidf = True  # Whether to use TF-IDF features
    tfidf_features = 100  # Number of TF-IDF features
    tune_hyperparams = False  # Whether to use grid search to optimize hyperparameters

    # Train enhanced VADER+Random Forest model
    print("\nTraining enhanced VADER+Random Forest model...")
    clf, feature_names, val_accuracy = train_enhanced_vader_model(
        train_df, val_df,
        use_tfidf=use_tfidf,
        tfidf_features=tfidf_features,
        tune_hyperparams=tune_hyperparams
    )

    # Load saved model data
    print("\nLoading best model...")
    model_data = joblib.load('vader_rf_results/enhanced_vader_rf_model.pkl')

    # Evaluate on test set
    print("\nEvaluating model on test set...")
    test_accuracy, report_df = evaluate_enhanced_vader_model(model_data, test_df)

    print("\nFinal results:")
    print(f"Validation set accuracy: {val_accuracy:.4f}")
    print(f"Test set accuracy: {test_accuracy:.4f}")
    print("\nDetailed classification report:")
    print(report_df)

    return model_data, val_accuracy, test_accuracy, report_df


# Test single sample
def predict_sample(text, model_data=None):
    """Predict the effectiveness of a single medication review"""
    if model_data is None:
        model_data = joblib.load('vader_rf_results/enhanced_vader_rf_model.pkl')

    clf = model_data['model']
    use_tfidf = model_data['use_tfidf']

    # Preprocess text
    processed_text, _ = preprocess_text(text)

    # Extract VADER features
    sample_features, _ = extract_enhanced_vader_features([processed_text])

    # If using TF-IDF, need to load vectorizer and extract features
    if use_tfidf:
        tfidf_vectorizer = joblib.load('vader_rf_results/enhanced_vader_tfidf_vectorizer.pkl')
        sample_tfidf = tfidf_vectorizer.transform([processed_text]).toarray()

        # Merge features
        sample_features = np.hstack((sample_features, sample_tfidf))

    # Get prediction and probabilities
    prediction = clf.predict(sample_features)[0]
    probabilities = clf.predict_proba(sample_features)[0]

    # Map prediction results
    class_names = ['Not effective(0)', 'Moderately effective(1)', 'Effective(2)']
    result = {
        'prediction': prediction,
        'class_name': class_names[prediction],
        'probabilities': {
            class_names[i]: round(prob * 100, 2)
            for i, prob in enumerate(probabilities)
        }
    }

    return result


if __name__ == "__main__":
    main()

    # Test predicting a single sample
    sample_text = "This medication helped a little bit with my symptoms, but not as much as I had hoped. I still experience some pain but it's better than before."
    result = predict_sample(sample_text)
    print("\nSample prediction test:")
    print(f"Text: {sample_text}")
    print(f"Predicted class: {result['class_name']}")
    print(f"Prediction probabilities: {result['probabilities']}")