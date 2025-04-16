import pandas as pd
import numpy as np
import os
import re
import sys
import nltk
import importlib
from textblob import TextBlob
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
import warnings

warnings.filterwarnings("ignore")

# Create output directories
os.makedirs('textblob_rf_results', exist_ok=True)
os.makedirs('vader_rf_results', exist_ok=True)


def download_required_corpora():
    """Download all required NLTK and TextBlob corpora"""
    print("Checking and downloading required NLTK and TextBlob resources...")

    # Required NLTK resources
    resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'punkt_tab', 'wordnet']

    for resource in resources:
        try:
            nltk.data.find(f'{resource}')
            print(f"✓ {resource} already downloaded")
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource)

    # For TextBlob corpus
    try:
        import subprocess
        print("Ensuring TextBlob corpora is downloaded...")
        subprocess.check_call([sys.executable, "-m", "textblob.download_corpora"])
        print("✓ TextBlob corpora download complete")
    except Exception as e:
        print(f"Warning: TextBlob corpora download error: {e}")
        print("You may need to manually run: python -m textblob.download_corpora")


# Download required resources at startup
import sys

download_required_corpora()


# Text preprocessing function
def preprocess_text(text):
    """Preprocess drug review text, preserving more information for TextBlob analysis"""
    if not isinstance(text, str):
        return ""

    # Keep original case for TextBlob POS tagging
    original_text = text

    # Convert to lowercase (for certain features only)
    text_lower = text.lower()

    # Remove URLs
    text_clean = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text_clean = re.sub(r'<.*?>', '', text_clean)

    # Preserve numbers, punctuation (important for drug dosages, emphasis)
    text_clean = re.sub(r'[^\w\s.,!?;:%\-+()]', ' ', text_clean)

    # Remove extra spaces
    text_clean = re.sub(r'\s+', ' ', text_clean).strip()

    return text_clean


# TextBlob feature extraction function
def extract_textblob_features(texts):
    """
    Extract text features using TextBlob, including sentiment, grammar, and language features
    """
    features = []
    feature_names = []

    # Add error handling for TextBlob processing
    def safe_textblob_process(text):
        try:
            return TextBlob(text)
        except Exception as e:
            print(f"Warning: TextBlob processing error: {e}")
            # Return an empty TextBlob object as fallback
            return TextBlob("")

    # Define drug effect keyword lists
    high_effect_words = ['great', 'excellent', 'amazing', 'completely', 'perfect',
                         'effective', 'cured', 'miracle', 'fantastic', 'wonderful']

    medium_effect_words = ['some', 'better', 'helped', 'improved', 'moderate',
                           'decent', 'noticeable', 'somewhat', 'partial', 'ok', 'okay']

    low_effect_words = ['little', "didn't help", "doesn't work", 'useless', 'waste', 'ineffective',
                        'no effect', 'bad', 'terrible', 'awful', 'horrible', 'worst']

    side_effect_words = ['side effect', 'reaction', 'nausea', 'headache', 'dizziness',
                         'vomiting', 'pain', 'rash', 'tired', 'fatigue']

    # Build feature name list
    base_feature_names = [
        'polarity', 'subjectivity',  # Basic TextBlob sentiment scores
        'polarity_abs', 'polarity_sign',  # Polarity derivative features
        'noun_count', 'verb_count', 'adj_count', 'adv_count',  # POS counts
        'sentence_count', 'word_count', 'avg_sentence_length',  # Text statistics
        'question_count', 'exclamation_count',  # Question and exclamation counts
        'capitalized_word_ratio',  # Ratio of capitalized words
        'high_effect_words', 'medium_effect_words', 'low_effect_words',  # Effect word counts
        'side_effect_words',  # Side effect word counts
        'numeric_count',  # Number count (dosage related)
        'comparative_count', 'superlative_count'  # Comparative and superlative counts
    ]

    feature_names.extend(base_feature_names)

    for text_idx, text in enumerate(tqdm(texts, desc="Extracting TextBlob features")):
        if not isinstance(text, str) or not text.strip():
            # Handle empty text
            features.append([0] * len(base_feature_names))
            continue

        # TextBlob analysis with error handling
        blob = safe_textblob_process(text)

        # Basic TextBlob features
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # Polarity derivative features
        polarity_abs = abs(polarity)
        polarity_sign = 1 if polarity > 0 else (-1 if polarity < 0 else 0)

        # POS statistics (nouns, verbs, adjectives, adverbs)
        pos_tags = blob.tags
        noun_count = len([tag for word, tag in pos_tags if tag.startswith('NN')])
        verb_count = len([tag for word, tag in pos_tags if tag.startswith('VB')])
        adj_count = len([tag for word, tag in pos_tags if tag.startswith('JJ')])
        adv_count = len([tag for word, tag in pos_tags if tag.startswith('RB')])

        # Comparative and superlative counts
        comparative_count = len([tag for word, tag in pos_tags if tag == 'JJR' or tag == 'RBR'])
        superlative_count = len([tag for word, tag in pos_tags if tag == 'JJS' or tag == 'RBS'])

        # Sentence and word statistics
        sentence_count = len(blob.sentences)
        word_count = len(blob.words)
        avg_sentence_length = word_count / max(1, sentence_count)

        # Question and exclamation counts
        question_count = len([s for s in blob.sentences if str(s).strip().endswith('?')])
        exclamation_count = len([s for s in blob.sentences if str(s).strip().endswith('!')])

        # Capitalized word statistics
        capitalized_words = len([word for word in blob.words if word[0].isupper() and len(word) > 1])
        capitalized_word_ratio = capitalized_words / max(1, word_count)

        # Keyword counts
        text_lower = text.lower()
        high_effect_count = sum(word in text_lower for word in high_effect_words)
        medium_effect_count = sum(word in text_lower for word in medium_effect_words)
        low_effect_count = sum(word in text_lower for word in low_effect_words)
        side_effect_count = sum(word in text_lower for word in side_effect_words)

        # Number count (may indicate dosage)
        numeric_count = len(re.findall(r'\d+', text))

        # Combine all features
        feature_vector = [
            polarity, subjectivity,
            polarity_abs, polarity_sign,
            noun_count, verb_count, adj_count, adv_count,
            sentence_count, word_count, avg_sentence_length,
            question_count, exclamation_count,
            capitalized_word_ratio,
            high_effect_count, medium_effect_count, low_effect_count,
            side_effect_count,
            numeric_count,
            comparative_count, superlative_count
        ]

        features.append(feature_vector)

    return np.array(features), feature_names


# Add TF-IDF features (optional)
def add_tfidf_features(train_texts, val_texts, test_texts, max_features=100):
    """Add TF-IDF features to capture important words in reviews"""
    print("Extracting TF-IDF features...")

    # Initialize TF-IDF vectorizer, limit feature count to avoid dimensionality explosion
    tfidf_vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=5,  # Must appear in at least 5 documents
        max_df=0.7,  # Must appear in at most 70% of documents
        ngram_range=(1, 2)  # Include single words and word pairs
    )

    # Fit and transform on training set
    train_tfidf = tfidf_vectorizer.fit_transform(train_texts).toarray()

    # Transform validation and test sets
    val_tfidf = tfidf_vectorizer.transform(val_texts).toarray()
    test_tfidf = tfidf_vectorizer.transform(test_texts).toarray()

    # Get feature names
    tfidf_feature_names = [f'tfidf_{word}' for word in tfidf_vectorizer.get_feature_names_out()]

    return train_tfidf, val_tfidf, test_tfidf, tfidf_feature_names, tfidf_vectorizer


# Train TextBlob+Random Forest model
def train_textblob_model(train_df, val_df, use_tfidf=True, tfidf_features=100, tune_hyperparams=False):
    """
    Train Random Forest classification model based on TextBlob features
    """
    start_time = time.time()

    # Preprocess text data
    print("Preprocessing text data...")
    train_texts = train_df['review'].apply(preprocess_text).values
    val_texts = val_df['review'].apply(preprocess_text).values

    # Extract TextBlob features
    print("Extracting TextBlob features...")
    train_textblob_features, feature_names = extract_textblob_features(train_texts)
    val_textblob_features, _ = extract_textblob_features(val_texts)

    # Combine with TF-IDF features (optional)
    if use_tfidf:
        train_tfidf, val_tfidf, _, tfidf_feature_names, tfidf_vectorizer = add_tfidf_features(
            train_texts, val_texts, val_texts, max_features=tfidf_features
        )

        # Merge features
        print(
            f"Combining TextBlob features({train_textblob_features.shape[1]}) and TF-IDF features({train_tfidf.shape[1]})...")
        train_features = np.hstack((train_textblob_features, train_tfidf))
        val_features = np.hstack((val_textblob_features, val_tfidf))

        # Merge feature names
        feature_names.extend(tfidf_feature_names)

        # Save TF-IDF vectorizer
        joblib.dump(tfidf_vectorizer, 'textblob_rf_results/textblob_tfidf_vectorizer.pkl')
    else:
        train_features = train_textblob_features
        val_features = val_textblob_features

    # Get labels
    train_labels = train_df['effectiveness']
    val_labels = val_df['effectiveness']

    # Print data dimensions
    print(f"Training features shape: {train_features.shape}")
    print(f"Validation features shape: {val_features.shape}")

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
            n_estimators=200,  # More trees for stability
            max_depth=None,  # Allow trees to grow fully
            min_samples_split=5,  # Prevent overfitting
            min_samples_leaf=2,  # Prevent overfitting
            max_features='sqrt',  # Use subset of features at each node
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
    plt.savefig('textblob_rf_results/textblob_feature_importance.png')
    plt.close()

    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(val_labels, val_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Effective(0)', 'Moderately Effective(1)', 'Effective(2)'],
                yticklabels=['Not Effective(0)', 'Moderately Effective(1)', 'Effective(2)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Validation Confusion Matrix')
    plt.tight_layout()
    plt.savefig('textblob_rf_results/textblob_validation_confusion_matrix.png')
    plt.close()

    # Save model and feature names
    model_data = {
        'model': clf,
        'feature_names': feature_names,
        'use_tfidf': use_tfidf
    }
    joblib.dump(model_data, 'textblob_rf_results/textblob_rf_model.pkl')

    # Calculate training time
    training_time = time.time() - start_time
    print(f"Model training completed, time elapsed: {timedelta(seconds=int(training_time))}")

    return clf, feature_names, val_accuracy


# Evaluate model on test set
def evaluate_textblob_model(model_data, test_df):
    """Evaluate model on test set and generate detailed report"""
    start_time = time.time()

    # Unpack model data
    clf = model_data['model']
    feature_names = model_data['feature_names']
    use_tfidf = model_data['use_tfidf']

    # Preprocess text
    print("Preprocessing test set text...")
    test_texts = test_df['review'].apply(preprocess_text).values

    # Extract TextBlob features
    print("Extracting test set TextBlob features...")
    test_textblob_features, _ = extract_textblob_features(test_texts)

    # If using TF-IDF, load vectorizer and extract features
    if use_tfidf:
        print("Extracting test set TF-IDF features...")
        tfidf_vectorizer = joblib.load('textblob_rf_results/textblob_tfidf_vectorizer.pkl')
        test_tfidf = tfidf_vectorizer.transform(test_texts).toarray()

        # Merge features
        test_features = np.hstack((test_textblob_features, test_tfidf))
    else:
        test_features = test_textblob_features

    # Get labels
    test_labels = test_df['effectiveness']

    # Generate predictions
    test_predictions = clf.predict(test_features)

    # Get prediction probabilities for analysis
    test_proba = clf.predict_proba(test_features)

    # Evaluate
    test_accuracy = accuracy_score(test_labels, test_predictions)
    print(f"Test set accuracy: {test_accuracy:.4f}")

    # Generate classification report
    print("\nClassification Report:")
    target_names = ['Not Effective(0)', 'Moderately Effective(1)', 'Effective(2)']
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
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('TextBlob+RF Model Confusion Matrix')
    plt.tight_layout()
    plt.savefig('textblob_rf_results/textblob_test_confusion_matrix.png')
    plt.close()

    # Analyze middle class ("Moderately Effective") predictions
    middle_class_idx = np.where(test_labels == 1)[0]
    middle_class_correct = np.sum(test_predictions[middle_class_idx] == 1)
    middle_class_accuracy = middle_class_correct / len(middle_class_idx) if len(middle_class_idx) > 0 else 0

    print(f"\nMiddle class ('Moderately Effective') prediction analysis:")
    print(f"Total samples: {len(middle_class_idx)}")
    print(f"Correctly predicted: {middle_class_correct}")
    print(f"Accuracy: {middle_class_accuracy:.4f}")

    # Analyze error cases
    error_indices = np.where(test_predictions != test_labels)[0]

    if len(error_indices) > 0:
        print(f"\nError prediction analysis (first 10):")
        for i, idx in enumerate(error_indices[:10]):
            print(f"Sample {idx}:")
            print(f"  Text: {test_texts[idx][:100]}...")
            print(f"  True Label: {test_labels.iloc[idx]} ('{target_names[test_labels.iloc[idx]]}')")
            print(f"  Predicted Label: {test_predictions[idx]} ('{target_names[test_predictions[idx]]}')")
            print(f"  Prediction Probabilities: {test_proba[idx]}")
            print()

    # Calculate evaluation time
    eval_time = time.time() - start_time
    print(f"Evaluation completed, time elapsed: {timedelta(seconds=int(eval_time))}")

    # Save prediction results
    results_df = pd.DataFrame({
        'review': test_texts,
        'true_label': test_labels,
        'predicted_label': test_predictions,
        'prob_class_0': test_proba[:, 0],
        'prob_class_1': test_proba[:, 1],
        'prob_class_2': test_proba[:, 2]
    })

    results_df.to_csv('textblob_rf_results/textblob_test_predictions.csv', index=False)

    return test_accuracy, report_df


# Main function
def main():
    """Main function: load data, train model, evaluate model"""

    # Ensure all required NLTK and TextBlob resources are available
    download_required_corpora()

    # Load datasets
    def load_datasets(base_dir='dataset'):
        print("Loading datasets...")
        train_path = os.path.join(base_dir, "train_drug_reviews2.csv")
        val_path = os.path.join(base_dir, "val_drug_reviews2.csv")
        test_path = os.path.join(base_dir, "test_drug_reviews2.csv")

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        # Select only needed columns
        train_df = train_df[['review', 'effectiveness']]
        val_df = val_df[['review', 'effectiveness']]
        test_df = test_df[['review', 'effectiveness']]

        # Check data distribution
        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        print(f"Test samples: {len(test_df)}")

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

    # Train TextBlob+Random Forest model
    print("\nTraining TextBlob+Random Forest model...")
    clf, feature_names, val_accuracy = train_textblob_model(
        train_df, val_df,
        use_tfidf=use_tfidf,
        tfidf_features=tfidf_features,
        tune_hyperparams=tune_hyperparams
    )

    # Load saved model data
    print("\nLoading best model...")
    model_data = joblib.load('textblob_rf_results/textblob_rf_model.pkl')

    # Evaluate on test set
    print("\nEvaluating model on test set...")
    test_accuracy, report_df = evaluate_textblob_model(model_data, test_df)

    print("\nFinal results:")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("\nDetailed classification report:")
    print(report_df)

    return model_data, val_accuracy, test_accuracy, report_df


# Compare VADER and TextBlob method results
def compare_models(vader_results=None, textblob_results=None):
    """Compare VADER and TextBlob methods' performance on drug effectiveness classification"""

    if vader_results is None:
        try:
            # Try to load VADER results
            vader_model = joblib.load('vader_rf_results/enhanced_vader_rf_model.pkl')
            vader_results = pd.read_csv('vader_rf_results/enhanced_vader_test_predictions.csv')
        except:
            print("VADER model results not found, will only show TextBlob results")

    if textblob_results is None:
        try:
            # Try to load TextBlob results
            textblob_model = joblib.load('textblob_rf_results/textblob_rf_model.pkl')
            textblob_results = pd.read_csv('textblob_rf_results/textblob_test_predictions.csv')
        except:
            print("TextBlob model results not found, will only show VADER results")

    # If both model results exist, compare them
    if vader_results is not None and textblob_results is not None:
        # Calculate accuracy for each class
        vader_acc = accuracy_score(vader_results['true_label'], vader_results['predicted_label'])
        textblob_acc = accuracy_score(textblob_results['true_label'], textblob_results['predicted_label'])

        # Per-class accuracy
        classes = [0, 1, 2]
        class_names = ['Not Effective(0)', 'Moderately Effective(1)', 'Effective(2)']

        vader_class_acc = []
        textblob_class_acc = []

        for cls in classes:
            # VADER
            vader_cls_idx = vader_results['true_label'] == cls
            vader_correct = sum(vader_results.loc[vader_cls_idx, 'predicted_label'] == cls)
            vader_total = sum(vader_cls_idx)
            vader_class_acc.append(vader_correct / vader_total if vader_total > 0 else 0)

            # TextBlob
            textblob_cls_idx = textblob_results['true_label'] == cls
            textblob_correct = sum(textblob_results.loc[textblob_cls_idx, 'predicted_label'] == cls)
            textblob_total = sum(textblob_cls_idx)
            textblob_class_acc.append(textblob_correct / textblob_total if textblob_total > 0 else 0)

        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Class': class_names,
            'VADER Accuracy': vader_class_acc,
            'TextBlob Accuracy': textblob_class_acc
        })

        # Add overall accuracy
        comparison_df = pd.concat([
            comparison_df,
            pd.DataFrame({
                'Class': ['Overall'],
                'VADER Accuracy': [vader_acc],
                'TextBlob Accuracy': [textblob_acc]
            })
        ])

        print("\nModel Comparison:")
        print(comparison_df)

        # Visualize comparison
        plt.figure(figsize=(10, 6))

        # Set up bar chart
        x = np.arange(len(comparison_df))
        width = 0.35

        plt.bar(x - width / 2, comparison_df['VADER Accuracy'], width, label='VADER+RF')
        plt.bar(x + width / 2, comparison_df['TextBlob Accuracy'], width, label='TextBlob+RF')

        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.title('VADER vs TextBlob Drug Effectiveness Classification Accuracy Comparison')
        plt.xticks(x, comparison_df['Class'])
        plt.ylim(0, 1.0)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig('vader_rf_results/vader_vs_textblob_comparison.png')
        plt.close()

        return comparison_df

    return None


# Test single sample
def predict_sample(text, model_data=None):
    """Predict the effectiveness of a single drug review"""
    if model_data is None:
        model_data = joblib.load('textblob_rf_results/textblob_rf_model.pkl')

    clf = model_data['model']
    use_tfidf = model_data['use_tfidf']

    # Preprocess text
    processed_text = preprocess_text(text)

    # Extract TextBlob features
    sample_features, _ = extract_textblob_features([processed_text])

    # If using TF-IDF, load vectorizer and extract features
    if use_tfidf:
        tfidf_vectorizer = joblib.load('textblob_rf_results/textblob_tfidf_vectorizer.pkl')
        sample_tfidf = tfidf_vectorizer.transform([processed_text]).toarray()

        # Merge features
        sample_features = np.hstack((sample_features, sample_tfidf))

    # Get prediction and probabilities
    prediction = clf.predict(sample_features)[0]
    probabilities = clf.predict_proba(sample_features)[0]

    # Map prediction results
    class_names = ['Not Effective(0)', 'Moderately Effective(1)', 'Effective(2)']
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

    # If both models have been trained, compare their performance
    try:
        compare_models()
    except:
        pass