# Text Analytics for Medication Feedback: A Multi-Model Approach to Understanding Patient Reviews

## Project Overview
This project implements a comprehensive text analytics solution to extract insights from patient-generated medication reviews. Using the Drug Review dataset from Kaggle (215,063 entries across 7 columns), we developed three complementary analytical components:

1. **Topic Modeling** - Identifying common themes in patient reviews
2. **Sentiment Analysis** - Evaluating patient satisfaction through multiple modeling approaches
3. **Review Retrieval** - Creating a search system for finding relevant medication reviews

## Dataset
The dataset contains patient reviews of specific drugs, associated medical conditions, and a 10-star rating reflecting overall patient satisfaction. The dataset includes:
- Review Number
- Patient ID
- Medical condition
- Review text
- Rating (1-10)
- Date
- Usefulness count
- Review length

## Methodology

### Data Preprocessing
- Removed missing values, HTML tags, special characters, and excess whitespace
- Implemented task-specific preprocessing:
  - Topic Modeling: Removed stopwords, filler words, and domain-specific terms
  - Sentiment Analysis: Applied lemmatization to standardize word forms
  - Review Retrieval: Minimal cleaning to preserve lexical richness

### Topic Modeling
- Implemented Latent Dirichlet Allocation (LDA) using Gensim
- Used coherence scores to determine the optimal number of topics (13)
- Trained the model with 13 topics and 10 passes
- Extracted top keywords per topic and visualized with word clouds
- Manually labeled topics and assigned each review to its dominant topic

### Sentiment Analysis
We implemented three distinct approaches to sentiment analysis:

#### 1. Traditional Machine Learning Models
- Converted ratings to three sentiment classes:
  - 1-4: Not Effective (24.9%)
  - 5-6: Moderately Effective (14.7%)
  - 7-10: Effective (60.4%)
- Applied TF-IDF vectorization with 5,000 most informative terms
- Evaluated multiple classifiers:
  - Naive Bayes (73% accuracy)
  - Logistic Regression (78% accuracy)
  - Linear SVM (78% accuracy)
  - Random Forest (88.15% accuracy)
- Conducted feature importance analysis to identify key predictive terms
- Performed error analysis, particularly on misclassification patterns

#### 2. Pre-trained Models
- Implemented BERT for sentiment classification
- Achieved 88.65% test accuracy and 88.7% validation accuracy
- Performed well on polar classes (Effective: F1 0.942, Not Effective: F1 0.856)
- Analyzed attention weights to identify highly predictive tokens
- Identified limitations with classifying the "Moderately Effective" class

#### 3. Lexicon-based Models
- Implemented VADER and TextBlob approaches
- VADER: 86.41% accuracy (86.57% validation)
- TextBlob: 86.24% accuracy (86.45% validation)
- Developed hybrid approaches combining lexicon scores with Random Forest
- Identified polarity and compound sentiment as the most influential features

### Review Retrieval
- Developed an interactive search tool using TF-IDF vectorization
- Implemented cosine similarity for relevance ranking
- Created filters for medical conditions and search keywords
- Added keyword highlighting for improved readability
- Built a responsive interface using ipywidgets

## Results and Analysis

### Topic Modeling
- Identified 13 topics covering areas such as:
  - Side effects and symptoms
  - Treatment effectiveness
  - Emotional impact
  - Sleep issues
  - Weight changes
  - Birth control and menstruation
- Visualized topic distribution across reviews
- Created word clouds for intuitive interpretation

### Sentiment Analysis
- Best model performance: BERT (88.65% accuracy)
- Random Forest achieved 88.15% accuracy
- Lexicon-based models performed well (VADER: 86.41%, TextBlob: 86.24%)
- All approaches struggled with the "Moderately Effective" class
- Feature analysis revealed:
  - Traditional models relied heavily on sentiment-laden adjectives
  - BERT identified specific tokens like "suspended" and "greed" as highly predictive
  - Lexicon models depended primarily on polarity and compound sentiment scores
- Consistent misclassification patterns across all approaches indicated inherent challenges with moderate sentiment expressions

### Review Retrieval
- Successfully returned ranked lists of relevant reviews
- Effectively filtered by medical condition
- Implemented highlighting of matching terms
- Identified limitations with TF-IDF's inability to understand synonyms or context

## Limitations and Future Work

### Topic Modeling
- Some topics produced overlapping or vague keyword clusters
- Results were sensitive to preprocessing decisions
- Imbalanced topic distribution
- LDA assumes word independence, missing contextual relationships

### Sentiment Analysis
- Limited performance on the "Moderately Effective" class
- Class imbalance affecting model training
- Long training times (exceeding 30 minutes) due to dataset size
- Challenges with contextual understanding and domain-specific terminology

### Review Retrieval
- Performance declined with large result sets
- Lacks deep contextual understanding
- Struggles with typos, synonyms, and word variations
- Keyword highlighting issues with partial matches

## Future Improvements
- Implement aspect-based sentiment analysis to evaluate sentiment for specific aspects like side effects or effectiveness
- Integrate clustering with review retrieval to group similar reviews
- Develop recommendation features based on shared experiences
- Optimize computational efficiency for large-scale analysis

## Conclusion
This project combined topic modeling, sentiment analysis, and review retrieval to provide a comprehensive understanding of patient feedback on medications. The multi-model approach to sentiment analysis demonstrated the strengths and limitations of different techniques, with contextual embeddings showing a slight advantage for capturing nuanced expressions. Together, these components offer both high-level insights and granular access to individual patient experiences.
