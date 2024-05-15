# Linguistic Analysis for Detecting and Analyzing Inappropriate Comments

## Project Overview
This project aims to detect and analyze inappropriate comments using linguistic analysis techniques. The dataset used for this project comprises tweets from Twitter. The following steps outline the methodology employed to achieve this goal.

## Methodology

### 1. Dataset Preparation
- **Dataset**: Collected from Twitter tweets.
- **Data Cleaning**: Performed initial data cleaning to handle missing values, remove unnecessary characters, and ensure consistency.

### 2. Linguistic Analysis
Performed various linguistic analysis techniques to extract meaningful features from the dataset:
- **Named Entity Recognition (NER)**: Identified and categorized entities (such as names, locations, etc.) in the text.
- **Part of Speech (POS) Tagging**: Labeled each word with its corresponding part of speech (noun, verb, adjective, etc.).
- **Dependency Parsing**: Analyzed grammatical structure and relationships between words in a sentence.

### 3. Feature Extraction
Converted the linguistic data into numerical vectors for model training:
- **Lemmatization**: Reduced words to their base or root form to handle variations of a word.
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Transformed text data into weighted numerical vectors based on the importance of terms in the dataset.

### 4. Model Training
Trained a machine learning model to detect inappropriate comments:
- **Algorithm**: Used Naive Bayes algorithm for classification.
- **Training**: Fed the vectorized dataset into the Naive Bayes classifier to learn patterns and relationships.

### 5. Model Evaluation
Evaluated the performance of the trained model:
- **ROC Score and ROC Curve**: Used Receiver Operating Characteristic (ROC) curve and ROC score to assess the model's ability to distinguish between appropriate and inappropriate comments.
- **Accuracy**: Achieved an accuracy of approximately 96% in detecting inappropriate comments.

### 6. Model Saving
- **Serialization**: Saved the trained model using Python's `pickle` module for future use and deployment.

### 7. API Integration
Connected the trained model with a web API to allow real-time predictions:
- **Flask API**: Developed a Flask API to serve the model and handle incoming requests for comment analysis.

### 8. User Interface
Created an interactive user interface to facilitate easy use of the model:
- **Streamlit**: Used Streamlit to build a web-based interface where users can input comments and receive predictions on their appropriateness.

## Conclusion
This project successfully demonstrates the use of linguistic analysis and machine learning to detect and analyze inappropriate comments. By leveraging Twitter tweets, performing comprehensive linguistic analysis, and implementing a robust machine learning model, we achieved a high accuracy rate. The integration with Flask API and Streamlit provides a seamless and user-friendly experience for real-time comment analysis.

## Repository Structure
```
- data/
  - twitter_dataset.csv
- notebooks/
  - data_preparation.ipynb
  - linguistic_analysis.ipynb
  - feature_extraction.ipynb
  - model_training.ipynb
  - model_evaluation.ipynb
- models/
  - trained_model.pkl
- api/
  - app.py
- interface/
  - streamlit_app.py
- README.md
- requirements.txt
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/linguistic-analysis-comment-detection.git
   cd linguistic-analysis-comment-detection
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Flask API:
   ```bash
   cd api
   python app.py
   ```
2. Run the Streamlit interface:
   ```bash
   cd interface
   streamlit run streamlit_app.py
   ```

## License
This project is licensed under the MIT License.

## Contact
For any inquiries, please contact [giridhar.chennuru3105@gmail.com].
