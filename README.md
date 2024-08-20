# X Sentiment Analysis

This project involves performing sentiment analysis on a dataset of tweets (or X's) to classify them into positive, negative, or neutral sentiments. The analysis is implemented in Python using various machine learning techniques and libraries.

## Project Overview

The project is implemented in a Jupyter Notebook and involves the following key steps:

1. **Data Preprocessing**: The dataset is cleaned and prepared for analysis. This includes removing unnecessary columns, handling missing values, and encoding categorical variables.

2. **Feature Extraction**: Text data is transformed into numerical features using techniques like TF-IDF (Term Frequency-Inverse Document Frequency).

3. **Model Training**: Various machine learning models are trained to classify the sentiment of tweets. Models include deep learning approaches using TensorFlow and Keras.

4. **Model Evaluation**: The performance of the models is evaluated using metrics like accuracy, precision, recall, and F1-score.

## Dataset

- **Dataset Source**: The dataset used for this analysis is named `train.csv`.
- **Number of Records**: 27,481 tweets.
- **Columns**:
  - `textID`: Unique identifier for each tweet.
  - `text`: The actual content of the tweet.
  - `selected_text`: The part of the text selected as the sentiment.
  - `sentiment`: The sentiment label for each tweet (`positive`, `negative`, `neutral`).

## Libraries and Tools

The following Python libraries are used in this project:

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `matplotlib` and `seaborn`: For data visualization.
- `tensorflow.keras`: For building and training deep learning models.
- `sklearn`: For preprocessing and evaluating machine learning models.
- `nltk`: For natural language processing tasks.
- `wordcloud`: For generating word clouds to visualize text data.

## Data Preprocessing

- **Text Cleaning**: The text data is cleaned by removing special characters, numbers, and stopwords.
- **Label Encoding**: The `sentiment` column is encoded into numerical values using `LabelEncoder`.
- **Feature Scaling**: The data is scaled using `MinMaxScaler` to normalize the input features.

## Model Training

- **Deep Learning Model**: A Sequential model from Keras with Dense and Dropout layers is used for classification.
- **Feature Extraction**: CountVectorizer and TfidfVectorizer are used to convert the text data into numerical features.

## Model Evaluation

- **Metrics**: The models are evaluated using accuracy, precision, recall, and F1-score. Confusion matrices are also generated to visualize the performance.
- **Visualization**: Word clouds and distribution plots are used to visualize the data and results.

## How to Run


