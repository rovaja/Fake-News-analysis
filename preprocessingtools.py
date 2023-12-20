"""Helper module for EDA notebook to perform data cleaning and preprocessing"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
nltk.download("stopwords")
nltk.download('punkt')

def check_null_values(df: pd.DataFrame, plot=True, return_nulls=False) -> pd.DataFrame:
    """Checks for null values of the given dataset"""
    if df.empty:
        print("Dataframe is empty")
    else:
        amount_of_nulls: pd.Series = pd.isnull(df).sum()
        total_amount: int = amount_of_nulls.sum()
        print(f"Total number of null values in data: {total_amount}")
        if total_amount > 0:
            columns_with_nan = amount_of_nulls[amount_of_nulls > 0].index.tolist(
            )
            print(
                f"Number of null values per column:\n{amount_of_nulls[amount_of_nulls>0]}"
            )
            if plot:
                df_proportions = (
                    (amount_of_nulls[amount_of_nulls > 0] /
                     df.shape[0]).rename("proportion").reset_index()
                )
                height = df.shape[1]/18*3
                plt.figure(figsize=(10, height))
                cols = ['red' if x >
                        0.50 else 'steelblue' for x in df_proportions.proportion]
                ax = df_proportions.pipe(
                    (sns.barplot, "data"), x="proportion", y='index', palette=cols)
                ax.set_xlim([0, 1])
                ax.set_title('Columns with null values')
                ax.set_ylabel('Columns')
                for container in ax.containers:
                    ax.bar_label(container, padding=0,
                                 color="black", fmt="{:.3%}")
            if return_nulls:
                return df[df.isnull().any(axis=1)]


def check_duplicated_rows(df: pd.DataFrame) -> list:
    """Checks for duplicated rows of the given dataset"""
    if df.empty:
        print("Dataframe is empty")
    else:
        duplicates_mask = df.duplicated(keep='first')
        total_amount: int = duplicates_mask.sum()
        print(f"Total number of duplicated rows in data: {total_amount}")

        if total_amount > 0:
            duplicates = df[duplicates_mask == True]
            print(duplicates)
            return duplicates.index.tolist()


def find_outliers_IRQ(df: pd.DataFrame, coefficient: float = 1.5) -> list[tuple[str, float]]:
    """Finds outliers from the given numerical features using 
    the Interquartile range and given coefficient.
    Return number of outliers as percentage for each column."""
    outlier_percentages = []
    total_samples = df.shape[0]

    for column in df.select_dtypes(include=[np.number]):
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - coefficient * iqr
        upper_bound = q3 + coefficient * iqr

        column_outliers = df[(df[column] < lower_bound) |
                             (df[column] > upper_bound)][column]

        percentage = column_outliers.count() / total_samples
        outlier_percentages.append((column, percentage))

    return outlier_percentages

def count_words(text):
    """Count number of words in input string"""
    return len(re.findall(r"\b\w+\b", text))

def text_cleaning(text, lowercase = False, websites=True, stopwords=True):
    """Preprocessing of text.
    - Make text lowercase, 
    - Remove text in square and angle brackets,
    - Remove websites, links etc
    - Remove punctuation,
    - Replace newlines with whitespace,
    - Replace multiple spaces with a single space,
    - Remove words containing numbers."""
    if lowercase:
        text = text.lower()
    
    if websites:
        text = re.sub('http?://\S+|https?://\S+|www\.\S+|pic.twitter.\S+', '', text)
        
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    pattern = re.compile(r"[’‘“”\"']")
    text = re.sub(pattern, ' ', text) 
    text = re.sub('\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub('\w*\d\w*', '', text)
    
    if stopwords:
        text = remove_stopwords(text)
        
    return text

def remove_stopwords(text):
    """Remove stopwords from the text."""
    tokens = nltk.tokenize.word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def get_top_ngram(corpus, n=None, stopwords: list = list(), top: int = 10):
    vector = CountVectorizer(ngram_range=(n, n), stop_words=stopwords).fit(corpus)
    bag_of_words = vector.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vector.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:top]

def find_special_symbols(text, characters: str = '@#'):
    """Find special symbols or words like hashtags and twitter handles from the raw text."""
    pattern = f"[{characters}]\w+"
    matches = re.findall(pattern, text)
    return matches

def drop_prefix(text: str, tag: str ='reuters',n: int=5):
    """Remove beginning of the text if the tag is found within the first n words."""
    words = str.split(text,' ')
    if tag in [word.lower() for word in words[:n]]:
        index = text.lower().find(tag.lower())
        return text[index + len(tag):].lstrip()
    else:
        return text

def uppercase_counter(text: str):
    """Counts the uppercase words in input string."""
    words = text.split()
    return sum(1 for word in words if word.isupper())