# https://www.dataquest.io/blog/tutorial-text-classification-in-python-using-spacy/
import string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
punctuations = string.punctuation
nlp = spacy.load('en')
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vectors
parser = English()

def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]
    return mytokens
    
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]
    def fit(self, X, y=None, **fit_params):
        return self
    def get_params(self, deep=True):
        return {}

def clean_text(text):
    return text.strip().lower()
    
bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)

df = pd.DataFrame.read_csv(csv_path)
X = df['text']
ylabels = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3)

# Logistic Regression Classifier
classifier = LogisticRegression()
# Create pipeline using Bag of Words
pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('classifier', classifier)])

# model generation
pipe.fit(X_train,y_train)
# Predicting with a test dataset
predicted = pipe.predict(X_test)
# Model Accuracy
print("Accuracy:",  metrics.accuracy_score(y_test, predicted))
print("Precision:", metrics.precision_score(y_test, predicted))
print("Recall:",    metrics.recall_score(y_test, predicted))