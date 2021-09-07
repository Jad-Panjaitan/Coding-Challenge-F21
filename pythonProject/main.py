import re
import string
import io
import pandas
import numpy

import nltk
import sklearn.utils
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# feed the training data into here.
# the twitter dataset doesn't come with headers, so i have to manually create them
header_list = ["polarity", "ID", "Date", "excise_this", "Handle", "tweet"]
# NTS: I can also change the encoding to encoding = "utf-8"
twitter_dataframe = pandas.read_csv(r'C:\Users\haris\Coding-Challenge-F21\archive\training.1600000.processed.noemoticon.csv'
    , names=header_list, usecols=["polarity", "tweet"]
    , encoding="ISO-8859-1")

# debugging: print(twitter_dataframe.head(20))

# shuffle, so when we split the data we do not constantly exclude a percentage of the positive reviews
# you may add in additional parameter: random_state = seed
# condensed_twitter_dataframe = sklearn.utils.shuffle(condensed_twitter_dataframe)
twitter_dataframe = sklearn.utils.shuffle(twitter_dataframe)
# debugging: print(twitter_dataframe.head(20))

# only sampling every 125 tweets or so from the dataframe as to spare my laptop
condensed_twitter_dataframe = twitter_dataframe[twitter_dataframe.index % 125 == 0].copy()
# debugging: print(condensed_twitter_dataframe.head(20))

# feed the training dataframe  you are interested in here
def preprocess(docu):
    # make everything lowercase
    docu = docu.lower()

    # remove URLs. as many alphanumerics as needed, then colon and 2 slashes
    # then as much alphanumerics as needed then ., anychar
    # match but dont capture anything that isnt whitespace or slashes
    docu = re.sub(r'\w+:\/{2}[\w-]+(\.[\w-]+)*(?:(?:\/[^\s/]*))*', '', docu)

    # remove twitter handles
    docu = re.sub('@[\w]+', '', docu)

    # strip away quotation marks, apostrophes, periods, hyphens, exclamation marks...
    # this regular expression is removing non-alphanumeric (except for spaces) characters
    docu = re.sub('[^a-z0-9 ]', '', docu)

    # tokenise the documents. Although we are training the ML on tweets, we are not using TweetTokenizer since it does preserve hashtags.
    docu_tokens = word_tokenize(docu)

    # remove stopwords: for each word in docu_tokens, if it is not in the list of stopwords, add it to scrubbed_tokens
    # i could write this in one sentence, but this is easier for me to read.
    scrubbed_tokens = []
    for word in docu_tokens:
        if word not in stopwords.words('english'):
            scrubbed_tokens.append(word)

    # lemmatizing each token. I chose to stem the tokens, because this is running on a potato.
    # lemmatized_tokens = []
    # lemmatizer = WordNetLemmatizer()
    # for word in scrubbed_tokens:
    #    lemmatized_word = lemmatizer.lemmatize(word)
    #    lemmatized_tokens.append(lemmatized_word)

    # return lemmatized_tokens

    # stemming each token
    stemmed_tokens = []
    stemmer = PorterStemmer()
    for word in scrubbed_tokens:
        stemmed_word = stemmer.stem(word)
        stemmed_tokens.append(stemmed_word)

    # return stemmed_tokens
    return stemmed_tokens


# debugging: print(preprocess("i, euripedes am an where's we ate.  @chde pop https://www.google.com/"))

# preprocess the tweets
preprocessor = lambda x: preprocess(x)
condensed_twitter_dataframe['preprocessed_tweet'] = pandas.DataFrame(
    condensed_twitter_dataframe.tweet.apply(preprocessor))

# split the twitter dataset into training and testing
tweet_train, tweet_test, polarity_train, polarity_test = model_selection.train_test_split(
    condensed_twitter_dataframe.tweet,
    condensed_twitter_dataframe.polarity,
    test_size=0.20)

# create model given training and testing data and print out its stats
def create_model(x_train, x_test, y_train, y_test, read_stats):
    vectorizer = TfidfVectorizer()
    classifier = MultinomialNB()
    model = Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])
    model.fit(x_train, y_train)

    predicted = model.predict(x_test)
    confusion_matrix(predicted, y_test)

    if read_stats:
        print(accuracy_score(predicted, y_test))
        print(precision_score(predicted, y_test, average='weighted'))
        print(recall_score(predicted, y_test, average='weighted'))

        print(classification_report(predicted, y_test))

    return model


sa_model = create_model(tweet_train, tweet_test, polarity_train, polarity_test, read_stats = False)

# read the input file
text_file = open(r'C:\Users\haris\Coding-Challenge-F21\input.txt')
s = text_file.read()
text_file.close()
specimen = pandas.DataFrame(nltk.tokenize.sent_tokenize(s.strip().replace('\n', ' '), language = 'english'))

# as the column is unnamed
specimen.rename(columns={0:"sentences"},inplace=True)

# we do a little preprocessing
sa_results = list(specimen['sentences'])
for sentence in specimen:
    sentence.lower()
    re.sub('[^a-z0-9 ]', '', sentence)
debug: print(sa_results)

# make a score for each sentence of text.
scores = []
predictor = lambda x: sa_model.predict(x)
scores.append(predictor(sa_results))
scores = scores[0]
print("Binary polarity of each sentence:", scores)

# calculate average score
average_score = 0.00
for score in scores:
    average_score = average_score + int(score)
average_score = average_score / (len(scores))
print("Polarity of the text from 0-4:", average_score)

exit()
