# ACM Research Coding Challenge (Fall 2021)

## [](https://github.com/ACM-Research/Coding-Challenge-F21#no-collaboration-policy)No Collaboration Policy

**You may not collaborate with anyone on this challenge.**  You  _are_  allowed to use Internet documentation. If you  _do_  use existing code (either from Github, Stack Overflow, or other sources),  **please cite your sources in the README**.

## [](https://github.com/ACM-Research/Coding-Challenge-F21#submission-procedure)Submission Procedure

Please follow the below instructions on how to submit your answers.

1.  Create a  **public**  fork of this repo and name it  `ACM-Research-Coding-Challenge-F21`. To fork this repo, click the button on the top right and click the "Fork" button.

2.  Clone the fork of the repo to your computer using  `git clone [the URL of your clone]`. You may need to install Git for this (Google it).

3.  Complete the Challenge based on the instructions below.

4.  Submit your solution by filling out this [form](https://acmutd.typeform.com/to/zF1IcBGR).

## Assessment Criteria 

Submissions will be evaluated holistically and based on a combination of effort, validity of approach, analysis, adherence to the prompt, use of outside resources (encouraged), promptness of your submission, and other factors. Your approach and explanation (detailed below) is the most weighted criteria, and partial solutions are accepted. 

## [](https://github.com/ACM-Research/Coding-Challenge-S21#question-one)Question One

[Sentiment analysis](https://en.wikipedia.org/wiki/Sentiment_analysis) is a natural language processing technique that computes a sentiment score for a body of text. This sentiment score can quantify how positive, negative, or neutral the text is. The following dataset in  `input.txt`  contains a relatively large body of text.

**Determine an overall sentiment score of the text in this file, explain what this score means, and contrast this score with what you expected.**  If your solution also provides different metrics about the text (magnitude, individual sentence score, etc.), feel free to add it to your explanation.   

**You may use any programming language you feel most comfortable. We recommend Python because it is the easiest to implement. You're allowed to use any library/API you want to implement this**, just document which ones you used in this README file. Try to complete this as soon as possible as submissions are evaluated on a rolling basis.

Regardless if you can or cannot answer the question, provide a short explanation of how you got your solution or how you think it can be solved in your README.md file. However, we highly recommend giving the challenge a try, you just might learn something new!

# JAD'S SOLUTION:

A simple TF-IDF model in Python, using scikit-learn for the modelling; pandas, nltk, and some vanilla Python packages like string and re for the preprocessing.
I trained the model using this dataset of [1.6 million tweets](https://www.kaggle.com/kazanova/sentiment140). Unfortunately, since my laptop is a potato I can only analyse 10,000 of those tweets.
It attempts to classify the sentiment of each sentence in the document by either 0 or 4 -- negative and positive, respectively. It then calculates the average sentiment of the document. 

Personally, I would say the polarity of the text above is, on a scale of 0 to 10, a 6 -- slightly positive. The first block of text is quite raucous, while the second one is kind and complimentary; however, despite the former being shorter, it is also more potent in its wording, mentioning murders and yelling and acts of rage. 
The model proposes an overall sentiment value of 1.6/4; (6/10) - (1.6/4) == 0.2, so the specific iteration of that model tells me that I overestimated the polarity -- or alternatively, the model underestimated the polarity.

Here are the polarities of each sentence the program parsed, if you're interested in that: [0, 0, 0, 0, 4, 0, 0, 0, 4, 4, 0, 0, 4, 4, 4]

## Notes

**Steps to cover, more or less:** 
0. Set everything to lowercase/uppercase. I don’t think this is needed for this text, specifically (.case) (python documentation duh)
1. Remove odd things like links and symbols. Again, I don’t think this is needed for this text. (re.sub) (python documentation ++ basic regex guide) 
2. Tokenise everything (tokenize, since this is American English) (NLTK documentation)
3. Remove stopwords, which are things like determiners, preposition, conjunctions. Give stopword list and say if such and such is in tokenise list then excise (geeksforgeeksguide)
4. Lemmatization and stemming (convert words to standardised form) (guru99)
5. Give dataset of word and sentiment (I suppose I'll choose the twitter dataset?), then train.
6. Train (this is the hard part.)
7. Run text through model

**PROBLEM: Anaconda isn't even working?**
SOLUTION: It has to do with the way my hand-me-down computer has partitioned drives that also have no more storage space. I suppose I'll have to switch to the other hand-me-down I have.

**PROBLEM: Everything in the twitter dataset is squished into one column.**
SOLUTION: Actually, it autonamed the headers incorrectly. I simply have to feed a list of headers to the read_csv function.

**PROBLEM: Odd preprocessing. Body becomes bodi, manager becomes manag, behave becomes behav...**
SOLUTION: This is a consequence of stemming, and it actually doesn't matter too much with regards to training the model. If I really wanted to ameliorate this, I can use lemmatisation. Also, you should only use one or the other.

**PROBLEM: Why does this take forever? I've been sitting here for 50 minutes...**
SOLUTION: Lemmatisation takes much longer than stemming. Also, I am analysing 1.6 million tweets. Hence, only analyse 10,000 of them.

**PROBLEM: What sort of model should I use? There is bag-of-words, TF-IDF, word2vec, BERT...**
SOLUTION: As it turns out, the linear algebra kicks in when you start working with word2vec and BERT. I'm good at math, but not THAT good. For now, I'll settle with TF-IDF.

**PROBLEM: The model is overfitting!**
SOLUTION: I'm only drawing from the first 10,000 rows, and in the csv the first 0.8 million or so rows are all negative.  I ended up randomising it, then drawing from one row out of every 125. Only then was it ready to train.

__ADDENDUM to STEP 6: Tweak the ratios of training size to sample size, along with the frequency at which I sample the rows. What brought the best compromise was sampling 1 row out of every 125, with 20% of that data being dedicated to testing.__

## Future Improvements

Clearly, the bigger the sample size the better. Only if I had one of those Alienware desktops they have in the ATEC computer labs... at least I have an understanding of the compromises people make when generating these models.

It seems that the NLTK tokenizer that tokenizes paragraphs into sentences does not work correctly. This is a major flaw, and I'm not sure if it's the issue with the function itself or if there some preprocessing that is to be performed.

Language such as that in literature is much more obtuse than that of, well, tweets. If there is a literature dataset that includes polarities, I would use that instead.

## Resources
As for the resources that I so graciously perused:

[The Python Standard Library](https://docs.python.org/3/library/index.html)

[pandas 1.3.2 documentation](https://pandas.pydata.org/)

[NLTK 3.6.2 documentation](https://www.nltk.org/)


[Stemming and Lemmatization in Python NLTK with Examples](https://www.guru99.com/stemming-lemmatization-python-nltk.html)

[Removing stop words with NLTK in Python](https://www.geeksforgeeks.org/removing-stop-words-nltk-python/)

[Text Classification with NLP: Tf-Idf vs Word2Vec vs BERT](https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794)

[Analyzing Sentiment of Movie Reviews - An AIS Workshop](https://www.youtube.com/watch?v=iD9fxZUcddc)

And plenty of Stack Overflow, of course.

 
