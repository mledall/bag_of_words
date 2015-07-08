# This code will follow the tutorial at https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words, for text treatment

# Import the pandas package, then use the "read_csv" function to read
# the labeled training data
import pandas as pd		# Allows to import the data
from bs4 import BeautifulSoup 	# Will be used to remove the HTML characters
import re			# This packages allows to remove punctuation and numbers
import nltk			# Allows to remove the stopwords (those words that carry not meaning, like 'and', 'the'...)
#nltk.download('all')		# Downloads the stopwords data sets
#print stopwords.words("english")
from nltk.corpus import stopwords	# Import a stopword list in English
from sklearn.feature_extraction.text import CountVectorizer		# Allows us to use bag-of-words learning and vectorize the set
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

#character and thius as welll

# Imports the data. The target is the sentiment
def train_data_import(data = "labeledTrainData.tsv"):
	return pd.read_csv(data, header=0, delimiter="\t", quoting=3)

def test_data_import(data = "testData.tsv"):
	return pd.read_csv(data, header=0, delimiter="\t", quoting=3)

# The following function takes in the raw review, and outputs the cleaned review
def clean_review( raw_review ):		# Removes the HTML, punctuation, numbers, stopwords...
	rm_html = BeautifulSoup(raw_review).get_text()	# removes html
	letters_only = re.sub("[^a-zA-Z]",           	# The pattern to search for; ^ means NOT
                   		  " ",                   	# The pattern to replace it with
                          rm_html )              	# The text to search
	lower_case = letters_only.lower()	         	# Convert to lower case
	words = lower_case.split()          	     	# Split into words
	stops = stopwords.words("english")
	stops.append('ve')
	stops = set(stops)
#	stops = set(stopwords.words("english"))		# It is faster to look through a set than a list
	meaningful_words = [w for w in words if not w in stops]	# Remove stop words from "words"
	return ' '.join(meaningful_words)			# Joins the words back together separated by a space


#print train_data_import()["review"][0]
print clean_review(train_data_import()["review"][0])

# The following function iterates clean_review over all reviews in the set
def clean_all_reviews( raw_train_data, N_articles ):
	cleaned_reviews = []
	for i in xrange(N_articles):
		cleaned_reviews.append(clean_review(raw_train_data["review"][i]))
		if ( (i+1) % 1000 == 0 ):
			print(' -- Clean review # %d' % i)
	return cleaned_reviews


# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
def Bag_of_Words(cleaned_reviews, n_features = 5000):
	vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None,    	# Allows to tokenize
                             preprocessor = None, 	# Allows to do some preprocessing
                             stop_words = None,   	# We could remove stopwords from here
                             max_features = n_features) 	# Chooses a given number of words, just a subset of the huge total number of words.
	# fit_transform() does two functions: First, it fits the model
	# and learns the vocabulary; second, it transforms our training data
	# into feature vectors. The input to fit_transform should be a list of 
	# strings.
	data_features = vectorizer.fit_transform(cleaned_reviews)
	data_features = data_features.toarray()
	return data_features, vectorizer


# Counts the words that appear in the reviews
def word_count():
	raw_train_data = train_data_import()
	N_articles = 2000#len(raw_train_data["review"][:])
	cleaned_reviews = clean_all_reviews(raw_train_data, N_articles)
	train_data_features, vectorizer = Bag_of_Words(cleaned_reviews)
	vocab = vectorizer.get_feature_names()
	dist = np.sum(train_data_features, axis=0)
	word_count = sorted(zip(dist,vocab),reverse = True)
	for count, tag in word_count:
	    print '{}: {}'.format(count, tag)


#word_count()

def Trainer(N_articles):
	print '- Import all training reviews'
	raw_train_data = train_data_import()
	print '- Start cleaning the training reviews'
	cleaned_reviews = clean_all_reviews(raw_train_data, N_articles)
	print '- Creating the bag-of-words with %d articles' % N_articles
	train_data_features, vectorizer = Bag_of_Words(cleaned_reviews, N_articles)
	print '- Trains a classifier'
	X_train, X_valid, Y_train, Y_valid = cross_validation.train_test_split(train_data_features, raw_train_data["sentiment"][:N_articles], train_size = 0.98, random_state=10)
	clf = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto', bootstrap=True, oob_score=False, n_jobs=4, random_state=None, verbose=0, min_density=None, compute_importances=None)
	forest = clf.fit( X_train, Y_train )
	score = clf.score(X_valid, Y_valid)
	print '- Finished training, score = %f' % score
	return forest, vectorizer, score

# Write a log file of results obtained thus far
def log_file(clf, classifier, score, N_train_articles):
	with open('Bag_of_words_log_file', 'a') as f:
		f.write('\n\n')
		f.write('- Classifier: %s \n' % classifier)
		f.write(' -- Score = %f , \n' % score)
		f.write(' -- Parameters: %s, \n' % clf.get_params(True))
		f.write(' -- Training data set: %f .' % N_train_articles)

# Creates a submission file
def submission_file():
	# Total number of articles = 25000
	N_train_articles = 3000
	clf, vectorizer, score = Trainer(N_train_articles)
	print '- Import test data'
	raw_test_data = test_data_import()				# Imports the test data
	N_test_articles = len(raw_test_data["id"][:])
	print '- Clean test data'
	cleaned_test_reviews = clean_all_reviews( raw_test_data, N_test_articles )	# Cleans the test data
	test_data_features = vectorizer.transform(cleaned_test_reviews)	# Not we do not fit here
	test_data_features = test_data_features.toarray()
	print '- Predicts sentiment' 
	predict = clf.predict(test_data_features)
	# Copy the results to a pandas dataframe with an "id" column and a "sentiment" column
	output = pd.DataFrame( data={"id":raw_test_data["id"], "sentiment":predict} )
	# Use pandas to write the comma-separated output file
	print '- Print submission file'
	output.to_csv( "Bag_of_Words_model_prediction.csv", index=False, quoting=3 )
	print '- Print log file'
	log_file(clf, 'RandomForest', score, N_train_articles)


def main_function():
	submission_file()

#main_function()



