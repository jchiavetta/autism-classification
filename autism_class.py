from conllu import parse
import csv, xlrd, re, random
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import *
import glob
from tqdm import tqdm
import pickle
from sklearn import tree
import spell
import symspell

#spellchecker object, lookup(string, 0, 2) will return a class object that can be retrieved with object[0].term
speller = symspell.SymSpell()
speller.create_dictionary('big.txt')

# Case number, fold number, and gold label, e.g. [['2010160242', '5', '0'], ['2010160243', '9', '1'], ... ]].
all_cases = []
with open('case_status_with_fold.csv') as f:
	reader = csv.reader(f)
	[all_cases.append(row) for row in reader]
all_cases = all_cases[1:]

# Separating out the test folds from the training folds.
train_cases = []
test_cases = []
for case in all_cases:
	if case[2] == '8':
		test_cases.append(case)
	else:
		train_cases.append(case)

# Map row index to document id, e.g. 355: 2010160242.
train_row2doc = {}
test_row2doc = {}
# Map document id to the gold classification label.
train_doc2label = {}
test_doc2label = {}
# Map row index to the gold classification label.
train_row2label = {}
test_row2label = {}

# Building the above dicts
for row in range(len(train_cases)):
	train_row2doc[row] = train_cases[row][0]
	train_doc2label[train_cases[row][0]] = train_cases[row][1]
	train_row2label[row] = train_cases[row][1]

for row in range(len(test_cases)):
	test_row2doc[row] = test_cases[row][0]
	test_doc2label[test_cases[row][0]] = test_cases[row][1]
	test_row2label[row] = test_cases[row][1]

# Building reverse dicts
train_doc2row = {}
for i in train_row2doc:
	train_doc2row[train_row2doc[i]] = i

test_doc2row = {}
for i in test_row2doc:
	test_doc2row[test_row2doc[i]] = i


# These conllu files hold the actual raw text data for each document.
path = '/Users/jeffreychiavetta/AHRQ-Autism-Case-Status-Assignment/data/autism_ehr_corenlp/*.conllu'
files = glob.glob(path)
print('files')
train_data = []
test_data = []
for row in tqdm(files):
	m = re.search('nlp\/(.*)\.', row)
	match = m.group(1)
	if match in train_doc2label:
		train_data.append((match, row))
	elif match in test_doc2label:
		test_data.append((match, row))

# This block does two things: it builds the training data list in the format [document id, gold label, list of word tokens],
# and it builds a frequency dictionary by adding every word that appears more than once in a document.
frequency_dict = {}
print('train_data_2')
train_data_2 = []
for entry in tqdm(train_data):
	token_list = []
	temp_vocab = []
	filter_dict = {}
	with open(entry[1]) as f:
		temp = f.readlines()
		for line in temp:
			# The word token appears after an index number and a tab.
			m = re.match('[0-9][0-9]?\\t(.*?)\\t', line)
			if m:
				m2 = m.group(1)
				m2 = m2.lower()
				# running a spell checker to locate and fix typos; this is crucial because the original
				# physicians' notes contain a very large number of errors.
				m3 = speller.lookup(m2, 0, 2)
				if m3:
					token_list.append(m3[0].term)
					# Only adds the chosen parts of speech to the vocabulary, as others are unlikely to be important.
					v = re.search('(NN)|(VB)|(JJ)', line)
					if v:
						temp_vocab.append(m3[0].term)
	for word in temp_vocab:
		if word in filter_dict:
			filter_dict[word] += 1
		else:
			filter_dict[word] = 1
	# Filters out a large number of remaining typos if set to > 1, reducing vocab size from 22,000~ to 8,000~
	# without noticeably reducing performance.
	temp_vocab_2 = [i for i in filter_dict if filter_dict[i] > 0]
	temp_vocab_2 = list(set(temp_vocab_2))
	for word in temp_vocab_2:
		if word in frequency_dict:
			frequency_dict[word] += 1
		else:
			frequency_dict[word] = 1
	train_data_2.append((entry[0], train_doc2label[entry[0]], token_list))

# Maps training document id to the list of words in that document.
train_doc2text = {}
for entry in train_data_2:
	train_doc2text[entry[0]] = entry[2]

# Same as for the training data but now there's no need to build another frequency_dict.
print('test_data_2')
test_data_2 = []
for entry in tqdm(test_data):
	token_list = []
	token_dict = {}
	with open(entry[1]) as f:
		temp = f.readlines()
		for line in temp:
			m = re.match('[0-9][0-9]?\\t(.*?)\\t', line)
			if m:
				m2 = m.group(1)
				m2 = m2.lower()
				# running a spell-checker to reduce typos
				m3 = speller.lookup(m2, 0, 2)
				if m3:
					token_list.append(m3[0].term)
	test_data_2.append((entry[0], test_doc2label[entry[0]], token_list))

# Maps test document id to the list of words in that document.
test_doc2text = {}
for entry in test_data_2:
	test_doc2text[entry[0]] = entry[2]

# Only words that appear in 2 or more separate documents get added.
raw_data_alt = []
for word in frequency_dict:
	if frequency_dict[word] >= 2:
		raw_data_alt.append(word)

# Building vocab: each word is assigned a unique integer for easy lookup, e.g. 'contact: 2355'.
raw_data_alt = list(set(raw_data_alt))
vocab = {}
for number in range(len(raw_data_alt)):
	vocab[raw_data_alt[number]] = number


# with open('spellchecked_train.txt', 'rb') as f:
# 	train_data_2 = pickle.load(f)

# with open('spellchecked_test.txt', 'rb') as f:
# 	test_data_2 = pickle.load(f)

# with open('spellchecked_vocab.txt', 'rb') as f:
# 	vocab = pickle.load(f)	

# Builds a reverse vocab, e.g. '2355: contact'.
reverse_vocab = {}
for word in vocab:
	reverse_vocab[vocab[word]] = word

# This builds the input matrix for the classifier. a vector of integers the size of the vocabulary, with a frequency count
# for words that appear in the document and 0 for every other word.
print('train_matrix')
train_matrix = []
for entry in tqdm(train_data_2):
	temp_vector = [0 for i in vocab]
	for word in entry[2]:
		if word in vocab:
			temp_vector[vocab[word]] += 1
	train_matrix.append(temp_vector)

# Same as above, for the test set
print('test_matrix')
test_matrix = []
for entry in tqdm(test_data_2):
	temp_vector = [0 for i in vocab]
	for word in entry[2]:
		if word in vocab:
			temp_vector[vocab[word]] += 1
	test_matrix.append(temp_vector)

# Builds train/test numpy arrays for the gold labels, aka 1s and 0s.
train_labels = [int(i[1]) for i in train_data_2]
test_labels = [int(i[1]) for i in test_data_2]
train_labels = np.asarray(train_labels)
test_labels = np.asarray(test_labels)

# Turns the input matrices into sci-kit sparse arrays.
train_array = np.asarray(train_matrix)
train_array = sp.csr_matrix(train_array)
test_array = np.asarray(test_matrix)
test_array = sp.csr_matrix(test_array)

def add_features(input_vectors, dataset): # e.g. train_X_tfidf, train_dataset
	"""Takes the documents and their featurized input vectors (indices must match).
	Converts the vectors into lists, adds custom features to the end, then converts them
	back into scikit sparse matrices which can be fed into a classifier.
	"""
	temp_matrix = []
	for row in range(len(dataset)): # 3980
		# features for columns OA - OE
		# y/n Down('s) Syndrome
		custom_feature_1 = 0.0
		# y/n ADHD
		custom_feature_2 = 0.0
		# y/n OCD
		custom_feature_3 = 0.0
		# y/n microcephaly
		custom_feature_4 = 0.0
		# y/n seizures
		custom_feature_5 = 0.0
		temp_string = ''
		# each row becomes an array
		a = input_vectors[row].toarray()
		# each row array becomes a list of length 40419, mostly zeroes
		b = list(a[0])
		# turns list into one searchable string
		temp_string = ' '.join(dataset[row][390:395])
		if re.search('Down(?:\'s)?', temp_string):
			custom_feature_1 = 1.0
		if re.search('ADD|[Dd]eficit', temp_string):
			custom_feature_2 = 1.0
		if re.search('OCD|[Oo]bsessive', temp_string):
			custom_feature_3 = 1.0
		if re.search('[Mm]icrocephaly', temp_string):
			custom_feature_4 = 1.0
		if re.search('[Ss]eizures?', temp_string):
			custom_feature_5 = 1.0
		b.append(custom_feature_1)
		b.append(custom_feature_2)
		b.append(custom_feature_3)
		b.append(custom_feature_4)
		b.append(custom_feature_5)
		temp_matrix.append(b)
	c = np.asarray(temp_matrix)
	d = sp.csr_matrix(c)
	return d

# Makes sure input vectors successfully map to vocabulary items (they do).
# index_check = []
# for i in range(len(train_matrix[0])):
# 	if train_matrix[0][i] > 0:
# 		index_check.append(reverse_vocab[i])


# takes the input matrices and builds tf-idf (term frequency - inverse document frequency) versions with scikit
tfidf_transformer = TfidfTransformer()
train_array_tfidf = tfidf_transformer.fit_transform(train_array)
test_array_tfidf = tfidf_transformer.fit_transform(test_array)


# Passed all checks for 1:1 index correspondence.

# This block trains a logistic regression classifier from sci-kit on the training data and then uses it on
# the test set, creating an array of predictions which it uses to display accuracy and other metrics.
clf_lr = LogisticRegression(penalty='l1')
print("Training logistic regression...")
clf_lr.fit(train_array, train_labels)
y_pred_lr = clf_lr.predict_proba(test_array)[:, 1]	
print("Logistic regression results:")
print('Accuracy: ', ((accuracy_score(test_labels, y_pred_lr.round(), normalize=False))/len(y_pred_lr))*100)
print('Precision: ', precision_score(test_labels, y_pred_lr.round()))
print('Recall: ', recall_score(test_labels, y_pred_lr.round()))
print('f1: ', f1_score(test_labels, y_pred_lr.round()))
print('\n')


# This block does the same but with a support vector machine.
clf_svc = SVC(kernel = 'linear', probability = True)
print("training SVC...")
clf_svc.fit(train_array, train_labels)
y_pred_svc = clf_svc.predict_proba(test_array)[:, 1]
print("SVC results:")
print('Accuracy: ', ((accuracy_score(test_labels, y_pred_svc.round(), normalize=False))/len(y_pred_svc))*100)
print('Precision ', precision_score(test_labels, y_pred_svc.round()))
print('Recall', recall_score(test_labels, y_pred_svc.round()))
print('F1: ', f1_score(test_labels, y_pred_svc.round()))