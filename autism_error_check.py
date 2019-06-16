# The following functions are for error analysis, so that I could
# provide detailed results to the research group.

def errorCheck(predictions, labels):
	""" Takes the array of predictions and the labels, returns lists of document ids 
	according to classification results.
	"""
	# Takes predicted float labels and rounds them to 0/1.
	pred_list = predictions.round().tolist()
	# Turns the list into ints which can be accuracy-checked against the test labels.
	pred_list_2 = [int(i) for i in pred_list]
	false_positives = []
	false_negatives = []
	true_positives = []
	true_negatives = []
	# Compares predicted values to gold labels.
	for i in range(len(pred_list_2)):
		if pred_list_2[i] == 0 and labels[i] == 1:
			false_negatives.append(test_row2doc[i])
		elif pred_list_2[i] == 1 and labels[i] == 0:
			false_positives.append(test_row2doc[i])
		elif pred_list_2[i] == 1 and labels[i] == 1:
			true_positives.append(test_row2doc[i])
		elif pred_list_2[i] == 0 and labels[i] == 0:
			true_negatives.append(test_row2doc[i])
	return true_positives, true_negatives, false_positives, false_negatives

true_positives, true_negatives, false_positives, false_negatives = errorCheck(y_pred_lr, test_labels)

def lengthCount(doc_list):
	""" Takes one of the output lists from errorCheck() and returns the average length of the document,
	aka how many words are in each.
	"""
	count = 0
	for i in doc_list:
		count += len(test_doc2text[i])
	avg = round(count / len(doc_list))
	return avg

def certaintyCount(doc_list):
	""" Takes an output list from errorCheck() and returns a number, where 0 indicates perfect certainty by the classifier.
	The higher the number, the less sure the classifier was of its answer.
	"""
	count = 0
	for i in doc_list:
		score = y_pred_lr[test_doc2row[i]]
		diff = abs(round(score) - score)
		count += diff
	avg = count / len(doc_list)
	return avg

def frequencyCount(doc_list):
	""" Takes an output list from errorCheck() and returns a percentage corresponding to how many of the words
	in the document appeared only one time.
	"""
	frequency = {}
	for entry in doc_list:
		for word in test_doc2text[entry]:
			if word in frequency:
				frequency[word] += 1
			else:
				frequency[word] = 1
	sortable = []
	for i in frequency:
		sortable.append((i, frequency[i]))
	sortable.sort(key=lambda x: x[1], reverse=True)
	final = [i[0] for i in sortable]
	one_count = 0
	for i in sortable:
		if i[1] == 1:
			one_count += 1
			print(one_count)
	one_ratio = one_count / len(sortable)
	return sortable, one_ratio


def weightInfo(clf_lr, num):
	weights = clf_lr.coef_.tolist()
	sorted_weights = sorted(weights[0])
	true_words = []
	false_words = []
	for w in sorted_weights[:num]:
		temp = weights[0].index(w)
		false_words.append(reverse_vocab[temp])
	for w in sorted_weights[-1:-(num+1):-1]:
		temp = weights[0].index(w)
		true_words.append(reverse_vocab[temp])
	return true_words, false_words

true_words, false_words = weightInfo(clf_lr, 500)

fn_words = []
fp_words = []

for doc in false_positives:
	for word in test_doc2text[doc]:
		fp_words.append(word)
fp_words = list(set(fp_words))


for doc in false_negatives:
	for word in test_doc2text[doc]:
		fn_words.append(word)
fn_words = list(set(fn_words))

fn_weights = []
fp_weights = []
fn_absent = []
fp_absent = []

for word in true_words:
	if word in fp_words:
		fp_weights.append(word)
	else:
		fp_absent.append(word)

for word in false_words:
	if word in fn_words:
		fn_weights.append(word)
	else:
		fn_absent.append(word)


fp_overview = []
for doc in false_positives:
	temp = list(set(test_doc2text[doc]))
	for word in temp:
		if word in true_words:
			fp_overview.append(word)

fp_overview = list(set(fp_overview))