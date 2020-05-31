from id3 import Id3Estimator
from id3 import export_graphviz
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from matplotlib import pyplot as plt
import re
import os
from export_tree import export_graph

"""
Set to choose model
  0: id3 - decision tree
  1: perceptron - modified internet
  2: perceptron - sklearn
  3: split decision trees
  4: ensemble - bagging
  5: ensemble - local trees + perceptron

"""
model = 0
debug = False # Change for verbose execution
"""
Set to choose number of folds - k
valid values 2..39
k = n is the same as leave one out
For stratified k-fold, maximum number of folds can go up to max(members in any class), in this case = 30
"""
num_folds = 3

"""
Perceptron parameters
l_rate: learning rate
n_epoch: number of epochs - iterations over the training set 
"""
l_rate = 1
n_epoch = 11

# Function to discard duplicate values
def unique(sequence):
  seen = set()
  return [x for x in sequence if not (x in seen or seen.add(x))]

# Perceptron code
# Make a prediction, given the threshold
def predict_wthreshold(row, weights, threshold):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	return 1.0 if activation >= threshold else 0.0

def train_weights_wthreshold(train, target, l_rate, n_epoch, threshold):
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		sum_error = 0.0
		for i, row in enumerate(train):
			prediction = predict_wthreshold(row, weights, threshold)
			error = int(target[i]) - prediction
			sum_error += error**2
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
		if debug: print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
	return weights
"""
  =================   ==============
  Classes                          2
  Samples per class      10(M),29(B)
  Samples total                   39
  Number of Features              63
  Features            real, positive
  =================   ==============

  Classes - Malignant(1) and Benign(0)
"""

data_fh = open("data.csv", "r")

data = np.empty((0,63)) # Initializing empty array
target = [] # Empty list

for line in data_fh:
  line = line.strip() # Discard trailing newline
  stats = line.split(',')[1:64] # First column contains the program name, second through second to last are feature values
  target.append(line.split(',')[64]) # Last column contains the labels
  try: # Checking if the numbers in the string can be represented as int - else float - required to convert string to numerical values
    num_list = list(map(int, stats))
  except ValueError:
    num_list = list(map(float, stats))
  
  temp_arr = np.asarray(num_list).reshape((1,63))
  data = np.vstack((data, temp_arr)) # Appending row to the array - vertical stack

feature_list_file = "features_65.txt"
feature_list_fh = open(feature_list_file, "r")

features = feature_list_fh.read()
feature_list_fh.close()

feature_list = features.split()
feature_list = unique(feature_list)

if (model == 0): # ID3 Decision tree
  estimator = Id3Estimator()
  max_estimator = Id3Estimator() # To store max model
  m_scores = np.zeros(num_folds)
  estimator = estimator.fit(data, np.ravel(target))
  tree = export_graphviz(estimator.tree_, 'tree.dot', np.asarray(feature_list).ravel())
  #tree = export_graph('tree.dot', 'tree_mine.txt')
  k_fold = StratifiedKFold(num_folds, shuffle=False) # K-fold data split function from sklearn
  max_acc_model = 0
  
  for k, (train, test) in enumerate(k_fold.split(data, np.ravel(target))): # Compute accuracy over k folds
    estimator.fit(data[train], np.ravel(target)[train])
    prediction = estimator.predict(data[test])
    #tree = export_graphviz(estimator.tree_, 'tree'+str(k)+'.dot', np.asarray(feature_list).ravel())
    yTest = np.ravel(target)[test] # target labels - test set
    m_scores[k] = 1 - (np.count_nonzero(prediction.astype(int) ^ yTest.astype(int)) / test.size) # Checking for errors and computing score
    if m_scores[k] > m_scores[max_acc_model]:
      max_acc_model = k
      max_acc_train = data[train] # Training set for the model with max accuracy
      max_acc_test = np.ravel(target)[test] # Test set for the model
      max_estimator = estimator

    print("Fold: " + str(k) + " Score: " + str(m_scores[k]))
  print("Accuracy: %0.2f (+/- %0.2f) " % (np.asarray(m_scores).mean(), np.asarray(m_scores).std()))
  print("Model with the best accuracy is for k = " + str(m_scores.argmax(axis=0)))

elif (model == 1): # Perceptron - internet
  weights = train_weights_wthreshold(data, np.ravel(target), l_rate, n_epoch, 0)
  weights_wo_bias = weights[1:]
  sorted_weights = np.argsort(np.asarray(weights_wo_bias)).ravel()
  abs_sorted_weights = np.argsort(abs(np.asarray(weights_wo_bias))).ravel()

  sorted_fh = open("Perceptron-sorted.txt", "w")
  abs_sorted_fh = open("Perceptron-absolute-sorted.txt", "w")
  
  for i in np.nditer(sorted_weights):
    print(feature_list[i], file=sorted_fh)

  for i in np.nditer(abs_sorted_weights):
    print(feature_list[i], file=abs_sorted_fh)
  
  k_fold = StratifiedKFold(num_folds, shuffle=False) # K-fold data split function from sklearn
  m_scores = np.zeros(num_folds)
  for k, (train, test) in enumerate(k_fold.split(data, np.ravel(target))): # Compute accuracy over k folds
    cv_weights = train_weights_wthreshold(data[train], np.ravel(target)[train], l_rate, n_epoch, 0)
    errors = 0
    for i in test:
      cv_prediction = predict_wthreshold(data[i], cv_weights, 0)
      if (cv_prediction != int(target[i])):
        errors += 1
      
    m_scores[k] = 1 - (errors / test.size) # Checking for errors and computing score

    print("Fold: " + str(k) + " Score: " + str(m_scores[k]))
  print("Accuracy: %0.2f (+/- %0.2f) " % (np.asarray(m_scores).mean(), np.asarray(m_scores).std()))

  roc_predict = np.zeros(39, dtype=int)

  #ROC Curve
  scale = 0.7
  pts_to_plt = int(80 * scale)
  fpr = np.zeros(pts_to_plt, dtype=float)
  tpr = np.zeros(pts_to_plt, dtype=float)
  plt.figure()
  thresh = -18
  thresh_tries = np.zeros(pts_to_plt, dtype=float)
  for j in range(40):
    fp = 0
    tp = 0
    fn = 0
    tn = 0
    for i in range (0,39):
      roc_predict[i] = predict_wthreshold(data[i], weights, thresh)
      if(int(np.asarray(target)[i]) == 0):
        if(roc_predict[i] != 0):
          fp += 1
        else:
          tn += 1
      else:
        if(roc_predict[i] != 1):
          fn += 1
        else:
          tp += 1
    
    if debug : print("Threshold: "+str(thresh))
    fpr[j] = fp/(fp+tn)
    tpr[j] = tp/(tp+fn)
    thresh_tries[j] = thresh
    if debug: print("FPR: " + str(fp/(fp+tn)))
    if debug: print("TPR: " + str(tp/(tp+fn)))
    if((thresh >= -10.5) and (thresh <= -9.5)): # Because of the quick drop in this range, trying to reduce granularity
      thresh += 0.07
    else:
      thresh += 1/scale

  weights_wo_bias = weights[1:]
  sorted_weights = np.argsort(np.asarray(weights_wo_bias)).ravel()
  abs_sorted_weights = np.argsort(abs(np.asarray(weights_wo_bias))).ravel()

  sorted_fh = open("Perceptron-sorted_roc.txt", "w")
  abs_sorted_fh = open("Perceptron-absolute-sorted_roc.txt", "w")
  
  for i in np.nditer(sorted_weights):
    print(feature_list[i], file=sorted_fh)

  for i in np.nditer(abs_sorted_weights):
    print(feature_list[i], file=abs_sorted_fh)

  #ROC PLOT
  plt.plot(fpr, tpr, lw = 2, ds='steps', label='perceptron')
  plt.plot([0,1],[0,1],'r--')
  plt.xlim([0,1.0])
  plt.ylim([0,1.0])
  plt.xlabel('FALSE POSITIVE RATE')
  plt.ylabel('TRUE POSITIVE RATE')
  plt.title('ROC')
  plt.legend(loc='lower right')
  plt.savefig('roc')
  plt.show()

  print("Post ROC based training")
  #Retraining Perceptron  
  weights = train_weights_wthreshold(data, np.ravel(target), l_rate, n_epoch, -9.428)
  m_scores = np.zeros(num_folds)
  for k, (train, test) in enumerate(k_fold.split(data, np.ravel(target))): # Compute accuracy over k folds
    cv_weights = train_weights_wthreshold(data[train], np.ravel(target)[train], l_rate, n_epoch, -9.428)
    errors = 0
    for i in test:
      cv_prediction = predict_wthreshold(data[i], cv_weights, -9.428)
      if (cv_prediction != int(target[i])):
        errors += 1
      
    m_scores[k] = 1 - (errors / test.size) # Checking for errors and computing score

    print("Fold: " + str(k) + " Score: " + str(m_scores[k]))
  print("Accuracy: %0.2f (+/- %0.2f) " % (np.asarray(m_scores).mean(), np.asarray(m_scores).std()))



elif (model == 2): # Perceptron - sklearn
  estimator = Perceptron(tol=1e-3, random_state = 0) # randomizes order in which the data is used for training the perceptron
  estimator = estimator.fit(data, np.ravel(target))
  sorted_weights = np.argsort(estimator.coef_).ravel()
  abs_sorted_weights = np.argsort(abs(estimator.coef_)).ravel()
  
  sorted_fh = open("Perceptron-sorted-sk.txt", "w")
  abs_sorted_fh = open("Perceptron-absolute-sorted-sk.txt", "w")
  
  for i in np.nditer(sorted_weights):
    print(feature_list[i], file=sorted_fh)

  for i in np.nditer(abs_sorted_weights):
    print(feature_list[i], file=abs_sorted_fh)

  m_scores = np.zeros(num_folds)
  k_fold = StratifiedKFold(num_folds, shuffle=False) # K-fold data split function from sklearn
  
  for k, (train, test) in enumerate(k_fold.split(data, np.ravel(target))): # Compute accuracy over k folds
    estimator.fit(data[train], np.ravel(target)[train])
    prediction = estimator.predict(data[test])
    yTest = np.ravel(target)[test] # target labels - test set
    m_scores[k] = 1 - (np.count_nonzero(prediction.astype(int) ^ yTest.astype(int)) / test.size) # Checking for errors and computing score

    print("Fold: " + str(k) + " Score: " + str(m_scores[k]))
  print("Accuracy: %0.2f (+/- %0.2f) " % (np.asarray(m_scores).mean(), np.asarray(m_scores).std()))

elif(model == 3):
  feature_split_file = "splits.txt"
  feature_split_fh = open(feature_split_file, "r")

  splits = feature_split_fh.read()
  feature_split_fh.close()
  splits = splits.split('\n')

  valid_splits = splits.copy()

  for i in splits:
    split_info = i.split(':')
    data_cols = split_info[1].split() # Columns in data.csv corressponding to this split
    estimator = Id3Estimator()
    
    estimator = estimator.fit(data[:,np.asarray(list(map(int,data_cols)))], np.ravel(target))
    tree = export_graphviz(estimator.tree_, 'data/'+ split_info[0] +'_'+'tree.dot', np.asarray(feature_list)[np.asarray(list(map(int,data_cols)))])
    tree_fh = open('data/'+ split_info[0] +'_'+'tree.dot', "r")
    valid = True # Flag for discarding inalid tree
    for line in tree_fh.readlines():
      if (re.match('(.*/.*)', line)):
        print("Split: " + split_info[0] + " does not completely classify the data")
        valid = False
        break

    tree_fh.close()
    if(not valid):
      os.remove('data/'+ split_info[0] +'_'+'tree.dot')
      valid_splits.remove(i)
    else: # Report Kfold validation accuracy
      m_scores = np.zeros(num_folds)
      k_fold = StratifiedKFold(num_folds, shuffle=True) # K-fold data split function from sklearn
      print(split_info[0] +' decision tree')
      for k, (train, test) in enumerate(k_fold.split(data, np.ravel(target))): # Compute accuracy over k folds
        estimator.fit(data[train], np.ravel(target)[train])
        prediction = estimator.predict(data[test])
        yTest = np.ravel(target)[test] # target labels - test set
        m_scores[k] = 1 - (np.count_nonzero(prediction.astype(int) ^ yTest.astype(int)) / test.size) # Checking for errors and computing score

        print("Fold: " + str(k) + " Score: " + str(m_scores[k]))
      print("Accuracy: %0.2f (+/- %0.2f) " % (np.asarray(m_scores).mean(), np.asarray(m_scores).std()))  

  print("Valid trees:")
  for element in valid_splits:
    print(element.split(':')[0] + "_tree")

    
# Ensemble methods
if(model == 4):
  estimator = BaggingClassifier(Id3Estimator(), n_estimators=5, max_samples=0.7, bootstrap=True)
  m_scores = np.zeros(num_folds)
  k_fold = StratifiedKFold(num_folds, shuffle=False) # K-fold data split function from sklearn
  
  for k, (train, test) in enumerate(k_fold.split(data, np.ravel(target))): # Compute accuracy over k folds
    estimator.fit(data[train], np.ravel(target)[train])
    prediction = estimator.predict(data[test])
    yTest = np.ravel(target)[test] # target labels - test set
    m_scores[k] = 1 - (np.count_nonzero(prediction.astype(int) ^ yTest.astype(int)) / test.size) # Checking for errors and computing score

    print("Fold: " + str(k) + " Score: " + str(m_scores[k]))
  print("Accuracy: %0.2f (+/- %0.2f) " % (np.asarray(m_scores).mean(), np.asarray(m_scores).std()))

 
if(model == 5):
# Instead of a global decision tree, trying to work with local trees and using these to train a perceptron - Experiment
# Extending model == 3
  feature_split_file = "splits.txt"
  feature_split_fh = open(feature_split_file, "r")

  splits = feature_split_fh.read()
  feature_split_fh.close()
  splits = splits.split('\n')

  valid_splits = splits.copy()

  percep_data = np.zeros((39,splits.__len__()), dtype='int')
  l = 0
  for i in splits:
    split_info = i.split(':')
    data_cols = split_info[1].split() # Columns in data.csv corressponding to this split
    local_est_data = np.zeros((39,data_cols.__len__()), dtype='float')
    estimator = Id3Estimator()

    for x in range(39):
      local_est_data[x,:] = data[x,np.asarray(list(map(int,data_cols)))]


    estimator = estimator.fit(local_est_data, np.ravel(target))
    tree = export_graphviz(estimator.tree_, 'data/'+ split_info[0] +'_'+'tree.dot', np.asarray(feature_list)[np.asarray(list(map(int,data_cols)))])
    tree_fh = open('data/'+ split_info[0] +'_'+'tree.dot', "r")
    percep_data[:,l] = estimator.predict(local_est_data)
    l += 1

  weights = train_weights_wthreshold(percep_data, np.asarray(target), l_rate, n_epoch, 0)
  weights_wo_bias = weights[1:]
  sorted_weights = np.argsort(np.asarray(weights_wo_bias)).ravel()
  abs_sorted_weights = np.argsort(abs(np.asarray(weights_wo_bias))).ravel()

  sorted_fh = open("Perceptron-sorted_ensemble.txt", "w")
  abs_sorted_fh = open("Perceptron-absolute-sorted_ensemble.txt", "w")
  
  for i in np.nditer(sorted_weights):
    print(splits[i].split(':')[0], file=sorted_fh)

  for i in np.nditer(abs_sorted_weights):
    print(splits[i].split(':')[0], file=abs_sorted_fh)  
  
  
  k_fold = StratifiedKFold(num_folds, shuffle=False) # K-fold data split function from sklearn
  m_scores = np.zeros(num_folds)

  for k, (train, test) in enumerate(k_fold.split(data, np.ravel(target))): # Compute accuracy over k folds
    percep_data = np.zeros((train.size,splits.__len__()), dtype='int')
    percep_data_test = np.zeros((test.size,splits.__len__()), dtype='int')
    l = 0
    for i in splits:
      split_info = i.split(':')
      data_cols = split_info[1].split() # Columns in data.csv corressponding to this split
      local_est_data = np.zeros((39,data_cols.__len__()), dtype='float')
      estimator = Id3Estimator()

      for x in range(39):
        local_est_data[x,:] = data[x,np.asarray(list(map(int,data_cols)))]


      estimator = estimator.fit(local_est_data[train], np.ravel(target)[train])
      percep_data[:,l] = estimator.predict(local_est_data[train])
      percep_data_test[:,l] = estimator.predict(local_est_data[test])
      l += 1

    weights = train_weights_wthreshold(percep_data, np.asarray(target)[train], l_rate, n_epoch, 0)
    errors = 0
    for x, i in enumerate(test):
      prediction = predict_wthreshold(percep_data_test[x,:], weights, 0) #Does not make sense
      if (prediction != int(target[i])):
        errors += 1
      
    m_scores[k] = 1 - (errors / test.size) # Checking for errors and computing score
    
    print("Fold: " + str(k) + " Score: " + str(m_scores[k]))
  print("Accuracy: %0.2f (+/- %0.2f) " % (np.asarray(m_scores).mean(), np.asarray(m_scores).std()))
