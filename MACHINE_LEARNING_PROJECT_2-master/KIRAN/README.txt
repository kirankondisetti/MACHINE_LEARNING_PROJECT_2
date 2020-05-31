The python script for assignment 2 has a few dependencies:
  1. sklean - https://scikit-learn.org/stable/
  2. id3 - https://pypi.org/project/decision-tree-id3/
  3. matplotlib - https://matplotlib.org/

To run a model, the parameter on line 25 in dt_plus_ensemble.py needs to be modified

Set to choose model
  0: id3 - decision tree
  1: perceptron - modified internet
  2: perceptron - sklearn
  3: split decision trees
  4: ensemble - bagging
  5: ensemble - local trees + perceptron

To run the script:
  python dt_plus_ensemble.py

For each model, the k-fold cross-validation metrics will be printed on the screen along with the accuracy

For model = 1 - perceptron
The ROC curve plotted will be displayed and the window needs to be closed inorder to proceed. The prompts include the k-fold validation metrics before and after changing the threshold.
The sorted weights can be found in:
  1. Perceptron-sorted.txt
  2. Perceptron-absolute-sorted.txt

For model = 3 - split decision trees
The k-fold cross validation metrics are output for each tree. The dot files can be found in the data directory

For model = 5
As this is an extension of model = 3, a decision tree for each of the splits is created and can be found in the data directory
The sorted weights can be found in:
  1. Perceptron-sorted_ensemble.txt
  2. Perceptron-absolute-sorted_ensemble.txt