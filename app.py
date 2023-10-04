import argparse
import numpy as np 

from models.example import Example
from models.indices import Indices
from knn_document_classifier import KNNClassifier


def read_examples(infile, indices=None):
    """ 
        Reads a .examples file and returns a list of Example instances 
        if indices is not None but an instance of Indices, 
        it is updated with potentially new words/indices while reading the examples
    """
    stream = open(infile)
    example = None
    examples = []
    update_indices = (indices is not None)

    while 1:
        line = stream.readline()
        if not line:
            break
        line = line[0:-1]

        if line.startswith("EXAMPLE_NB"): 
            columns = line.split('\t')
            gold_class = columns[3]
            example_number = columns[1]
            example = Example(example_number, gold_class)
            examples.append(example)
            if update_indices:
                indices.add_class(gold_class)
        
        elif line and example is not None:
            (word, value) = line.split('\t')
            example.add_feature(word, float(value.replace('x', '')))
            if update_indices:
                indices.add_word(word)  

    return examples


def build_matrix(examples, indices, is_train=True):
    
    num_of_rows = len(examples)
    num_of_columns = indices.get_words_size()

    X = np.zeros((num_of_rows, num_of_columns))
    y = np.zeros((num_of_rows, 1))

    for index, example in enumerate(examples):
        for word, value in example.vector.items():
            feature_index = indices.index_from_word(word, create_new=is_train)
            if feature_index is not None:
                X[index][feature_index] = value
            gold_class = indices.index_from_class(example.gold_class)
            y[index, 0] = gold_class  
    
    return (X, y)


usage = """ Document Classification with KNN

  prog [options] TRAIN_FILE TEST_FILE

  TRAIN_FILE and TEST_FILE must have *.examples format 

"""

parser = argparse.ArgumentParser(usage=usage)
parser.add_argument('train_file', help='File d\'exemples, utilized as neighbors', default=None)
parser.add_argument('test_file', help='File d\'exemples, utilized for KNN evaluation', default=None)
parser.add_argument("-k", '--k', default=1, type=int, help='Hyperparameter K : number of neighbors for classification(all the values of k will be tested). Default=1')
parser.add_argument('-w', '--weight_neighbors', action="store_true", default=False, help="Weighting the neighbors. Default=False")
args = parser.parse_args()

indices = Indices()
train_examples = read_examples(args.train_file, indices)
test_examples = read_examples(args.test_file)

(X_train, y_train) = build_matrix(train_examples, indices)
(X_test, y_test) = build_matrix(test_examples, indices, is_train=False)

knn = KNNClassifier(n_neighbors=args.k)
knn.fit(X_train, y_train, indices)
# y_pred = knn.predict(X_test)
knn.predict(X_test)
# accuracy = knn.evaluate(y_test, y_pred)
# print(f"for K={knn.k} the accuracy is: {accuracy}")

# In case of getting accuracies for different Ks, run the below code
# for k in range(1,args.k + 1):
#     knn = KNNClassifier(n_neighbors=k)
#     knn.fit(X_train, y_train, indices)
#     y_pred = knn.predict(X_test)
#     accuracy = knn.evaluate(y_test, y_pred)
#     print(f"for K={k} the accuracy is: {accuracy}")