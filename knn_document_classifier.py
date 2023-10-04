import argparse
import statistics
import numpy as np

class Example:
    """
        Parameters:
        ---------- 
        - example_number : int

        - vector : {feature: float value}
            vector representation of features and their corresponding non-null values

        - gold_class : str
            the corresponsing gold class for this example
    """
    def __init__(self, example_number, gold_class):
        self.example_number = example_number
        self.gold_class = gold_class
        self.vector = {}
    
    def add_feature(self, feature_name, value):
        self.vector[feature_name] = value

class Indices:
    """ 
        class to handle the correspondences from documents' classes to indices and from words to indices correspondences

        Parameters: 
        ----------
        - classes: list
            list containing the corresponding classes for documents
        
        - class_index_dict : dict
            dict representation of classes and their corresponding indices
        
        - words : list
            unique words in the whole dataset to be considered as the features
        
        - word_index_dict : dict
            dict representation of the features and their corresponding indices in the BOW vector
    """
    def __init__(self):
        self.classes = []
        self.class_index_dict = {}
        self.words = []
        self.word_index_dict = {}
    
    def add_word(self, word):
        if word not in self.word_index_dict:
            self.word_index_dict[word] = len(self.words)
            self.words.append(word)
    
    def add_class(self, class_lable):
        if class_lable not in self.class_index_dict:
            self.class_index_dict[class_lable] = len(self.classes)
            self.classes.append(class_lable)  
    
    def get_words_size(self):
        return len(self.words)
    
    def get_classes_size(self):
        return len(self.classes)
    
    def index_from_class(self, class_lable):

        """ if class_label is already known: return its index
            otherwise, add it to the classes list and return its new index
        """

        if class_lable not in self.class_index_dict:
            self.class_index_dict[class_lable] = len(self.classes)
            self.classes.append(class_lable)
        return self.class_index_dict[class_lable]
            
    def index_from_word(self, word, create_new=False):
        
        """ if word is already known : returns its index
            otherwise, either add it to the words list and return its new index,
            or returns None, if create_new is True
        """
        
        if word in self.word_index_dict:
            return self.word_index_dict[word]
        if not create_new:
            return None        
        self.word_index_dict[word] = len(self.words)
        self.words.append(word)    
        return self.word_index_dict[word]
    
    def class_from_index(self, index):
        return [key for key, value in self.class_index_dict.items() if index == value]
        
class KNNClassifier:    
    
    def __init__(self, n_neighbors=1,  use_weight=False, metric='cos', idf=False):
        """
        Parameters:
        
        n_neighbors : int, default=1
            Number of neighbors to consider

        use_weight : boolean, default=False
            if True, weight points by the inverse of their distance
            if False, all points in each neighborhood are weighted equally.

        metric : {'cos', 'dist'}, default='cos'
            Distance metric for searching neighbors. Possible values:
            - 'cos'  : cos similarity between Xtrain and Xtest
            - 'dist' : distance 
        
        idf : boolean, default=False
            Use TF.IDF values in the BOW vectors
        """
        self.k = n_neighbors
        self.use_weight = use_weight
        self.metric = metric
        self.idf = idf

    def fit(self, X, y, indices):
        
        self.X_train = X
        self.y_train = y
        self.weights = np.identity(X.shape[1])
        self.indices = indices
        return self

    def predict(self, X_test):
        
        num_samples = np.shape(X_test)[0]
        y_pred = np.zeros((num_samples,1))
        cos_matrix = cos_similarity(X_test, X_train)
        cos_sorted_indices = np.argsort(-cos_matrix)
        euclidean_matrix = euclidean_dist(X_test, X_train.T)
        weights = np.vectorize(inverse)(euclidean_matrix)
        weights_sorted = np.zeros(weights.shape)
        # print(cos_matrix)
        # print(euclidean_matrix)
        # print(weights)
    
        for i in range(0, weights.shape[0]):
            weights_sorted[i] = weights[i][cos_sorted_indices[i]]
        
        # print(weights_sorted)
        for sample in range(num_samples):
                k_neighbors_indices = [self.y_train[idx][0] for idx in cos_sorted_indices[sample][:self.k]]
                k_neighbors_classes = sorted([self.indices.class_from_index(index)[0] for index in k_neighbors_indices])
                print(k_neighbors_classes)
                estimated_class = statistics.mode(k_neighbors_classes)
                y_pred[sample] = self.indices.index_from_class(estimated_class)
        
        return y_pred

    
    def evaluate(self, y_real, y_pred):
        num_of_samples = np.shape(y_real)[0]
        num_correct_preds = 0
        for sample in range(0,num_of_samples):
            if y_real[sample] == y_pred[sample]:
                num_correct_preds += 1
        accuracy = num_correct_preds / num_of_samples
        return accuracy


def normalize_row_vectors(X):
        """
            returns the normalized version of the row vectors in X(each row vector of X divided by its norm)
        """
        row_square_sum = np.sum(X*X, axis=1)
        row_sqrt = np.sqrt(row_square_sum)
        X_normalized = np.divide(X,row_sqrt[:,None], out=X, where=X!=0) 
        return X_normalized

def cos_similarity(X1, X2):
    """
        return the cosine similarity between each row of X1 and each row of X2 matrices
    """
    X1_norm = normalize_row_vectors(X1)
    X2_norm = normalize_row_vectors(X2)
    cos_matrix = np.matmul(X1_norm, X2_norm.T)
    return cos_matrix

def euclidean_dist(X1, X2):
    """
        returns the euclidean distance between each row of X1 with each row of X2
    """
    euclidean_matrix = np.zeros((X1.shape[0], X2.shape[1]))
    for i in range(X1.shape[0]):
            for j in range(0, X2.shape[1]):
                euclidean_matrix[i, j] = np.sqrt(np.sum((X1[i] - X2[:,j]) ** 2))
    return euclidean_matrix


def inverse(n):
    return 1/n

def read_examples(infile, indices=None):
    """ 
        Reads a .examples file and returns a list of Example instances 
        if indices is not None but an instance of Indices, 
        it is updated with potentially new words/indices while reading the examples
    """
    stream = open(infile)
    example = None
    examples = []
    update_indices = (indices != None)

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
        
        elif line and example != None:
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
            if feature_index != None:
                X[index][feature_index] = value
            gold_class = indices.index_from_class(example.gold_class)
            y[index, 0] = gold_class  
    
    return (X, y)


usage = """ Document Classification with KNN

  prog [options] TRAIN_FILE TEST_FILE

  TRAIN_FILE and TEST_FILE must have *.examples format 

"""

parser = argparse.ArgumentParser(usage = usage)
parser.add_argument('train_file', help='File d\'exemples, utilized as neighbors', default=None)
parser.add_argument('test_file', help='File d\'exemples, utilized for KNN evaluation', default=None)
parser.add_argument("-k", '--k', default=1, type=int, help='Hyperparameter K : number of neighbors for classification(all the values of k will be tested). Default=1')
parser.add_argument('-w', '--weight_neighbors', action="store_true", default=False,help="Weighting the neighbors. Default=False")
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