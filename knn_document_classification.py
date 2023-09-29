import argparse
import statistics
import numpy as np

class Example:
    
    def __init__(self, example_number, gold_class):
        self.example_number = example_number
        self.gold_class = gold_class
        self.vector = {}
    
    def add_feature(self, feature_name, value):
        self.vector[feature_name] = value

class Indices:
    
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
        if class_lable not in self.class_index_dict:
            self.class_index_dict[class_lable] = len(self.classes)
            self.classes.append(class_lable)
        return self.class_index_dict[class_lable]
            
    def index_from_word(self, word, create_new=False):
        if word in self.word_index_dict:
            return self.word_index_dict[word]
        if not create_new:
            return None        
        self.word_index_dict[word] = len(self.words)
        self.words.append(word)    
        return self.word_index_dict[word]
    
    def class_from_index(self, index):
        return [key for key, value in self.class_index_dict.items() if index == value]
        

def normalize_row_vectors(X):
        row_square_sum = np.sum(X*X, axis=1)
        row_sqrt = np.sqrt(row_square_sum)
        X_normalized = np.divide(X,row_sqrt[:,None], out=X, where=X!=0) 
        return X_normalized

class KNN:

    def __init__(self, X, y, k=1, neighbors_weight=False, trace=False):
        
        self.X_train = X
        self.X_train_normalized = normalize_row_vectors(X)
        self.y_train = y
        self.k = k
        self.neighbors_weight = neighbors_weight
        self.trace = trace

    def evaluate(self, X_test, y_test, indices):

        num_test_points = np.shape(X_test)[0]
        num_correct_estimation = 0  
        accuracies = []          

        X_test_normalized = normalize_row_vectors(X_test)
        cos_matrix = np.matmul(X_test_normalized, self.X_train_normalized.T)
        cos_sorted_indices = np.argsort(-cos_matrix)
    
        for k in range(1, self.k + 1):
            num_correct_estimation = 0            
            for sample in range(num_test_points):
                y_real = y_test[sample][0]
                k_neighbors_indices = [self.y_train[idx][0] for idx in cos_sorted_indices[sample][:k]]
                k_neighbors_classes = sorted([indices.class_from_index(index)[0] for index in k_neighbors_indices])
                estimated_class = statistics.mode(k_neighbors_classes)
                y_est = indices.index_from_class(estimated_class)
                if y_est == y_real:
                    num_correct_estimation += 1

            accuracy = num_correct_estimation / num_test_points
            accuracies.append(accuracy)
            print(f"Accuracy for K = {k} is: {accuracy} ({num_correct_estimation}/{num_test_points})")
        
        return accuracies


def read_examples(infile, indices=None):
    
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

def build_matrices(examples, indices, is_train=True):
    
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
parser.add_argument('train_file', help='fichier d\'exemples, utilized as neighbors', default=None)
parser.add_argument('test_file', help='fichier d\'exemples, utilized for KNN evaluation', default=None)
parser.add_argument("-k", '--k', default=1, type=int, help='Hyperparameter K : number of neighbors for classification(all the values of k will be tested). Default=1')
parser.add_argument('-v', '--trace',action="store_true",default=False,help="A utiliser pour déclencher un mode verbeux. Default=False")
parser.add_argument('-w', '--weight_neighbors', action="store_true", default=False,help="Pondération des voisins. Default=False")
args = parser.parse_args()

indices = Indices()
train_examples = read_examples(args.train_file, indices)
test_examples = read_examples(args.test_file)

(X_train, y_train) = build_matrices(train_examples, indices)
(X_test, y_test) = build_matrices(test_examples, indices, is_train=False)

# knn_classifier = KNN(X=X_train, y=y_train, k=args.k, neighbors_weight = args.weight_neighbors, trace=args.trace)
# score = knn_classifier.evaluate(X_test, y_test, indices)




















class KNNClassifier:    
    
    def __init__(self, n_neighbors=1,  use_weight=False, metric='cos'):
    
        """
    Parameters:
    
    n_neighbors : int, default=1
        Number of neighboring samples

    use_weight : boolean, default=False

    metric : {'cos', 'dist'}, default='cos'
        Distance metric for searching neighbors. Possible values:
        - 'cos'  : cos similarity between Xtrain and Xtest
        - 'dist' : distance 
    
    """
        self.k = n_neighbors
        self.use_weight = use_weight
        self.metric = metric

    def fit(self, X, y, indices):
        
        self.X_train_norm = normalize_row_vectors(X)
        self.y_train = y
        self.indices = indices
        return self

    def predict(self, X_test):
        
        num_samples = np.shape(X_test)[0]
        X_test_norm = normalize_row_vectors(X_test)
        y_est = np.zeros((num_samples,1))
        cos_matrix = np.matmul(X_test_norm, self.X_train_norm.T)
        cos_sorted_indices = np.argsort(-cos_matrix)
                    
        for sample in range(num_samples):
                k_neighbors_indices = [self.y_train[idx][0] for idx in cos_sorted_indices[sample][:self.k]]
                k_neighbors_classes = sorted([self.indices.class_from_index(index)[0] for index in k_neighbors_indices])
                estimated_class = statistics.mode(k_neighbors_classes)
                y_est[sample] = self.indices.index_from_class(estimated_class)
        
        return y_est

    
    def evaluate(self, y_true, y_pred):
        num_of_samples = np.shape(y_true)[0]
        num_correct_preds = 0
        for sample in range(0,num_of_samples):
            if y_true[sample] == y_pred[sample]:
                num_correct_preds += 1
        accuracy = num_correct_preds / num_of_samples
            
        return accuracy


knn = KNNClassifier(n_neighbors=args.k)
knn.fit(X_train, y_train, indices)
y_pred = knn.predict(X_test)
accuracy = knn.evaluate(y_test, y_pred)
print(accuracy)
