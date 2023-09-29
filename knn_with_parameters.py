import argparse
import statistics
import numpy as np

class KNNClassifier:    
    
    def __init__(self, n_neighbors=1,  use_weight=False, metric='cos', indices=None):
    
        """
        Parameters:
    
    n_neighbors : int, default=1
        Number of neighboring samples to use for imputation
    
    use_weight : boolean, default=False

    metric : {'cos'}, default='cos'
        Distance metric for searching neighbors. Possible values:

        - 'cos'  : cos similarity between Xtrain and Xtest
        - 'dist' : distance 
    
    indices: Indices, default=None

    """
        self.k = n_neighbors
        self.use_weight = use_weight
        self.metric = metric
        self.indices = indices

    def fit(self, X, y, indices):
        
        self.X_train_norm = normalize_row_vectors(X)
        self.y_train = y
        self.indices = indices
        return self

    def predict(self, X_test):
        
        num_samples = np.shape(X_test)[0]
        X_test_norm = normalize_row_vectors(X_test)
        cos_matrix = np.matmul(X_test_norm, self.X_train_norm.T)
        cos_sorted_indices = np.argsort(-cos_matrix)
        
        for sample in range(num_samples):
                # y_real = y_test[sample][0]
                k_neighbors_indices = [self.y_train[idx][0] for idx in cos_sorted_indices[sample][:self.k]]
                k_neighbors_classes = sorted([self.indices.class_from_index(index)[0] for index in k_neighbors_indices])
                estimated_class = statistics.mode(k_neighbors_classes)
                y_est = self.indices.index_from_class(estimated_class)
                # if y_est == y_real:
                #     num_correct_estimation += 1
        return y_est

    
    def evaluate(self, y_test, y_est):
        print(y_test)
        print(y_est)
        y_real = y_test[sample][0]
        # self.X_train = X
        # self.y_train = y


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

                
    # def evaluate(self, X_test, y_test, indices):

    #     num_test_points = np.shape(X_test)[0]
    #     num_correct_estimation = 0  
    #     accuracies = []          

    #     X_test_normalized = normalize_row_vectors(X_test)
    #     cos_matrix = np.matmul(X_test_normalized, self.X_train_normalized.T)
    #     cos_sorted_indices = np.argsort(-cos_matrix)
    
    #     for k in range(1, self.k + 1):
    #         num_correct_estimation = 0            
    #         for sample in range(num_test_points):
    #             y_real = y_test[sample][0]
    #             k_neighbors_indices = [self.y_train[idx][0] for idx in cos_sorted_indices[sample][:k]]
    #             k_neighbors_classes = sorted([indices.class_from_index(index)[0] for index in k_neighbors_indices])
    #             estimated_class = statistics.mode(k_neighbors_classes)
    #             y_est = indices.index_from_class(estimated_class)
    #             if y_est == y_real:
    #                 num_correct_estimation += 1

    #         accuracy = num_correct_estimation / num_test_points
    #         accuracies.append(accuracy)
    #         print(f"Accuracy for K = {k} is: {accuracy} ({num_correct_estimation}/{num_test_points})")
        
    #     return accuracies


def normalize_row_vectors(X):
        row_square_sum = np.sum(X*X, axis=1)
        row_sqrt = np.sqrt(row_square_sum)
        X_normalized = np.divide(X,row_sqrt[:,None], out=X, where=X!=0) 
        return X_normalized
