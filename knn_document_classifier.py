import statistics
import numpy as np

from utils.matrix_utils import cos_similarity, euclidean_distance
from utils.math_utils import inverse


class KNNClassifier:

    def __init__(self, n_neighbors=1, cos_or_dist=True, use_weight=False, idf=False):
        """
        Parameters:

        n_neighbors : int, default=1
            Number of neighbors to consider

        cos_or_dist: boolean, default=True
            Distance metric for searching neighbors. 
            if True, use cosine similarity, else use distance 

        use_weight : boolean, default=False
            if True, weight points by the inverse of their distance
            if False, all points in each neighborhood are weighted equally.
        
        idf : boolean, default=False
            Use TF.IDF values in the BOW vectors
        """
        self.k = n_neighbors
        self.use_cos = cos_or_dist
        self.use_weight = use_weight
        self.idf = idf

    def fit(self, X, y, indices):

        self.X_train = X
        self.y_train = y
        self.weights = np.identity(X.shape[1])
        self.indices = indices
        return self

    def predict(self, X_test):
        
        num_samples = np.shape(X_test)[0]
        y_pred = np.zeros((num_samples, 1))
        cos_matrix = cos_similarity(X_test, self.X_train)
        cos_sorted_indices = np.argsort(-cos_matrix)       
        euclidean_matrix = euclidean_distance(X_test, self.X_train.T)
        weights = np.vectorize(inverse)(euclidean_matrix)
        weights_sorted = np.zeros(weights.shape)
        # print(cos_matrix)
        print(euclidean_matrix)
        print(weights)
        print(weights_sorted)
    
        for i in range(0, weights.shape[0]):
            weights_sorted[i] = weights[i][cos_sorted_indices[i]]
        
        for sample in range(num_samples):
            k_neighbors_indices = [self.y_train[idx][0] for idx in cos_sorted_indices[sample][:self.k]]
            k_neighbors_classes = sorted([self.indices.class_from_index(index)[0] for index in k_neighbors_indices])
            # print(k_neighbors_classes)
            estimated_class = statistics.mode(k_neighbors_classes)
            y_pred[sample] = self.indices.index_from_class(estimated_class)
        
        return y_pred

    def evaluate(self, y_real, y_pred):
        num_of_samples = np.shape(y_real)[0]
        num_correct_preds = 0
        for sample in range(0, num_of_samples):
            if y_real[sample] == y_pred[sample]:
                num_correct_preds += 1
        accuracy = num_correct_preds / num_of_samples
        return accuracy
