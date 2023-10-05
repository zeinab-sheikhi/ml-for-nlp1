import numpy as np

from collections import Counter
from utils.matrix_utils import cos_similarity, euclidean_distance
from utils.math_utils import inverse


class KNNClassifier:

    def __init__(self, n_neighbors=1, cos_or_dist=True, use_weight=False):
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
        """
        self.k = n_neighbors
        self.use_cosine = cos_or_dist
        self.use_weight = use_weight

    def fit(self, X, y, indices):

        self.X_train = X
        self.y_train = y
        self.weights = np.identity(X.shape[1])
        self.indices = indices
        return self

    def predict(self, X_test):
        
        num_samples = np.shape(X_test)[0]
        y_pred = np.zeros((num_samples, 1))
        distance_matrix = self.build_distance_matrix(X_test)
        sorted_distance_indices = self.sort_distance(distance_matrix)
        
        if self.use_weight:
            self.weights = self.build_weight_matrix(distance_matrix)
            sorted_weights = self.sort_weight(sorted_distance_indices)
        
        for sample in range(num_samples):
            k_neighbors_indices = []
            k_neighbors_weights = []
            k_neighbors_classes = []
            
            for idx in sorted_distance_indices[sample][:self.k]:
                k_neighbors_indices.append(self.y_train[idx][0])
                if self.use_weight:
                    k_neighbors_weights.append(sorted_weights[sample][idx])
                        
            k_neighbors_classes = dict(Counter(sorted([self.indices.class_from_index(index)[0] for index in k_neighbors_indices])))
            estimated_class = self.get_gold_class(k_neighbors_classes, k_neighbors_weights, use_weight=self.use_weight)
            y_pred[sample] = self.indices.index_from_class(estimated_class)
            
        return y_pred

    def build_distance_matrix(self, X_test):
        
        if self.use_cosine:
            return cos_similarity(X_test, self.X_train)
        else:
            return euclidean_distance(X_test, self.X_train.T)

    def sort_distance(self, distance_matrix):
        if self.use_cosine:
            return np.argsort(-distance_matrix) 
        else:
            return np.argsort(distance_matrix) 

    def build_weight_matrix(self, distance_matrix):
        if self.use_cosine:
            return distance_matrix
        else:
            return np.vectorize(inverse)(distance_matrix)

    def sort_weight(self, sort_indices):
        sorted_weights = np.zeros(self.weights.shape)
        for i in range(0, self.weights.shape[0]):
            sorted_weights[i] = self.weights[i][sort_indices[i]]
        
        return sorted_weights        
    
    def get_gold_class(self, k_neighbors_classes, k_neighbors_weights, use_weight=False):
        
        max_value = max(k_neighbors_classes.values())
        indices_of_classes_with_max_value = [list(k_neighbors_classes.keys()).index(key) for key, value in k_neighbors_classes.items() if value == max_value]
        
        index_of_max_value = 0
        
        if len(indices_of_classes_with_max_value) == 1 and not use_weight:
            index_of_max_value = indices_of_classes_with_max_value[0]
        else:            
            index_of_max_value = 0
            maximum = 0
            for i in indices_of_classes_with_max_value:
                max_weight = k_neighbors_weights[i]
                if max_weight > maximum:
                    maximum = max_weight 
                    index_of_max_value = i

        estimated_class = list(k_neighbors_classes.keys())[index_of_max_value]
        return estimated_class
    
    def evaluate(self, y_real, y_pred):
        
        num_of_samples = np.shape(y_real)[0]
        num_correct_preds = 0
        for sample in range(0, num_of_samples):
            if y_real[sample] == y_pred[sample]:
                num_correct_preds += 1
        accuracy = num_correct_preds / num_of_samples
        return accuracy
