# # First version of Knn
# class KNN:

#     def __init__(self, X, y, k=1, neighbors_weight=False, trace=False):
        
#         self.X_train = X
#         self.X_train_normalized = normalize_row_vectors(X)
#         self.y_train = y
#         self.k = k
#         self.neighbors_weight = neighbors_weight
#         self.trace = trace

#     def evaluate(self, X_test, y_test, indices):

#         num_test_points = np.shape(X_test)[0]
#         num_correct_estimation = 0  
#         accuracies = []          

#         X_test_normalized = normalize_row_vectors(X_test)
#         cos_matrix = np.matmul(X_test_normalized, self.X_train_normalized.T)
#         cos_sorted_indices = np.argsort(-cos_matrix)
    
#         for k in range(1, self.k + 1):
#             num_correct_estimation = 0            
#             for sample in range(num_test_points):
#                 y_real = y_test[sample][0]
#                 k_neighbors_indices = [self.y_train[idx][0] for idx in cos_sorted_indices[sample][:k]]
#                 k_neighbors_classes = sorted([indices.class_from_index(index)[0] for index in k_neighbors_indices])
#                 estimated_class = statistics.mode(k_neighbors_classes)
#                 y_est = indices.index_from_class(estimated_class)
#                 if y_est == y_real:
#                     num_correct_estimation += 1

#             accuracy = num_correct_estimation / num_test_points
#             accuracies.append(accuracy)
#             print(f"Accuracy for K = {k} is: {accuracy} ({num_correct_estimation}/{num_test_points})")
        
#         return accuracies

# knn_classifier = KNN(X=X_train, y=y_train, k=args.k, neighbors_weight = args.weight_neighbors, trace=args.trace)
# score = knn_classifier.evaluate(X_test, y_test, indices)
