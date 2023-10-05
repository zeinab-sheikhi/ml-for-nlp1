import argparse

from models.indices import Indices
from knn_document_classifier import KNNClassifier
from utils.utils import read_examples, bow_with_tfidf, build_matrix

usage = """ Document Classification with KNN

  prog [options] TRAIN_FILE TEST_FILE

  TRAIN_FILE and TEST_FILE must have *.examples format 

"""

parser = argparse.ArgumentParser(usage=usage)
parser.add_argument('train_file', help='File d\'exemples, utilized as neighbors', default=None)
parser.add_argument('test_file', help='File d\'exemples, utilized for KNN evaluation', default=None)
parser.add_argument("-k", '--k', default=1, type=int, help='Hyperparameter K : number of neighbors considered for classification. Default=1')
parser.add_argument("-m", '--metric', action="store_false", default=True, help='Hyperparameter cos_or_dist :  whether to use cos similarity or distance. Default=cos')
parser.add_argument('-w', '--use_weight', action="store_true", default=False, help="Hyperparameter use_weight: whether or not to weight the neighbors. Default=False")
parser.add_argument('-i', '--use_idf', action="store_true", default=False, help="Hyperparameter use_idf: whether or not to use TF.IDF values instead of TF in the BOW vectors. Default=False")
args = parser.parse_args()

indices = Indices()

train_documents = read_examples(args.train_file, indices)
dev_documents = read_examples(args.test_file)

if args.use_idf:
    bow_with_tfidf(train_documents, indices)

(X_train, y_train) = build_matrix(train_documents, indices)
(X_test, y_test) = build_matrix(dev_documents, indices, is_train=False)


knn = KNNClassifier(n_neighbors=args.k, use_weight=args.use_weight, cos_or_dist=args.metric)
knn.fit(X_train, y_train, indices)
y_pred = knn.predict(X_test)
accuracy = knn.evaluate(y_test, y_pred)
print(f"for K={knn.k} the accuracy is: {accuracy}")

# In case of getting accuracies for different Ks, run the below code
# for k in range(1, args.k + 1):
#     knn = KNNClassifier(n_neighbors=k, use_weight=True, cos_or_dist=True)
#     knn.fit(X_train, y_train, indices)
#     y_pred = knn.predict(X_test)
#     accuracy = knn.evaluate(y_test, y_pred)
#     print(f"for K={knn.k} the accuracy is: {accuracy}")

