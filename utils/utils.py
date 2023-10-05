import numpy as np

from models.document import Document


def read_examples(infile, indices=None):
    """ 
        Reads a .examples file and returns a list of Example instances 
        if indices is not None but an instance of Indices, 
        it is updated with potentially new words/indices while reading the examples
    """
    stream = open(infile)
    document = None
    documents = []
    update_indices = (indices is not None)

    while 1:
        line = stream.readline()
        if not line:
            break
        line = line[0:-1]

        if line.startswith("EXAMPLE_NB"): 
            columns = line.split('\t')
            gold_class = columns[3]
            document_number = columns[1]
            document = Document(document_number, gold_class)
            documents.append(document)
            if update_indices:
                indices.add_class(gold_class)
        
        elif line and document is not None:
            (word, value) = line.split('\t')
            document.add_word_tf(word, float(value.replace('x', '')))
            if update_indices:
                indices.add_word(word)  
                indices.update_df(word)

    return documents


def bow_with_tfidf(documents_list, indices):
    """
        use tf.idf value in the BOW value
    """
    indices.create_idf(total_docs_num=len(documents_list))
    for document in documents_list:
        for word, tf in document.word_tf.items():
            idf = indices.idf[word]
            document.word_tf[word] = tf * idf
            
    return documents_list


def build_matrix(documents, indices, is_train=True):
    
    num_of_rows = len(documents)
    num_of_columns = indices.get_words_size()

    X = np.zeros((num_of_rows, num_of_columns))
    y = np.zeros((num_of_rows, 1))

    for index, document in enumerate(documents):
        for word, value in document.word_tf.items():
            feature_index = indices.index_from_word(word, create_new=is_train)
            if feature_index is not None:
                X[index][feature_index] = value
            gold_class = indices.index_from_class(document.gold_class)
            y[index, 0] = gold_class  
    
    return (X, y)
