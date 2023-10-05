from utils.math_utils import log


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
        
        - idf: dict
            dict representation of inverse document frequency for each word
    """
    def __init__(self):
        self.classes = []
        self.class_index_dict = {}
        self.words = []
        self.word_index_dict = {}
        self.idf = {}
    
    def add_word(self, word):
        if word not in self.word_index_dict:
            self.word_index_dict[word] = len(self.words)
            self.words.append(word)
    
    def add_class(self, class_lable):
        if class_lable not in self.class_index_dict:
            self.class_index_dict[class_lable] = len(self.classes)
            self.classes.append(class_lable)  
    
    def update_df(self, word):        
        if word not in self.idf:
            self.idf[word] = 1
        else:
            self.idf[word] += 1

    def create_idf(self, total_docs_num):
        self.idf = {key: log(total_docs_num / (df + 1)) for key, df in self.idf.items()}
    
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
