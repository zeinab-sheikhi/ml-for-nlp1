class Document:
    """
        Parameters:
        ---------- 
        - document_number : int

        - word_tf : { word: float value}
            vector representation of all the words in the document and their corresponding tf value

        - gold_class : str
            the corresponsing gold class for this document
    """
    def __init__(self, document_number, gold_class):
        self.document_number = document_number
        self.gold_class = gold_class
        self.word_tf = {}
    
    def add_word_tf(self, word, tf_value):
        self.word_tf[word] = tf_value
