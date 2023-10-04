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
