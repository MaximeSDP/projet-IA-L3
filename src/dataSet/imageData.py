class ImageData:

    def __init__(self, name_path, y_true_class):
        self.name_path = name_path
        self.resized_image = None
        self.X_histo = None
        self.y_true_class = y_true_class
        self.y_predicted_class = None
        self.X_gradient = None
