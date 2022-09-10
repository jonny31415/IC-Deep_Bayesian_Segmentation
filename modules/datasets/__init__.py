
# --- Global Variables
# Mean and std from Cityscapes
dataset_mean = [0.485, 0.456, 0.406]
dataset_std = [0.229, 0.224, 0.225]
# dataset_mean = [0.5, 0.5, 0.5]
# dataset_std = [0.2, 0.2, 0.2]
# # ---

class DatasetBase:
    def __init__(self):
        self.folder_path_imgs = 'joao'
        self.folder_path_labels = ''