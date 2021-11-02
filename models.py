import git
import tempfile
import shutil
import os
import tensorflow.keras as keras

class Astromer():
    def __init__(self, pretrained_weights):
        self.pretrained_weights = pretrained_weights
        self.git_link = 'https://github.com/HarshVardhanGoyal/test_model.git'
        self.model_name = 'Test_model.h5'
        self.model_path = os.path.join(os.getcwd(), self.model_name)

    def load_model(self, overwrite=False):
        
        # Creating a temporary directory and getting the saved weights
        test_dir = tempfile.mkdtemp()
        git.Repo.clone_from(self.git_link, test_dir, branch='main', depth=1)
        if os.path.exists(self.model_path):
            if overwrite:
                print("The saved weights already exists")
            else:
                os.remove(self.model_path)
        shutil.move(os.path.join(test_dir, self.model_name), os.getcwd())
        # shutil.rmtree(test_dir)

        # Loading the model
        if os.path.exists(self.model_path):
            self.model = keras.models.load_model(self.model_path)
        return self.model
            
