import numpy as np
import torch

class EarlyStopping:
    """earlystopping class"""

    def __init__(self, patience=5, verbose=False, path='checkpoint_model.pth'):

        self.patience = patience        # stopping patience
        self.verbose = verbose          # display flag
        self.counter = 0                # current counter value
        self.best_score = None          # best score
        self.early_stop = False         # early stop flag
        self.val_loss_min = np.inf      # previous best score memory
        self.path = path                # best model storage path

    def __call__(self, val_loss, model):
        """
        Method to calculate whether the minimum loss was updated in the actual training loop
        """
        score = -val_loss    

        if self.best_score is None:  
            self.best_score = score   
            self.checkpoint(val_loss, model)  
        elif score < self.best_score:  
            self.counter += 1  
            if self.verbose:  
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  
            if self.counter >= self.patience:  
                self.early_stop = True
        else:  
            self.best_score = score  
            self.checkpoint(val_loss, model)  
            self.counter = 0  

        return self.counter

    def checkpoint(self, val_loss, model):
        '''Method to save the model when the validation loss decreases'''
        if self.verbose:  
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.module.state_dict(), self.path)  
        self.val_loss_min = val_loss  
