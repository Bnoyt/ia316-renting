from utils import *


class Policy(object):
    def __init__(self,params,seed):
        self.params = params
        self.seed = seed
        self.list_length = self.params["list_length"]
        self.init()
        
    def init(self):
        self.rng = np.random.RandomState(self.seed)
        
    def get_action(self,context):
        
        raise NotImplementedError("You must implement the get_action method")