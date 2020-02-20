from utils import *


class Policy(object):
    def __init__(self,params,seed):
        self.params = params
        self.seed = seed
        self.list_length = self.params["list_length"]
        self.init()
        
    def init(self):
        self.rng = np.random.RandomState(self.seed)
        self.history = []
        
    def get_action(self,context):
        
        raise NotImplementedError("You must implement the get_action method")
        
    def get_history(self):
        return pd.DataFrame(self.history,columns = ["user_id","bike_id","price_proposed","accepted"])
        
    def update(self,context,bikes,days,prices,result):
        
        user_id = context["user_id"]
        
        if result == "INVALID":
            pass
        elif result == "REFUSED":
            rows = []
            for b,p in zip(bikes,prices):
                self.history.append([user_id,b,p,0])
        else: 
            index = bikes.index(result)
            self.history.append([user_id,bikes[index],prices[index],1])
        
        
class RandomPolicy(Policy):
    def __init__(self,params,seed):
        self.params = params
        self.seed = seed
        self.list_length = self.params["list_length"]
        self.init()
        
        self.__name__ = f"RandomPolicy_{params['price']}"
        
    def get_action(self,context):
        
        indexes = np.array([i for i in range(len(context["bikes_available"]))])
        
        self.rng.shuffle(indexes)
        
        
        bikes_available = [context["bikes_available"][i] for i in indexes]
        bikes_availability = [context['bikes_availability'][i] for i in indexes]
        prices = [self.params["price"] for i in indexes]
        
        return bikes_available[:self.list_length],bikes_availability[:self.list_length],prices[:self.list_length]
    
    
class SmarterRandomPolicy(Policy):
    def __init__(self,params,seed):
        self.params = params
        self.seed = seed
        self.list_length = self.params["list_length"]
        self.init()
        
        self.__name__ = f"SmarterRandomPolicy_{params['price']}"
        
    def get_action(self,context):
        
        
        a = [len(d) for d in context["bikes_availability"]]
        
        
        chosen_index = np.argmax(a)      
        
        return [context["bikes_available"][chosen_index]],[context['bikes_availability'][chosen_index]],[self.params["price"]]