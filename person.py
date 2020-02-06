from utils import *



class Person(object):
    
    def __init__(self,params,price_appetence,seed,user_id):
        
        """
        A person is defined by its public info (that the website can see) and its private params (that we don't know, but influence the behaviour)
        """
        
        self.params = params
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.id = user_id
        self.price_appetence = price_appetence
        
        self.alter_params()

    
    def alter_params(self):
        pass
    
    
    
    @classmethod
    def randomPerson(cls,seed,dim,clusters,user_id):
        rng = np.random.RandomState(seed)
        
        n = len(clusters)
        c = clusters[rng.randint(n)]
        
        
        
        params = np.array([max(0,rng.normal(s,1)) for s in c])
        
        price_appetence = rng.uniform(1000)
        
        
        return cls(params,price_appetence,seed,user_id)
        
        
    def get_best_reward(self):
        pass
        
    def get_proba(self,bike,price,days,days_wanted):
        
        proba = similarity(bike.params,self.params)
        
        
        #print(np.exp(-price/self.price_appetence))
        
        proba = proba * np.exp(-price/self.price_appetence)
        
        proba = proba * np.exp(-abs(len(days) - len(days_wanted)))
        
        
        
        
        
        return proba
    
    def get_reward(self,bike,days,days_wanted):
        
        reward = similarity(bike.params,self.params)
       
        
        reward = reward * self.price_appetence
        
        reward = reward * np.exp(-abs(len(days) - len(days_wanted)))
        
        
        
        
        
        return reward
    
    def get_best_reward(self,bike_list,days_list,days_wanted):
        
        rewards = []
        
        for (bike,days) in zip(bike_list,days_list):
            rewards.append(self.get_reward(bike,days,days_wanted))
            
        return np.max(rewards)
        
    def which_bike(self,bike_list,days_list,price_list,days_wanted):
        
        
        probas = []
        
        for (bike,days,price) in zip(bike_list,days_list,price_list):
            probas.append(self.get_proba(bike,price,days,days_wanted))
            
            
        #print(probas)
            
        is_chosen = []
        prices = []
        
        for (i,p),price in zip(enumerate(probas),price_list):
            if self.rng.uniform() < p:
                is_chosen.append(i)
                prices.append(price)
                
        if len(is_chosen) == 0:
            return None,None
        else:
            a = self.rng.randint(len(is_chosen))
            
            return is_chosen[a],prices[a]
            
            
    
        