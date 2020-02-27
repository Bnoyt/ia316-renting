from utils import *
from person import Person
from bike import Bike


class Environnement(object):
    
    def __init__(self,params,seed):
        self.experiment_length = params["experiment_length"]
        self.seed = seed
        
        self.n_users = params["n_users"]
        self.n_clusters = params["n_clusters"]
        
        self.dim = params["dim"]
        
        self.users_per_day = params["users_per_day"]
        self.n_bikes_per_user = params["n_bikes_per_user"]
        
        self.bike_overlapping = params["bike_overlapping"]
        
    def init(self,seed):
        
        
        self.rng = np.random.RandomState(hash(seed*self.seed))
        
        #price_expectation = self.rng.uniform(2000)
        price_expectation=500
        
        self.clusters = [np.array([self.rng.uniform() for i in range(self.dim)]) for j in range(self.n_clusters)]
        
        
        self.peoples = [Person.randomPerson(hash(s*seed),self.dim,self.clusters,s,price_expectation) for s in range(self.n_users)]
        
        
        
        self.bikes = [Bike(c,self.experiment_length,i) for (i,c) in enumerate(self.clusters)]
        
        
        self.t = 0
        
        self.users_seen = 0
        
        self.current_user = self.rng.choice(self.peoples)
        
        self.days_wanted = self.get_wanted_days()
        
        
        self.history = {
            "user_id":[],
            "days_wanted":[],
            "bike_proposed":[],
            "price_proposed":[],
            "days_proposed":[],
            "answer":[],
            "reward":[],
            "best_reward":[],
            "probas":[],
            'best_bike':[],
        }
        
    
    
        
        
    def get_wanted_days(self):
        day1 = int(self.rng.exponential(10) + self.t)
        day2 = int(self.rng.exponential(5) + 1 + day1)
        
        days = list(range(day1,day2))
        
        return days
    
    def get_history(self):
        return pd.DataFrame(self.history)
    
        
    def get_context(self):
        
        return {
                "day":self.t,
                "user_id":self.current_user.id,
                "days_wanted":self.days_wanted,
                "bikes_available":[bike.id for bike in self.bikes],
                "bikes_availability":[bike.available_days(self.days_wanted) for bike in self.bikes]
            }
        
        
        
    
    def act(self,bike_list,days_list,price_list):
        
        if self.t == self.experiment_length:
            return "Experiment_over"
        
        bike_proposed = bike_list[:self.n_bikes_per_user]
        bike_list = [self.bikes[bike_id] for bike_id in bike_list][:self.n_bikes_per_user]
        days_list = days_list[:self.n_bikes_per_user]
        price_list = price_list[:self.n_bikes_per_user]
        
        
        indice_chosen,price,probas = self.current_user.which_bike(bike_list,days_list,price_list,self.days_wanted)
        
        
        
        context = self.get_context()
        best_reward,best_bike,best_days = self.current_user.get_best_reward([self.bikes[b] for b in context["bikes_available"]],context["bikes_availability"],self.days_wanted)
        
        

        result = None
        
        if indice_chosen != None:
            chosen_bike = bike_list[indice_chosen]
            
            result = chosen_bike.rent(days_list[indice_chosen],price_list[indice_chosen])
            #result = self.bikes[best_bike].rent(best_days,price_list[0])
            
            if result:
                result = chosen_bike.id
                
            if self.bike_overlapping:
                result = chosen_bike.id

            
        reward = price
        if result == None:
            result = "REFUSED"
            reward = 0
        if result == -1:
            result = "INVALID"
            reward = 0
        
            
        self.history["user_id"].append(self.current_user.id)
        self.history["days_wanted"].append(self.days_wanted)
        self.history["bike_proposed"].append(bike_proposed)
        self.history["price_proposed"].append(price_list)
        self.history["days_proposed"].append(days_list)
        self.history["answer"].append(result)
        self.history["reward"].append(reward)
        self.history["best_reward"].append(best_reward)
        self.history["probas"].append(probas)
        self.history["best_bike"].append(best_bike)
            
        
        
        
        self.users_seen =+ 1
        self.t = self.users_seen % self.users_per_day
        
        self.current_user = self.rng.choice(self.peoples)
        
        self.days_wanted = self.get_wanted_days()
        
        
        

        return result
        