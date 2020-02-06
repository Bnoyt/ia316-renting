from utils import *

class Bike(object):
    def __init__(self,params,experiment_length,bike_id):
        self.params = params
        self.experiment_length = experiment_length
        
        days = list(range(self.experiment_length))
        price = [0]*self.experiment_length
        rented = [False]*self.experiment_length
        self.id = bike_id
        
        
        self.renting_data = pd.DataFrame({
            "day":days,
            "price":price,
            "rented":rented
        })
        
    def available_days(self,days):
        return self.renting_data[(self.renting_data.day.isin(days)) & (self.renting_data.rented == False)].day.values
    
    
    def rent(self,days,price):
        
        v = self.renting_data.rented.loc[self.renting_data.day.isin([days])].astype(int).sum()
        
        if v > 0:
         
            
            
            
            return -1
        
        
        
        
        else:

            self.renting_data.rented.loc[self.renting_data.day.isin([days])] = True
            self.renting_data.price.loc[self.renting_data.day.isin([days])] = price
            
            return True