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
    

class Thomson_SamplingPolicy(Policy):
    #NB il faut mettre nb_clusters à 2 sinon avec 5 ça me choisissait dans les 5 vélos pour une raison que j'ai pas encore cherché à comprendre
    """
    Exemple de policy qui fait un Thomson Sampling
    
    """
    
    def __init__(self,params,seed):
        self.params = params
        self.seed = seed
        self.list_length = self.params["list_length"]
        self.n_bikes = self.params["n_bikes"]
        self.init()
        self.__name__ = f"Thomson_SamplingPolicy_{params['price']}"
        
        #initialise une béta distribution correspondant au g des slides du cours
        # one distribution per bike (arm) per user
        # a=1 et b=1 par défaut, nb c'est peut etre une mauvaise initialisation 
        self.prior = [[(1,1) for i in range(self.n_bikes)] for j in range(self.params['n_users'])]
        
    def get_action(self,context):
        
        df = self.get_history() 
        df = df[df.user_id == context['user_id']]
        
        df_env = env.get_history()
        df_env= df_env[df_env.user_id == context['user_id']]
        
        if df.shape[0]<self.params["n_bikes"]:
            return [context["bikes_available"][df.shape[0]]],[context['bikes_availability'][df.shape[0]]],[self.params["price"]]
            #return [context["bikes_available"][df.shape[0]]],[context['days_wanted']],[self.params["price"]]
        
        if len(df) < 10:
            a = [len(d) for d in context["bikes_availability"]]
            chosen_index = self.rng.randint(0,len(a))  
            return [context["bikes_available"][chosen_index]],[context['bikes_availability'][chosen_index]],[self.params["price"]]
            #return [context["bikes_available"][chosen_index]],[context['days_wanted']],[self.params["price"]]

        # Nombre de fois que chaque vélo a été choisi
        choices = df.groupby('bike_id').accepted.sum().tolist()

        # Reward cumulé par vélo
        df_env['bike_proposed']=df_env['bike_proposed'].apply(lambda x : x[0])
        rwrds=df_env[['bike_proposed','reward']].groupby(['bike_proposed']).sum().reward.tolist()
        
        for bike_index in range(self.n_bikes):
            tmp=self.prior[context['user_id']][bike_index]
            #pour b je prends la valeur absolue mais c'est bizarre ca devrait rester positif je crois (peut etre que ca reste positif mais que ca tombe à 0, il faut que je regarde)
            #je rajoute 0.00001 parce que ca tombe a 0 sinon (pareil il faudra que je réfléchisse à cette constante si je la laisse)
            self.prior[context['user_id']][bike_index]=(tmp[0]+rwrds[bike_index],abs(tmp[1]+1-rwrds[bike_index])+0.000001)
        
        samples = [np.random.beta(x[0],x[1]) for x in self.prior[context['user_id']]]
        chosen_index = np.argmax(samples)
        
        return [context["bikes_available"][chosen_index]],[context['bikes_availability'][chosen_index]],[self.params["price"]]
        #return [context["bikes_available"][chosen_index]],[context['days_wanted']],[self.params["price"]]

    
class EGreedyPolicy(Policy):
    """
    Example de policy qui fait un epsilon greedy (mais qui marche pas trop trop)
    
    """
    
    
    def __init__(self,params,seed):
        self.params = params
        self.seed = seed
        self.list_length = self.params["list_length"]
        self.n_bikes = self.params["n_bikes"]
        self.init()
        self.eps = self.params["eps"]
        self.__name__ = f"EGreedyPolicy_{params['price']}"
        

            
        
    def get_action(self,context):
        
        df = self.get_history() # La méthode pour récupérer l'historique
        
        df = df[df.user_id == context['user_id']]
        
        if self.rng.uniform() < self.eps or len(df) < 10:
        
            a = [len(d) for d in context["bikes_availability"]]

            chosen_index = self.rng.randint(0,len(a))  

            #return [context["bikes_available"][chosen_index]],[context['bikes_availability'][chosen_index]],[self.params["price"]]
            return [context["bikes_available"][chosen_index]],[context['days_wanted']],[self.params["price"]]

        
        
        else:
            bikes_ids = pd.DataFrame({
                'bike_id':context["bikes_available"]
            })
            
            
            df = df.merge(bikes_ids,on='bike_id').groupby('bike_id').accepted.mean().reset_index().sort_values('accepted',ascending=False)
            
            chosen_index = context["bikes_available"].index(df.bike_id.values[0])
            
            return [context["bikes_available"][chosen_index]],[context['bikes_availability'][chosen_index]],[self.params["price"]]
            #return [context["bikes_available"][chosen_index]],[context['days_wanted']],[self.params["price"]]

            
           
            
 
            
            
class UCBPolicy(Policy):
    """
    Exemple de UCB policy
    
    """
    
    
    def __init__(self,params,seed):
        self.params = params
        self.seed = seed
        self.list_length = self.params["list_length"]
        self.n_bikes = self.params["n_bikes"]
        self.init()
        self.__name__ = f"UCBPolicy_{params['price']}"
        
        
        
    def get_action(self,context):
        
        df = self.get_history() 
        u = df.copy()
        
        #on veut s'assurer que les Nt(a) ne valent pas 0 donc on doit attribuer chaque vélo au moins une fois
        if df.shape[0]<self.params["n_bikes"]:
            return [context["bikes_available"][df.shape[0]]],[context['bikes_availability'][df.shape[0]]],[self.params["price"]]
        
        else :

            df = df[df.user_id == context['user_id']]

            if len(df) < 10:
                a = [len(d) for d in context["bikes_availability"]]
                chosen_index = self.rng.randint(0,len(a))  
                return [context["bikes_available"][chosen_index]],[context['bikes_availability'][chosen_index]],[self.params["price"]]
                #return [context["bikes_available"][chosen_index]],[context['days_wanted']],[self.params["price"]]

            
            else :
                
                #bikes_ids a une seule colonne 'bike_id' qui contient les id des bikes
                bikes_ids = pd.DataFrame({
                        'bike_id':context["bikes_available"]
                    })

                df = df.merge(bikes_ids,on='bike_id').groupby('bike_id').accepted.mean().reset_index().sort_values('accepted',ascending=True)

                #crée la colonne contenant le Nt(a) du dénominateur dans ucb
                
                
                df2 = u[u.user_id == context['user_id']].groupby('bike_id').accepted.sum().reset_index()
                df2.columns=['bike_id','sum_accepted']

                df = df.merge(df2,on='bike_id')

                #on suppose c=1
                df['arg_ucb']=df.apply(lambda x : (x['accepted'] + np.sqrt(context['day']/x['sum_accepted'])),axis=1)
                
                
                correction =    [    np.exp(-abs(len(days) - len(context['days_wanted']))) for days in context['bikes_availability']]
                correction = pd.DataFrame({"correction":correction,"bike_id":context["bikes_available"]})
                
                df = df.merge(correction,on="bike_id")
                
                df["arg_ucb"] = df["arg_ucb"]*df.correction

                df=df.reset_index().sort_values(by=['arg_ucb'],ascending=False)

                chosen_index = context["bikes_available"].index(df.bike_id.values[0])

                return [context["bikes_available"][chosen_index]],[context['bikes_availability'][chosen_index]],[self.params["price"]]
                #return [context["bikes_available"][chosen_index]],[context['days_wanted']],[self.params["price"]]



from keras.layers import Embedding, Flatten, Dense, Dropout,Input
from keras.layers import Dot
from keras.models import Model


class DeepPolicy(Policy):
    """
    Example de policy qui fait un epsilon greedy (mais qui marche pas trop trop)
    
    """
    
    
    def __init__(self,params,seed):
        self.params = params
        self.seed = seed
        self.list_length = self.params["list_length"]
        self.n_bikes = self.params["n_bikes"]
        self.init()
        self.eps = self.params["eps"]
        self.__name__ = f"DeepPolicy_{params['price']}"
        
        
    def init(self):
        self.rng = np.random.RandomState(self.seed)
        self.history = []
        
        
        user_inputs = Input(shape=(1,))
        bike_inputs = Input(shape=(1,))
        
        user_embedding = Embedding(output_dim=self.params["embedding_size"],
                                        input_dim=self.params["n_users"],
                                        input_length=1,
                                        name='user_embedding')
        bike_embedding = Embedding(output_dim=self.params["embedding_size"],
                                        input_dim=self.params["n_bikes"],
                                        input_length=1,
                                        name='bike_embedding')
        
        user_vecs = Flatten()(user_embedding(user_inputs))
        bike_vecs = Flatten()(bike_embedding(bike_inputs))
        
        final_layer = Dot(axes=1)([user_vecs, bike_vecs])
        
        
        self.model = Model(inputs=[user_inputs,bike_inputs],outputs=[final_layer])
        
        
        self.model.compile(optimizer="adam", loss='mae') 
        

            
        
    def get_action(self,context):
        
        df = self.get_history() # La méthode pour récupérer l'historique
        
        df = df[df.user_id == context['user_id']]
        
        if len(df) < 100:
        
            a = [len(d) for d in context["bikes_availability"]]


            chosen_index = self.rng.randint(0,len(a))  

            return [context["bikes_available"][chosen_index]],[context['bikes_availability'][chosen_index]],[self.params["price"]]
            #return [context["bikes_available"][chosen_index]],[context['days_wanted']],[self.params["price"]]
        
        
        else:
            
            X1,X2 =   df[["user_id","bike_id"]].values[:,0],df[["user_id","bike_id"]].values[:,1]
            y = df["accepted"].values
            
            
            
            
            self.model.fit([X1,X2],y,verbose=0)
            
            
            X1,X2 = [context["user_id"] for v in context["bikes_available"]], [v for v in context["bikes_available"]]
            
            
            ratings = self.model.predict([X1,X2])
            
            correction =    [    np.exp(-abs(len(days) - len(context['days_wanted']))) for days in context['bikes_availability']]
            
        
            ratings = [x*y for (x,y) in zip(ratings,correction)]
            
            chosen_index = np.argmax(ratings)
            
            return [context["bikes_available"][chosen_index]],[context['bikes_availability'][chosen_index]],[self.params["price"]]
            #return [context["bikes_available"][chosen_index]],[context['days_wanted']],[self.params["price"]]
            
            
            
            
            
            
         
            
            
            
            
            
            
         