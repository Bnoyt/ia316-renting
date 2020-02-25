from utils import *


class Tester(object):
    def __init__(self,policies,env,params):
        self.params = params
        self.n_steps = params["n_steps"]
        self.n_rep = params["n_rep"]
        
        self.history = {p.__name__:[] for p in policies}
        self.policies = policies
        self.env = env
        self.ran = False
        
        
    def run(self):
                
        for policy in self.policies:
        
            for j in range(self.n_rep):
                print( f"Testing policy |{policy.__name__}| rep {j}/{self.n_rep}")

                self.env.init(j)
                policy.init()

                for i in tqdm(range(self.n_steps)):
                    context = self.env.get_context()
                    b,d,p = policy.get_action(context)
                    result = self.env.act(b,d,p)
                    policy.update(context,b,d,p,result)

                self.history[policy.__name__].append(self.env.get_history())
                
        self.ran = True
                
    def plotByPolicy(self,policy_name):
        
        assert policy_name in self.history, "Policy name unknown"
        assert self.ran, "You must run the test by tester.run() before plotting results"
        
        dfs = self.history[policy_name]
        
        regrets = np.array([(df.best_reward - df.reward).cumsum().values for df in dfs])

        regret = np.array([(df.best_reward - df.reward).values for df in dfs])
        
        plt.figure(figsize=(15,7))
        plt.style.use("bmh")
        plt.plot(regrets.mean(axis=0), color='blue',label="Average")
        plt.plot(np.quantile(regrets, 0.05,axis=0), color='grey', alpha=0.5,label="Quantile = 0.05")
        plt.plot(np.quantile(regrets, 0.95,axis=0), color='grey', alpha=0.5,label="Quantile = 0.95")
        plt.title('Mean regret: {:.2f} for {}'.format(regret.mean(),policy_name))
        plt.legend()
        plt.xlabel('steps')
        plt.ylabel('regret')
        plt.show()
        
    def plotAllPoliciesQuantiles(self):
        for n in self.history:
            self.plotByPolicy(n)
            
            
    def plotAllPoliciesCompared(self):
        plt.figure(figsize=(15,7))
        plt.style.use("bmh")
        for policy_name in self.history:
            dfs = self.history[policy_name]
            regrets = np.array([(df.best_reward - df.reward).cumsum().values for df in dfs])
            plt.plot(regrets.mean(axis=0), label=policy_name)
        plt.xlabel('steps')
        plt.ylabel('regret')  
        plt.legend()
        plt.title("Comparison of the regret of all policies")
            


