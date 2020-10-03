import numpy as np
import pandas as pd

class GradientFreeResourceAllocator:
    def __init__(self, depth=3, lmda=1, n_restarts=10, n_trials=int(1e3),\
                allocationMethod='variablePrecision'):
        # General parameters with default values
        self.lmda = lmda
        self.n_trials = n_trials
        self.n_restarts = n_restarts
        self.allocationMethod = allocationMethod
        
        # Fixed attributes
        self.costType = 'dkl'
        self.randomizeStartState = True

        # Parameters specific for Huys' planning task
        self.depth = depth
        self.searchBudget = list(range(depth,(depth * 2**depth)+1,depth))
        q = np.array([170,100, 10, 10, 60, 10, 50, 60, 70, 60,120,50,\
                      140, 70, 30, 30, 30, 30, 20, 80, 30, 80, 80,70,\
                      100, 20,-20,  0,-20, 50,-20, 50, 50,100,100,20,\
                      120, 40,-40,-90,  0,-50,  0,-40, 70,  0,120, 0,\
                      140, 20,-20,-70,-20,-70, 20,-20,-70,-20,-20,20])
        self.q = q[ (len(q) - 12*depth) : ]

        # Task environment
        from task import HuysTask
        self.env = HuysTask(depth=self.depth)

    
    # Methods
    @staticmethod
    def kl_mvn(m0, S0, m1, S1):
        """
        Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
        Diagonal covariances are assumed.  Divergence is expressed in nats.
        """    
        # store inv diag covariance of S1 and diff between means
        N = m0.shape[0]
        iS1 = np.linalg.inv(S1)
        diff = m1 - m0

        # kl is made of three terms
        tr_term   = np.trace(iS1 @ S0)
        det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0))
        quad_term = diff.T @ np.linalg.inv(S1) @ diff 
        return .5 * (tr_term + det_term + quad_term - N)

    
    def reshapeSigma(self, sigma):
        # Reshaping sigma to be the right size for Huys' planning task
        if self.allocationMethod=='variablePrecision':   # memories can have varying certainty
            sigma = np.array(list(sigma) + [1]*12)
        elif self.allocationMethod=='equalPrecision':    # all memories have same uncertainty
            sigma = [float(sigma)]*(12*(self.depth-1)) + [1]*12
        if self.costType=='ent':
            sigma = np.clip(sigma,1,100)     # setting upper and lower bounds
        return sigma


    def cost(self, sigma):
        """
        Returns the cost associated with a given resource allocation.
        """
        sigma = self.reshapeSigma(sigma)
        sigma_base = [100]*(12*(self.depth-1)) + [1]*12
        return {
            'ent': - self.lmda * np.sum(2*np.log(sigma)),
            'dkl': self.lmda * self.kl_mvn(self.q, np.diag(np.square(sigma)), \
                        self.q, np.diag(np.square( sigma_base )))
        }.get(self.costType, 0)     # 0 is default if x is not found


    def expectedRewards(self, sigma, searchBudget):
        """
        Returns mean reward obtained across n_trials for given search budget
        and resource allocation vector, sigma (uncertainty across memories).
        """
        # Unpacking environment variables
        T = self.env.transition_matrix
        R = self.env.reward_matrix
        n_states = len(T)   # length of transition matrix
        sigma = self.reshapeSigma(sigma)

        def onehot(ind, size=n_states):
            a = np.zeros((size))
            a[ind] = 1
            return a

        if self.randomizeStartState:
            start_state = np.random.choice(np.arange(6), size=self.n_trials)

        rewardsObtained = []
        for s0 in start_state:
            # Draw samples from the q distribution once for each trial
            q = np.random.normal(self.q, sigma)
            
            # Define search parameters
            N_accesses = searchBudget
            paths = [[s0]]*N_accesses
            rewards = [0]*N_accesses
            
            # Define array to disallow re-exploring states (convoluted)
            N_visitsLeft = []
            for i in range(self.depth+1):
                N_visitsLeft += 2**i * [int(2**(self.depth-i))]
            N_visitsLeft = np.array(N_visitsLeft)
            aa = '1'

            # Run the trial according to tree policy (currently depth-first search)
            s = s0   # initially
            while N_accesses > 0:
                # Thompson sampling: returns binary action 0 or 1
                a = np.argmax(q[2*s:2*s+2])
                s1 = np.nonzero(np.dot(onehot(s), T))[0][a]     # next state

                # disallow re-exploring paths (convoluted; not implemented in paper)
                aa += str(a)
                a_idx = int(aa,2) - 1
                if N_visitsLeft[a_idx] == 0:
                    s1 = np.argmax(np.dot(onehot(s), T) - onehot(s1))
                    aa = aa[:-1] + str(abs(int(aa[-1])-1))  # stupid hack 
                    a_idx = int(aa,2) - 1

                # Getting rewards and updating time left
                r = R[s,s1]
                N_visitsLeft[a_idx] -= 1
                N_accesses -= 1
                
                # store paths accessed in working memory
                for index, row in enumerate(paths):
                    if row[-1] == s:
                        paths[index] = paths[index] + [s1]
                        rewards[index] += r
                        break
                
                # setup for the next step
                if s1 < n_states-6:
                    s = s1      # if state non-terminal, curr_state = s1 (next_state)
                else:
                    s = s0      # if state terminal, curr_state = s0 (initial/root)
                    aa = '1'

            # Keep only paths that are fully discovered
            count = sum(map(lambda x: len(x) == self.depth+1, paths))
            paths = paths[:count]
            rewards = rewards[:count]

            # reward obtained for the trial (and path chosen commented)
            reward = max(rewards)   # bestPath = paths[np.argmax(rewards)]
            rewardsObtained += [reward]

        return np.mean(rewardsObtained)


    def optimize(self, searchBudget):
        """
        Optimally allocate resources across memories for a given search budget.
        """
        import itertools
        import multiprocessing as mp

        # Define an output queue
        output = mp.Queue()

        # Define the objective function
        def obj_func(sigma, searchBudget=searchBudget):
            expectedReward = self.expectedRewards(sigma=sigma, searchBudget=searchBudget)
            cost           = self.cost(sigma=sigma)
            return (-expectedReward + cost)

        # Define the optimisation function with output sent to queue
        def optimize_local(output=output):
            # 
            if self.allocationMethod=='variablePrecision':
                # importing CMAES libarary
                import cma

                # setting up parameters and running the optimization            
                x0 = 50 + 15*np.random.randn(12*(self.depth-1))
                res = cma.fmin(obj_func, x0, 30, options={'bounds':[1,100],\
                             'tolfun':1, 'maxfevals': int(1e4)})
                sigma_opt = res[0] 

            elif self.allocationMethod=='equalPrecision':
                # importing Bayesian optimization libraries
                import GPy, GPyOpt
                from GPyOpt.methods import BayesianOptimization

                # setting up parameters and running the optimization
                kernel = GPy.kern.Matern52(input_dim=1)
                domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (0,100)}]
                optimizer = BayesianOptimization(obj_func, domain=domain, kernel=kernel)
                optimizer.run_optimization(max_iter=50)
                sigma_opt = optimizer.X[optimizer.Y.argmin()]

            # appending the result (scalar sigma) to output queue
            output.put(sigma_opt)

        # Setup a list of processes that we want to run
        processes = [mp.Process(target=optimize_local) for x in range(self.n_restarts)]

        # Run processes
        for p in processes:
            p.start()

        # Exit the completed processes
        for p in processes:
            p.join()

        # Get process results from the output queue
        results = [output.get() for p in processes]

        return results


    def run_optimization(self, saveResults=True):
        """
        Find optimal resource allocation for all possible search budgets.
        """
        results = [] 
        for budget in self.searchBudget:
            print(f"\n\nOptimizing for search budget = {budget} \n")
            result = self.optimize(budget)
            results += [result]

        # Saving the results
        if saveResults:
            pass    # decide on a folder structure and save shit
        return np.array(results)



if __name__ == "__main__":
    # Defining the two models
    varPrecModel = GradientFreeResourceAllocator()
    eqPrecModel  = GradientFreeResourceAllocator(allocationMethod='equalPrecision')

    # Optimizing memory allocation and plotting results
    varPrecModel.results = varPrecModel.run_optimization()

    # Plotting results
    import plotting
    f_m, ax_m           = plotting.plot_table(varPrecModel, tableColour='mean')
    f_s, ax_s           = plotting.plot_table(varPrecModel, tableColour='std')
    f,ax1,ax2           = plotting.plot_curves(varPrecModel)
    g,bx1               = plotting.plot_dprime(varPrecModel, table=True)
    cx                  = plotting.plot_dprime(varPrecModel, table=False)

    # Saving files
    # import pickle
    # fh = open('varPrecModel.pkl','wb')
    # pickle.dump(varPrecModel, fh, pickle.HIGHEST_PROTOCOL)
    # fh.close()