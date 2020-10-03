import itertools
import numpy as np
import pandas as pd
from time import time
from task import HuysTask

env = HuysTask(depth=3)
n_trials = int(6e4)

# Hyper params
beta        = 10
N_traj      = 10
nRestarts   = 1
searchBudget= range(3, 25, 3)
a1_         = 0.1
a2_         = 1

# initializing params
lmda= 1
a1  = 0.1   # learning rate for q
a2  = 0.1   # learning rate for u
y   = 1     # gamma
e   = 0.1   # epsilon for e-greedy

# Defining environment variables
t  = env.transitions
T  = env.transition_matrix
R  = env.reward_matrix
t1 = np.reshape(np.repeat(np.arange(env.depth*6,(env.depth+1)*6),2),(6,2))
    # Creating terminal states to add to the list of transitions
t  = np.vstack((t,t1,t1))   # Adding terminal states to list of transitions
tlist = list(np.mod(t,6)+1) # Human-readable list of states (idxed 1-6 as in Huys et al.)
sTerminal = 6*env.depth     # All states >= sTerminal are Terminal states
nTerminal = 12              # No. of terminal states x 2 (x2 is a silly hack for lack of motivation)

def softmax(x,beta):
    x = np.array(x)
    b = np.max(beta*x)      # This is a trick to avoid overflow errors
    y = np.exp(beta*x - b)  # during numerical calculations
    return y/y.sum()

def policy(q,sigma,s,t,beta):
    # Soft thompson sampling: softmax applied instead of max
    ind_s   = (t[:,0]==s)       # index for transition list where state is s
    zeta_s  = np.random.randn(len(q[ind_s]))
    p       = softmax(q[ind_s] + zeta_s*sigma[ind_s],beta=beta)  # choice probability vector   
    a       = np.random.choice(np.arange(len(p)), p=p)
    return a, ind_s, zeta_s, p

# Defining the cost function
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


df_allRuns  = []                # List to collect results (q,sigma) from all runs
expR_allRuns= []                # List to collect expected rewards from all runs
cost_allRuns= []                # List to collect cost from all runs
obj_allRuns = []                # List to collect objective from all runs

for budget in searchBudget:

    for run in range(nRestarts):

        expR       = []             # List to collect expected rewards(t)
        cost       = []             # List to collect cost(t)
        objective  = []             # List to collect objective(t)
        # Initializing Gaussian approximated q-distribution
        sigma0  = 40                # Initial sigma for Q ~ N(q,sigma)
        sigmaBase = 100             # Sigma for base distribtion
        sigmaT  = 1                 # sigma for terminal states (defines min possible sigma)
        q       = np.zeros(len(t))
        sigma   = sigma0*np.ones(len(q))
        sigma[-nTerminal:] = sigmaT     # sigma for terminal states
        q[(env.depth-1)*12:env.depth*12] = env.rewards[(env.depth-1)*12:env.depth*12]   # Last depth level memories known
        sigma[(env.depth-1)*12:env.depth*12] = sigmaT   # depth 3 memories known
        start = time()

        for ii in range(n_trials):

            if np.mod(ii,10000)==0:
                print(f"Run {run+1}/{nRestarts}, Trial {ii}/{n_trials}")

            s0 = np.random.randint(6)    # starting state
            s  = s0
            # nSamples = 0

            while s < sTerminal:    # nSamples <= budget:  could change
                a, i_s, z,p = policy(q, sigma, s, t, beta)  # Take action acc. to policy
                s1          = t[i_s][a][1]                  # (s'|s,a)
                r           = R[s,s1]                       # r(s,a,s')
                i_sa        = i_s * (t[:,1]==s1)            # idx for [s,a] pair
                _,_,_,p1    = policy(q, sigma, s1, t, beta) # computing pi(a'|s')
                q[i_sa]     = q[i_sa] + a1*(r + y*np.dot(p1,q[t[:,0]==s1]) - q[i_sa]) # on-policy update
                # q[idx]  = q[idx] + a1*(r + y*max(q[t[:,0]==s1]) - q[idx]) # q-learning update
                s = s1
                # nSamples += 1
                
            # Update sigma at the end of each trial by stochastic gradient descent:
            grads   = []

            for i in range(int(N_traj)): # Sampling N_traj to make gradient more stable
                
                # Initialising some variables
                grad = np.zeros(len(t))     # Vector of gradients, one for each sigma(s,a)
                nSmpl= 0
                rewards_workingMem = []

                while nSmpl < budget:

                    s = s0
                    r = 0

                    while s < sTerminal: 

                        a, i_s, z,p = policy(q, sigma, s, t, beta)  # Take action acc. to policy
                        s1          = t[i_s][a][1]                  # (s'|s,a)
                        r          += R[s,s1]                       # r(s,a,s')

                        # Maybe advantage gradients will work fine. Not sure, but def try later
                        # i_sa        = i_s * (t[:,1]==s1)            # idx for [s,a] pair          
                        # adv         = q[i_sa] - np.mean(q[i_s])     # A(s,a) = q(s,a) - mean(q(s,.))

                        g           = -beta * np.multiply(z, p)     # grad(log(pi(a'|s))) for all a' available from s
                        g[a]       += beta * z[a]                   # grad(log(pi(a|s))) for a taken
                        grad[i_s]  += g                             # updating vector of gradients for all a
                        
                        s           = s1                            # state update for next step
                        nSmpl      += 1

                    rewards_workingMem.append(r)

                r  = max(rewards_workingMem)
                grads   += [(r*grad)]   # adding it to the list of gradients for all trajectories

            grad_cost = (sigma/(sigmaBase**2) - 1/sigma)   # gradient of DKL cost term
            grad_mean = np.mean(grads, axis=0)          # mean gradient of expected rewards across N_traj
            
            # Updating sigmas for all but depth-n memories
            sigma[:-(nTerminal+12)]  += a2 * (grad_mean - lmda*grad_cost)[:-(nTerminal+12)]
            sigma                    = np.clip(sigma, 1, 100)

            if np.mod(ii,10000)==0:
                ##### Computing and storing objective at end of training ########
                # Expected rewards
                rewards = []
                for i in range(int(1e4)):
                    s0  = np.random.randint(0,6)
                    nSamples = 0
                    rewards_workingMem = []         
                    while nSamples < budget:
                        s   = s0
                        r   = 0 
                        while s < sTerminal:
                            # Draw q, choose next action, get reward, next state
                            a,ind_s,_,_ = policy(q,sigma,s,t,beta)
                            s1          = t[ind_s][a][1]
                            r          += R[s,s1]
                            s           = s1
                            nSamples   += 1
                        rewards_workingMem.append(r)
                    r = max(rewards_workingMem)
                    rewards += [r]  # reward obtained for the trial
                expR = np.mean(rewards)

                # Cost
                mu    = q[:-(nTerminal+12)]
                S1    = np.diag(np.square(sigma[:-(nTerminal+12)]))
                S2    = np.diag(np.square([sigmaBase]*len(mu)))
                cost  = lmda * kl_mvn(mu, S1, mu, S2)

                # Objective
                objective = (expR - cost)

                # Computing table of memories
                d = {'transitions':tlist, 'q':np.around(q,0), 'sigma':np.around(sigma,1)}
                df= pd.DataFrame(d)
                
                # Storing results in a dict        
                df_allRuns   += [df.iloc[:-nTerminal]]
                expR_allRuns += [expR]
                cost_allRuns += [cost]
                obj_allRuns  += [objective]

                print(f"Objective = {np.around(objective,1)}, "+\
                    f"Reward = {np.around(expR,1)}, "+\
                    f"Cost = {np.around(cost,1)}")

    print(f"\n Budget {budget} \n")


###### Saving ###################
import pickle
path = f"../figures/"    # initialResults/gradientBasedLearning/roleOfTime

f = open(path+f"unmod_df_depth{env.depth}", 'wb')
pickle.dump(df_allRuns, f, protocol=pickle.HIGHEST_PROTOCOL)
f.close()

np.save(path + f"unmod_R_depth{env.depth}", expR_allRuns)
np.save(path + f"unmod_C_depth{env.depth}", cost_allRuns)
np.save(path + f"unmod_O_depth{env.depth}", obj_allRuns)


# Loading results from unmodified objective
d4 = pickle.load(open(path+f"unmod_df_depth4",'rb'))
o4 = (np.load(path + f"unmod_R_depth4")).reshape((8,15))
i4 = np.argmax(o4,axis=1) + np.arange(0,120,15)
d4 = d4[i4]

# ###### Plotting ###################
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()

# sb = np.arange(12.5,101,12.5)

# # Convergence plots
# f,ax = plt.subplots()
# ax.errorbar(sb, dfPlot['o_diff'], dfPlot['oe_diff'])
# ax.set_xlabel('Search budget')
# ax.set_ylabel('Objective difference (Flexible - Constrained)')

# ax2 = ax.twinx()
# ax2.errorbar(sb, dfPlot['o_var'], dfPlot['oe_var'])
# ax2.errorbar(sb, dfPlot['o_eq'], dfPlot['oe_eq'])
# ax2.set_ylabel('Objective')

# plt.show()