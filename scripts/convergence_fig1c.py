import itertools
import numpy as np
import pandas as pd
from time import time, sleep
from task import Maze
from scipy.stats import entropy

env = Maze()
n_trials = int(5e3)

# Hyper params
# beta     = 10.**(np.arange(1,2))    # inverse temp param for softmax
# N_traj   = 10.**(np.arange(2,3))    # number of trajectories to average stoch. gradient
# iterProd = itertools.product(beta,N_traj)
beta        = 10
N_traj      = 10
nRestarts   = 2

# initializing params
lmda= 0.1
a1  = 0.1   # learning rate for q
a2  = 0.1   # learning rate for u
yma = 0.9   # gamma
e   = 0.1   # epsilon for e-greedy

def softmax(x,beta):
    x = np.array(x)
    b = np.max(beta*x)      # This is a trick to avoid overflow errors
    y = np.exp(beta*x - b)  # during numerical calculations
    return y/y.sum()

def policy(q,sigma,s,beta):
    # Soft thompson sampling: softmax applied instead of max
    x, y    = s
    zeta_s  = np.random.randn(len(env.actions))
    p       = softmax(q[x,y,:] + zeta_s*sigma[x,y,:], beta=beta)  # choice probability vector   
    a       = np.random.choice(np.arange(len(p)), p=p)
    return a, zeta_s, p

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

for sigma0 in [1,3,4,5]:
    print(f"\n sigma_0 = {sigma0} \n\n")
    df_allRuns  = []                # List to collect results (q,sigma) from all runs
    expR_allRuns= []                # List to collect expected rewards from all runs
    cost_allRuns= []                # List to collect cost from all runs
    obj_allRuns = []                # List to collect objective from all runs
    for run in range(nRestarts):
        expR       = []             # List to collect expected rewards(t)
        cost       = []             # List to collect cost(t)
        objective  = []             # List to collect objective(t)
        # Initializing Gaussian approximated q-distribution
        # sigma0  = 3               # Initial sigma for Q ~ N(q,sigma)
        sigmaBase = 5               # Sigma for base distribtion
        sigmaT  = 0.1               # sigma for terminal states (defines min possible sigma)

        q       = np.zeros(env.q_size)
        sigma   = sigma0*np.ones(env.q_size)
        
        start = time()
        for ii in range(n_trials):
            if np.mod(ii,100)==0:
                print(f"Run {run+1}/{nRestarts}, Trial {ii}/{n_trials}, time={np.around(time()-start,1)}")
            s  = env.START_STATE    # starting state

            while s not in env.GOAL_STATES:
                a, z, p = policy(q, sigma, s, beta)     # Take action acc. to policy
                s1, r   = env.step(s, a)                # (s'|s,a)
                _,_,p1  = policy(q, sigma, s1, beta)    # computing pi(a'|s')
                x, y    = s
                x1, y1  = s1
                q[x,y,a]= q[x,y,a] + a1*(r + yma*np.dot(p1,q[x1,y1,:]) - q[x,y,a]) # on-policy update
                # q[x,y,a]= q[x,y,a] + a1*(r + yma*np.max(q[x1,y1,:]) - q[x,y,a]) # q-learning update
                s = s1
            
            # Update sigma at the end of each trial by stochastic gradient descent:
            grads   = []
            for i in range(int(N_traj)): # Sampling N_traj to make gradient more stable
                # Initialising some variables
                grad = np.zeros(env.q_size) # Vector of gradients, one for each sigma(s,a)
                s    = env.START_STATE      # initial state
                while s not in env.GOAL_STATES:
                    a, z, p      = policy(q, sigma, s, beta)    # Take action acc. to policy
                    s1, r        = env.step(s, a)               # (s'|s,a)
                    x, y         = s
                    adv          = q[x,y,a] - np.mean(q[x,y,:]) # A(s,a) = q(s,a) - mean(q(s,.))
                    grad[x,y,:] -= (beta*z*p) * adv             # grad(log(pi(a'|s))) for all a' available from s
                    grad[x,y,a] += (beta*z[a]) * adv            # grad(log(pi(a|s))) for a taken
                    s            = s1                           # state update for next step
                grads   += [grad]   # adding it to the list of gradients for all trajectories
            grad_cost = (sigma/(sigmaBase**2) - 1/sigma)    # gradient of DKL cost term
            grad_mean = np.mean(grads, axis=0)              # mean gradient of expected rewards across N_traj
            # Updating sigmas for all but depth-n memories
            sigma += a2 * (grad_mean - lmda*grad_cost)
            sigma  = np.clip(sigma, 0.01, 100)

            # Compute objective every 1000 trials to plot convergence
            if np.mod(ii,100)==0:
                ##### Computing and storing objective for convergence ########
                # Expected rewards
                rewards = []
                for i in range(int(100)):
                    s = env.START_STATE  # random initial state
                    r = 0                # reward accumulated so far
                    while s not in env.GOAL_STATES:
                        # Draw q, choose next action, get reward, next state
                        a, z, p = policy(q, sigma, s, beta)     # Take action acc. to policy
                        s1, r_  = env.step(s, a)                # (s'|s,a)
                        r      += r_
                        s       = s1
                    rewards += [r]  # reward obtained for the trial
                expR += [np.mean(rewards)]

                # Cost
                mu    = q.flatten()
                S1    = np.diag(np.square(sigma.flatten()))
                S2    = np.diag(np.square([sigmaBase]*len(mu)))
                cost += [lmda * kl_mvn(mu, S1, mu, S2)]

                # Objective
                objective += [ (expR[-1] - cost[-1]) ]

                # Some pretty printing to satisfy the restless mind
                print(f"Objective = {objective[-1]}")
                
        # df_allRuns   += [d]
        expR_allRuns += [expR]
        cost_allRuns += [cost]
        obj_allRuns  += [objective]


    # # ###### Saving ###################
    import pickle
    path = f"./figures/" 
    np.save(path + f"convergence_R_sigma0{int(sigma0*10)}_{int(lmda*100)}_{nRestarts}runs", expR_allRuns)
    np.save(path + f"convergence_C_sigma0{int(sigma0*10)}_{int(lmda*100)}_{nRestarts}runs", cost_allRuns)
    np.save(path + f"convergence_O_sigma0{int(sigma0*10)}_{int(lmda*100)}_{nRestarts}runs", obj_allRuns)


# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Convergence plots
f,ax = plt.subplots()

o = f"O"    # O: objective, C: cost, R: reward
a1  = np.load(path + f"convergence_{o}_sigma010_10_{nRestarts}runs.npy")
a10 = np.load(path + f"convergence_{o}_sigma030_10_{nRestarts}runs.npy")
a30 = np.load(path + f"convergence_{o}_sigma040_10_{nRestarts}runs.npy")
a50 = np.load(path + f"convergence_{o}_sigma050_10_{nRestarts}runs.npy")

e1  = np.std(a1, axis=0)
e10 = np.std(a10, axis=0)
e30 = np.std(a30, axis=0)
e50 = np.std(a50, axis=0)
a1  = np.mean(a1, axis=0)
a10 = np.mean(a10, axis=0)
a30 = np.mean(a30, axis=0)
a50 = np.mean(a50, axis=0)

x = np.arange(len(a1))*100
ax.errorbar(x,a1,e1)
ax.errorbar(x,a10,e10)
ax.errorbar(x,a30,e30)
ax.errorbar(x,a50,e50)

ax.legend(['$\sigma_0$=1','$\sigma_0$=3','$\sigma_0$=4','$\sigma_0$=5'], fontsize=16, loc='center right')
ax.set_xlabel('Number of trials', fontsize=16)
ax.set_ylabel('Objective (-)', fontsize=16)
ax.set_title('Convergence for diff. initial resources', fontsize=16)
ax.tick_params(labelsize=11)
ax.set_ylim([-50,0])

# Cost
o = f"C"    # O: objective, C: cost, R: reward
a1  = np.load(path + f"convergence_{o}_sigma010_10_{nRestarts}runs.npy")
a10 = np.load(path + f"convergence_{o}_sigma030_10_{nRestarts}runs.npy")
a30 = np.load(path + f"convergence_{o}_sigma040_10_{nRestarts}runs.npy")
a50 = np.load(path + f"convergence_{o}_sigma050_10_{nRestarts}runs.npy")

a1[:,0] = 50

e1  = np.std(a1, axis=0)
e10 = np.std(a10, axis=0)
e30 = np.std(a30, axis=0)
e50 = np.std(a50, axis=0)
a1  = np.mean(a1, axis=0)
a10 = np.mean(a10, axis=0)
a30 = np.mean(a30, axis=0)
a50 = np.mean(a50, axis=0)

ax2 = ax.twinx()
ax2.errorbar(x,a1,e1, ls='--')
ax2.errorbar(x,a10,e10, ls='--')
ax2.errorbar(x,a30,e30, ls='--')
ax2.errorbar(x,a50,e50, ls='--')

ax2.set_ylabel('Cost (--)', fontsize=16)
ax2.set_title('Convergence for diff. initial resources', fontsize=16)
ax2.tick_params(labelsize=11)
ax2.set_ylim([0,30])
plt.show()