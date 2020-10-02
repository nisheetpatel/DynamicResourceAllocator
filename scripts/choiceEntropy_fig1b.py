import itertools
import numpy as np
import pandas as pd
from time import time, sleep
from task import Maze
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

env = Maze()
n_trials = int(10e3)

# Hyper params
beta        = 10
N_traj      = 10
nRestarts   = 1

# initializing params
lmda= 0.1
a1  = 0.2   # learning rate for q
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

df_allRuns  = []                # List to collect results (q,sigma) from all runs
expR_allRuns= []                # List to collect expected rewards from all runs
cost_allRuns= []                # List to collect cost from all runs
obj_allRuns = []                # List to collect objective from all runs
for run in range(nRestarts):
    expR       = []             # List to collect expected rewards(t)
    cost       = []             # List to collect cost(t)
    objective  = []             # List to collect objective(t)
    # Initializing Gaussian approximated q-distribution
    sigma0  = 3               # Initial sigma for Q ~ N(q,sigma)
    sigmaBase = 5               # Sigma for base distribtion
    sigmaT  = 0.1               # sigma for terminal states (defines min possible sigma)

    q       = np.zeros(env.q_size)
    sigma   = sigma0*np.ones(env.q_size)
    
    start = time()
    for ii in range(n_trials):
        if np.mod(ii,100)==0 | (ii >= env.switch_time & ii < env.switch_time+100 & np.mod(ii,10)==0):
            print(f"Run {run+1}/{nRestarts}, Trial {ii}/{n_trials}, time={np.around(time()-start,1)}")

            # Computing choice entropy
            choiceEntropy = np.zeros((q.shape[0],q.shape[1]))
            for x in range(q.shape[0]):
                for y in range(q.shape[1]):
                    if ([x,y] in env.obstacles) | ([x,y] in env.GOAL_STATES):
                        choiceEntropy[x,y] = np.nan
                    else:
                        samples = q[x,y,:] + sigma[x,y,:]*np.random.randn(int(1e5),len(env.actions))
                        choice = np.argmax(samples, axis=1)
                        count  = np.bincount(choice)
                        jj     = np.nonzero(count)[0]
                        p      = count[jj]/int(1e5)
                        choiceEntropy[x,y] = entropy(p)

            # Plotting
            fig = plt.figure()
            ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])
            ax.imshow(choiceEntropy)
            fig.colorbar(plt.cm.ScalarMappable())
            plt.savefig(f"./figures/ChoiceEntropyEvolution/{ii}")
            plt.close()

        if ii == env.switch_time:
            env.switch()

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