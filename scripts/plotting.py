"""
Plotting for the Huys task. The functions have been written
for the object in resourceAllocator.py (gradient-free 
optimization - either CMAES or Bayesian Optimization for 
the equal precision model).
Also plots comparison of gradient-based with gradient-free
after the relevant simulation results have been produced.
"""

import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib, matplotlib.pyplot as plt
sns.set()


def plot_table(obj, tableColour='mean'):
    # Setting up memory columns other than resourceAllocVector
    a   = np.array([' left','right'])
    s   = np.arange(6) + 1
    r   = obj.env.rewards[:12]
    s1  = ((obj.env.transitions[:12]+1)[:,1] - 6)
    sa  = [sa for sa in itertools.product(s,a)] * (obj.depth-1)
    rs  = list(zip(r,s1)) * (obj.depth-1)
    q   = obj.q[:-12]
    d   = {'(s,a)': sa, '(r,s\')': rs, 'q_mu(s,a)':q}

    # Setting up resource allocation vectors for table of memories
    nMems = obj.results.shape[-1]
    means = np.transpose( np.mean(obj.results, axis=1) )
    stds  = np.transpose( np.std(obj.results, axis=1) ) 
    cols  = np.around( (np.array(obj.searchBudget) / \
            (obj.depth * 2**obj.depth) * 100), decimals=1)
    
    # Table of optimal resource allocation vectors for different search budgets
    df = pd.DataFrame(d)
    if len(means)==1:       # if equal precision => broadcast
        means = np.array(list(means)*len(df))
        nMems = len(df)
    dm = pd.DataFrame(means.astype(int), columns=cols)
    df = pd.merge(df, dm, left_index=True, right_index=True)
    vals = dm.values
    extraCols = 3

    # Setting up figure
    fig = plt.figure() 
    ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])

    # Depth dependent parameters:
    if obj.depth==3:
        fh = 6
        cbx= 0.42
        sc = 1.2
        tx = 0.4
    elif obj.depth>=4:
        fh = obj.depth*3 - 2
        cbx= 0.32
        sc = 1.35
        tx = 0.2

    fig.set_figheight(fh) 
    fig.set_figwidth(8) 

    # setting colourmap either based on means only, or both means & std 
    if tableColour=='mean': 
        norm = plt.Normalize(0,100)
        colours = plt.cm.viridis(norm(vals))
        colours = np.concatenate((np.array([[[0]*4]*extraCols]*nMems),colours), axis=1)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis), ax=ax\
            , cax= fig.add_axes([0.42, 0.04, 0.54, 0.02]), orientation='horizontal')
    elif tableColour=='std':
        norm = plt.Normalize(0,50)
        colours = plt.cm.Greens_r(norm(stds))
        colours = np.concatenate((np.array([[[0]*4]*extraCols]*nMems),colours), axis=1)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.Greens_r), ax=ax\
            , cax= fig.add_axes([cbx, 0.04, 0.54, 0.02]), orientation='horizontal') 

    # Plotting the table of memories according to the colourmap 
    the_table=matplotlib.table.table(ax, cellText=df.values, rowLabels=None, \
                colLabels=df.columns, loc='center', cellColours=colours)
    fig.text(tx,0.925,'Resource allocation, i.e. q_sigma(s,a), for search budget (%)')
    the_table.scale(1,sc)  # depth 4: 1.35
    plt.show()
    return fig, ax


# Plotting <reward>, cost, and overall objective vs search budget
def plot_curves(obj):
    # Initialising arrays
    reward      = np.nan * np.ones( len(obj.searchBudget) )
    reward_std  = np.nan * np.ones( len(obj.searchBudget) )
    cost        = np.nan * np.ones( len(obj.searchBudget) )
    cost_std    = np.nan * np.ones( len(obj.searchBudget) )
    objective   = np.nan * np.ones( len(obj.searchBudget) )
    obj_std     = np.nan * np.ones( len(obj.searchBudget) )

    # Simulating to get expected rewards, cost, and obj as funcs of sigma_opt 
    for idx, budget in enumerate(obj.searchBudget):
        sigma = obj.results[idx,:,:]    # size (n_restarts x n_mems)
        r = np.zeros(len(sigma))
        c = np.zeros(len(sigma))
        o = np.zeros(len(sigma))
        for i in range(len(sigma)):
            r[i] = obj.expectedRewards(sigma=sigma[i], searchBudget=budget)
            c[i] = obj.cost(sigma=sigma[i])
            o[i] = r[i] - c[i]
        reward[idx]     = np.mean(r)
        reward_std[idx] = np.std(r)
        cost[idx]       = np.mean(c)
        cost_std[idx]   = np.std(c)
        objective[idx]  = np.mean(o)
        obj_std[idx]    = np.std(o)
    
    # Storing the arrays in a pandas dataframe
    d = {'Reward': reward, 'Cost': cost, 'Objective': objective,\
        'err_reward': reward_std, 'err_cost': cost_std, 'err_obj': obj_std}
    obj.dfResults = pd.DataFrame(d)

    # Setting up the figure subplots and axes
    f, (ax,ax2) = plt.subplots(nrows=2,ncols=1,sharex=True)#,facecolor='w')
    xaxis = np.array(obj.searchBudget) / (obj.depth * 2**obj.depth) * 100
    ax2.set_xlabel('Search Budget (% tree explored)', fontsize = 16)
    ax.set_ylabel('(a.u.)', fontsize = 16)

    # Plot same data on both axis (hack to get different colour for each curve)
    ax.errorbar(xaxis, reward, yerr = reward_std, label = 'Reward')
    ax.errorbar(xaxis, objective, yerr = obj_std, label = 'Objective')
    ax.errorbar(xaxis, cost, yerr = cost_std, label = 'Cost')
    ax2.errorbar(xaxis, reward, yerr = reward_std, label = 'Reward')
    ax2.errorbar(xaxis, objective, yerr = obj_std, label = 'Objective')
    ax2.errorbar(xaxis, cost, yerr = cost_std, label = 'Cost')

    # Set axis limits
    ax.set_ylim(50,80)
    ax2.set_ylim(0,15)

    # Hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop='off')

    # Making the kinks or diagonal break-lines between the subplots
    d = .01  # size of diagonal kinks
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    # Show plot with legend
    ax2.legend()
    plt.show()
    return f, ax, ax2


# Plotting the table of memories
def plot_dprime(obj, table=True):
    # Setting up memory columns other than resourceAllocVector
    states = np.arange(6) + 1
    q = obj.q[:-12]

    # Calculating d' and setting column names
    cols = np.around((np.array(obj.searchBudget) / \
            (obj.depth * 2**obj.depth) * 100), decimals=1)
    m1 = (q[1:len(q):2]).reshape(int(len(q)/2), 1)      # mean for action 'right'
    m2 = (q[0:len(q)+1:2]).reshape(int(len(q)/2), 1)    # mean for action 'left'
    dprime = np.zeros( (len(m1),len(obj.searchBudget),obj.results.shape[1]) )
    for i in range(obj.results.shape[1]):
        s1 = np.transpose(obj.results[:, i, 1:len(q):2])
        s2 = np.transpose(obj.results[:, i, 0:len(q)+1:2])
        dprime[:,:,i] = abs(m1-m2) / np.sqrt( s1**2 + s2**2 )
    dprime_mean = np.mean(dprime, axis=2)
    nMems = len(m1)

    # Creating dataframe for plotting
    d = {'State': list(states)*(obj.depth-1), 'Reward diff.': abs(m1-m2).ravel().astype(str)}
    dm = pd.DataFrame(np.around(dprime_mean, decimals=1), columns=cols)
    vals = dm.values
    df = pd.merge(pd.DataFrame(d), dm, left_index=True, right_index=True)
    obj.dprime = df

    if table:
        # setting plotting values right
        extraCols = 2

        # setting up figure
        fig = plt.figure() 
        ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])

        # Depth dependent parameters:
        if obj.depth==3:
            fh = 4
            cbx= 0.42
            sc = 1.2
            tx = 0.3
        elif obj.depth>=4:
            fh = 2*obj.depth - 2
            cbx= 0.32
            sc = 1.35
            tx = 0.2

        fig.set_figheight(fh) 
        fig.set_figwidth(8) 
    
        # setting colourmap based on mean dprime values
        norm = plt.Normalize(0, 4)
        colours = plt.cm.viridis_r(norm(vals))
        colours = np.concatenate((np.array([[[0]*4]*extraCols]*nMems),colours), axis=1)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis_r), ax=ax\
            , cax= fig.add_axes([cbx, 0.1, 0.54, 0.02]), orientation='horizontal')
    
        # Plotting the table of memories according to the colourmap 
        the_table=matplotlib.table.table(ax, cellText=df.values, rowLabels=None, \
                    colLabels=df.columns, loc='center', cellColours=colours)
        fig.text(tx,0.875,'d\', i.e., discriminability between q-distributions, for search budget (%)')
        the_table.scale(1,sc)
        plt.show()
        return fig, ax
    else:
        # Dataframe of all dprime values 
        dprime_full = dprime.transpose(1,0,2).reshape(dprime.shape[1],-1).transpose()
        d = {'State': list(np.repeat(states, dprime.shape[2])) * (obj.depth-1), \
            'Reward diff.': np.repeat(abs(m1-m2).ravel(), dprime.shape[2])}
        dm = pd.DataFrame(np.around(dprime_full, decimals=1), columns=cols)
        dfull = pd.merge(pd.DataFrame(d), dm, left_index=True, right_index=True)

        # Figure settings
        palette = sns.color_palette("mako_r", len(obj.searchBudget))
        sns.set()

        # separating different depths
        df_depth = list(range(obj.depth-1))
        for i in range(obj.depth-1):
            blockLen = 6 * obj.n_restarts
            df_depth[i] = pd.melt(dfull.iloc[i*blockLen:(i+1)*blockLen,1:], id_vars=['Reward diff.'], \
                        value_vars=cols, var_name='Search budget', value_name='d-prime')
            # Plotting figure for each depth
            f = plt.figure()
            ax1 = f.subplots()
            ax1 = sns.lineplot(x='Reward diff.', y='d-prime', hue='Search budget', \
                            data=df_depth[i], palette=palette).set_title(f'Depth {i+1} memories')
            plt.show()
        return ax1



############### Plotting comparison with CMAES ##################
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import itertools
import pickle
sns.set()

# Defining the cost functions
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

# Initialising the lists
x               = []
method          = []
expectedRewards = []
cost            = []
objective       = []

# Initiliazing path and other variables
nRuns   = 5
path    = "./figures/"  # initialResults/gradientBasedLearning/q-learning_adv

# Gradient-based optimization for all depths
depth = 3
for lmda,N_traj in itertools.product([0.15,0.3,1],[1,10]):
    method += [f'$\\nabla_\sigma$, $N_{{traj}}$={N_traj}, $q_{{*}}$']*nRuns
    x      += [f'Depth {depth}, lambda={lmda}']*nRuns
    expectedRewards += list(np.load(path+f"expR_depth{depth}_{int(lmda*100)}_{N_traj}.npy"))
    # computing the cost
    df_allRuns = pickle.load(open(path +\
            f"df_allRuns_depth{depth}_{int(lmda*100)}_{N_traj}",'rb'))
    for df in df_allRuns:
        q          = df['q'][:-12]
        sigma      = df['sigma'][:-12]
        cost      += [lmda * kl_mvn(q, np.diag(np.square(sigma)), \
                            q, np.diag(np.square( [100]*(len(q)) )))]
    objective += list(np.array(expectedRewards)[-nRuns:] - np.array(cost)[-nRuns:])

lmda = 1
for depth,N_traj in itertools.product([4,5],[1,10]):
    method += [f'$\\nabla_\sigma$, $N_{{traj}}$={N_traj}, $q_{{*}}$']*nRuns
    x      += [f'Depth {depth}, lambda={lmda}']*nRuns
    expectedRewards += list(np.load(path+f"expR_depth{depth}_{int(lmda*100)}_{N_traj}.npy"))
    # computing the cost
    df_allRuns = pickle.load(open(path+f"df_allRuns_depth{depth}_{int(lmda*100)}_{N_traj}",'rb'))
    for df in df_allRuns:
        q          = df['q'][:-12]
        sigma      = df['sigma'][:-12]
        cost      += [lmda * kl_mvn(q, np.diag(np.square(sigma)), \
                            q, np.diag(np.square( [100]*(len(q)) )))]
    # Objective = Expected rewards - cost
    objective += list(np.array(expectedRewards)[-nRuns:] - np.array(cost)[-nRuns:])

# Depth 3, lambda = {1, 0.3, 0.15}; gradient-free (CMAES) optimization
x       +=  ['Depth 3, lambda=1']*1000 #+\
            # ['Depth 3, lambda=0.3']*1000 +\
            # ['Depth 3, lambda=0.15']*1000
method  += ['Gradient-free (CMAES)']*1000

for i in [100]: #,30,15]:
    dfResults = pickle.load(open(f'./initialResults/Depth3/DKL_lmda{i}_variablePrecision/dfResults','rb'))
    meanR = dfResults.iloc[0]['Reward']
    stdR  = dfResults.iloc[0]['err_reward']
    meanC = dfResults.iloc[0]['Cost']
    stdC  = dfResults.iloc[0]['err_cost']
    meanO = dfResults.iloc[0]['Objective']
    stdO  = dfResults.iloc[0]['err_obj']
    expectedRewards  += list(meanR + np.random.randn(1000)*stdR)
    cost             += list(meanC + np.random.randn(1000)*stdC)
    objective        += list(meanO + np.random.randn(1000)*stdO)

# Depth 4, lambda = 1; gradient-free (CMAES) optimization
x      += ['Depth 4, lambda=1']*1000
method += ['Gradient-free (CMAES)']*1000
model   = pickle.load(open("./initialResults/Depth4/varPrecModel.pkl",'rb'))
meanR   = model.dfResults.iloc[0]['Reward']
stdR    = model.dfResults.iloc[0]['err_reward']
meanC   = model.dfResults.iloc[0]['Cost']
stdC    = model.dfResults.iloc[0]['err_cost']
meanO   = model.dfResults.iloc[0]['Objective']
stdO    = model.dfResults.iloc[0]['err_obj']
expectedRewards  += list(meanR + np.random.randn(1000)*stdR)
cost             += list(meanC + np.random.randn(1000)*stdC)
objective        += list(meanO + np.random.randn(1000)*stdO)

# Creating final dataframe for plotting
df = pd.DataFrame({'x':x, 'Method':method, 'Expected Rewards':expectedRewards,\
                   'Cost':cost, 'Objective':objective})

# Plotting
toPlot  = ["Expected Rewards", "Cost", "Objective"]
color   = ["Blues_d", "Greens_d", "Reds_d"]
for y,colr in zip(toPlot,color):
    fig, ax = plt.subplots()
    fig.set_figwidth(10)
    ax = sns.barplot(x="x", y=y, hue="Method",data=df, ci="sd", palette=colr)
    plt.show()
# ax.set_ylim([])


######################## Comparison #####################
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import itertools
import pickle
sns.set()

# Initialising the lists
x               = []
method          = []
expectedRewards = []
cost            = []
objective       = []

# Initiliazing path and other variables
nRuns   = 5
basePath   = f"./initialResults/gradientBasedLearning/onPolicy-learning"
subPath    = [f"/", f"_return/", f"_qpiSamples/", f"_qpiMean/", f"_adv/"]
gradUpdate = [f"$r(\\tau)$", f"$G_t$", f"$q_\pi$", f"$meanq_\pi$", f"A"]

for subP, label in zip(subPath, gradUpdate):
    depth = 3
    for lmda,N_traj in itertools.product([0.15,0.3,1],[10]):
        method += [label]*nRuns
        x      += [f'Depth {depth}, lambda={lmda}']*nRuns
        expectedRewards += list(np.load(basePath+subP+f"expR_depth{depth}_{int(lmda*100)}_{N_traj}.npy"))
        # computing the cost
        df_allRuns = pickle.load(open(basePath + subP +\
                f"df_allRuns_depth{depth}_{int(lmda*100)}_{N_traj}",'rb'))
        for df in df_allRuns:
            q          = df['q'][:-12]
            sigma      = df['sigma'][:-12]
            cost      += [lmda * kl_mvn(q, np.diag(np.square(sigma)), \
                                q, np.diag(np.square( [100]*(len(q)) )))]
        objective += list(np.array(expectedRewards)[-nRuns:] - np.array(cost)[-nRuns:])

    lmda = 1
    for depth,N_traj in itertools.product([4,5],[10]):
        method += [label]*nRuns
        x      += [f'Depth {depth}, lambda={lmda}']*nRuns
        expectedRewards += list(np.load(basePath+subP+f"expR_depth{depth}_{int(lmda*100)}_{N_traj}.npy"))
        # computing the cost
        df_allRuns = pickle.load(open(basePath+subP+f"df_allRuns_depth{depth}_{int(lmda*100)}_{N_traj}",'rb'))
        for df in df_allRuns:
            q          = df['q'][:-12]
            sigma      = df['sigma'][:-12]
            cost      += [lmda * kl_mvn(q, np.diag(np.square(sigma)), \
                                q, np.diag(np.square( [100]*(len(q)) )))]
        # Objective = Expected rewards - cost
        objective += list(np.array(expectedRewards)[-nRuns:] - np.array(cost)[-nRuns:])

    depth = 10
    if subP in [f"/", f"_qpiMean/", f"_adv/"]:
        method += [label]*nRuns
        x      += [f'Depth {depth}, lambda={lmda}']*nRuns
        expectedRewards += list(np.load(basePath+subP+f"expR_depth{depth}_{int(lmda*100)}_{N_traj}.npy"))
        # computing the cost
        df_allRuns = pickle.load(open(basePath+subP+f"df_allRuns_depth{depth}_{int(lmda*100)}_{N_traj}",'rb'))
        for df in df_allRuns:
            q          = df['q'][:-12]
            sigma      = df['sigma'][:-12]
            cost      += [lmda * kl_mvn(q, np.diag(np.square(sigma)), \
                                q, np.diag(np.square( [100]*(len(q)) )))]
        # Objective = Expected rewards - cost
        objective += list(np.array(expectedRewards)[-nRuns:] - np.array(cost)[-nRuns:])


# Creating final dataframe for plotting
df = pd.DataFrame({'x':x, 'Method':method, 'Expected Rewards':expectedRewards,\
                   'Cost':cost, 'Objective':objective})

# Plotting
toPlot  = ["Expected Rewards", "Cost", "Objective"]
color   = ["Blues_d", "Greens_d", "Reds_d"]
for y,colr in zip(toPlot,color):
    fig, ax = plt.subplots()
    fig.set_figwidth(10)
    ax = sns.barplot(x="x", y=y, hue="Method",data=df, ci="sd", palette=colr)
    plt.show()
# ax.set_ylim([])