import gym
import numpy as np

class GradientBasedResourceAllocator:                                         #
    def __init__(self, episodes=int(5e3), learning_q=0.2, learning_sigma=0.1,
                discount=1, nRestarts=1, nTraj=10, environment='MountainCar',
                policy='Thompson', exploration=10, lmda=0.1, gradient='A',
                explore_slowly=True, render=False):
        self.episodes       = episodes
        self.nRestarts      = nRestarts     # currently not implemented
        self.learning_q     = learning_q
        self.learning_sigma = learning_sigma
        self.discount       = discount
        self.nTraj          = nTraj
        self.policy         = policy
        self.exploration    = exploration
        self.explore_slowly = explore_slowly
        self.lmda           = lmda
        self.gradient       = gradient
        env_dict = {'Huys': 1,              # currently not implemented
                    'TwoStepMaze': 2,       # currently not implemented
                    'GridworldMaze': 3,     # currently not implemented
                    'MountainCar': gym.make('MountainCar-v0')}
        self.env = env_dict[environment]
        self.envName = environment
        self.render = render

        # Initialising q-distribution: N(q, diag(sigma)^2)
        self.q  = np.random.uniform(low = -1, high = 1, 
                                    size = (19,15,3) ) #self.env.q_size)
        self.sigma0     = 1
        self.sigmaBase  = 5
        self.sigma      = np.ones(self.q.shape) * self.sigma0

    def softmax(self, x):
        x = np.array(x)
        b = np.max(self.exploration*x)      # Trick to avoid overflow errors
        y = np.exp(self.exploration*x - b)  # during numerical calculations
        return y/y.sum()

    # the cost function
    @staticmethod
    def kl_mvn(m0, S0, m1, S1):
        """
        Kullback-Liebler divergence from Gaussian m0,S0 to Gaussian m1,S1.
        Diagonal covariances are assumed.  Divergence is expressed in nats.
        """    
        # store inv diag covariance of S1 and diff between means
        N = m0.shape[0]
        iS1 = np.linalg.inv(S1)
        diff = m1 - m0

        # three terms of the KL divergence
        tr_term   = np.trace(iS1 @ S0)
        #det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0))
        det_term  = np.trace(np.ma.log(S1)) - np.trace(np.ma.log(S0))
        quad_term = diff.T @ np.linalg.inv(S1) @ diff 
        return .5 * (tr_term + det_term + quad_term - N)

    def computeCost(self):
        # Computing the cost of the current memories
        q = self.q.flatten()
        S1 = np.diag(np.square(self.sigma.flatten()))
        S0 = np.diag(np.square(np.ones(len(self.sigma.flatten()))*5))
        return (self.lmda * self.kl_mvn(q,S1,q,S0))

    def act(self, state):        
        # Soft thompson sampling: softmax applied instead of max
        if self.policy == 'Thompson':
            x, y    = state
            zeta_s  = np.random.randn(self.env.action_space.n)
            prob_a  = self.softmax(self.q[x,y,:] + zeta_s*self.sigma[x,y,:])
            action  = np.random.choice(np.arange(len(prob_a)), p=prob_a)
            return action, zeta_s, prob_a
        else:
            raise ValueError('Currently only supporting policy: Thompson')

    def learn(self):
        # Initialize variables to track rewards and explore efficiently
        reward_list = []
        ave_reward_list = []
        reduction = 0

        if self.envName == 'MountainCar':
            # Reduction in exploration parameter
            reduction = 1.5*self.exploration/self.episodes

            def discretizeState(state):
                s = (state-self.env.observation_space.low)*np.array([10,100])
                s = np.around(s, 0).astype(int)
                return s
        
        for ii in range(self.episodes):
            # Initializing some variables
            done = False
            tot_reward, reward = 0,0
            state = self.env.reset()
            state = discretizeState(state)

            while not done:
                # Render environment for last five episodes
                if (ii >= (self.episodes - 5)) & (self.render==True):
                    self.env.render()

                # Determine next action
                action, _, _ = self.act(state)
                    
                # Get next state and reward
                state2, reward, done, info = self.env.step(action)
                s1 = discretizeState(state2)

                _,_,p1  = self.act(s1)
                x, y    = state
                x1, y1  = s1

                # Allow for terminal states
                if done and state2[0] >= 0.5:
                    self.q[x,y,action] = reward

                # Adjust Q value for current state
                else:
                    delta = reward + \
                            self.discount * np.dot(p1, self.q[x1,y1,:]) -\
                            self.q[x,y,action]
                    self.q[x,y,action] += self.learning_q * delta
                    # self.q[x,y,a] += self.learning_q * (r + self.discount\
                    #              *np.max(self.q[x1,y1,:]) - self.q[x,y,a])

                # Update variables
                tot_reward += reward
                state = s1

            # Track rewards
            reward_list.append(tot_reward)
            
            # Efficient exploration
            if self.explore_slowly:
                self.exploration = min(1.5**(ii/1000), 10)

            if (ii+1) % 100 == 0:
                ave_reward = np.mean(reward_list)
                ave_reward_list.append(ave_reward)
                reward_list = []
                print(f'Episode {ii+1}, \
                        Reward = {np.around(ave_reward,2)}')

            # Update sigma at the end of each trial by SGD:
            grads   = []
            for jj in range(int(self.nTraj)):
                # Initialising some variables
                grad  = np.zeros(self.q.shape)
                done = False
                tot_reward, reward = 0,0
                state = self.env.reset()
                state = discretizeState(state)
                r = []

                while not done:
                    # Determine next action
                    action, z, prob_a = self.act(state)
                    
                    # Get next state and reward
                    state2, reward, done, info = self.env.step(action)
                    s1 = discretizeState(state2)
                    r.append(reward)

                    # Compute advantage/Q/R gradient
                    x, y              = state
                    
                    if self.gradient == 'A':
                        advantage = self.q[x,y,action] - np.mean(self.q[x,y,:])
                    elif self.gradient == 'Q':
                        advantage = self.q[x,y,action]
                    elif self.gradient == 'R':
                        advantage = 1

                    grad[x,y,:]      -= (self.exploration * z * prob_a) *\
                                        advantage
                    grad[x,y,action] += (self.exploration * z[action]) *\
                                        advantage
                    
                    # Update state for next step, add total reward
                    state       = s1
                    tot_reward += reward
                
                if self.gradient == 'R':
                    rturn = np.sum(r)
                    grads += [np.dot(rturn,grad)]
                else:
                    # Collect sampled stochastic gradients for all trajectories
                    grads   += [grad]
                    reward_list.append(tot_reward)

            # Compute average gradient across sampled trajs & cost
            grad_cost = (self.sigma/(self.sigmaBase**2) - 1/self.sigma)
            grad_mean = np.mean(grads, axis=0)

            # Updating sigmas
            self.sigma += self.learning_sigma * \
                            (grad_mean - self.lmda * grad_cost)
            self.sigma  = np.clip(self.sigma, 0.01, 10)
        
        self.env.close()
    
        return np.mean(ave_reward_list[:-3])


if __name__ == '__main__':
    # Script for plots
    from dynamicResourceAllocator import GradientBasedResourceAllocator
    import numpy as np

    def discretizeState(obj, state):
        s = (state-obj.env.observation_space.low)*np.array([10,100])
        s = np.around(s, 0).astype(int)
        return s

    # Train and compute learnt objective
    cost, expR, objective = [], [], []
    gradients = ['R', 'Q', 'A']
    for gradient in gradients:
        obj = GradientBasedResourceAllocator(gradient=gradient)
        rew = obj.learn()

        # Append to list of exp rewards, cost, and objective
        expR.append(rew)
        cost.append(obj.computeCost())
        objective.append(expR[-1] - cost[-1])
    
    # Plotting
    import matplotlib.pyplot as plt

    # Value function (Fig 2B)
    value = np.max(obj.q, axis=2)   # average across runs for cleaner plot
    
    fig = plt.figure()
    ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])
    im = ax.imshow(value.transpose(), cmap='viridis')
    fig.colorbar(im)
    plt.show()

    # Computing and plotting choice entropy (Fig 2C)
    from scipy.stats import entropy

    choiceEntropy = np.zeros((obj.q.shape[0],obj.q.shape[1]))
    for x in range(obj.q.shape[0]):
        for y in range(obj.q.shape[1]):
            samples = obj.q[x,y,:] + obj.sigma[x,y,:] * \
                        np.random.randn(obj.episodes, 3)
            choice = np.argmax(samples, axis=1)
            count  = np.bincount(choice)
            ii     = np.nonzero(count)[0]
            p      = count[ii] / obj.episodes
            choiceEntropy[x,y] = entropy(p)
    choiceEntropy = np.around(choiceEntropy, 1)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])
    im = ax.imshow(choiceEntropy.transpose(), cmap='cividis')
    fig.colorbar(im)
    plt.show()

    # Comparing the different gradients (Fig 2D)
    fig = plt.figure()
    ax = fig.add_subplot()
    for ii in range(len(gradients)):
        plt.bar(ii, objective[ii])
    plt.xticks([0, 1, 2], gradients)
    plt.show()