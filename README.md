
### Name : NATHIN R
### Reg.No. 212222230090

# Q Learning Algorithm


## AIM
To develop a Python program to find the optimal policy for the given RL environment using Q-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT

Develop a Python program to implement Q-Learning for finding the optimal policy in a given Reinforcement Learning (RL) environment. Evaluate the learned policy by comparing state values obtained using Q-Learning with those computed via the Monte Carlo method. The program should demonstrate convergence of Q-Learning and analyze the accuracy of the state values. Finally, visualize and interpret the results to assess the effectiveness of both approaches.

## Q LEARNING ALGORITHM

### Step 1:
Initialize Q-table and hyperparameters.
### Step 2:
Choose an action using the epsilon-greedy policy and execute the action, observe the next state, reward, and update Q-values and repeat until episode ends.
### Step 3:
After training, derive the optimal policy from the Q-table.
### Step 4:
Implement the Monte Carlo method to estimate state values.
### Step 5:
Compare Q-Learning policy and state values with Monte Carlo results for the given RL environment.

## Q LEARNING FUNCTION


```py
def q_learning(env,
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    # Epsilon-greedy action-selection strategy
    select_action = lambda state, Q, epsilon: \
        np.argmax(Q[state]) if np.random.random() > epsilon \
        else np.random.randint(len(Q[state]))

    # Decay schedule for alpha and epsilon
    alphas = decay_schedule(
        init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(
        init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    # Iterating over episodes
    for e in tqdm(range(n_episodes), leave=False):
        # Reset environment
        state, done = env.reset(), False

        # Interaction loop for online learning
        while not done:
            # Select an action
            action = select_action(state, Q, epsilons[e])
            next_state, reward, done, _ = env.step(action)

            # Full experience tuple (s, a, s', r, d)
            # Update code continues...
            # Update code continues...
                # Calculate TD target
            td_target = reward + gamma * np.max(Q[next_state]) * (not done)

                # Calculate TD error
            td_error = td_target - Q[state][action]

                # Update Q-value
            Q[state][action] += alphas[e] * td_error

                # Update state
            state = next_state

                # Track Q-function and policy
        Q_track[e] = Q.copy()
        pi_track.append(np.argmax(Q, axis=1))

    # Final policy
    V = np.max(Q, axis=1)
    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return Q, V, pi, Q_track, pi_track
```

## OUTPUT:



![image](https://github.com/user-attachments/assets/4bcafe3a-61fc-4672-8c77-649517306065)



![image](https://github.com/user-attachments/assets/fa7384bc-5220-4e37-b6ff-7ff83d3d4022)



![image](https://github.com/user-attachments/assets/816e1307-d685-4fc7-a6e8-26884a2ba074)



## RESULT:

Thus, Q-Learning outperformed Monte Carlo in finding the optimal policy and state values for the RL problem.
