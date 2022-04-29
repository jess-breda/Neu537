def take_action(cur_s, V, epsilon):
    """Takes an action and returns the next state (nxt_s) and a bool indicating
    if the agent hit the wall. If hit_wall = True, cur_s = nxt_s. Epsilon value determines
    which action policy to use."""

    if np.random.uniform (0,1) <= epsilon:
        nxt_s, hit_wall = take_random_action(cur_s)
    else:
        nxt_s, hit_wall = take_greedy_action(cur_s, V)
    
    return nxt_s, hit_wall

def take_random_action(cur_s):
    """Randomly selects movement from 4 options given current state (cur_s) and then
    checks for wall hit & adjusts"""

    actions=["up","down","left","right"]
    walls = np.array([-1,7])
    action = np.random.choice(actions)

    # make move
    if action == "up":
        nxt_s = (cur_s[0] - 1, cur_s[1])
    elif action == "down":
        nxt_s = (cur_s[0] + 1, cur_s[1])
    elif action == "left":
        nxt_s = (cur_s[0] , cur_s[1] - 1)
    elif action == "right":
        nxt_s = (cur_s[0] , cur_s[1] + 1)

    valid_nxt_s, hit_wall = check_boundaries(cur_s, nxt_s, walls)

    return valid_nxt_s, hit_wall

def take_greedy_action(cur_s, V):
    """Takes a greedy action given 4 options,  given current state (cur_s) and Values (V),
    checks for wall hit & adjusts"""

    walls = np.array([-1,7])

    # go through all possible moves:
    up = (cur_s[0] - 1, cur_s[1])
    down = (cur_s[0] + 1, cur_s[1])
    left = (cur_s[0], cur_s[1] - 1)
    right = (cur_s[0], cur_s[1] + 1)
    
    moves = [up, down, left, right]
    
    Vs = [-100] * len(moves) # init some empty space that's indexable
    for idx, move in enumerate(moves):

        # there's no value associated with boundary, so you need to adjust and grab
        # the value for the current state
        if move[0] in walls or move[1] in walls:
            valid_move = cur_s
        else:
            valid_move = move
        
        Vs[idx] = V[valid_move[0], valid_move[1]]

    # select move with the highest value
    nxt_s = moves[np.argmax(Vs)]

    # check boundary
    valid_nxt_s, hit_wall = check_boundaries(cur_s, nxt_s, walls)

    return valid_nxt_s, hit_wall

def check_boundaries(cur_s, nxt_s, walls):
    if nxt_s[0] in walls or nxt_s[1] in walls:
        return cur_s, True
    else:
        return nxt_s, False

def give_reward(valid_nxt_s, hit_wall):
    "Gives the reward for the action taken, and adjusts for hitting goal state or wall"
    # init 
    win_big = (6,6)
    win_small = (3,3)

    # determine
    if hit_wall:
        reward = -5
    elif valid_nxt_s == win_big:
        reward = 100
    elif valid_nxt_s == win_small:
        reward = 10
    else:
        reward = -1
    return reward

def update_value(V, cur_s, valid_nxt_s, reward, discount, lr=0.01):
    "TD algorithm"
    
    exp_V = V[cur_s[0], cur_s[1]] # expected value given state
    nxt_V = V[valid_nxt_s[0], valid_nxt_s[1]] # next value given state + 1
    prediction_error = reward + (discount*nxt_V) - exp_V 
    
    # update value
    V[cur_s[0], cur_s[1]] = exp_V + (lr * prediction_error)

    return V

def simulate_agent(n_tsteps, epsilon, discount=0.95):
    "Simulates agent for a single run given n_steps, epsilon and discount factor"
   
    # init
    rows = 7
    columns = 7
    start = (0,0)
    win_big =(6,6)
    win_small = (3,3)
    
    # create space
    V = np.random.normal(0,1, size=(rows, columns))
    total_reward = 0 # will add to each loop
    avg_reward = np.zeros(n_tsteps)

    # simulate agent
    for i in range(n_tsteps):

        # get state at time t and adjust
        if i == 0:
            cur_s = start
        elif nxt_s == win_big or nxt_s == win_small:
            cur_s = start
        else:
            cur_s = nxt_s
        
        # get state at time t + 1 given action & policy
        nxt_s, hit_wall = take_action(cur_s, V, epsilon)

        # get reward at time t + 1 & update reward history
        reward = give_reward(nxt_s, hit_wall)
        total_reward += reward
        avg_reward[i] = total_reward / (i + 1)
        
        # update values
        V = update_value(V, cur_s, nxt_s, reward, discount=discount)
    
    return V, avg_reward

def simulate_multi_parameter(epsilons, discount, averaged = True, runs=5, n_tsteps=50000):
    "Simulate agent for multiple runs w/ the ability to average across runs"

    Vs = []
    Rs = []

    for epsilon in epsilons:
        print(f"startings runs for epsilon: {epsilon}")

        runs_Vs = np.zeros((n_runs, 7, 7))
        runs_Rs = np.zeros((n_runs, n_tsteps))

        for run in range(runs):
            runs_Vs[run,:,:], runs_Rs[run, :] = simulate_agent(n_tsteps, epsilon, discount=discount)

        if averaged:
            Vs.append(np.mean(runs_Vs, axis = 0))
            Rs.append(np.mean(runs_Rs, axis = 0))

        else:
            Vs.append(runs_Vs)
            Rs.append(runs_Rs)

    return Vs, Rs


