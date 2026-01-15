import numpy as np
import matplotlib.pyplot as plt

# --- ADD THESE TWO LINES HERE ---
import matplotlib
matplotlib.use('Agg') # This allows saving files without a GUI window
# --------------------------------

# ---------------------------------------------------------
# 1. Objective Function (The problem we want to solve)
# ---------------------------------------------------------
def objective_function(x):
    """
    Sphere function: f(x) = sum(x^2)
    Global minimum is 0 at x = [0, 0, ..., 0]
    """
    return np.sum(x**2)

# ---------------------------------------------------------
# 2. Whale Optimization Algorithm (WOA) Implementation
# ---------------------------------------------------------
def whale_optimization_algorithm(obj_func, dim, search_agents_no, max_iter, lb, ub):
    
    # Initialize position of whales
    whales_pos = np.random.uniform(lb, ub, (search_agents_no, dim))
    
    # Initialize Leader (Best Whale)
    leader_pos = np.zeros(dim)
    leader_score = float("inf")  # We minimize, so init with infinity
    
    # Convergence curve to store the best score at each iteration
    convergence_curve = []
    
    print("Optimization started...")
    
    # Main Loop
    for t in range(max_iter):
        
        # Linearly decreasing 'a' from 2 to 0
        # This controls the transition from exploration to exploitation
        a = 2 - t * (2 / max_iter)
        
        for i in range(search_agents_no):
            
            # Boundary handling: Check if whales are out of search space
            whales_pos[i, :] = np.clip(whales_pos[i, :], lb, ub)
            
            # Calculate fitness
            fitness = obj_func(whales_pos[i, :])
            
            # Update Leader
            if fitness < leader_score:
                leader_score = fitness
                leader_pos = whales_pos[i, :].copy() 
        
        # Update the position of each whale
        for i in range(search_agents_no):
            
            r1 = np.random.random() # Random number [0, 1]
            r2 = np.random.random() # Random number [0, 1]
            
            A = 2 * a * r1 - a  # Equation (2.3) in WOA paper
            C = 2 * r2          # Equation (2.4) in WOA paper
            
            # Parameters for spiral update
            b = 1
            l = (np.random.random() * 2) - 1 # Random number [-1, 1]
            p = np.random.random()           # Probability to switch mechanism
            
            if p < 0.5:
                if abs(A) < 1:
                    # SHRINKING ENCIRCLING MECHANISM (Exploitation)
                    # Update position towards the leader
                    D = abs(C * leader_pos - whales_pos[i, :])
                    whales_pos[i, :] = leader_pos - A * D
                else:
                    # SEARCH FOR PREY (Exploration)
                    # Select a random whale
                    random_whale_index = np.random.randint(0, search_agents_no)
                    random_whale_pos = whales_pos[random_whale_index, :]
                    
                    D = abs(C * random_whale_pos - whales_pos[i, :])
                    whales_pos[i, :] = random_whale_pos - A * D
            
            else:
                # SPIRAL UPDATING POSITION (Bubble-net attacking)
                # Equation (2.5) in WOA paper
                distance_to_leader = abs(leader_pos - whales_pos[i, :])
                whales_pos[i, :] = distance_to_leader * np.exp(b * l) * np.cos(2 * np.pi * l) + leader_pos
        
        # Store best score
        convergence_curve.append(leader_score)
        
        if t % 5 == 0:
            print(f"Iteration {t}: Best Fitness = {leader_score}")

    return leader_score, leader_pos, convergence_curve

# ---------------------------------------------------------
# 3. Execution Block
# ---------------------------------------------------------
if __name__ == "__main__":
    # Parameters
    DIM = 30                # Number of dimensions (variables)
    SEARCH_AGENTS = 50      # Population size (number of whales)
    MAX_ITER = 100          # Maximum iterations
    LOWER_BOUND = -100      # Lower bound of search space
    UPPER_BOUND = 100       # Upper bound of search space
    
    # Run WOA
    best_score, best_pos, curve = whale_optimization_algorithm(
        objective_function, DIM, SEARCH_AGENTS, MAX_ITER, LOWER_BOUND, UPPER_BOUND
    )
    
    print("\n-------------------------------------------")
    print("Optimization Finished!")
    print(f"Best Fitness Found: {best_score}")
    print(f"Best Position: {best_pos[:5]} ... (showing first 5 dims)")
    print("-------------------------------------------")
    
    # Plotting the Convergence Curve (Crucial for your Demo)
    plt.figure(figsize=(10, 6))
    plt.plot(curve, color='blue', linewidth=2)
    plt.title('WOA Convergence Curve', fontsize=16)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Best Fitness (Cost)', fontsize=14)
    plt.grid(True)
    plt.yscale('log') # Log scale helps see convergence better for 0 targets
    plt.savefig('WOA_convergence.png') # Saves the image for your PPT
    plt.show()