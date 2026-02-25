import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_data_from_tb(log_dir, tag):
    ea = EventAccumulator(log_dir)
    ea.Reload()
    if tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        return [e.value for e in events]
    return []

def plot_dagger_learning_curve(env_name, bc_dir, dagger_dir, expert_return):
    bc_returns = get_data_from_tb(bc_dir, 'Eval_AverageReturn')
    bc_stds = get_data_from_tb(bc_dir, 'Eval_StdReturn')
    
    dagger_returns = get_data_from_tb(dagger_dir, 'Eval_AverageReturn')
    dagger_stds = get_data_from_tb(dagger_dir, 'Eval_StdReturn')
    
    iterations = np.arange(len(dagger_returns))
    
    plt.figure(figsize=(10, 6))
    
    # Plot DAgger
    plt.errorbar(iterations, dagger_returns, yerr=dagger_stds, label='DAgger', fmt='-o', capsize=5)
    
    # Plot BC (constant line since it's just 1 iteration)
    if bc_returns:
        plt.axhline(y=bc_returns[0], color='r', linestyle='--', label=f'BC (Return: {bc_returns[0]:.2f})')
        if bc_stds:
            plt.fill_between(iterations, bc_returns[0] - bc_stds[0], bc_returns[0] + bc_stds[0], color='r', alpha=0.2)
            
    # Plot Expert
    plt.axhline(y=expert_return, color='g', linestyle='-.', label=f'Expert (Return: {expert_return:.2f})')
    
    plt.xlabel('DAgger Iterations')
    plt.ylabel('Mean Return')
    plt.title(f'DAgger Learning Curve - {env_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{env_name}_dagger_curve.png')
    plt.close()
    print(f"Saved {env_name}_dagger_curve.png")

def generate_bc_table(log_dirs):
    print("\nBehavior Cloning Results:")
    print("-" * 60)
    print(f"{'Environment':<20} | {'Mean Return':<15} | {'Std Return':<15}")
    print("-" * 60)
    
    for env, log_dir in log_dirs.items():
        returns = get_data_from_tb(log_dir, 'Eval_AverageReturn')
        stds = get_data_from_tb(log_dir, 'Eval_StdReturn')
        
        if returns and stds:
            print(f"{env:<20} | {returns[0]:<15.2f} | {stds[0]:<15.2f}")
        else:
            print(f"{env:<20} | {'N/A':<15} | {'N/A':<15}")
    print("-" * 60)

if __name__ == '__main__':
    # Define the directories (update these based on actual generated folders)
    data_dir = 'data'
    
    # Find the latest directories for each experiment
    dirs = os.listdir(data_dir)
    
    bc_dirs = {}
    dagger_dirs = {}
    
    for d in dirs:
        if d.startswith('q1_bc_'):
            env = d.split('_')[3]
            # Keep the latest directory for each environment
            if env not in bc_dirs or os.path.getmtime(os.path.join(data_dir, d)) > os.path.getmtime(bc_dirs[env]):
                bc_dirs[env] = os.path.join(data_dir, d)
        elif d.startswith('q2_dagger_'):
            env = d.split('_')[3]
            if env not in dagger_dirs or os.path.getmtime(os.path.join(data_dir, d)) > os.path.getmtime(dagger_dirs[env]):
                dagger_dirs[env] = os.path.join(data_dir, d)
            
    # Generate BC Table
    generate_bc_table(bc_dirs)
    
    # Expert returns (approximate values based on typical expert performance)
    expert_returns = {
        'Ant-v4': 4713.65,
        'HalfCheetah-v4': 4205.77,
        'Hopper-v4': 3772.67,
        'Walker2d-v4': 5566.84
    }
    
    # Generate DAgger plots
    for env in dagger_dirs:
        if env in bc_dirs and env in expert_returns:
            plot_dagger_learning_curve(env, bc_dirs[env], dagger_dirs[env], expert_returns[env])
