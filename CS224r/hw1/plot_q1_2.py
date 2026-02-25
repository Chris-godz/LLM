import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_latest_dir(pattern):
    dirs = glob.glob(pattern)
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)

def get_eval_return(log_dir):
    if not log_dir:
        return None
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    try:
        # Get the last value of Eval_AverageReturn
        return event_acc.Scalars('Eval_AverageReturn')[-1].value
    except Exception as e:
        print(f"Error reading {log_dir}: {e}")
        return None

def main():
    steps = [1000, 2000, 5000, 10000]
    returns = []
    
    # 1000 steps (default run)
    dir_1000 = get_latest_dir('data/q1_bc_ant_Ant-v4_*')
    ret_1000 = get_eval_return(dir_1000)
    returns.append(ret_1000)
    
    # 2000 steps
    dir_2000 = get_latest_dir('data/q1_q1_bc_ant_steps_2000_Ant-v4_*')
    ret_2000 = get_eval_return(dir_2000)
    returns.append(ret_2000)
    
    # 5000 steps
    dir_5000 = get_latest_dir('data/q1_q1_bc_ant_steps_5000_Ant-v4_*')
    ret_5000 = get_eval_return(dir_5000)
    returns.append(ret_5000)
    
    # 10000 steps
    dir_10000 = get_latest_dir('data/q1_q1_bc_ant_steps_10000_Ant-v4_*')
    ret_10000 = get_eval_return(dir_10000)
    returns.append(ret_10000)
    
    print("Steps:", steps)
    print("Returns:", returns)
    
    plt.figure(figsize=(8, 6))
    plt.plot(steps, returns, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
    plt.title('Behavior Cloning Performance vs. Training Steps (Ant-v4)', fontsize=14)
    plt.xlabel('Number of Agent Train Steps per Iteration', fontsize=12)
    plt.ylabel('Eval Average Return', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(steps)
    
    # Add expert performance line for reference
    # From previous runs, expert return for Ant is around 4713
    plt.axhline(y=4713.65, color='r', linestyle='--', label='Expert Return (~4713.65)')
    plt.legend()
    
    plt.savefig('q1_2_hyperparameter_tuning.png', dpi=300, bbox_inches='tight')
    print("Saved plot to q1_2_hyperparameter_tuning.png")

if __name__ == '__main__':
    main()
