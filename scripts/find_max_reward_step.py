from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import glob

log_dir = "logs/runs/swing_experiment_1"
# The specific event file with the high reward
target_file = "events.out.tfevents.1766625359.UgoLabs.19680.0"
event_file = os.path.join(log_dir, target_file)

import datetime

try:
    ea = EventAccumulator(event_file)
    ea.Reload()
    tags = ea.Tags()['scalars']
    if 'Reward/Episode_Total' in tags:
        events = ea.Scalars('Reward/Episode_Total')
        max_reward = -float('inf')
        max_step = -1
        max_time = 0
        for e in events:
            if e.value > max_reward:
                max_reward = e.value
                max_step = e.step
                max_time = e.wall_time
        
        print(f"Max Reward: {max_reward} at Step: {max_step}")
        print(f"Wall Time: {datetime.datetime.fromtimestamp(max_time)}")
        
        # Check specific steps
        for step in [90, 100, 110]:
            for e in events:
                if e.step == step:
                    print(f"Reward at Step {step}: {e.value}")
    else:
        print("Reward/Episode_Total tag not found.")
except Exception as e:
    print(f"Error: {e}")
