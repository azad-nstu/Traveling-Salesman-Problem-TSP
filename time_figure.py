import matplotlib.pyplot as plt
import numpy as np
import re

# Data
num_nodes = [5, 10, 13, 15, 25, 50, 100, 250, 500, 750, 1000]
brute_force = [1, 259, 462921, None, None, None, None, None, None, None, None]
brute_force = [int(value) if value is not None else 0 for value in brute_force]  # Replace None with 0
NN = [1, 2, 2, 2, 3, 3, 5, 13, 46, 96, 169]
genetic_alg = [39, 50, 61, 62, 62, 69, 109, 469, 1113, 1634, 2320]
proposed_2step_NN = [1, 2, 2, 3, 3, 3, 6, 20, 73, 157, 283]

# Bar width
bar_width = 0.2

# Set the index for the bars
index = np.arange(len(num_nodes))

# Plot bars
brute_force_bars = plt.bar(index, brute_force, bar_width, label='Brute Force')
NN_bars = plt.bar(index + bar_width, NN, bar_width, label='NN')
genetic_alg_bars = plt.bar(index + 2*bar_width, genetic_alg, bar_width, label='Genetic Alg')
proposed_2step_NN_bars = plt.bar(index + 3*bar_width, proposed_2step_NN, bar_width, label='Proposed 2-Step NN')

# Add labels, title, legend, and logarithmic scale
plt.xlabel('Number of Cities')
plt.ylabel('Execution Time (ms)')
plt.title('Comparison of different algorithms for Execution Time Vs Number of Cities')
plt.xticks(index + bar_width * 1.5, num_nodes)
plt.legend()
plt.yscale('log')  # Set y-axis to logarithmic scale

# Add text annotations with the same color as the bars
for bars in [brute_force_bars, NN_bars, genetic_alg_bars, proposed_2step_NN_bars]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, '%d' % int(height), ha='center', va='bottom', color=bar.get_facecolor())

# Save the figure
plt.savefig('time_comparison.png')

# Show plot
plt.tight_layout()
plt.show()