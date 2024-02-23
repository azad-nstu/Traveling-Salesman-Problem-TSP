import matplotlib.pyplot as plt
import numpy as np

# Data
num_nodes = [5, 10, 13, 15, 25, 50, 100, 250, 500, 750, 1000]
brute_force = [207341, 647105, 628904, None, None, None, None, None, None, None, None]
brute_force = [value if value is not None else 0 for value in brute_force]
NN = [269101, 1628604, 1801463, 1820321, 950756, 1371802, 3179955, 2609719, 1999082, 2824495, 1780054]
genetic_alg = [207341, 647105, 661905, 628203, 1134269, 6812631, 24086044, 58526357, 142035502, 262161280, 389579130]
proposed_2step_NN = [269101, 1628604, 1823748, 1820321, 862827, 1139795, 3088761, 1944536, 1703980, 2387702, 4126961]

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
plt.ylabel('Distance/Cost')
plt.title('Comparison of different algorithms for Distance Vs Number of Cities')
plt.xticks(index + bar_width * 1.5, num_nodes)
plt.legend()
plt.yscale('log')  # Set y-axis to logarithmic scale

# Add text annotations with the same color as the bars
for bars in [brute_force_bars, NN_bars, genetic_alg_bars, proposed_2step_NN_bars]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, '%d' % int(height), ha='center', va='bottom', color=bar.get_facecolor())

# Save the figure
plt.savefig('distance_comparison.png')

# Show plot
plt.tight_layout()
plt.show()