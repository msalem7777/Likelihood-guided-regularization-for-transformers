import matplotlib.pyplot as plt
import numpy as np

# Define data
methods = ["Dropconnect", "Dropout", "Ising"]
metrics = ["Accuracy", "Recall", "Precision", "F1", "FPR"]
training_sizes = [300, 6000, 18000]
data = {
    300: {
        "Dropconnect": [0.549, 0.541, 0.545, 0.537, 0.050],
        "Dropout": [0.489, 0.480, 0.485, 0.458, 0.057],
        "Ising": [0.466, 0.454, 0.438, 0.420, 0.059],
    },
    6000: {
        "Dropconnect": [0.923, 0.922, 0.922, 0.922, 0.009],
        "Dropout": [0.907, 0.905, 0.906, 0.905, 0.010],
        "Ising": [0.911, 0.909, 0.910, 0.909, 0.010],
    },
    18000: {
        "Dropconnect": [0.967, 0.967, 0.967, 0.967, 0.004],
        "Dropout": [0.958, 0.957, 0.957, 0.957, 0.005],
        "Ising": [0.955, 0.955, 0.955, 0.955, 0.005],
    },
}

# Function to plot radar chart for all training sizes in one figure
def plot_combined_radar(data, metrics, methods, training_sizes):
    # Number of variables
    num_vars = len(metrics)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the radar chart

    # Create a 3x1 plot
    fig, axs = plt.subplots(3, 1, figsize=(8, 24), subplot_kw=dict(polar=True))  # Increased vertical spacing with larger height (24)

    for idx, training_size in enumerate(training_sizes):
        ax = axs[idx]
        ax.set_title(f"Training Size: {training_size}", fontsize=16, pad=40)  # Increased title padding to 40 for more space
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=15)  # Further increased font size for metric labels
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=12)  # Increase font size for grid labels

        # Plot for each method
        for method in methods:
            values = data[training_size][method]
            values += values[:1]  # Close the circle
            ax.plot(angles, values, label=method, linewidth=2)
            ax.fill(angles, values, alpha=0.1)

    # Add a shared legend for all plots below the subplots
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', fontsize=12, ncol=len(methods), bbox_to_anchor=(0.5, -0.1))  # Single legend
    plt.tight_layout(h_pad=5)  # Add extra vertical padding between subplots
    plt.show()

# Generate radar chart
plot_combined_radar(data, metrics, methods, training_sizes)

#%%
training_sizes = [300, 6000, 18000]
data = {
    300: {
        "Dropconnect": [0.276, 0.262, 0.155, 0.165, 0.081],
        "Dropout": [0.453, 0.441, 0.446, 0.401, 0.061],
        "Ising": [0.289, 0.275, 0.178, 0.185, 0.079],
    },
    6000: {
        "Dropconnect": [0.796, 0.790, 0.794, 0.786, 0.023],
        "Dropout": [0.895, 0.893, 0.895, 0.893, 0.012],
        "Ising": [0.849, 0.846, 0.848, 0.846, 0.017],
    },
    18000: {
        "Dropconnect": [0.906, 0.905, 0.906, 0.905, 0.010],
        "Dropout": [0.954, 0.954, 0.954, 0.953, 0.005],
        "Ising": [0.922, 0.921, 0.921, 0.921, 0.009],
    },
}

# Generate radar chart
plot_combined_radar(data, metrics, methods, training_sizes)
# %%
training_sizes = [300, 6000, 18000]
data = {
    300: {
        "Dropconnect": [0.605, 0.605, 0.608, 0.601, 0.044],
        "Dropout": [0.560, 0.561, 0.582, 0.550, 0.049],
        "Ising": [0.583, 0.583, 0.580, 0.570, 0.046],
    },
    6000: {
        "Dropconnect": [0.841, 0.841, 0.841, 0.841, 0.018],
        "Dropout": [0.823, 0.823, 0.824, 0.822, 0.020],
        "Ising": [0.828, 0.828, 0.827, 0.826, 0.019],
    },
    18000: {
        "Dropconnect": [0.902, 0.902, 0.902, 0.902, 0.011],
        "Dropout": [0.881, 0.881, 0.882, 0.881, 0.013],
        "Ising": [0.879, 0.879, 0.879, 0.879, 0.013],
    },
}

# Generate the radar charts
plot_combined_radar(data, metrics, methods, training_sizes)
# %%
training_sizes = [300, 6000, 18000]
data = {
    300: {
        "Dropconnect": [0.377, 0.379, 0.319, 0.299, 0.069],
        "Dropout": [0.547, 0.546, 0.560, 0.528, 0.050],
        "Ising": [0.376, 0.378, 0.306, 0.296, 0.069],
    },
    6000: {
        "Dropconnect": [0.755, 0.754, 0.751, 0.743, 0.027],
        "Dropout": [0.812, 0.811, 0.812, 0.808, 0.021],
        "Ising": [0.777, 0.776, 0.775, 0.769, 0.025],
    },
    18000: {
        "Dropconnect": [0.818, 0.818, 0.818, 0.814, 0.020],
        "Dropout": [0.875, 0.875, 0.876, 0.874, 0.014],
        "Ising": [0.835, 0.835, 0.834, 0.832, 0.018],
    },
}

# Generate the radar charts
plot_combined_radar(data, metrics, methods, training_sizes)
# %%
training_sizes = [250, 5000, 15000]
data = {
    250: {
        "Dropconnect": [0.257, 0.257, 0.264, 0.252, 0.083],
        "Dropout": [0.224, 0.225, 0.223, 0.189, 0.086],
        "Ising": [0.254, 0.254, 0.258, 0.239, 0.083],
    },
    5000: {
        "Dropconnect": [0.497, 0.496, 0.501, 0.497, 0.056],
        "Dropout": [0.435, 0.435, 0.443, 0.428, 0.063],
        "Ising": [0.459, 0.459, 0.459, 0.455, 0.060],
    },
    15000: {
        "Dropconnect": [0.661, 0.661, 0.663, 0.661, 0.038],
        "Dropout": [0.591, 0.590, 0.593, 0.588, 0.045],
        "Ising": [0.577, 0.577, 0.575, 0.573, 0.047],
    },
}

# Generate the radar charts
plot_combined_radar(data, metrics, methods, training_sizes)
# %%
training_sizes = [250, 5000, 15000]
data = {
    250: {
        "Dropconnect": [0.257, 0.257, 0.264, 0.252, 0.083],
        "Dropout": [0.220, 0.220, 0.220, 0.179, 0.087],
        "Ising": [0.239, 0.239, 0.232, 0.208, 0.085],
    },
    5000: {
        "Dropconnect": [0.395, 0.395, 0.401, 0.384, 0.067],
        "Dropout": [0.420, 0.420, 0.427, 0.410, 0.064],
        "Ising": [0.428, 0.428, 0.435, 0.421, 0.064],
    },
    15000: {
        "Dropconnect": [0.479, 0.479, 0.482, 0.472, 0.058],
        "Dropout": [0.552, 0.552, 0.555, 0.546, 0.050],
        "Ising": [0.558, 0.558, 0.557, 0.555, 0.049],
    },
}

# Generate the radar charts
plot_combined_radar(data, metrics, methods, training_sizes)
# %%
training_sizes = [250, 5000, 15000]
data = {
    250: {
        "Dropconnect": [0.046, 0.046, 0.041, 0.036, 0.010],
        "Dropout": [0.016, 0.016, 0.003, 0.003, 0.010],
        "Ising": [0.037, 0.037, 0.017, 0.016, 0.010],
    },
    5000: {
        "Dropconnect": [0.220, 0.221, 0.223, 0.216, 0.008],
        "Dropout": [0.127, 0.128, 0.126, 0.108, 0.009],
        "Ising": [0.145, 0.146, 0.131, 0.119, 0.009],
    },
    15000: {
        "Dropconnect": [0.574, 0.575, 0.575, 0.565, 0.004],
        "Dropout": [0.769, 0.770, 0.771, 0.768, 0.002],
        "Ising": [0.642, 0.642, 0.639, 0.633, 0.004],
    },
}

# Generate the radar charts
plot_combined_radar(data, metrics, methods, training_sizes)
# %%
training_sizes = [250, 5000, 15000]
data = {
    250: {
        "Dropconnect": [0.029, 0.029, 0.041, 0.036, 0.010],
        "Dropout": [0.018, 0.018, 0.004, 0.003, 0.010],
        "Ising": [0.034, 0.034, 0.009, 0.011, 0.010],
    },
    5000: {
        "Dropconnect": [0.116, 0.117, 0.091, 0.081, 0.009],
        "Dropout": [0.069, 0.069, 0.051, 0.046, 0.009],
        "Ising": [0.134, 0.135, 0.119, 0.104, 0.009],
    },
    15000: {
        "Dropconnect": [0.196, 0.196, 0.176, 0.154, 0.008],
        "Dropout": [0.300, 0.302, 0.292, 0.278, 0.007],
        "Ising": [0.478, 0.477, 0.483, 0.452, 0.005],
    },
}

# Generate the radar charts
plot_combined_radar(data, metrics, methods, training_sizes)
# %%
