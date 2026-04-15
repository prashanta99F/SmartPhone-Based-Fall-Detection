import matplotlib.pyplot as plt

# Your exact data
models = ['Statistical Baseline', 'Random Forest ML']
accuracies = [80.0, 94.77]

# Create a professional-looking figure
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the bars (Blue for statistical, Green for ML to show improvement)
bars = ax.bar(models, accuracies, color=['#4C72B0', '#55A868'], width=0.5)

# Format the axes and title
ax.set_ylabel('Overall Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Detection Accuracy: Deterministic vs. Machine Learning', fontsize=14, fontweight='bold')
ax.set_ylim(0, 110) # Gives a little breathing room at the top

# Automatically add the exact numbers on top of the bars!
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),  # 5 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add a subtle grid behind it for readability
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the final graph
plt.show()