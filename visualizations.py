import matplotlib.pyplot as plt

# display prediction results
def prediction_plot(predicted, ground_truth, error):
    plt.figure(figsize=(4,5.15))
    plt.plot(ground_truth, predicted, "-o", label="Predicted")
    plt.plot(ground_truth, ground_truth, linestyle="dashed", label="Ground Truth")
    plt.fill_between(ground_truth, predicted+error, predicted-error, alpha=0.3)
    plt.xlabel("Ground Truth Temperature (°C)", fontweight="bold", fontsize=12)
    plt.ylabel("Predicted Temperature (°C)", fontweight="bold", fontsize=12)
    plt.title("Protein Shake Network Prediction", fontweight="bold", fontsize=14)

    for i in range(4):
        label = ground_truth[i]
        plt.annotate(f"({predicted[i]:.2f}, {label})", (predicted[i], label))

    label = ground_truth[4]
    plt.annotate(f"({predicted[4]:.2f}, {label})", (predicted[4]-5, label))
    plt.legend()
    plt.show()
