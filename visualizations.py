import matplotlib.pyplot as plt

# display prediction results
def prediction_plot(predicted, ground_truth, error, title):
    plt.figure(figsize=(4,5.15))
    plt.plot(ground_truth, predicted, "-o", label="Predicted")
    plt.plot(ground_truth, ground_truth, linestyle="dashed", label="Ground Truth")
    plt.fill_between(ground_truth, predicted+error, predicted-error, alpha=0.3)
    plt.xlabel("Ground Truth Temperature (°C)", fontweight="bold", fontsize=12)
    plt.ylabel("Predicted Temperature (°C)", fontweight="bold", fontsize=12)
    plt.title(title, fontweight="bold", fontsize=14)

    num = len(ground_truth)
    for i in range(num-1):
        label = ground_truth[i]
        plt.annotate(f"({label}, {predicted[i]:.2f})", (label+1, predicted[i]))

    label = ground_truth[num-1]
    plt.annotate(f"({label}, {predicted[4]:.2f})", (label-5, predicted[num-1]))
    plt.legend()
    plt.show()
