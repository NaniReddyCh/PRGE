import json, matplotlib.pyplot as plt

def plot_results(results_file="results.json"):
    with open(results_file) as f: results = json.load(f)
    steps, acc = results["steps"], results["accuracy"]
    plt.figure(figsize=(6,4))
    plt.plot(steps, acc, marker="o")
    plt.xlabel("Training Steps")
    plt.ylabel("Validation Accuracy")
    plt.title("P-RGE Fine-tuning Progress")
    plt.grid(True)
    plt.tight_layout()
    return plt
