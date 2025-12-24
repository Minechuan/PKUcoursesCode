import matplotlib.pyplot as plt

# Data as provided
data = """
alpha=0.70 lambda=0.00 natural_accuracy=0.5848 robustness=0.3525 score=0.6449
alpha=0.70 lambda=0.10 natural_accuracy=0.5863 robustness=0.3600 score=0.6532
alpha=0.70 lambda=0.20 natural_accuracy=0.5855 robustness=0.3589 score=0.6517
alpha=0.70 lambda=0.40 natural_accuracy=0.5841 robustness=0.3594 score=0.6514
alpha=0.70 lambda=0.60 natural_accuracy=0.5836 robustness=0.3624 score=0.6542
alpha=0.70 lambda=0.80 natural_accuracy=0.5807 robustness=0.3608 score=0.6512
alpha=0.70 lambda=1.00 natural_accuracy=0.5808 robustness=0.3594 score=0.6498
alpha=0.70 lambda=1.20 natural_accuracy=0.5775 robustness=0.3623 score=0.6511
alpha=0.70 lambda=1.40 natural_accuracy=0.5774 robustness=0.3604 score=0.6491
alpha=0.70 lambda=1.60 natural_accuracy=0.5776 robustness=0.3552 score=0.6440
alpha=0.70 lambda=1.80 natural_accuracy=0.5746 robustness=0.3592 score=0.6465
"""


# Parse data
alphas = []
natural_acc = []
robustness = []
scores = []

for line in data.strip().splitlines():
    parts = line.split()
    a = float(parts[1].split('=')[1])
    na = float(parts[2].split('=')[1])
    ro = float(parts[3].split('=')[1])
    sc = float(parts[4].split('=')[1])
    alphas.append(a)
    natural_acc.append(na)
    robustness.append(ro)
    scores.append(sc)

# Plot
plt.figure()
plt.plot(alphas, natural_acc, label='Natural Accuracy')
plt.plot(alphas, robustness, label='Robustness')
plt.plot(alphas, scores, label='Score')
plt.xlabel('Lambda')
plt.ylabel('Value')
plt.title('Natural Accuracy, Robustness, and Score vs Lambda')
plt.legend()
plt.grid(True)
plt.show()