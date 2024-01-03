import matplotlib.pyplot as plt
import seaborn as sns

class ConfusionMatrixPlotter:
    def __init__(self, cm):
        self.cm = cm

    def plot(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
