import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


num_classes = 45
num_images_per_class = 500

filename2 = "matrix/vit-gldv-similarity-quad_clean/matrix.npy"


cosine_similarities = np.load(filename2)
length = cosine_similarities.shape[0]


true_labels = np.zeros_like(cosine_similarities)
for i in range(0, true_labels.shape[0], num_images_per_class):
    true_labels[i : i + num_images_per_class, i : i + num_images_per_class] = 1


num_thresholds = 51
thresholds = np.linspace(0, 1, num_thresholds)
chunk_size = num_images_per_class


precision_list, recall_list, accuracy_list = [], [], []

binary_predictions = np.zeros_like(cosine_similarities)

for t, threshold in enumerate(thresholds):
    print("Threshold: ", threshold)
    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0

    for start in tqdm(range(0, length, chunk_size)):
        end = min(start + chunk_size, length)
        chunk_true_labels = true_labels[start:end, :]
        chunk_cosine_similarities = cosine_similarities[start:end, :]

        binary_predictions[start:end, :] = (
            chunk_cosine_similarities >= threshold
        ).astype(int)

        tp = np.sum(
            np.logical_and(
                binary_predictions[start:end, :] == 1, chunk_true_labels == 1
            )
        )
        total_tp += tp
        total_tp -= min(tp, min(chunk_size, end - start))
        total_tn += np.sum(
            np.logical_and(
                binary_predictions[start:end, :] == 0, chunk_true_labels == 0
            )
        )
        total_fp += np.sum(
            np.logical_and(
                binary_predictions[start:end, :] == 1, chunk_true_labels == 0
            )
        )
        total_fn += np.sum(
            np.logical_and(
                binary_predictions[start:end, :] == 0, chunk_true_labels == 1
            )
        )
    accuracy = (
        (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn)
        if (total_tp + total_fp + total_fn + total_tn) > 0
        else 0
    )
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)

precision_list.pop()
precision_list.append(1)
print(recall_list)
print(precision_list)
print(accuracy_list)

from sklearn.metrics import auc

area = auc(recall_list, precision_list)
print(f"Area under curve: {area:.10f}")

linestyles = ["solid", "dotted", "dashed"]

plt.figure(figsize=(10, 6))
plt.plot(thresholds, recall_list, linestyle=linestyles[0], label="Atkūrimas")
plt.plot(thresholds, precision_list, linestyle=linestyles[1], label="Preciziškumas")
plt.plot(thresholds, accuracy_list, linestyle=linestyles[2], label="Tikslumas")
plt.xlabel("Slenkstinė vertė")
plt.ylabel("Metrikos vertė")
plt.legend()
plt.show()

plt.plot(recall_list, precision_list, linestyle="-")
plt.grid(True)
plt.fill_between(recall_list, precision_list, alpha=0.2, color="b")
plt.title("Atkūrimo preciziškumo kreivė")
plt.xlabel("Atkūrimas")
plt.ylabel("Preciziškumas")
plt.show()
