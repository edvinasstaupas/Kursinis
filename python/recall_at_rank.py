import numpy as np
from tqdm import tqdm

classes = 1000
num_images_per_class = 25

filename1 = 'matrix/final/matrix.npy'
filename2 = 'matrix/resnet/matrix.npy'

# # Load similarity matrix
cosine_similarities1 = np.load(filename1)
# cosine_similarities2 = np.load(filename2)

# Calculate true positives, false positives, and total positives for various thresholds
num_thresholds = 51  # Adjust the number of thresholds
thresholds = np.linspace(0, 1, num_thresholds)

# Specify the values of k for Recall at K
ks = [1, 10, 50]  # Adjust the values of k as needed

def calc_recall(cosine_similarities):
    recall_at_ks_list = np.zeros((len(thresholds), len(ks)))
    recall_at_ks_list_2 = np.zeros((len(thresholds), len(ks)))
    length = cosine_similarities.shape[0]
    print(length)
    for t, threshold in enumerate(thresholds):
        print("Threshold: ", threshold)
        loc_at_k = np.zeros(len(ks))
        for i in tqdm(range(length)):
            class_id = int(i // num_images_per_class)
            row = cosine_similarities[i, :]
            row_with_class = []
            for j in range(row.shape[0]):
                row_with_class.append((row[j], j // num_images_per_class == class_id))
            sorted_row_with_class = sorted(row_with_class, key=lambda x: x[0], reverse=True)
            del sorted_row_with_class[0]
            for l, k in enumerate(ks):
                k_items = sorted_row_with_class[:k]
                k_items = [x for x in k_items if x[0] >= threshold]
                count_true = sum(1 for x in k_items if x[1])
                loc_at_k[l] += count_true / (num_images_per_class - 1)
        for l, k in tqdm(enumerate(ks)):
            recall_at_ks_list[t][l] = loc_at_k[l] / length
            recall_at_ks_list_2[t][l] = loc_at_k[l]
        print(recall_at_ks_list)

    print("Recalls:")
    print(recall_at_ks_list)
    return recall_at_ks_list

recalls_model = calc_recall(cosine_similarities1)
# recalls_resnet = calc_recall(cosine_similarities2)