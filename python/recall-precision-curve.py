import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Assuming there are 1000 classes with 25 images each
num_classes = 1000
num_images_per_class = 25

filename2 = 'matrix/final/matrix.npy'

# Load similarity matrix
cosine_similarities = np.load(filename2)
length = cosine_similarities.shape[0]

# Create ground truth labels for positive and negative samples
true_labels = np.zeros_like(cosine_similarities)
for i in range(0, true_labels.shape[0], num_images_per_class):
    true_labels[i:i + num_images_per_class, i:i + num_images_per_class] = 1

# Calculate true positives, false positives, and total positives for various thresholds
num_thresholds = 51  # Adjust the number of thresholds
thresholds = np.linspace(0, 1, num_thresholds)
chunk_size = num_images_per_class  # Adjust this based on your available memory

# Lists to store metrics for each threshold
precision_list, recall_list, accuracy_list = [], [], []

binary_predictions = np.zeros_like(cosine_similarities)

for t, threshold in enumerate(thresholds):
    print("Threshold: ", threshold)
    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0

    for start in tqdm(range(0, length, chunk_size)):
        end = min(start + chunk_size, length)
        chunk_true_labels = true_labels[start:end, :]
        chunk_cosine_similarities = cosine_similarities[start:end, :]

        # Apply threshold to cosine similarities to get binary predictions for the current chunk
        binary_predictions[start:end, :] = (chunk_cosine_similarities >= threshold).astype(int)

        # Calculate confusion matrix for the current chunk
        tp = np.sum(np.logical_and(binary_predictions[start:end, :] == 1, chunk_true_labels == 1))
        total_tp += tp
        total_tp -= min(tp, min(chunk_size, end-start))  # Subtract the number of true positives that are the same image
        # if tp <= min(chunk_size, end-start):
        #     total_fp += min(tp, min(chunk_size, end-start))
        total_tn += np.sum(np.logical_and(binary_predictions[start:end, :] == 0, chunk_true_labels == 0))
        total_fp += np.sum(np.logical_and(binary_predictions[start:end, :] == 1, chunk_true_labels == 0))
        total_fn += np.sum(np.logical_and(binary_predictions[start:end, :] == 0, chunk_true_labels == 1))
    accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn) if (total_tp + total_fp + total_fn + total_tn) > 0 else 0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    # print(recall)
    # print(precision)

precision_list.pop()
precision_list.append(1)
print(recall_list)
print(precision_list)
print(accuracy_list)

# recall_list=np.array([0.9476133333333333, 0.9373666666666667, 0.92646, 0.9138866666666666, 0.9001033333333334, 0.8851933333333334, 0.8690966666666666, 0.8511233333333333, 0.83244, 0.8122533333333334, 0.7908133333333334, 0.7683033333333333, 0.7450533333333333, 0.72039, 0.69523, 0.6698333333333333, 0.6429133333333333, 0.6159566666666667, 0.5882966666666667, 0.55982, 0.5314766666666667, 0.5023433333333334, 0.47328333333333333, 0.4442933333333333, 0.41551333333333335, 0.3867833333333333, 0.3582166666666667, 0.33061666666666667, 0.30316333333333334, 0.27569, 0.24942666666666666, 0.22359666666666667, 0.19822, 0.17398666666666668, 0.15089333333333332, 0.12922333333333333, 0.10881333333333333, 0.09025666666666667, 0.07331, 0.05802, 0.04429333333333333, 0.03271333333333333, 0.022963333333333332, 0.015453333333333333, 0.009643333333333334, 0.005463333333333333, 0.0027266666666666667, 0.0011266666666666667, 0.00032, 6.333333333333333e-05, 6.666666666666667e-06])
# precision_list=np.array([0.0011979708627902173, 0.0012407787100887297, 0.001289946918446216, 0.0013446641665367204, 0.001406178318362456, 0.0014751290950430346, 0.0015522809767823111, 0.0016370498678663274, 0.0017322716375229713, 0.0018373299333525675, 0.0019537518563685543, 0.002082877413124648, 0.0022269247654097715, 0.0023853062301891536, 0.002562751495847633, 0.0027625572489983377, 0.002981818719261599, 0.0032292957998917658, 0.003505928746029313, 0.0038137033633373002, 0.0041642118632987885, 0.004555392974623581, 0.005001154085386539, 0.005509885575044495, 0.006095116757957649, 0.006765634761841319, 0.007541466486632415, 0.00845809921520505, 0.009527745477021067, 0.010770085812218416, 0.012266139223335762, 0.014046246091052358, 0.016170747605428974, 0.01875655361802837, 0.02194763171342603, 0.02593944941482845, 0.030952037805405324, 0.03746867462710525, 0.045977468051168306, 0.05715280149202764, 0.07182003913132776, 0.09279325277509881, 0.12233844186748592, 0.16818429167422455, 0.2395264116575592, 0.35553145336225594, 0.5471571906354515, 0.7971698113207547, 1.0, 1.0, 1])
# accuracy_list=np.array([0.24145434617384695, 0.2755629905196208, 0.31130427937117483, 0.3483148285931437, 0.38624178247129887, 0.42464114564582583, 0.4631989343573743, 0.5015378407136285, 0.5392935941437658, 0.5761823688947558, 0.6119669842793711, 0.646389327573103, 0.6792743037721509, 0.7104804928197128, 0.739933101324053, 0.7675469450778031, 0.793279430377215, 0.8171046873874955, 0.8390745933837354, 0.8591891899675987, 0.8775311268450738, 0.894136818272731, 0.909095637425497, 0.9224796959878395, 0.9343905084203368, 0.9448983559342373, 0.9541263218528742, 0.9621480539221569, 0.9690745965838633, 0.9749944813792552, 0.9799969118764751, 0.9841868106724269, 0.9876524692987719, 0.9904686683467339, 0.9927292771710868, 0.9945054282171287, 0.9958738317532702, 0.9969006664266571, 0.997649963598544, 0.9981767590703629, 0.9985329269170767, 0.9987643217728709, 0.9989038505540222, 0.9989814216568663, 0.9990198263930558, 0.9990356990279611, 0.9990404128165127, 0.9990407680307213, 0.9990402688107525, 0.9990400224008961, 0.99903996799872])

# [0.9173766666666666, 0.9072233333333334, 0.8965033333333333, 0.8853466666666666, 0.8733433333333334, 0.8607733333333333, 0.8470766666666667, 0.8324466666666667, 0.8171333333333334, 0.8015366666666667, 0.7850333333333334, 0.7679933333333333, 0.7500866666666667, 0.7319433333333333, 0.7128366666666667, 0.6930433333333333, 0.67281, 0.65163, 0.6305833333333334, 0.6086266666666666, 0.5864033333333334, 0.56354, 0.5403633333333333, 0.51701, 0.49306333333333335, 0.46885, 0.44398, 0.41924, 0.39375666666666664, 0.3683, 0.34291, 0.31705333333333335, 0.29155, 0.26682666666666666, 0.24205, 0.21782666666666667, 0.19320666666666667, 0.16912333333333332, 0.14587, 0.12304, 0.10151666666666667, 0.08149, 0.06258333333333334, 0.046213333333333335, 0.03202666666666667, 0.02018, 0.01098, 0.004866666666666667, 0.00142, 0.00019, 6.666666666666667e-06]
# [0.0014383360325249278, 0.0014895140274615627, 0.001544706062453954, 0.001604450727616206, 0.001668371920646543, 0.0017371225295041985, 0.0018100589498014083, 0.0018878478497471652, 0.001971354547351375, 0.0020620902340555297, 0.0021589052734132847, 0.002263610009463819, 0.002375589798711135, 0.002497780375144697, 0.002628476917274739, 0.0027693078382028833, 0.002922600625291709, 0.003087068131384738, 0.0032692769912925504, 0.0034656028553281632, 0.0036821245319168705, 0.003917531219847231, 0.004176816656560156, 0.004464174826484105, 0.004779485760432085, 0.0051300817005311755, 0.005515461753146129, 0.005950905737391941, 0.006431387580023482, 0.006973845774558901, 0.007593486960834413, 0.00828847403142032, 0.009090696496082941, 0.010042145447269266, 0.011140707838062912, 0.01244813910340177, 0.013945677253883891, 0.015736677621580313, 0.017918885320116682, 0.02052727376872227, 0.023782712455253673, 0.02798423532993971, 0.03320734293501619, 0.04063830036698754, 0.05132478632478633, 0.06734898208922016, 0.09369666628740471, 0.14834383255435887, 0.2903885480572597, 0.76, 1]
# [0.38848493459738387, 0.4160480723228929, 0.44358241529661185, 0.47098426977079083, 0.49816515220608826, 0.5249775911036442, 0.5513848425937038, 0.5773093099723989, 0.6026696555862234, 0.6274110260410416, 0.6514526981079243, 0.6747945501820073, 0.6973504092163687, 0.7191180895235809, 0.7400485395415817, 0.7601127117084684, 0.7793219888795552, 0.7976426097043882, 0.8150769998799952, 0.8316073218928757, 0.8472732669306773, 0.8620197095883836, 0.8758753646145846, 0.8888476211048442, 0.9009467930717229, 0.9122000656026241, 0.922611899675987, 0.9322104404176167, 0.9410184055362214, 0.9490459554382176, 0.9563445417816713, 0.962925035401416, 0.9688101284051363, 0.9740434129365174, 0.9786463202528101, 0.982658749549982, 0.9861103372134885, 0.9890470818832753, 0.9915047801912077, 0.9935217600704028, 0.9951369510780431, 0.9964008000320013, 0.9973508156326253, 0.9980369518780752, 0.9985023912956519, 0.9987910492419697, 0.9989485403416136, 0.9990178103124125, 0.9990379935197408, 0.9990400864034561, 0.99903996799872]

from sklearn.metrics import auc
area = auc(recall_list, precision_list)
print(f'Area under curve: {area:.10f}')

linestyles = ['solid', 'dotted', 'dashed']
# Plotting metrics
plt.figure(figsize=(10, 6))
plt.plot(thresholds, recall_list, linestyle=linestyles[0], label='Atkūrimas')
plt.plot(thresholds, precision_list, linestyle=linestyles[1], label='Preciziškumas')
plt.plot(thresholds, accuracy_list, linestyle=linestyles[2], label='Tikslumas')
plt.xlabel('Slenkstinė vertė')
plt.ylabel('Metrikos vertė')
plt.legend()
plt.show()

plt.plot(recall_list, precision_list, linestyle='-')
plt.grid(True)
plt.fill_between(recall_list, precision_list, alpha=0.2, color='b')
plt.title('Atkūrimo preciziškumo kreivė')
plt.xlabel('Atkūrimas')
plt.ylabel('Preciziškumas')
plt.show()



