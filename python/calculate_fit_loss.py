import matplotlib.pyplot as plt
import re
import numpy as np

# The provided text
text = """
1251/1251 [==============================] - ETA: 0s - loss: 0.8892
Epoch 1: saving model to training_1/checkpoints_new/cp-0001.ckpt
1251/1251 [==============================] - 4413s 4s/step - loss: 0.8892 - val_loss: 0.8957
Epoch 2/100
1251/1251 [==============================] - ETA: 0s - loss: 0.8698
Epoch 2: saving model to training_1/checkpoints_new/cp-0002.ckpt
1251/1251 [==============================] - 2658s 2s/step - loss: 0.8698 - val_loss: 0.8888
Epoch 3/100
1251/1251 [==============================] - ETA: 0s - loss: 0.8627
Epoch 3: saving model to training_1/checkpoints_new/cp-0003.ckpt
1251/1251 [==============================] - 2644s 2s/step - loss: 0.8627 - val_loss: 0.8867
Epoch 4/100
1251/1251 [==============================] - ETA: 0s - loss: 0.8583
Epoch 4: saving model to training_1/checkpoints_new/cp-0004.ckpt
1251/1251 [==============================] - 2546s 2s/step - loss: 0.8583 - val_loss: 0.8836
Epoch 5/100
1251/1251 [==============================] - ETA: 0s - loss: 0.8570
Epoch 5: saving model to training_1/checkpoints_new/cp-0005.ckpt
1251/1251 [==============================] - 2666s 2s/step - loss: 0.8570 - val_loss: 0.8839
Epoch 6/100
1251/1251 [==============================] - ETA: 0s - loss: 0.8531
Epoch 6: saving model to training_1/checkpoints_new/cp-0006.ckpt
1251/1251 [==============================] - 2793s 2s/step - loss: 0.8531 - val_loss: 0.8812
Epoch 7/100
1251/1251 [==============================] - ETA: 0s - loss: 0.8500
Epoch 7: saving model to training_1/checkpoints_new/cp-0007.ckpt
1251/1251 [==============================] - 2676s 2s/step - loss: 0.8500 - val_loss: 0.8797
Epoch 8/100
1251/1251 [==============================] - ETA: 0s - loss: 0.8492
Epoch 8: saving model to training_1/checkpoints_new/cp-0008.ckpt
1251/1251 [==============================] - 2598s 2s/step - loss: 0.8492 - val_loss: 0.8795
Epoch 9/100
1251/1251 [==============================] - ETA: 0s - loss: 0.8469
Epoch 9: saving model to training_1/checkpoints_new/cp-0009.ckpt
1251/1251 [==============================] - 2613s 2s/step - loss: 0.8469 - val_loss: 0.8808
Epoch 10/100
1251/1251 [==============================] - ETA: 0s - loss: 0.8456
Epoch 10: saving model to training_1/checkpoints_new/cp-0010.ckpt
1251/1251 [==============================] - 2533s 2s/step - loss: 0.8456 - val_loss: 0.8773
Epoch 11/100
1251/1251 [==============================] - ETA: 0s - loss: 0.8445
Epoch 11: saving model to training_1/checkpoints_new/cp-0011.ckpt
1251/1251 [==============================] - 2550s 2s/step - loss: 0.8445 - val_loss: 0.8774
Epoch 12/100
1251/1251 [==============================] - ETA: 0s - loss: 0.8438
Epoch 12: saving model to training_1/checkpoints_new/cp-0012.ckpt
1251/1251 [==============================] - 2526s 2s/step - loss: 0.8438 - val_loss: 0.8781
Epoch 13/100
1251/1251 [==============================] - ETA: 0s - loss: 0.8424
Epoch 13: saving model to training_1/checkpoints_new/cp-0013.ckpt
1251/1251 [==============================] - 2560s 2s/step - loss: 0.8424 - val_loss: 0.8755
Epoch 14/100
1251/1251 [==============================] - ETA: 0s - loss: 0.8417
Epoch 14: saving model to training_1/checkpoints_new/cp-0014.ckpt
1251/1251 [==============================] - 2570s 2s/step - loss: 0.8417 - val_loss: 0.8748
Epoch 15/100
1251/1251 [==============================] - ETA: 0s - loss: 0.8408
Epoch 15: saving model to training_1/checkpoints_new/cp-0015.ckpt
1251/1251 [==============================] - 2524s 2s/step - loss: 0.8408 - val_loss: 0.8750
Epoch 16/100
1251/1251 [==============================] - ETA: 0s - loss: 0.8397
Epoch 16: saving model to training_1/checkpoints_new/cp-0016.ckpt
1251/1251 [==============================] - 2576s 2s/step - loss: 0.8397 - val_loss: 0.8741
Epoch 17/100
1251/1251 [==============================] - ETA: 0s - loss: 0.8393
Epoch 17: saving model to training_1/checkpoints_new/cp-0017.ckpt
1251/1251 [==============================] - 2544s 2s/step - loss: 0.8393 - val_loss: 0.8745
Epoch 18/100
1251/1251 [==============================] - ETA: 0s - loss: 0.8386
Epoch 18: saving model to training_1/checkpoints_new/cp-0018.ckpt
1251/1251 [==============================] - 2550s 2s/step - loss: 0.8386 - val_loss: 0.8722
Epoch 19/100
1251/1251 [==============================] - ETA: 0s - loss: 0.8385
Epoch 19: saving model to training_1/checkpoints_new/cp-0019.ckpt
1251/1251 [==============================] - 2533s 2s/step - loss: 0.8385 - val_loss: 0.8730
Epoch 20/100
1251/1251 [==============================] - ETA: 0s - loss: 0.8375
Epoch 20: saving model to training_1/checkpoints_new/cp-0020.ckpt
1251/1251 [==============================] - 2553s 2s/step - loss: 0.8375 - val_loss: 0.8724
Epoch 21/100
1251/1251 [==============================] - ETA: 0s - loss: 0.8373
Epoch 21: saving model to training_1/checkpoints_new/cp-0021.ckpt
1251/1251 [==============================] - 2536s 2s/step - loss: 0.8373 - val_loss: 0.8722
Epoch 22/100
1251/1251 [==============================] - ETA: 0s - loss: 0.8364
Epoch 22: saving model to training_1/checkpoints_new/cp-0022.ckpt
1251/1251 [==============================] - 2549s 2s/step - loss: 0.8364 - val_loss: 0.8718
Epoch 23/100
1251/1251 [==============================] - ETA: 0s - loss: 0.8356
Epoch 23: saving model to training_1/checkpoints_new/cp-0023.ckpt
1251/1251 [==============================] - 2546s 2s/step - loss: 0.8356 - val_loss: 0.8721
Epoch 24/100
1251/1251 [==============================] - ETA: 0s - loss: 0.8363
Epoch 24: saving model to training_1/checkpoints_new/cp-0024.ckpt
1251/1251 [==============================] - 2635s 2s/step - loss: 0.8363 - val_loss: 0.8724
Epoch 25/100
1251/1251 [==============================] - ETA: 0s - loss: 0.8351
Epoch 25: saving model to training_1/checkpoints_new/cp-0025.ckpt
1251/1251 [==============================] - 3425s 3s/step - loss: 0.8351 - val_loss: 0.8721
Epoch 26/100
1251/1251 [==============================] - ETA: 0s - loss: 0.8350
Epoch 26: saving model to training_1/checkpoints_new/cp-0026.ckpt
1251/1251 [==============================] - 2784s 2s/step - loss: 0.8350 - val_loss: 0.8709
Epoch 27/100
1251/1251 [==============================] - ETA: 0s - loss: 0.8347
Epoch 27: saving model to training_1/checkpoints_new/cp-0027.ckpt
1251/1251 [==============================] - 2676s 2s/step - loss: 0.8347 - val_loss: 0.8711
Epoch 28/100
1251/1251 [==============================] - ETA: 0s - loss: 0.8347
Epoch 28: saving model to training_1/checkpoints_new/cp-0028.ckpt
1251/1251 [==============================] - 5345s 4s/step - loss: 0.8347 - val_loss: 0.8701
Epoch 29/100
1251/1251 [==============================] - ETA: 0s - loss: 0.8338
Epoch 29: saving model to training_1/checkpoints_new/cp-0029.ckpt
1251/1251 [==============================] - 5516s 4s/step - loss: 0.8338 - val_loss: 0.8718
Epoch 30/100
1251/1251 [==============================] - ETA: 0s - loss: 0.8341
Epoch 30: saving model to training_1/checkpoints_new/cp-0030.ckpt
1251/1251 [==============================] - 2824s 2s/step - loss: 0.8341 - val_loss: 0.8719
Epoch 31/100
1251/1251 [==============================] - ETA: 0s - loss: 0.8330
Epoch 31: saving model to training_1/checkpoints_new/cp-0031.ckpt
1251/1251 [==============================] - 2698s 2s/step - loss: 0.8330 - val_loss: 0.8714
Epoch 32/100
1251/1251 [==============================] - ETA: 0s - loss: 0.8336
Epoch 32: saving model to training_1/checkpoints_new/cp-0032.ckpt
1251/1251 [==============================] - 2548s 2s/step - loss: 0.8336 - val_loss: 0.8713
Epoch 33/100
1251/1251 [==============================] - ETA: 0s - loss: 0.8326
Epoch 33: saving model to training_1/checkpoints_new/cp-0033.ckpt
1251/1251 [==============================] - 2675s 2s/step - loss: 0.8326 - val_loss: 0.8705
2024-01-06 19:49:22,621 - INFO - Finished fitting
"""

# Use regular expression to find all occurrences of val_loss: followed by a floating-point number
val_loss_pattern = re.compile(r'val_loss: (\d+\.\d+)')
loss_pattern = re.compile(r'ETA: 0s - loss: (\d+\.\d+)')

# Find all matches in the text
matches1 = val_loss_pattern.findall(text)
matches2 = loss_pattern.findall(text)

# Convert the matched values to a NumPy array
val_loss_array = np.array([float(match) for match in matches1])
loss_array = np.array([float(match) for match in matches2])

# Generate x values from 1 to the length of the array
x_values = np.arange(1, len(loss_array) + 1)

# Plotting
font = {'size': 14}
plt.rc('font', **font)

plt.figure(figsize=(7, 6))

plt.plot(x_values, val_loss_array, marker='.', linestyle='-', color='b', label='Validacijos nuostolių funkcijos reikšmės')
plt.plot(x_values, loss_array, marker='.', linestyle='-', color='k', label='Mokymo nuostolių funkcijos reikšmės')
plt.axvline(x=28, color='g', linestyle='--', label='28 epocha')
plt.axhline(y=0.8701, color='y', linestyle='--', label='Mažiausia validacijos nuostolių funkcijos reikšmė')

plt.xlabel('Epocha')
plt.ylabel('Nuostolių funckijos reikšmės')

# Set integer ticks on the x-axis, showing every second value
plt.xticks(x_values[::2], map(int, x_values[::2]))

plt.grid(True)
plt.legend()
plt.show()