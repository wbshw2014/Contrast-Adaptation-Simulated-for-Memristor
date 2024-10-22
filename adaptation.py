import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PIL import Image


def sams_to_pics(dataset, label=None, background=None, pad=4, num_show=1, print_label=None, return_value=None):
    # np.random.seed(7)
    rand_idx = np.random.choice(dataset.shape[0], num_show)

    if background is not None:
        x = background[rand_idx]
    else:
        x = dataset[rand_idx]

    if label is not None:
        y = label[rand_idx]

    num = x.shape[0]
    col = int(np.ceil(np.sqrt(num)))
    row = int(np.ceil(num / col))
    width = x.shape[1]
    height = x.shape[2]
    nml = 1.0
    x_min = 7
    Q = 1.0

    all_x = np.ones((row * (width + pad) + pad, col * (height + pad) + pad))

    # if Mode in ['scotopic', 'S', 's']:
    #     all_x = np.ones((row*(width+pad)+pad, col*(height+pad)+pad))
    #     # x_min = (Devices().scotopic_avg_values[0] - Devices().scotopic_sd_values[0])
    #     # nml = (Devices().scotopic_avg_values[10] + Devices().scotopic_sd_values[10]) - x_min
    # elif Mode in ['photopic', 'P', 'p']:
    #     if Time == 0:
    #         all_x = np.zeros((row * (width + pad) + pad, col * (height + pad) + pad))
    #     # x_min = (Devices().photopic_avg_values[10] - Devices().photopic_sd_values[10])
    #     # nml = (Devices().photopic_avg_values[0] + Devices().photopic_sd_values[0]) - x_min
    # else:
    #     all_x = np.ones((row * (width + pad) + pad, col * (height + pad) + pad))
    # Q = ((Devices().photopic_avg_values[0] + Devices().photopic_sd_values[0]) - (
    #             Devices().scotopic_avg_values[0] - Devices().scotopic_sd_values[0]))

    n = 0
    for r in range(row):
        Label = []
        for c in range(col):
            if n > num:
                break

            # all_x[r*height+(r+1)*pad:r*height+(r+1)*pad+height,
            # c*width+(c+1)*pad:c*width+(c+1)*pad+width] = ((x[n]*Q)-x_min)/nml/np.max(np.abs(dataset[n]))

            all_x[r * height + (r + 1) * pad:r * height + (r + 1) * pad + height,
            c * width + (c + 1) * pad:c * width + (c + 1) * pad + width] = x[n]  # (x[n] - np.min(dataset[n])) / (
            # np.max(dataset[n]) - np.min(dataset[n]))

            if label is not None:
                Label.append(y[n])

            n += 1
        if print_label is True:
            print(Label)

    Figure = plt.figure()
    # plt.clf()
    rgb = [[1, 1, 1], [1, 1, 0.986666679382324], [1, 1, 0.973333358764648], [1, 1, 0.959999978542328],
           [1, 1, 0.946666657924652], [1, 1, 0.933333337306976], [1, 1, 0.920000016689301], [1, 1, 0.906666696071625],
           [1, 1, 0.893333315849304], [1, 1, 0.879999995231628], [1, 1, 0.866666674613953], [1, 1, 0.758333325386047],
           [1, 1, 0.649999976158142], [1, 1, 0.541666686534882], [1, 1, 0.433333337306976], [1, 1, 0.324999988079071],
           [1, 1, 0.216666668653488], [1, 1, 0.108333334326744], [1, 1, 0], [1, 0.977777779102325, 0],
           [1, 0.955555558204651, 0], [1, 0.933333337306976, 0], [1, 0.911111116409302, 0], [1, 0.888888895511627, 0],
           [1, 0.866666674613953, 0], [1, 0.844444453716278, 0], [1, 0.822222232818604, 0], [1, 0.800000011920929, 0],
           [1, 0.777777791023254, 0], [1, 0.75555557012558, 0], [1, 0.733333349227905, 0], [1, 0.711111128330231, 0],
           [1, 0.688888907432556, 0], [1, 0.666666686534882, 0], [1, 0.644444465637207, 0], [1, 0.622222244739532, 0],
           [1, 0.600000023841858, 0], [1, 0.577777802944183, 0], [1, 0.555555582046509, 0], [1, 0.533333361148834, 0],
           [1, 0.51111114025116, 0], [1, 0.488888889551163, 0], [1, 0.466666668653488, 0], [1, 0.444444447755814, 0],
           [1, 0.422222226858139, 0], [1, 0.400000005960464, 0], [1, 0.37777778506279, 0], [1, 0.355555564165115, 0],
           [1, 0.333333343267441, 0], [1, 0.311111122369766, 0], [1, 0.288888901472092, 0], [1, 0.266666680574417, 0],
           [1, 0.244444444775581, 0], [1, 0.222222223877907, 0], [1, 0.200000002980232, 0], [1, 0.177777782082558, 0],
           [1, 0.155555561184883, 0], [1, 0.133333340287209, 0], [1, 0.111111111938953, 0], [1, 0.0888888910412788, 0],
           [1, 0.0666666701436043, 0], [1, 0.0444444455206394, 0], [1, 0.0222222227603197, 0], [1, 0, 0]]
    cmap1 = colors.ListedColormap(rgb, name='my_color')

    # plt.imshow(all_x, cmap='gray')
    plt.imshow(all_x, vmin=0.0, vmax=1, cmap='gray')
    plt.axis('off')
    # plt.pause(0.01)
    plt.close()
    if return_value is None:
        return Figure
    else:
        return all_x


def save_image(data, path, file_name):
    if isinstance(data, Image.Image):
        if not os.path.exists(path):
            os.makedirs(path)

        save_path = os.path.join(path, file_name + '.jpg')
        data.save(save_path)
        print("Pics saved to：", save_path)
    elif isinstance(data, plt.Figure):
        if not os.path.exists(path):
            os.makedirs(path)

        save_path = os.path.join(path, file_name + '.png')
        data.savefig(save_path)
        print("Pics saved to：", save_path)
    else:
        print('Error：Pics unsaved！')


class Devices:
    def __init__(self):
        self.array_size = (28, 28)
        # np.random.seed(7)
        initial_state = np.random.random(self.array_size)
        self.initial_state = (initial_state - np.min(initial_state)) / (np.max(initial_state) - np.min(initial_state))

        '''
        self.scotopic_avg_values = {0: 1, 0.5: 1.65009, 1: 2.12631, 2: 2.65893, 4: 3.24304, 5: 3.42807, 6: 3.5719,
                                    8: 3.77626, 10: 3.91438,
                                    0.18: 1.05899, 0.19: 1.08500, 0.2: 1.11422, 0.3: 1.33796, 0.4: 1.50986,
                                    0.6: 1.76926, 0.8: 1.96678}
        self.photopic_avg_values = {0: 100, 0.5: 89.14337, 1: 79.24408, 2: 65.88302, 4: 47.86126, 5: 40.91912,
                                    6: 34.81524, 8: 24.35489, 10: 15.50496,
                                    0.18: 99.19838, 0.19: 99.31838, 0.2: 98.41533, 0.3: 94.81886, 0.4: 91.78367,
                                    0.6: 86.79991, 0.8: 82.73724}

        self.scotopic_sd_values = {0: 9.03595E-17, 0.5: 0.07117, 1: 0.13005, 2: 0.1882, 4: 0.22809, 5: 0.23827,
                                   6: 0.24787, 8: 0.26461, 10: 0.27853,
                                   0.18: 0.0077, 0.19: 0.00900, 0.2: 0.01233, 0.3: 0.03529, 0.4: 0.05465, 0.6: 0.0859,
                                   0.8: 0.10991}
        self.photopic_sd_values = {0: 9.36837E-15, 0.5: 3.39617, 1: 5.01, 2: 6.12937, 4: 7.14954, 5: 7.5823, 6: 8.0078,
                                   8: 8.82997, 10: 9.61507,
                                   0.18: 0.36404, 0.19: 0.52404, 0.2: 0.69301, 0.3: 1.93885, 0.4: 2.78516, 0.6: 3.86304,
                                   0.8: 4.54494}
        '''

        # 20231120 && 20240328
        self.photopic_avg_values = {0: 232.34208, 0.5: 225.3568, 1: 216.26624, 2: 203.03712, 4: 184.49248,
                                    5: 177.18848, 6: 170.73792, 8: 159.67872, 10: 150.28256,
                                    0.2: 231.8368, 0.3: 229.7688, 0.4: 227.54336, 0.6: 223.31104, 0.8: 219.59184}
        self.photopic_sd_values = {0: 11.3124, 0.5: 10.59276, 1: 10.68361, 2: 11.97841, 4: 14.08166,
                                   5: 14.86301, 6: 15.48735, 8: 16.58045, 10: 17.50504,
                                   0.2: 11.1437, 0.3: 10.88631, 0.4: 10.69837, 0.6: 10.53383, 0.8: 10.56703}

        # self.photopic_background_avg_values = {0: 107.9476, 0.5: 94.68507, 1: 82.952, 2: 67.58533, 4: 47.36853,
        #                                        5: 39.668, 6: 32.92507, 8: 21.4704, 10: 11.90533,
        #                                        0.2: 105.99307, 0.3: 101.59013, 0.4: 97.88053, 0.6: 91.87787,
        #                                        0.8: 87.05253}
        # self.photopic_background_sd_values = {0: 5.85116, 0.5: 6.22131, 1: 5.76906, 2: 5.241, 4: 5.28859,
        #                                       5: 5.66272, 6: 6.12181, 8: 7.15756, 10: 8.20654,
        #                                       0.2: 6.01057, 0.3: 6.24804, 0.4: 6.28572, 0.6: 6.1354, 0.8: 5.94608}
        self.photopic_background_avg_values = {0: 100, 0.5: 87.68517, 1: 76.83396, 2: 62.62714, 4: 43.93682,
                                               5: 36.81599, 6: 30.58028, 8: 19.98352, 10: 11.13333,
                                               0.2: 98.17771, 0.3: 94.08368, 0.4: 90.64389, 0.6: 85.08781,
                                               0.8: 80.62481}
        self.photopic_background_sd_values = {0: 9.21085E-15, 0.5: 2.37039, 1: 3.10373, 2: 3.80164, 4: 4.84758,
                                              5: 5.36593, 6: 5.86492, 8: 6.81544, 10: 7.70519,
                                              0.2: 0.6148, 0.3: 1.56053, 0.4: 2.07439, 0.6: 2.58984, 0.8: 2.88121}

        self.scotopic_avg_values = {0: 8.40107, 0.5: 13.86107, 1: 17.95973, 2: 22.50133, 4: 27.36853,
                                    5: 28.8936, 6: 30.074, 8: 31.70733, 10: 32.7952,
                                    0.2: 9.34427, 0.3: 11.2124, 0.4: 12.66333, 0.6: 14.8788, 0.8: 16.58453}
        self.scotopic_sd_values = {0: 0.72955, 0.5: 1.02748, 1: 1.38354, 2: 1.80926, 4: 2.09241,
                                   5: 2.15563, 6: 2.20433, 8: 2.27792, 10: 2.36007,
                                   0.2: 0.76072, 0.3: 0.85708, 0.4: 0.94646, 0.6: 1.10612, 0.8: 1.25148}

        self.scotopic_background_avg_values = {0: 0.64122, 0.5: 1.56156, 1: 2.48989, 2: 3.72156, 4: 5.10656,
                                               5: 5.54911, 6: 5.91644, 8: 6.495, 10: 6.91222,
                                               0.2: 0.77833, 0.3: 1.07089, 0.4: 1.32656, 0.6: 1.77222, 0.8: 2.15411}
        self.scotopic_background_sd_values = {0: 0.30333, 0.5: 0.4104, 1: 0.43963, 2: 0.44615, 4: 0.48772,
                                              5: 0.49191, 6: 0.48195, 8: 0.45753, 10: 0.44633,
                                              0.2: 0.32581, 0.3: 0.36892, 0.4: 0.39141, 0.6: 0.42186, 0.8: 0.43396}

        '''
        # 20231120 - 2
        self.scotopic_avg_values={0:8.40107	, 0.2:9.34427, 0.3:11.2124, 0.4:12.66333, 0.6:14.8788, 0.8:16.58453}
        self.photopic_avg_values={0:232.34208, 0.2:231.8368, 0.3:229.7688, 0.4:227.54336, 0.6:223.31104, 0.8:219.59184}

        self.scotopic_sd_values={0:0.72955, 0.2:0.76072, 0.3:0.85708, 0.4:0.94646, 0.6:1.10612, 0.8:1.25148}
        self.photopic_sd_values={0:11.3124, 0.2:11.1437, 0.3:10.88631, 0.4:10.69837, 0.6:10.53383, 0.8:10.56703}
        '''

    def scotopic_array(self, Time):
        scotopic_array = self.scotopic_avg_values[Time] + (self.initial_state * 2 - 1) * self.scotopic_sd_values[Time]

        # scotopic_array[scotopic_array < 8] = 8
        # scotopic_array = (scotopic_array-8) / 92

        scotopic_array[scotopic_array < self.scotopic_avg_values[0]] = self.scotopic_avg_values[
            0]  # + self.scotopic_sd_values[0]
        scotopic_array = (scotopic_array) / 100

        return scotopic_array

    def scotopic_background_array(self, Time):
        scotopic_array = self.scotopic_background_avg_values[Time] + (self.initial_state * 2 - 1) * \
                         self.scotopic_background_sd_values[Time]

        # scotopic_array[scotopic_array < 8] += 8
        # scotopic_array = (scotopic_array-8) / 92

        scotopic_array += self.scotopic_avg_values[0] - self.scotopic_background_avg_values[0] - \
                          self.scotopic_background_sd_values[0]
        scotopic_array = (scotopic_array) / 100

        return scotopic_array

    def photopic_array(self, Time):
        photopic_array = self.photopic_avg_values[Time] + (self.initial_state * 2 - 1) * self.photopic_sd_values[Time]

        # photopic_array[photopic_array > 100] = 100
        # photopic_array = (photopic_array-8) / 92

        photopic_array[photopic_array > self.photopic_background_avg_values[0]] = self.photopic_background_avg_values[0]
        photopic_array = (photopic_array) / 100

        return photopic_array

    def photopic_background_array(self, Time):
        photopic_array = self.photopic_background_avg_values[Time] + (self.initial_state * 2 - 1) * \
                         self.photopic_background_sd_values[Time]

        # photopic_array[photopic_array > 100] = 100
        # photopic_array = (photopic_array-8) / 92

        photopic_array[photopic_array > self.photopic_background_avg_values[0]] = self.photopic_background_avg_values[0]
        photopic_array = (photopic_array) / 100

        return photopic_array

    def scotopic_avg(self, Time):
        return -1.10728 * np.exp(-Time / 0.47821) - 2.34232 * np.exp(-Time / 3.94965) + 4.08897

    def photopic_avg(self, Time):
        return 20.04986 * np.exp(-Time / 0.76214) + 103.30367 * np.exp(-Time / 9.10795) - 18.68358


class Adaptation:
    def __init__(self, Time, pattern):
        self.Time = Time
        self.pattern = pattern.copy()
        self.D = Devices()

        self.scotopic_begin = self.D.scotopic_background_array(self.Time) * np.ones(np.shape(self.pattern))
        self.photopic_begin = self.D.photopic_background_array(self.Time) * np.ones(np.shape(self.pattern))

        self.scotopic_array = self.D.scotopic_array(self.Time) * np.ones(np.shape(self.pattern))
        self.photopic_array = self.D.photopic_array(self.Time) * np.ones(np.shape(self.pattern))

    def scotopic_adapt(self):
        scotopic_mix = self.pattern.copy()
        scotopic_mix[self.pattern <= 0] = self.scotopic_begin[self.pattern <= 0]
        scotopic_mix[self.pattern > 0] = self.scotopic_array[self.pattern > 0]

        return scotopic_mix

    def photopic_adapt(self):
        photopic_mix = self.pattern.copy()
        photopic_mix[self.pattern <= 0] = self.photopic_begin[self.pattern <= 0]
        photopic_mix[self.pattern > 0] = self.photopic_array[self.pattern > 0]

        return photopic_mix


def adapt(Time, dataset):
    A = Adaptation(Time, dataset)
    pattern = A.pattern
    scotopic_begin = A.scotopic_begin
    photopic_begin = A.photopic_begin

    scotopic_adapt = A.scotopic_adapt()

    photopic_adapt = A.photopic_adapt()

    return scotopic_adapt, photopic_adapt


# figure_0, Figure_1 = adapt(10, x_test, y_test, num_show=100)


def load_dataset(mode, time, dataset='new'):
    path = os.path.join(os.getcwd(), 'dataset', dataset)

    file_names = os.listdir(path)

    # load your dataset here
    # (x_train, y_train), (x_test, y_test) = load_data('Emnist')

    name = []
    if mode == 'normal' or mode == 'n' or mode == 'N':
        name = 'normal_adapt_'
        # return name + str(time), (x_train, y_train), (x_test, y_test)
    elif mode == 'scotopic' or mode == 's' or mode == 'S':
        name = 'scotopic_adapt_'
    elif mode == 'photopic' or mode == 'p' or mode == 'P':
        name = 'photopic_adapt_'
    else:
        print('File name not exist!')

    files = [
        'train_' + 'labels_' + name + str(time) + '.npy', 'train_' + 'images_' + name + str(time) + '.npy',
        'test_' + 'labels_' + name + str(time) + '.npy', 'test_' + 'images_' + name + str(time) + '.npy'
    ]

    paths = []
    for fname in files:
        paths.append(os.path.join(path, fname))

    X_train, Y_train, X_test, Y_test = [], [], [], []

    for i in range(len(files)):
        file = files[i]
        if file in file_names:
            if i == 0:
                Y_train = np.load(paths[i])
            elif i == 1:
                X_train = np.load(paths[i])
            elif i == 2:
                Y_test = np.load(paths[i])
            elif i == 3:
                X_test = np.load(paths[i])
        else:
            if i == 0:
                np.save(paths[i], y_train)
                Y_train = y_train
            elif i == 1:
                scotopic_train, photopic_train = adapt(time, x_train)
                if name == 'scotopic_adapt_':
                    np.save(paths[i], scotopic_train)
                    X_train = scotopic_train
                elif name == 'photopic_adapt_':
                    np.save(paths[i], photopic_train)
                    X_train = photopic_train
            elif i == 2:
                np.save(paths[i], y_test)
            elif i == 3:
                scotopic_test, photopic_test = adapt(time, x_test)
                if name == 'scotopic_adapt_':
                    np.save(paths[i], scotopic_test)
                    X_test = scotopic_test
                elif name == 'photopic_adapt_':
                    np.save(paths[i], photopic_test)
                    X_test = photopic_test

    return name + str(time), (X_train, Y_train), (X_test, Y_test)


Acc = []

np.random.seed(4)

# scotopic_begin = None
# photopic_begin = None

file_name, (x_train, y_train), (x_test, y_test) = load_dataset('s', 8, 'Emnist')

X_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
# x_test /= np.max(x_test)
# X_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))
# X_test = (x_test - 9) / 224

X_test = np.copy(x_test)

A = Adaptation(8, x_test)
scotopic_begin = (A.scotopic_begin)  # - np.min(x_test)) / (np.max(x_test) - np.min(x_test))
photopic_begin = (A.photopic_begin)  # - np.min(x_test)) / (np.max(x_test) - np.min(x_test))

X_test = (X_test - np.min(X_test)) / (np.max(X_test) - np.min(X_test))

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(400, activation='relu'),
    Dense(26, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train
history = model.fit(X_train, y_train, epochs=10, batch_size=256,
                    validation_data=(X_test, y_test))  # epochs=10, batch_size=32

y_pred_proba = model.predict(X_test)

y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true, y_pred)

# access Precision, Recall, F1-score, AUC-ROC, Log-loss
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
auc_roc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
log_loss_value = log_loss(y_test, y_pred_proba)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'AUC-ROC: {auc_roc}')
print(f'Log-loss: {log_loss_value}')

# visualize acc and loss curves
plt.figure(num='acc-loss', figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['acc'], label='Training Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
# plt.show()
# plt.close()

Acc.append(history.history['val_acc'][-1])

# plt.figure()
# plt.plot(Time_list, Acc[:len(Acc) // 2], '-.')
# plt.plot(Time_list, Acc[len(Acc) // 2:], '-.')
plt.show()
