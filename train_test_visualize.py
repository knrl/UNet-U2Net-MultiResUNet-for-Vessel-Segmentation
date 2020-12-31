'''

    @author: mehmetkaanerol

'''
import random
import pickle
import numpy as np
import tensorflow as tf
from models.unet import get_Unet
from models.resunet import get_resUnet
from models.usquarednet import get_U2net

def get_data():
    X_train = pickle.load(open("X_train.pickle", "rb"))
    Y_train = pickle.load(open("Y_train.pickle", "rb"))
    X_test = pickle.load(open("X_test.pickle", "rb"))
    Y_test = pickle.load(open("Y_test.pickle", "rb"))
    X_test_mask = pickle.load(open("X_test_mask.pickle", "rb"))
    return X_train, X_test, Y_train, Y_test, X_test_mask

def eval_only_FOV(preds, y_tests, masks):
    TP, TN, FP, FN = 0, 0, 0, 0
    pred_list, test_list = [], []
    j = 0
    patch_size = len(preds)
    for a in range(patch_size):
        mask = masks[a]
        y_test = y_tests[a]
        pred = preds[a]
        for i in range(H):
            for k in range(W):
                if (mask[i][k] == 1):
                    if (y_test[i][k] == 1 and pred[i][k] == 1):
                        TP += 1
                    elif (y_test[i][k] == 0 and pred[i][k] == 0):
                        TN += 1
                    elif (y_test[i][k] == 0 and pred[i][k] == 1):
                        FP += 1
                    elif (y_test[i][k] == 1 and pred[i][k] == 0):
                        FN += 1
                    pred_list.append(pred[i][k])
                    test_list.append(y_test[i][k])
    return TP, TN, FP, FN, pred_list, test_list

# Load data
X_train, X_test, Y_train, Y_test, X_test_mask = get_data()

# determine data size, normalize
W, H = 128, 128
inputs = tf.keras.layers.Input((H, W, 1))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

# Train
# Select model, optimize, compile, fit
model = get_Unet(s)

tf.keras.optimizers.Adam(
    learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam'
)

model.compile(optimizer='Adam', loss='binary_crossentropy')#, metrics=['accuracy', 'Recall', 'AUC'])
results = model.fit(X_train, Y_train, batch_size=32, epochs=5, validation_split=0.1)#, callbacks=[lr_finder], verbose=False)

# Test
# Predict
preds_test = model.predict(X_test, verbose=False)

# Normalize test data
preds_test_t = (preds_test > 0.49).astype(np.uint8)
Y_test_t = Y_test / 255.
Y_test_t = Y_test.astype(np.uint8)
X_test_mask = X_test_mask / 255.
X_test_mask = X_test.astype(np.uint8)

# Evaluate only fovea (mask) area
TP, TN, FP, FN, pred_list, test_list = eval_only_FOV(preds_test_t, Y_test_t, X_test_mask)

# Evaluation metrics
from sklearn.metrics import roc_auc_score
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)
AUC_ROC = roc_auc_score(test_list, pred_list)
sen = TP / (TP + FN)
spe = TN / (TN + FP)
print(precision, " ", recall, " ", f1_score, " ", AUC_ROC, " ", spe, " ", sen)


# Visualize test results
# merge the images
fig, axs = plt.subplots(nrows=5, ncols=3, figsize=(15, 30), subplot_kw={'xticks': [], 'yticks': []})

i, k, j, d = 0, 0, 0, 0
for ax in axs.flat:
    if (i % 3 == 0):
        ax.imshow(preds_test[j].reshape(W, H), cmap='gray')
        ax.set_title(str(j) + " prediction")
        j += 1
    elif (i % 3 == 1):
        ax.imshow(Y_test[k].reshape(W, H),cmap='gray')
        ax.set_title(str(k) + " ground truth")
        k += 1
    else:
        ax.imshow(X_test[d].reshape(W, H),cmap='gray')
        ax.set_title(str(d) + " image")
        d += 1
    i += 1
