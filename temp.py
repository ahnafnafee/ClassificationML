import cv2
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc
import matplotlib.cm as cm
from collections import Counter
import glob
from matplotlib.image import imread
from enum import Enum
from scipy.linalg import svd
import scipy.stats as ss
import collections

# For testing
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


imp_data = np.genfromtxt('spambase.data', delimiter=',')

def train_test_split(data, train_size, random_state):
    '''Splitting testing and training data'''

    # Resetting random seed
    np.random.seed(random_state)

    n = len(data)

    # Rows shuffled
    np.random.shuffle(data)

    # Calculates array index for splitting
    spltIdx = int(np.ceil((2/3)*n))

    # Training-validation data split
    data_train, data_test = data[:spltIdx,:], data[spltIdx:,:]

    # Training data
    x_tr, y_tr = np.hsplit(data_train, [-1])
    # Testing Data
    x_tt, y_tt = np.hsplit(data_test, [-1])



    # Separating class label from data
    class_label_tr = data_train[:, -1].astype(int)
    dataset_tr = data_train[:, :-1]

    class_label_tt = data_test[:, -1].astype(int)
    dataset_tt = data_test[:, :-1]

    # Filtering features with low std
    # dataset_tr = std_filter(dataset_tr, 0)
    # dataset_tt = std_filter(dataset_tt, 0)

    # dataset_tr = (dataset_tr - np.mean(dataset_tr)) / np.std(dataset_tr)
    # dataset_tt = (dataset_tt - np.mean(dataset_tt)) / np.std(dataset_tt)

    # x_tr = (x_tr - np.mean(x_tr)) / np.std(x_tr)
    # x_tt = (x_tt - np.mean(x_tt)) / np.std(x_tt)

    return x_tr, y_tr, x_tt, y_tt
    # return dataset_tr, class_label_tr, dataset_tt, class_label_tt


def std_filter(data, std_val):
    '''Filters out features with low std'''

    # Standardizing the matrix
    standardized_mat = (data - np.mean(data)) / np.std(data)

    std_mat = np.std(standardized_mat, axis = 0)
    col_num = standardized_mat.shape[1]
    low_idx = []

    for i in range(col_num):
        if std_mat[i] <= std_val:
            low_idx.append(i)

    temp = 0
    for j in low_idx:
        standardized_mat = np.delete(standardized_mat, j - temp, 1)
        temp += 1

    return standardized_mat



class ClassifierEvaluation:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true.astype(int)
        self.y_pred = y_pred.astype(int)
        unique, counts = np.unique(y_true, return_counts=True)
        self.y_true_dict = dict(zip(unique, counts))
        unique, counts = np.unique(y_pred, return_counts=True)
        self.y_pred_dict = dict(zip(unique, counts))
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

    def eval(self):
        for i in range(len(self.y_true)):
            if (self.y_true[i] == self.y_pred[i] == 1):
                self.TP += 1
            elif (self.y_true[i] == self.y_pred[i] == 0):
                self.TN += 1
        self.FP = len(self.y_true) - self.TP
        self.FN = len(self.y_true) - self.TN
        print(self.TP, self.TN, self.FP, self.FN)

    def get_precision(self):
        precision = self.TP/(self.TP + self.FP)
        return precision

    def get_recall(self):
        recall = self.TP/(self.TP + self.FN)
        return recall

    def get_fmeasure(self):
        fmeasure = (2 * self.get_precision() * self.get_recall())/(self.get_precision() + self.get_recall())
        return recall

    def get_accuracy(self):
        accuracy = (self.TP + self.TN) /(self.TP + self.TN + self.FP + self.FN)

        return accuracy





class NaiveBayes:
    def __init__(self, x, y):
        self.n_samples, self.n_features = x.shape
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)

        self.features = x
        self.target = y.flatten()

        self.classes = np.unique(y)
        self.mean_data = np.zeros((self.n_classes, self.n_features), dtype=np.float64)
        self.std_data = np.zeros((self.n_classes, self.n_features), dtype=np.float64)
        self.prior_data = np.zeros(self.n_classes)
        

    def get_target(self):
        print(np.mean(self.mean_data.flatten()))
        print(np.mean(self.std_data.flatten()))
        # print(self.features[0])
        print(self.target.shape)
        print(self.features.shape)
        return self.target


    def class_sep(self):
        '''Separates spam and not spam rows'''

        data = self.features
        label = self.target

        label = label.reshape(label.shape[0], 1)

        spIdx_lst = np.where(~label.any(axis=1))[0]
        notIdx_sp_lst = np.where(label.any(axis=1))[0]

        d_list = data.tolist()
        sp_list = []
        not_sp_list = []

        for index in spIdx_lst:
            sp_list += [d_list[index]]

        for index in notIdx_sp_lst:
            not_sp_list += [d_list[index]]


        sp_data = np.asarray(sp_list)
        not_sp_data = np.asarray(not_sp_list)

        self.mean_data[0, :] = sp_data.mean(axis=0)
        self.std_data[0, :] = sp_data.std(axis=0)
        self.prior_data[0] = sp_data.shape[0] / float(self.n_samples)

        self.mean_data[1, :] = not_sp_data.mean(axis=0)
        self.std_data[1, :] = not_sp_data.std(axis=0)
        self.prior_data[1] = not_sp_data.shape[0] / float(self.n_samples)


    def fit(self):

        for idx, c in enumerate(self.classes):
            temp = self.features[self.target==c]
            self.mean_data[idx, :] = temp.mean(axis=0)
            self.std_data[idx, :] = temp.std(axis=0)
            self.prior_data[idx] = temp.shape[0] / float(self.n_samples)


    def get_stats(self):
        return self.mean_data, self.std_data, self.prior_data

    
    def calc_posterior(self, x):
        '''Calculates posterior prob for each class'''

        posteriors = []

        for i in range(self.n_classes):
            prior = np.log(self.prior_data[i])
            n_log = np.log(self.norm_pdf(x, i))
            n_log = np.nan_to_num(n_log, nan=10^-8, posinf=10^8, neginf=10^-8)
            posterior = np.sum(n_log)
            posterior = prior + posterior
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]


    def predict(self, x):
        preds = [self.calc_posterior(i) for i in x]
        # preds = [self.calc_posterior(i) for i in x[:5]]
        return np.asarray(preds, dtype=np.float64)


    def norm_pdf(self, data, c_idx):
        '''Calculates norm pdf'''

        mean = self.mean_data[c_idx]
        std = self.std_data[c_idx]

        numerator = np.exp((-1/2)*((data-mean)**2) / (2 * (std**2)))
        denominator = std * np.sqrt(2 * np.pi)

        return numerator / denominator

x_tr, y_tr, x_tt, y_tt = train_test_split(imp_data, train_size=2/3, random_state=0)
print(type(y_tt))

g_nb = NaiveBayes(x_tr, y_tr)

# g_nb.class_sep()
g_nb.fit()
print(g_nb.get_target())
# print(g_nb.get_stats())

predictions = g_nb.predict(x_tt)
print("LEN: ", len(predictions))
gb_ce = ClassifierEvaluation(y_tt, predictions)
gb_ce.eval()
# gb_ce.get_accuracy()
print("Naive Bayes classification accuracy", gb_ce.get_accuracy())
print()
print()
# g_nb.predict(x_tt)

# predict(x_tr, y_tr)

GaussNB = GaussianNB()
GaussNB.fit(x_tr, y_tr)
y_expect = y_tt
y_predict = GaussNB.predict(x_tt)
print("LEN: ", len(y_predict))
print()
accuracy_score(y_expect,y_predict)

