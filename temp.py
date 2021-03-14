imp_data = np.genfromtxt('q1_2.csv', delimiter=',')

x_tr, y_tr = np.hsplit(imp_data, [-1])
og_mean = np.mean(x_tr)
og_std = np.std(x_tr)
x = (x_tr - og_mean) / og_std

label = y_tr
n_samples, n_features = x.shape
classes = np.unique(y_tr)
n_classes = len(classes)

mean_data = np.zeros((n_classes, n_features), dtype=np.float64)
std_data = np.zeros((n_classes, n_features), dtype=np.float64)
prior_data = np.zeros(n_classes)
label = label.reshape(label.shape[0], 1)

noIdx_lst = np.where(~label.any(axis=1))[0]
yesIdx_lst = np.where(label.any(axis=1))[0]


d_list = x.tolist()
no_list = []
yes_list = []

for index in noIdx_lst:
    no_list += [d_list[index]]

for index in yesIdx_lst:
    yes_list += [d_list[index]]


no_data = np.asarray(no_list)
yes_data = np.asarray(yes_list)


mean_data[0, :] = no_data.mean(axis=0)
std_data[0, :] = no_data.std(axis=0)
prior_data[0] = no_data.shape[0] / float(n_samples)

mean_data[1, :] = yes_data.mean(axis=0)
std_data[1, :] = yes_data.std(axis=0)
prior_data[1] = yes_data.shape[0] / float(n_samples)

print("Mean:", mean_data)
print()
print("Std:", std_data)
print()
print("Priors:", prior_data)

x_tt = np.asarray([[242, 4.56]], dtype=np.float32)
x_test = (x_tt - og_mean) / og_std
print(x_test)


def calc_posterior(x):
    posteriors = []

    for i in range(n_classes):
        prior = prior_data[i]
        n_pdf = norm_pdf(x, i)
        n_pdf = np.prod(np.nan_to_num(n_pdf, nan=10^-8, posinf=10^8, neginf=10^-12))
        posterior = prior * n_pdf
        posteriors.append(posterior)
    return classes[np.argmax(posteriors)]


def norm_pdf(data, c_idx):

    mean = mean_data[c_idx]
    std = std_data[c_idx]

    numerator = np.exp(- (data-mean)**2 / (2 * (std**2)))
    denominator = std * np.sqrt(2 * np.pi)

    return numerator / denominator


preds = [calc_posterior(i) for i in x_test]
print(preds)
