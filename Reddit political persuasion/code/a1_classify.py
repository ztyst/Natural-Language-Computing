import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier


def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    diag_sum = np.trace(C)
    total_elements = np.sum(C)
    accuracy = diag_sum / total_elements if total_elements != 0 else 0

    return accuracy


def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    pred_class = C.diagonal().reshape(-1)
    deno = np.sum(C, axis=0).reshape(-1)

    if len(pred_class) != len(deno):
        print("INCONSISTENT DIMENSION")
        exit()
    recall = [float(pred_class[i])/float(deno[i]) if deno[i] != 0 else 0 for i in range(len(pred_class))]

    return recall


def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    pred_class = C.diagonal().reshape(-1)
    deno = np.sum(C, axis=1).reshape(-1)

    if len(pred_class) != len(deno):
        print("INCONSISTENT DIMENSION")
        exit()
    precision = [float(pred_class[i])/float(deno[i]) if deno[i] != 0 else 0 for i in range(len(pred_class))]

    return precision


def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:
       i: int, the index of the supposed best classifier
    '''
    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
        iBest = 0
        best_accuracy = -1

        def clf_execute(clf, X_train, X_test, y_train, y_test, outf, classifier_name, index, best_accuracy, iBest):
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            conf_matrix = confusion_matrix(y_test, y_pred)
            rec = recall(conf_matrix)
            prec = precision(conf_matrix)
            accu = accuracy(conf_matrix)
            if accu > best_accuracy:
                best_accuracy = accu
                iBest = index
            outf.write(f'Results for {classifier_name}:\n')
            outf.write(f'\tAccuracy: {accu:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in rec]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in prec]}\n')
            outf.write(f'\tConfusion Matrix: \n{conf_matrix}\n\n')

            return iBest, best_accuracy

        # SGDClassifier
        # classifier_name = "SGDClassifier"
        clf_sgd = SGDClassifier()
        iBest, best_accuracy = clf_execute(clf_sgd, X_train, X_test, y_train, y_test, outf, "SGDClassifier", 1, best_accuracy, iBest)

        # GaussianNB
        # classifier_name = "GaussianNB"
        clf_gau = GaussianNB()
        iBest, best_accuracy = clf_execute(clf_gau, X_train, X_test, y_train, y_test, outf, "GaussianNB", 2, best_accuracy, iBest)

        # RandomForestClassifier
        # classifier_name = "RandomForestClassifier"
        # maximum depth of 5, and 10 estimators
        clf_ran = RandomForestClassifier(max_depth=5, n_estimators=10)
        iBest, best_accuracy = clf_execute(clf_ran, X_train, X_test, y_train, y_test, outf, "RandomForestClassifier",3, best_accuracy, iBest)

        # MLPClassifier:
        # classifier_name = "MLPClassifier"
        # α= 0.05
        clf_mlp = MLPClassifier(alpha=0.05)
        iBest, best_accuracy = clf_execute(clf_mlp, X_train, X_test, y_train, y_test, outf, "MLPClassifier",4, best_accuracy, iBest)

        # AdaBoostClassifier
        # classifier_name = "AdaBoostClassifier"
        clf_ada = AdaBoostClassifier()
        iBest, best_accuracy =clf_execute(clf_ada, X_train, X_test, y_train, y_test, outf, "AdaBoostClassifier",5, best_accuracy, iBest)

    return iBest


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        if iBest == 1:
            clf = SGDClassifier()
        elif iBest == 2:
            clf = GaussianNB()
        elif iBest == 3:
            clf = RandomForestClassifier(max_depth=5, n_estimators=10)
        elif iBest == 4:
            clf = MLPClassifier(alpha=0.05)
        elif iBest == 5:
            clf = AdaBoostClassifier()
        
        size_list = [1000, 5000, 10000, 15000, 20000]
        for num_train in size_list:
            idx = np.random.choice(X_train.shape[0], num_train, replace=False)
            # print(X_train.shape, y_train.shape)
            random_X_train = X_train[idx, :]
            random_y_train = y_train[idx]
            if num_train == 1000:
                X_1k, y_1k = random_X_train, random_y_train
            clf.fit(random_X_train, random_y_train)
            y_pred = clf.predict(X_test)
            conf_matrix = confusion_matrix(y_test, y_pred)
            accu = accuracy(conf_matrix)
            outf.write(f'{num_train}: {accu:.4f}\n')

    return (X_1k, y_1k)


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        k_range = [5, 50]

        if i == 1:
            clf = SGDClassifier()
        elif i == 2:
            clf = GaussianNB()
        elif i == 3:
            clf = RandomForestClassifier(max_depth=5, n_estimators=10)
        elif i == 4:
            clf = MLPClassifier(alpha=0.05)
        elif i == 5:
            clf = AdaBoostClassifier()

        for k_feat in k_range:

            selector = SelectKBest(f_classif, k=k_feat)
            X_new = selector.fit_transform(X_train, y_train)
            p_values = selector.pvalues_
            # for each number of features k_feat, write the p-values for
            # that number of features:
            outf.write(f'{k_feat} p-values: {[round(pval, 4) for pval in p_values]}\n')

        selector = SelectKBest(f_classif, k=5)

        # 1k training set
        X_new_1k = selector.fit_transform(X_1k, y_1k)
        X_test_new = selector.transform(X_test)
        clf.fit(X_new_1k, y_1k)
        y_pred_1k= clf.predict(X_test_new)
        conf_matrix_1k = confusion_matrix(y_test, y_pred_1k)
        accuracy_1k = accuracy(conf_matrix_1k)
        outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')
        indices_1k = selector.get_support(indices=True)

        # 32k training set
        X_new = selector.fit_transform(X_train, y_train)
        X_test_new = selector.transform(X_test)
        clf.fit(X_new, y_train)
        y_pred= clf.predict(X_test_new)
        conf_matrix = confusion_matrix(y_test, y_pred)
        accuracy_full = accuracy(conf_matrix)
        outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')
        top_5 = set(selector.get_support(indices=True))

        feature_intersection = set(set(top_5) & set(indices_1k))

        outf.write(f'Chosen feature intersection: {feature_intersection}\n')
        
        outf.write(f'Top-5 at higher: {top_5}\n')



def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
        '''
    def generate_accuracy(clf, X_train, X_test, y_train, y_test):
        clf.fit(X_train, y_train)
        y_pred= clf.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred)
        accu = accuracy(conf_matrix)

        return accu

    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        # Prepare kfold_accuracies, then uncomment this, so it writes them to outf.
        X_full = np.concatenate((X_train, X_test),axis=0)
        y_full = np.concatenate((y_train, y_test),axis=0)

        kf = KFold(n_splits=5,shuffle=True)
        kfold_accuracies_sum = []
        for train, test in kf.split(X_full):
            temp = []
            clf_sgd = SGDClassifier()
            sgd_accuracy = generate_accuracy(clf_sgd, X_full[train], X_full[test], y_full[train], y_full[test])
            
            clf_gau = GaussianNB()
            gau_accuracy = generate_accuracy(clf_gau, X_full[train], X_full[test], y_full[train], y_full[test])
            
            clf_ran = RandomForestClassifier(max_depth=5, n_estimators=10)
            ran_accuracy = generate_accuracy(clf_ran, X_full[train], X_full[test], y_full[train], y_full[test])

            clf_mlp = MLPClassifier(alpha=0.05)
            mlp_accuracy = generate_accuracy(clf_mlp, X_full[train], X_full[test], y_full[train], y_full[test])

            clf_ada = AdaBoostClassifier()
            ada_accuracy = generate_accuracy(clf_ada, X_full[train], X_full[test], y_full[train], y_full[test])
            
            temp.append(sgd_accuracy)
            temp.append(gau_accuracy)
            temp.append(ran_accuracy)
            temp.append(mlp_accuracy)
            temp.append(ada_accuracy)
            kfold_accuracies_sum.append(temp)
            outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in temp]}\n')
   
        a = np.array(kfold_accuracies_sum)
        # kfold_accuracies = a.mean(axis=0)
        # outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')

        p_values = []
        best_clf = a[:, i-1]
        for j in range(5):
            if j != i-1:
                S = ttest_rel(a[:, j], best_clf)
                p_values.append(S.pvalue)

        outf.write(f'p-values: {[round(pval, 4) for pval in p_values]}\n')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    data = np.load(args.input)
    data = data["arr_0"]
    X = data[:,0:-1]
    y = data[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    iBest = class31(args.output_dir, X_train, X_test, y_train, y_test)
    X_1k, y_1k = class32(args.output_dir, X_train, X_test, y_train, y_test, iBest)
    class33(args.output_dir, X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    class34(args.output_dir, X_train, X_test, y_train, y_test, iBest)

