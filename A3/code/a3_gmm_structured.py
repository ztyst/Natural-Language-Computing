from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random

dataDir = "/u/cs401/A3/data/"
# dataDir = "E:\\CloudStation\\UOT\\CSC401\\Natural-Language-Computing\\A3\\data"


class theta:
    def __init__(self, name, M=8, d=13):
        """Class holding model parameters.
        Use the `reset_parameter` functions below to
        initialize and update model parameters during training.
        """
        self.name = name
        self._M = M
        self._d = d
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))

    def precomputedForM(self, m):
        """Put the precomputedforM for given `m` computation here
        This is a function of `self.mu` and `self.Sigma` (see slide 32)
        This should output a float or equivalent (array of size [1] etc.)
        NOTE: use this in `log_b_m_x` below
        """
        result = np.sum(np.divide(np.square(self.mu[m]), (2*self.Sigma[m])))
        result += self._d /2 * np.log(2*np.pi)
        result += 0.5 * np.log(np.product(self.Sigma[m]))
        
        return result

    def reset_omega(self, omega):
        """Pass in `omega` of shape [M, 1] or [M]
        """
        omega = np.asarray(omega)
        assert omega.size == self._M, "`omega` must contain M elements"
        self.omega = omega.reshape(self._M, 1)

    def reset_mu(self, mu):
        """Pass in `mu` of shape [M, d]
        """
        mu = np.asarray(mu)
        shape = mu.shape
        assert shape == (self._M, self._d), "`mu` must be of size (M,d)"
        self.mu = mu

    def reset_Sigma(self, Sigma):
        """Pass in `sigma` of shape [M, d]
        """
        Sigma = np.asarray(Sigma)
        shape = Sigma.shape
        assert shape == (self._M, self._d), "`Sigma` must be of size (M,d)"
        self.Sigma = Sigma


def log_b_m_x(m, x, myTheta):
    """ Returns the log probability of d-dimensional vector x using only
        component m of model myTheta (See equation 1 of the handout)

    As you'll see in tutorial, for efficiency, you can precompute
    something for 'm' that applies to all x outside of this function.
    Use `myTheta.preComputedForM(m)` for this.

    Return shape:
        (single row) if x.shape == [d], then return value is float (or equivalent)
        (vectorized) if x.shape == [T, d], then return shape is [T]

    You should write your code such that it works for both types of inputs.
    But we encourage you to use the vectorized version in your `train`
    function for faster/efficient computation.
    """
    mean = myTheta.mu[m]
    sigma = myTheta.Sigma[m]

    precomp = myTheta.precomputedForM(m)
    if x.ndim == 1:
        logbmx = - np.sum(0.5 * np.divide(np.square(x), sigma) - np.divide(x * mean, sigma)) - precomp
    else:
        logbmx = - np.sum(0.5 * np.divide(np.square(x),sigma) - np.divide(x * mean , sigma), axis=1) - precomp
        
    return logbmx


def cal_log_Bs(X, M, T, myTheta):
    log_Bs = np.zeros((M, T))

    for m in range(M):
        log_Bs[m] = log_b_m_x(m, X, myTheta)

    return log_Bs  
    

def log_p_m_x(log_Bs, myTheta):
    """ Returns the matrix of log probabilities i.e. log of p(m|X;theta)

    Specifically, each entry (m, t) in the output is the
        log probability of p(m|x_t; theta)

    For further information, See equation 2 of handout

    Return shape:
        same as log_Bs, np.ndarray of shape [M, T]

    NOTE: For a description of `log_Bs`, refer to the docstring of `logLik` below
    """
    full_logpmx = np.log(myTheta.omega) + log_Bs

    maxterm = np.max(full_logpmx, axis=0, keepdims=True) # M,
    logpmx =  full_logpmx - (maxterm + np.log(np.sum(np.exp(full_logpmx - maxterm), axis=0)))

    return logpmx


def logLik(log_Bs, myTheta):
    """ Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency.

        See equation 3 of the handout
    """
    omega = myTheta.omega
    max_term = np.max(np.log(omega) + log_Bs, axis=0, keepdims=True)
    
    result = max_term + np.log(np.sum(np.exp(np.log(omega) + log_Bs - max_term),axis=0))
    result = np.sum(result)

    return result


def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    """ Train a model for the given speaker. Returns the theta (omega, mu, sigma)"""
    myTheta = theta(speaker, M, X.shape[1])
    T, d = X.shape # T, d

    omega_template = np.zeros((M, 1))
    mu_template = np.zeros((M, d))
    Sigma_template = np.zeros((M, d))
    omega_template.fill(1/M) # M, 1
    mu_template = X[np.random.choice(T,size=M,replace=False),:] #M, d
    Sigma_template.fill(1.0) #M, d

    myTheta.reset_omega(omega_template)
    myTheta.reset_mu(mu_template)
    myTheta.reset_Sigma(Sigma_template)

    i = 0
    prev_L = -np.inf
    improvement = np.inf

    while i <= maxIter and improvement >= epsilon:
        log_Bs = cal_log_Bs(X, M, T, myTheta)

        logpmx = log_p_m_x(log_Bs, myTheta)
        L = logLik(log_Bs, myTheta)

        pmx = np.exp(logpmx)
        prob_sum = np.sum(pmx, axis=1) 

        for m in range(M):
            myTheta.omega[m] = prob_sum[m] / T
            myTheta.mu[m] = np.dot(pmx[m], X) / prob_sum[m]
            myTheta.Sigma[m] = np.divide(np.dot(pmx[m], np.square(X)), prob_sum[m])-np.square(myTheta.mu[m])

        improvement = L - prev_L
        prev_L = L
        i +=1

    return myTheta


def test(mfcc, correctID, models, k=5):
    """ Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    """
    bestModel = -1
    M,d = models[0].mu.shape
    T = mfcc.shape[0]
    sname, loglik = [],[]

    for i in range(len(models)):
        log_Bs = cal_log_Bs(mfcc, M, T, models[i])

        l = logLik(log_Bs, models[i])
        sname.append(models[i].name)
        loglik.append(l)
    
    sname_sorted = [x for _,x in sorted(zip(loglik,sname),reverse=True)]
    loglik.sort(reverse=True)
    bestModel = sname.index(sname_sorted[0])

    out = open("gmmLiks.txt", "a")
    if k > 0:
        out.write(models[correctID].name)
        out.write("\n")
        for idx in range(k):
            out.write("{},{}\n".format(sname_sorted[idx], loglik[idx]))
        # out.write("=================================\n")
    
    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    # print("TODO: you will need to modify this main block for Sec 2.3")
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20
    # train a model for each speaker, and reserve data for testing

    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)

            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), "*npy")
            random.shuffle(files)

            testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
            testMFCCs.append(testMFCC)

            X = np.empty((0, d))

            for file in files:
                myMFCC = np.load(os.path.join(dataDir, speaker, file))
                X = np.append(X, myMFCC, axis=0)

            trainThetas.append(train(speaker, X, M, epsilon, maxIter))

    # evaluate
    numCorrect = 0

    for i in range(0, len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k)
    accuracy = 1.0 * numCorrect / len(testMFCCs)
    # print("M: {0} MaxIter: {1}  Accuracy: {2}".format(M, maxIter, accuracy))

