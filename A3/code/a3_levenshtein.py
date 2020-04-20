import os
import re
import numpy as np
import statistics

dataDir = '/u/cs401/A3/data/'

def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    >>> wer("who is there".split(), "is there".split())                         
    0.333 0 0 1                                                                           
    >>> wer("who is there".split(), "".split())                                 
    1.0 0 0 3                                                                           
    >>> wer("".split(), "who is there".split())                                 
    Inf 0 3 0                                                                           
    """
    n = len(r)
    m = len(h)
    if n == 0:
        return (np.Inf, 0, m, 0)
    if m == 0:
        return (np.Inf, 0, n, 0)
    if n == 0 and m ==0:
        return (0, 0, 0, 0)
    calc = np.zeros((n+1, m+1))
    calc[0] = np.arange(m+1)
    calc[:,0] = np.arange(n+1)

    backtrace = np.zeros((n+1,m+1))
    backtrace[0] = np.full(m+1, 2)
    backtrace[:,0] = np.full(n+1, 3)
    backtrace[0][0] = 0

    for i in range(1,n+1):
        for j in range(1, m+1):
            insertion = calc[i,j-1] + 1
            deletion = calc[i-1,j] + 1
            substitution = calc[i-1,j-1] + 1

            match = np.Inf
            if r[i-1] == h[j-1]:
                match = calc[i-1,j-1]
            else:
                match = substitution
            
            calc[i,j] = min(deletion, match, insertion)

            if calc[i,j] == substitution:
                backtrace[i,j] = 1
            elif calc[i,j] == insertion:
                backtrace[i,j] = 2
            elif calc[i,j] == deletion:
                backtrace[i,j] = 3
            elif calc[i,j] == calc[i-1,j-1]:
                backtrace[i,j] = 4

    sub, ins, dels = 0,0,0
    row, col = n, m
    while row != 0 or col != 0:
        if backtrace[row, col] == 1:
            sub += 1
            row, col = row-1, col-1
        elif backtrace[row, col] == 2:
            ins += 1
            row, col = row, col-1
        elif backtrace[row, col] == 3:
            dels +=1
            row, col = row-1, col
        elif backtrace[row,col] == 4:
            row, col = row-1, col-1

    return round((sub+ins+dels)/n, 3), sub, ins, dels
        
    
def preproc(content):
    after_process = []
    for line in content:
        line = line.lower()
        line = re.sub(r"[{!\"#$%&'()*+,\-.\/:;<=>?@^_`{|}~}+]", "", line)
        line = re.sub(r'\s{1,}', ' ', line)
        after_process.append(line)
    return after_process


if __name__ == "__main__":
    gwer = []
    kwer = []
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print( speaker )

            with open(dataDir + "/" + speaker + "/transcripts.Google.txt", "r") as google:
                google_content = google.readlines() 
            with  open(dataDir + "/" + speaker + "/transcripts.Kaldi.txt", "r") as kaldi:
                kaldi_content = kaldi.readlines()
            with open(dataDir + "/" + speaker + "/transcripts.txt", "r") as human:
                human_content = human.readlines()
            
            google_content = preproc(google_content)
            kaldi_content = preproc(kaldi_content)
            human_content = preproc(human_content)
  
            if len(google_content) == 0  or len(kaldi_content) == 0 or len(human_content) == 0:
                print("Ignore speaker " + speaker)
                continue
            
            for i in range(len(human_content)):
                WER, nS, nI, nD = Levenshtein(human_content[i].split(), google_content[i].split())
                print("{0} {1} {2} {3} S:{4}, I:{5}, D:{6}".format(speaker, "Google", i, WER, nS, nI, nD))
                gwer.append(WER)

                WER, nS, nI, nD = Levenshtein(human_content[i].split(), kaldi_content[i].split())
                print("{0} {1} {2} {3} S:{4}, I:{5}, D:{6}".format(speaker, "Kaldi", i, WER, nS, nI, nD))
                kwer.append(WER)
            
            print("============================================================")

    print("Google TOTAL WER AVERAGE " + str(np.mean(gwer)) + " STD " + str(np.std(gwer)))
    print("Kaldi TOTAL WER AVERAGE " + str(np.mean(kwer)) + " STD " + str(np.std(kwer)))