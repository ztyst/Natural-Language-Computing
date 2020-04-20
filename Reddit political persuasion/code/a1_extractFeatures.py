import numpy as np
import argparse
import json
import re
import spacy
import pandas as pd
import math
from collections import defaultdict


# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}

# Filepath
WORDLIST_DIR = "/u/cs401/Wordlists/"
FEATS_DIR = "/u/cs401/A1/feats/"

BGL_LOC = WORDLIST_DIR + "BristolNorms+GilhoolyLogie.csv"
WA_LOC = WORDLIST_DIR + "Ratings_Warriner_et_al.csv"

LEFT_ID_LOC = FEATS_DIR + "Left_IDs.txt"
CENTER_ID_LOC = FEATS_DIR + "Center_IDs.txt"
RIGHT_ID_LOC = FEATS_DIR + "Right_IDs.txt"
ALT_ID_LOC = FEATS_DIR + "Alt_IDs.txt"

LEFT_FEATS_LOC = FEATS_DIR + "Left_feats.dat.npy"
CENTER_FEATS_LOC = FEATS_DIR + "Center_feats.dat.npy"
RIGHT_FEATS_LOC = FEATS_DIR + "Right_feats.dat.npy"
ALT_FEATS_LOC = FEATS_DIR + "Alt_feats.dat.npy"


def create_matching_regex(pattern):
    regex =r"(\s|^)("
    for item in pattern:
        regex += item
        regex += '|'
    regex = regex[:-1] + r")\/\w+"
    return regex


def read_csv(csv_loc):
    csv = pd.read_csv(csv_loc)
    return csv


BGL = pd.read_csv(BGL_LOC)
BGL = BGL[['WORD','AoA (100-700)', 'IMG', 'FAM']]
WARRINGER = pd.read_csv(WA_LOC)
WARRINGER = WARRINGER[['Word','V.Mean.Sum', 'A.Mean.Sum', 'D.Mean.Sum']]
LEFT_ID = open(LEFT_ID_LOC,"r").read().split("\n")
LEFT_FEATS = np.load(LEFT_FEATS_LOC)
CENTER_ID = open(CENTER_ID_LOC, "r").read().split("\n")
CENTER_FEATS = np.load(CENTER_FEATS_LOC)
RIGHT_FEATS = np.load(RIGHT_FEATS_LOC)
RIGHT_ID = open(RIGHT_ID_LOC, "r").read().split("\n")
ALT_FEATS = np.load(ALT_FEATS_LOC)
ALT_ID = open(ALT_ID_LOC,"r").read().split("\n")


def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    feats = np.zeros((1, 173))

    sentence_num = comment.count("\n")
    comment = re.sub(r"\n", " ", comment)

    # 1.  Number of words in uppercase (≥3 letters long)
    upper_word_count = len(re.findall(r"\b[A-Z]{3,}\/", comment))
    feats[0,0] = upper_word_count

    word_list = comment.split()

    def replacement(match):
        return match.group(0).lower()
    
    for i in range(len(word_list)):
        word_list[i] = re.sub(r"\b(\w*[A-Z]*\w*)\/", replacement, word_list[i])
    lower_comment = ' '.join(word_list)

    # 2.  Number of first-person pronouns
    fp_regex = create_matching_regex(FIRST_PERSON_PRONOUNS)
    fp_count = len(re.findall(fp_regex, lower_comment))
    feats[0,1] = fp_count
    
    # 3.  Number of second-person pronouns
    sp_regex = create_matching_regex(SECOND_PERSON_PRONOUNS)
    sp_count = len(re.findall(sp_regex, lower_comment))
    feats[0,2] = sp_count

    # 4.  Number of third-person pronouns
    tp_regex = create_matching_regex(THIRD_PERSON_PRONOUNS)
    tp_count = len(re.findall(tp_regex, lower_comment))
    feats[0,3] = tp_count

    # 5.  Number of coordinating conjunctions
    # conj_count = len(re.findall(r"\b(\S+)\/(CC)", lower_comment))
    conj_count = len(re.findall(r"\/(CC)\b", lower_comment))
    feats[0, 4] = conj_count

    # 6.  Number of past-tense verbs
    # pt_count = len(re.findall(r"\b(\S+)\/(VBD)", lower_comment))
    pt_count = len(re.findall(r"\/(VBD)", lower_comment))
    feats[0, 5] = pt_count
    
    # 7. Number of future tense
    # ft_count = len(re.findall(r"\b(will\/MD|will\/MD \w+\/RB|go\/VBG to\/TO) \w+\/VB", lower_comment))
    ft_count = len(re.findall(r"\b(will\/MD|will\/MD \w+\/RB|go\/VBG to\/TO|i'll\/\w+|she'll\/\w+|he'll\/\w+|it'll\/\w+|you'll\/\w+|we'll\/\w+|they'll\/\w+) \w+\/(VBP|VB)", lower_comment))
    feats[0, 6] = ft_count

    # 8.  Number of commas
    comma_count = len(re.findall(r"\,\/", lower_comment))
    feats[0,7] = comma_count

    # 9. Number of multi-character punctuation tokens
    mp_count = len(re.findall(r"\s(_|[^\w\s]){2,}\/", lower_comment))
    feats[0,8] = mp_count

    # 10. Number of common nouns
    # cn_count = len(re.findall(r"\b(\S+)\/(NNS|NN)\b", lower_comment))
    cn_count = len(re.findall(r"\/(NNS|NN)\b", lower_comment))
    feats[0,9] = cn_count

    # 11. Number of proper nouns
    # pn_count = len(re.findall(r"\b(\S+)\/(NNPS|NNP)\b", lower_comment))
    pn_count = len(re.findall(r"\/(NNPS|NNP)\b", lower_comment))
    feats[0,10] = pn_count

    # 12. Number of adverbs
    # adverb_count = len(re.findall(r"\b(\S+)\/(RBS|RBR|RB)\b", lower_comment))
    adverb_count = len(re.findall(r"\/(RBS|RBR|RB)\b", lower_comment))
    feats[0,11] = adverb_count

    # 13. Number of wh-words
    # wwords_count =  len(re.findall(r"\b(\S+)\/(WDT|WP$|WP|WRB)\b", lower_comment))
    wwords_count =  len(re.findall(r"\/(WDT|WP$|WP|WRB)\b", lower_comment))
    feats[0,12] = wwords_count

    # 14. Number of slang acronyms
    slang_regex = create_matching_regex(SLANG)
    slang_count = len(re.findall(slang_regex, lower_comment))
    feats[0,13] = slang_count

    # 15. Average length of sentences, in tokens
    token_count = len(re.findall(r"\/\S+", lower_comment))
    feats[0,14] = token_count / float(sentence_num) if sentence_num != 0 else 0

    # 16. Average length of tokens, excluding punctuation-only tokens, in characters
    token_nopunc =re.findall(r"([^\w\s]*\w+([^\w\s]*\w*)*)\/", lower_comment)
    token_nopunc_list = [tup[0] for tup in token_nopunc]
    feats[0,15] = len("".join(token_nopunc_list)) / len(token_nopunc_list) if len(token_nopunc_list)!=0 else 0

    # 17. Number of sentences.
    feats[0,16] = sentence_num

    warringer_filter = WARRINGER.loc[WARRINGER['Word'].isin(token_nopunc_list)]
    bgl_filter = BGL.loc[BGL['WORD'].isin(token_nopunc_list)]

    word_list = bgl_filter['WORD'].tolist()
    aoa_list = bgl_filter['AoA (100-700)'].tolist()
    img_list = bgl_filter['IMG'].tolist()
    fam_list = bgl_filter['FAM'].tolist()

    tok_bgl_dict = defaultdict(list)
    for i in range(len(word_list)):
        tok_bgl_dict[word_list[i]].append(aoa_list[i])
        tok_bgl_dict[word_list[i]].append(img_list[i])
        tok_bgl_dict[word_list[i]].append(fam_list[i])

    word_list = warringer_filter['Word'].tolist()
    vms_list = warringer_filter['V.Mean.Sum'].tolist()
    ams_list = warringer_filter['A.Mean.Sum'].tolist()
    dms_list = warringer_filter['D.Mean.Sum'].tolist()

    tok_warringer_dict = defaultdict(list)
    for i in range(len(word_list)):
        tok_warringer_dict[word_list[i]].append(vms_list[i])
        tok_warringer_dict[word_list[i]].append(ams_list[i])
        tok_warringer_dict[word_list[i]].append(dms_list[i])

    aoa, img, fam = [], [], []
    vms, ams, dms = [], [], []
    for tok in token_nopunc_list:
        if tok in tok_bgl_dict:
            aoa.append(tok_bgl_dict[tok][0])
            img.append(tok_bgl_dict[tok][1])
            fam.append(tok_bgl_dict[tok][2])
        if tok in tok_warringer_dict:
            vms.append(tok_warringer_dict[tok][0])
            ams.append(tok_warringer_dict[tok][1])
            dms.append(tok_warringer_dict[tok][2])

    # 18. Average of AoA (100-700) from Bristol, Gilhooly, and Logie norms
    aoa_mean = np.mean(aoa) if len(aoa)!= 0 else 0
    feats[0,17] = aoa_mean if not math.isnan(aoa_mean) else 0

    # 19. Average of IMG from Bristol, Gilhooly, and Logie norms
    img_mean = np.mean(img) if len(img) != 0 else 0
    feats[0,18] = img_mean if not math.isnan(img_mean) else 0

    # 20. Average of FAM from Bristol, Gilhooly, and Logie norms
    fam_mean = np.mean(fam) if len(fam) != 0 else 0
    feats[0,19] = fam_mean if not math.isnan(fam_mean) else 0

    # 21. Standard deviation of AoA (100-700) from Bristol, Gilhooly, and Logie norms
    aoa_std = np.std(aoa) if len(aoa) != 0 else 0
    feats[0,20] = aoa_std if not math.isnan(aoa_std) else 0

    # 22. Standard deviation of IMG from Bristol, Gilhooly, and Logie norms
    img_std = np.std(img) if len(img) != 0 else 0
    feats[0,21] = img_std if not math.isnan(img_std) else 0

    # 23. Standard deviation of FAM from Bristol, Gilhooly, and Logie norms
    fam_std = np.std(fam) if len(fam) != 0 else 0
    feats[0,22] = fam_std if not math.isnan(fam_std) else 0

    # 24. Average of V.Mean.Sum from WARRINGER norms
    vms_mean = np.mean(vms) if len(vms) != 0 else 0
    feats[0,23] = vms_mean if not math.isnan(vms_mean) else 0

    # 25. Average of A.Mean.Sum from WARRINGER norms
    ams_mean = np.mean(ams) if len(ams) != 0 else 0
    feats[0,24] = ams_mean if not math.isnan(ams_mean) else 0

    # 26. Average of D.Mean.Sum from WARRINGER norms
    dms_mean = np.mean(dms) if len(dms) != 0 else 0
    feats[0,25] = dms_mean if not math.isnan(dms_mean) else 0

    # 27. Standard deviation of V.Mean.Sum from WARRINGER norms
    vms_std = np.std(vms) if len(vms) != 0 else 0
    feats[0,26] = vms_std if not math.isnan(vms_std) else 0

    # 28. Standard deviation of A.Mean.Sum from WARRINGER norms
    ams_std = np.std(ams) if len(ams) != 0 else 0
    feats[0,27] = ams_std if not math.isnan(ams_std) else 0

    # 29. Standard deviation of D.Mean.Sum from WARRINGER norms
    dms_std = np.std(dms) if len(dms) != 0 else 0
    feats[0,28] = dms_std if not math.isnan(dms_std) else 0

    return feats
    

def extract2(feats, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feats: np.array of length 173FEATS_DIR + "Alt_IDs.txt"
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (this
        function adds feature 30-173). This should be a modified version of
        the parameter feats.
    '''
    if comment_class == "Left":
        try:
            index = LEFT_ID.index(comment_id)
            feats[29:-1] = LEFT_FEATS[index]
        except ValueError:
            pass
    elif comment_class == "Center":
        try:
            index = CENTER_ID.index(comment_id)
            feats[29:-1] = CENTER_FEATS[index]
        except ValueError:
            pass
    elif comment_class == "Right":
        try:
            index = RIGHT_ID.index(comment_id)
            feats[29:-1] = RIGHT_FEATS[index]
        except ValueError:
            pass
    else:
        try:
            index = ALT_ID.index(comment_id)
            feats[29:-1] = ALT_FEATS[index]
        except ValueError:
            pass

    return feats


def main(args):
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))

    for i in range(len(feats)):
        # print(">>>>>>>>>>>>>>START EXTRACT1")
        feats[i][:173] = extract1(str(data[i]["body"]))
        # print(">>>>>>>>>>>>>>START EXTRACT2")
        feats[i] = extract2(feats[i], data[i]["cat"], data[i]["id"])
        if data[i]["cat"] == "Left":
            feats[i][173] = 0
        elif data[i]["cat"] == "Center":
            feats[i][173] = 1
        elif data[i]["cat"] == "Right":
            feats[i][173] = 2
        else:
            feats[i][173] = 3

    np.savez_compressed(args.output, feats)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()

    main(args)
    
