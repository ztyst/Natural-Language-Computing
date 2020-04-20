# Copyright 2020 University of Toronto, all rights reserved

'''Calculate BLEU score for one reference and one hypothesis

You do not need to import anything more than what is here
'''

from math import exp  # exp(x) gives e^x


def grouper(seq, n):
    '''Extract all n-grams from a sequence

    An n-gram is a contiguous sub-sequence within `seq` of length `n`. This
    function extracts them (in order) from `seq`.

    Parameters
    ----------
    seq : sequence
        A sequence of words or token ids representing a transcription.
    n : int
        The size of sub-sequence to extract.

    Returns
    -------
    ngrams : list
    '''
    # assert False, "Fill me"
    ngrams = []
    for i in range(len(seq)):
        if i+n <= len(seq):
            sub_sequent = seq[i:i+n]
            ngrams.append(sub_sequent)

    return ngrams


def n_gram_precision(reference, candidate, n):
    '''Calculate the precision for a given order of n-gram

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The order of n-gram precision to calculate

    Returns
    -------
    p_n : float
        The n-gram precision. In the case that the candidate has length 0,
        `p_n` is 0.
    '''
    # assert False, "Fill me"
    p_n = 0
    match = 0
    if len(candidate) != 0:
        ref_n_grams = grouper(reference, n)
        cand_n_grams = grouper(candidate, n)
        for cand in cand_n_grams:
            if cand in ref_n_grams:
                match += 1
        p_n = match / len(cand_n_grams)
    return p_n
    


def brevity_penalty(reference, candidate):
    '''Calculate the brevity penalty between a reference and candidate

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)

    Returns
    -------
    BP : float
        The brevity penalty. In the case that the candidate transcription is
        of 0 length, `BP` is 0.
    '''
    # assert False, "Fill me"
    BP = 0
    if len(candidate) != 0:
        # nume = len(reference)
        # deno = len(candidate)
        brev_pen = len(reference) / len(candidate)
        BP = 1 if brev_pen < 1 else exp(1-brev_pen)
        # print(BP)
    return BP



def BLEU_score(reference, hypothesis, n):
    '''Calculate the BLEU score

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The maximum order of n-gram precision to use in the calculations,
        inclusive. For example, ``n = 2`` implies both unigram and bigram
        precision will be accounted for, but not trigram.

    Returns
    -------
    bleu : float
        The BLEU score
    '''
    bleu = 0.0
    p_n = 1
    for i in range(n):
        p_i = n_gram_precision(reference, hypothesis, i+1)
        p_n = p_n * p_i
    BP = brevity_penalty(reference,hypothesis)
    bleu = BP * (p_n ** (1/2))
    return bleu
    # assert False, "Fill me"
