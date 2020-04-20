import sys
import argparse
import os
import json
import re
import spacy
import html


nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)


def preproc1(comment , steps=range(1, 5)):
    ''' This function pre-processes a single comment

    Parameters:
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step

    Returns:
        modComm : string, the modified comment
    '''
    modComm = comment
    if 1 in steps:  # replace newlines with spaces
        modComm = re.sub(r"\n{1,}", " ", modComm)
    if 2 in steps:  # unescape html
        modComm = html.unescape(modComm)
        modComm = html.unescape(modComm)
    if 3 in steps:  # remove URLs
        modComm = re.sub(r"(http|www)\S+", "", modComm)
    if 4 in steps:  # remove duplicate spaces
        modComm = re.sub(r'\s{1,}', ' ', modComm)

    if 5 in steps:
    # TODO: get Spacy document for modComm
        new_modComm = ""
        utt = nlp(modComm)

    # TODO: use Spacy document for modComm to create a string.
    # Make sure to:
    #    * Insert "\n" between sentences.
    #    * Split tokens with spaces.
    #    * Write "/POS" after each token.
        for sent in utt.sents:
            for token in sent:
                if not str(token.lemma_).startswith('-'):
                    new_modComm += str(token.lemma_)
                else:
                    new_modComm += str(token)
                new_modComm += '/' + str(token.tag_)
                new_modComm += " "
            new_modComm = new_modComm[:-1]
            new_modComm += "\n"

    modComm = new_modComm
    return modComm


def main(args):
    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print( "Processing " + fullFile)

            data = json.load(open(fullFile))

            # TODO: select appropriate args.max lines
            # TODO: read those lines with something like `j = json.loads(line)`
            # TODO: choose to retain fields from those lines that are relevant to you
            # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...)
            # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
            # TODO: replace the 'body' field with the processed text
            # TODO: append the result to 'allOutput'
            start_line = args.ID[0] % len(data)
            end_line = args.max + start_line
            lines = []
            if end_line > len(data):
                end_line = len(data)
                end_line1 = end_line % len(data)
                lines = data[:end_line1] + data[start_line:]
            else:
                lines = data[start_line: end_line]
            for line in lines:
                j = json.loads(line)
                # rel_key = ['id', 'body', 'ups', 'downs','score', 'controversiality', 'subreddit', 'author']
                rel_key = ['id', 'body']
                dict_j = {key: j[key] for key in rel_key}
                dict_j["cat"] = file
                dict_j["body"] = preproc1(dict_j["body"],steps=[1,2,3,4,5])
                allOutput.append(dict_j)


    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='1002109470')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", type=int, help="The maximum number of comments to read from each file", default=10000)
    parser.add_argument("--a1_dir", help="The directory for A1. Should contain subdir data. Defaults to the directory for A1 on cdf.", default='/Users/zty-tom/Google/UOT/CSC401/A1')

    args = parser.parse_args()

    if (args.max > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)

    indir = os.path.join(args.a1_dir, 'data')
    main(args)
