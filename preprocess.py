import pandas as pd


""" This program processes the annotated data into lexicon format. """

def anno_to_lexicon(annofile):
    lexicons = {}
    lexicons_neg = {}
    df_anno = pd.read_excel(annofile)
    for index, row in df_anno.iterrows():
        if not isinstance(row['Standard Symptom'], str):
            continue
        symExps = row['Symptom Expressions'].strip('$$$').split('$$$')
        stdSyms = row['Standard Symptom'].strip('$$$').split('$$$')
        symCuis = row['Symptom CUIs'].strip('$$$').split('$$$')
        negFlag = row['Negation Flag'].strip('$$$').split('$$$')

        for i in range(len(symExps)):
            if negFlag[i] == '1':
                print(symExps[i], stdSyms[i], symCuis[i])
                lexicons_neg[symExps[i]] = (stdSyms[i], symCuis[i])
            else:
                lexicons[symExps[i]] = (stdSyms[i], symCuis[i])

    print(lexicons_neg)
    print(lexicons)
    return lexicons, lexicons_neg


def output_dict(lexdic, outfile):
    out = open(outfile, 'w')
    for k, v in lexdic.items():
        out.write('\t'.join(v) + '\t' + k + '\n')



annofile = 's4.xlsx'
lexicons, lexicons_neg = anno_to_lexicon(annofile)
output_dict(lexicons, 'tmp1')
output_dict(lexicons_neg, 'tmp_neg')