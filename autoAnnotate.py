import os
import pandas as pd
import itertools
from nltk.tokenize import word_tokenize
import re
import Levenshtein
from fuzzywuzzy import fuzz

import EvaluationScript as es
import IAA_Calculator as iaacal


class covidSymptomDetection:
    """ This class detects COVID symptoms from the social media data and generates sequence symptom annotations. """

    def __init__(self):
        self.symplexicon = {}
        self.negtrigs = []
        self.threshold_lev = 0.88
        self.threshold_fuzz = 95

    def fit(self, filepath_symlex='./COVID-Twitter-Symptom-Lexicon.txt', filepath_negtrigs='./neg_trigs.txt'):
        self.symplexicon = symptomlexicon_to_dict(filepath_symlex)
        self.negtrigs = negtrigs_to_list(filepath_negtrigs)

    def set_threshold(self, threshold):
        self.threshold_lev = threshold

    def set_ratio(self, ratio):
        self.threshold_fuzz = ratio

    def annotate(self, posts='./s4.xlsx', outfile='.outputs/autoAnnotations.xlsx'):
        df_anno = pd.DataFrame(columns=['ID', 'TEXT', 'Symptom Expressions', 'Symptom CUIs', 'Negation Flag'])
        df_posts = pd.read_excel(posts)
        for index, row in df_posts.iterrows():
            if not isinstance(row['TEXT'], str):
                continue
            text = row['TEXT'].lower()
            symptoms = find_symptoms(text, self.symplexicon, self.threshold_lev, self.threshold_fuzz)
            # print(symptoms)
            symptoms = check_negation(text, self.negtrigs, symptoms)

            # remove negation conflicts
            # rms = []
            # for (idx_s, idx_t), (symp, cui, negflag) in symptoms.items():
            #     if negflag == 1:
            #         for (i_s, i_t), (s, c, n) in symptoms.items():
            #             if c == cui and n == 0:
            #                 rms.append((idx_s, idx_t))
            #                 break
            # for f in rms:
            #     del symptoms[f]

            # print(symptoms)
            df_anno.loc[index, 'ID'] = row['ID']
            df_anno.loc[index, 'TEXT'] = row['TEXT']
            df_anno.loc[index, 'Symptom Expressions'] = '$$$' + '$$$'.join([x[0] for x in symptoms.values()]) + '$$$'
            df_anno.loc[index, 'Symptom CUIs'] = '$$$' + '$$$'.join([x[1] for x in symptoms.values()]) + '$$$'
            df_anno.loc[index, 'Negation Flag'] = '$$$' + '$$$'.join([str(x[2]) for x in symptoms.values()]) + '$$$'

        df_anno.to_excel(outfile, index=False, header=True)



def symptomlexicon_to_dict(filepath_symlex):
    """ This function reads the symtom lexicon file as a dictionary,
    in which the key is the symptom expression and the value is the CUI code. """
    file_symlex = open(filepath_symlex)
    # file_symlex2 = open(filepath_symlex2)
    symplexicon = {}
    for line in file_symlex:
        s = line.strip().split('\t')
        symplexicon[s[2].lower()] = s[1]
    # for line in file_symlex2:
    #     s = line.strip().split('\t')
    #     symplexicon[s[0].lower()] = s[1]
    # print(symplexicon)
    return symplexicon


def negtrigs_to_list(filepath_negtrigs):
    """ This function reads the negation file as a list. """
    negtrigs = []
    for line in open(filepath_negtrigs):
        negtrigs.append(line.strip().lower())
    # print(negtrigs)
    return negtrigs


def find_symptoms(text, symplexicon, threshold_lev, threshold_fuzz):
    """ This function searches for symptoms in a text using Levenshtein distance. """
    symptoms = {}   # dictionary to store the indexes of symptoms:
    # key = (start, end), value = (symptom expressions, symptom CUIs, negation flag)
    words = word_tokenize(text)
    # print(words)
    for symp, cui in symplexicon.items(): # for each standard symptom in the symptom lexicon
        # find exact matches
        for match in re.finditer(r'\b' + symp + r'\b', text):
            idx_start = len(word_tokenize(text[:match.start()]))
            idx_end = len(word_tokenize(text[:match.end()])) - 1
            symptoms[(idx_start, idx_end)] = (symp, cui, 0)
        # find fuzzy matches
        window_size = len(symp.split())
        for (idx, window) in run_sliding_window_through_text(words, window_size):
            windowStr = ' '.join(window)
            # using levenshtein distance
            similar_score = Levenshtein.ratio(windowStr, symp)
            if similar_score >= threshold_lev:
                symptoms[(idx, idx+window_size-1)] = (windowStr, cui, 0)
            # # using fuzz ratio
            # if fuzz.ratio(windowStr, symp) >= threshold_fuzz or\
            #         fuzz.partial_ratio(windowStr, symp) >= threshold_fuzz or\
            #         fuzz.token_sort_ratio(windowStr, symp) >= threshold_fuzz:
            #     symptoms[(idx, idx+window_size-1)] = (windowStr, cui, 0)


    # print(symptoms)
    return symptoms


def run_sliding_window_through_text(words, window_size):
    """
    Generate a window sliding through a sequence of words
    """
    word_iterator = iter(words)  # creates an object which can be iterated one element at a time
    word_window = tuple(itertools.islice(word_iterator,
                                         window_size))  # islice() makes an iterator that returns selected elements from the the word_iterator
    idx = 0
    yield (idx, word_window)
    # now to move the window forward, one word at a time
    for w in word_iterator:
        idx += 1
        word_window = word_window[1:] + (w,)
        yield (idx, word_window)


def check_negation(text, negtrigs, symptoms):
    nearest_neg = dict.fromkeys(symptoms.keys(), 100)
    for neg in negtrigs:    # for each negation indicator
        for match in re.finditer(r'\b' + neg + r'\b', text):
            mat = re.search(r'[;?!\\.]', text[match.end():])    # end the negation scope
            if mat != None:
                text_after = text[match.end():match.end()+mat.end()]
            else:
                text_after = text[match.end():]
            words_before = word_tokenize(text[:match.end()])    # words before the negation indicator
            idx_neg = len(words_before) - 1    # index of the last word of neg
            words_after = word_tokenize(text_after) # words after the negation indicator
            for (idx_s, idx_t), (symp, cui, negflag) in symptoms.items():
                if idx_s > idx_neg and idx_s <= idx_neg+3 and idx_s <= idx_neg+len(words_after):
                # the first word of the symptom is within the negation scope
                    if idx_s - idx_neg < nearest_neg[(idx_s, idx_t)]:
                    # find the nearest negation indicator
                        neg_symp = neg + ' ' + symp
                        symptoms[(idx_s, idx_t)] = (neg_symp, cui, 1)
                        nearest_neg[(idx_s, idx_t)] = idx_s - idx_neg

    return symptoms


def create_dir(dirName):
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    # else:
    #     print("Directory ", dirName, " already exists")


def train1():
    filepath_goldStd = 'Assignment1GoldStandardSet.xlsx'
    filepath_labeled = 's4.xlsx'

    # to find suitable threshold for Levenshtein, and suitable negation scope
    model = covidSymptomDetection()
    model.fit()

    for lev in [0.75, 0.8, 0.85, 0.87, 0.9]:
    # for lev in [0.87, 0.88, 0.89]:
        print('==== File: {} == Threshold: {} == Scope: {} ===='.format(filepath_labeled, lev, 4))
        filepath_out = './outputs/autoAnnos_s4_scope4_lev{}.xlsx'.format(lev)
        model.set_threshold(lev)
        model.annotate(filepath_labeled, filepath_out)

        (tp, fp, fn), recall, precision, f1 = es.evaluate(filepath_labeled, filepath_out)

    # # for lev in [0.75, 0.8, 0.85, 0.87, 0.9]:
    # for lev in [0.87, 0.88, 0.89]:
    #     print('==== File: {} == Threshold: {} == Scope: {} ===='.format(filepath_goldStd, lev, 4))
    #     filepath_out = './outputs/autoAnnos_gold_scope4_lev{}.xlsx'.format(lev)
    #     model.set_threshold(lev)
    #     model.annotate(filepath_goldStd, filepath_out)
    #
    #     (tp, fp, fn), recall, precision, f1 = es.evaluate(filepath_goldStd, filepath_out)


def train2():
    filepath_symlex = './COVID-Twitter-Symptom-Lexicon.txt'
    filepath_labeled = 's4.xlsx'
    filepath_goldStd = 'Assignment1GoldStandardSet.xlsx'

    # to use different negation indicators, lev = 0.87
    filepath_negtrigs2 = './neg_trigs2.txt'
    filepath_out = './outputs/autoAnnos_s4_negtrigs2_scope3_lev0.87.xlsx'
    model = covidSymptomDetection()
    model.fit(filepath_negtrigs2)
    model.annotate(filepath_labeled, filepath_out)

    (tp, fp, fn), recall, precision, f1 = es.evaluate(filepath_labeled, filepath_out)


    # 'neg_trigs2.txt'
    # to remove negation conflicts, to use standard symptoms, to change lev
    filepath_negtrigs2 = './neg_trigs2.txt'
    filepath_symlex2 = './COVID-Twitter-Symptom-Lexicon_new.txt'
    filepath_out = './outputs/autoAnnos_s4_negtrigs2_scope3_lev0.87_2lexicons.xlsx'
    model = covidSymptomDetection()
    model.set_threshold(0.87)
    model.fit(filepath_symlex2, filepath_negtrigs2)
    model.annotate(filepath_goldStd, filepath_out)

    (tp, fp, fn), recall, precision, f1 = es.evaluate(filepath_goldStd, filepath_out)


def train3():
    filepath_goldStd = 'Assignment1GoldStandardSet.xlsx'
    filepath_negtrigs2 = './neg_trigs2.txt'
    filepath_symlex3 = './COVID-Twitter-Symptom-Lexicon_new2.txt'
    filepath_out = './outputs/autoAnnos_gold_negtrigs2_scope3_lev0.88_3lexicons.xlsx'

    # to also use my annotated data as a part of symptom lexicons
    model = covidSymptomDetection()
    model.fit(filepath_symlex3, filepath_negtrigs2)
    model.set_threshold(0.88)
    model.annotate(filepath_goldStd, filepath_out)

    (tp, fp, fn), recall, precision, f1 = es.evaluate(filepath_goldStd, filepath_out)


def main():
    filepath_symlex = './COVID-Twitter-Symptom-Lexicon.txt'
    filepath_negtrigs = './neg_trigs.txt'
    filepath_goldStd = 'Assignment1GoldStandardSet.xlsx'
    filepath_labeled = 's4.xlsx'
    create_dir('./outputs/')
    filepath_out = './outputs/autoAnnotations.xlsx'


    # train to find optimal parameters and rules
    # ============================================================= #

    # train1()
    # # to use:
    # # lev threshold: 0.88 (0.85-0.9); basic negation scope: 3;

    # train2()
    # # to use:
    # # neg_trigs2, 2lexicons

    # train3()
    # # to use:
    # # neg_trigs2, 3lexicons


    # Annotation Agreement:
    # ============================================================= #
    # # used annotations for gold standard:
    # # 'outputs/autoAnnos_gold_negtrigs2_scope3_lev0.88_3lexicons.xlsx'
    annofile = 'outputs/autoAnnos_gold_negtrigs2_scope3_lev0.88_3lexicons.xlsx'
    iaacal.run(filepath_goldStd, annofile)
    # # cohen_kappa_score = 0.7349194828397553


    # Predict for unlabeled data (negtrigs2_scope3_lev0.88_3lexicons)
    # ============================================================= #
    # filepath_negtrigs2 = './neg_trigs2.txt'
    # filepath_symlex3 = './COVID-Twitter-Symptom-Lexicon_new2.txt'
    # filepath_posts = './UnlabeledSet2.xlsx'
    # filepath_out = './autoAnnotations_unlabeledSet2.xlsx'
    # model = covidSymptomDetection()
    # model.fit(filepath_symlex3, filepath_negtrigs2)
    # model.annotate(filepath_posts, filepath_out)


if __name__ == '__main__':
    main()