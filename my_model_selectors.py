import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences
import sys #rima


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    #p was calculated as dim(A)+dim(B)+dim(pi)
    #where A =state transition probability matrix
    #B = state output probability matrix
    #pi =initial state distribution
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            Best_BIC=float('inf')
            for i in range(self.min_n_components,self.max_n_components +1):
                hmm_model=self.base_model(i)
                try:
                    #old (wrong) BIC calculation
                    #BIC=- 2 * hmm_model.score(self.X, self.lengths) + ((i**2)+(2*i)) * np.log(sum(self.lengths))
                    #Correction on 10/10/2017

                    BIC=- 2 * hmm_model.score(self.X, self.lengths) + (i**2+2*i*self.X.shape[1] -1 ) * np.log(sum(self.lengths))
                    if BIC < Best_BIC:
                        Best_BIC=BIC
                        best_num_components=i
                except:
                    pass
            return self.base_model(best_num_components)
        except:
            pass

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            Best_DIC=float('-inf')
            for i in range(self.min_n_components,self.max_n_components +1):
                try:
                    hmm_model=self.base_model(i)
                    other_words_score=[]
                    for word in self.words:
                        if word !=self.this_word:
                            X_competing_word, lengths_competing_word =self.hwords[word]
                            other_words_score.append(hmm_model.score(X_competing_word, lengths_competing_word))
                        avg_other_words_score= sum(other_words_score)/len(other_words_score)
                        DIC=hmm_model.score(self.X, self.lengths)-avg_other_words_score
                        if DIC > Best_DIC:
                            Best_DIC=DIC
                            best_num_components=i
                except:
                    pass
            return self.base_model(best_num_components)
        except:
            pass

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            best_likelihood=float('-inf')
            if (len(self.words[self.this_word]) < 3) :
                #In case we don't have many examples to be used for training/test
                #is it acceptable to choose the simplest of all models to avoid overfitting?
                return self.base_model(self.min_n_components)
            else:
                kf = KFold(n_splits = 3, shuffle = False, random_state = None)
            for i in range(self.min_n_components,self.max_n_components +1):
                likelihood_list=[]
                for cv_train_idx, cv_test_idx in kf.split(self.sequences):
                    self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
                    X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                    hmm_model= self.base_model(i)
                    likelihood_list.append(hmm_model.score(X_test, lengths_test))
                avg_likelihood=sum(likelihood_list)/float(len(likelihood_list))
                if avg_likelihood > best_likelihood:
                    best_likelihood=avg_likelihood
                    best_num_components =i
            return self.base_model(best_num_components)
        except:
            pass
