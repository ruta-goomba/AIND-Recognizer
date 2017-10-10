import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


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
    where L is the likelihood of the fitted model, p is the number of parameters,
    and N is the number of data points
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # TODO implement model selection based on BIC scores
        best_num_components = self.min_n_components
        best_score = float("inf")
        for n_comp in range(self.min_n_components, self.max_n_components):
            try:
                model = self.base_model(n_comp)
                logL = model.score(self.X, self.lengths)
                p = (model.startprob_.size - 1) + (model.transmat_.size - 1) + model.means_.size + model.covars_.diagonal().size
                bic_score = -2*logL + p*np.log(np.sum(self.lengths))
                if bic_score < best_score:
                    best_num_components = n_comp
                    best_score = bic_score
            except:
                continue
        return self.base_model(best_num_components)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_num_components = self.min_n_components
        best_score = float("-inf")
        for n_comp in range(self.min_n_components, self.max_n_components):
            try:
                model = self.base_model(n_comp)
                logL = model.score(self.X, self.lengths)
                all_scores = [model.score(self.hwords[word][0], self.hwords[word][1]) for word in self.words if word != self.this_word]
                dic_score = logL - np.mean(all_scores)
                if dic_score > best_score:
                    best_num_components = n_comp
                    best_score = dic_score
            except:
                continue
        return self.base_model(best_num_components)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        best_num_components = self.min_n_components
        best_score = float("-inf")
        for n_comp in range(self.min_n_components, self.max_n_components):
            try:
                test_score = float("inf")
                split_method = KFold()
                all_scores = []
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    train_X, train_lengths = combine_sequences(cv_train_idx, self.sequences)
                    test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)
                    model = GaussianHMM(n_components=n_comp, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(train_X, train_lengths)
                    all_scores.append(model.score(test_X, test_lengths))
                logP_avg = np.mean(all_scores)
                if logP_avg > best_score:
                    best_num_components = n_comp
                    best_score = logP_avg
            except:
                continue
        return self.base_model(best_num_components)
