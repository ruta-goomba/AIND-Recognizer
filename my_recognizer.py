import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    all_xlengths = test_set.get_all_Xlengths()
    # iterate through test set items
    for key in all_xlengths:
        this_X,this_lengths = test_set.get_item_Xlengths(key)
        best_score = float("-inf")
        best_guess_word = ""
        probs = {}
        # iterate through models
        for model_key in models.keys():
            try:
                model_score = models[model_key].score(this_X,this_lengths)
                probs[model_key] = model_score
                if model_score > best_score:
                    best_score = model_score
                    best_guess_word = model_key
            except:
                probs[model_key] = float("-inf")
                continue
        guesses.append(best_guess_word)
        probabilities.append(probs)
    return probabilities, guesses
