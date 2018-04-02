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
    for word_id in range(0, len(test_set.get_all_Xlengths())):
        word_probability = {}
        feature_sequences, lengths = test_set.get_item_Xlengths(word_id)
        for word, model in models.items():
                try:
                    score = model.score(feature_sequences, lengths)
                    word_probability[word] = score
                except:
                    pass
        probabilities.append(word_probability)
        guessed_word=max(word_probability, key = word_probability.get)
        #guesses.append(guessed_word)
        guesses.append(''.join( c for c in guessed_word if  c not in '0123456789' ))#Get rid of digits end of recognized word
    return probabilities, guesses
