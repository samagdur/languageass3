import os
import random
import time
import argparse
import string
from collections import defaultdict
import numpy as np
from sklearn.neighbors import NearestNeighbors

from tqdm import tqdm


"""
This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2020 by Dmytro Kalpakchi.
"""


class Word2Vec(object):
    def __init__(self, filenames, dimension=300, window_size=2, nsample=10,
                 learning_rate=0.025, epochs=3, use_corrected=True, use_lr_scheduling=True):
        """
        Constructs a new instance.
        
        :param      filenames:      A list of filenames to be used as the training material
        :param      dimension:      The dimensionality of the word embeddings
        :param      window_size:    The size of the context window
        :param      nsample:        The number of negative samples to be chosen
        :param      learning_rate:  The learning rate
        :param      epochs:         A number of epochs
        :param      use_corrected:  An indicator of whether a corrected unigram distribution should be used
        """
        self.__pad_word = '<pad>'
        self.__sources = filenames
        self.__H = dimension
        self.__lws = window_size
        self.__rws = window_size
        self.__C = self.__lws + self.__rws
        self.__init_lr = learning_rate
        self.__lr = learning_rate
        self.__nsample = nsample
        self.__epochs = epochs
        self.__nbrs = None
        self.__use_corrected = use_corrected
        self.__use_lr_scheduling = use_lr_scheduling

        self.__vocab = set()
        self.index_mapper = dict()
        self.word_mapper = dict()


    def init_params(self, W, w2i, i2w):
        self.__W = W
        self.__w2i = w2i
        self.__i2w = i2w

    ##
    ## @brief      A function cleaning the line from punctuation and digits
    ##
    ##             The function takes a line from the text file as a string,
    ##             removes all the punctuation and digits from it and returns
    ##             all words in the cleaned line.
    ##
    ## @param      line  The line of the text file to be cleaned
    ##
    ## @return     A list of words in a cleaned line
    ##
    def clean_line(self, line):
        # YOUR CODE HERE
        import re
        line_without_punctuation = re.sub(r'[^\w\s]', '', line)
        pattern = r'[0-9]'
        cleaned_string = re.sub(pattern, '', line_without_punctuation)
        return cleaned_string.split()


    ##
    ## @brief      A generator function providing one cleaned line at a time
    ##
    ##             This function reads every file from the source files line by
    ##             line and returns a special kind of iterator, called
    ##             generator, returning one cleaned line a time.
    ##
    ##             If you are unfamiliar with Python's generators, please read
    ##             more following these links:
    ## - https://docs.python.org/3/howto/functional.html#generators
    ## - https://wiki.python.org/moin/Generators
    ##
    ## @return     A generator yielding one cleaned line at a time
    ##
    def text_gen(self):
        for fname in self.__sources:
            with open(fname, encoding='utf8', errors='ignore') as f:
                for line in f:
                    yield self.clean_line(line)

    @property
    def vocab_size(self):
        return self.__V
        

    def build_vocabulary(self):
        # YOUR CODE HERE
        size_vocab = 0
        ctr = 0
        indexmapper = {}
        for text_line in self.text_gen():
            for one_word in text_line:
                indexmapper[ctr] = one_word
                ctr += 1
                self.__vocab.add(one_word)
                size_vocab_new = len(self.__vocab)
                if size_vocab_new != size_vocab:
                    self.index_mapper[one_word] = size_vocab_new
                    self.word_mapper[size_vocab_new] = one_word
                    size_vocab = size_vocab_new

    def calculate_unigram_distributions(self):
        counts = {}
        total_words = 0
        for text_line in self.text_gen():
            for one_word in text_line:
                try:
                    counts[one_word] += 1
                except KeyError:
                    counts[one_word] = 1
                total_words += 1
        unigram_probabilities = {key: val / total_words for key, val in counts.items()}

        self.unigram_probabilities = unigram_probabilities

        return unigram_probabilities

    def calculate_modified_unigram_distributions(self):
        counts = {}
        total_words = 0
        for text_line in self.text_gen():
            for one_word in text_line:
                try:
                    counts[one_word] += 1
                except KeyError:
                    counts[one_word] = 1
                total_words += 1

        modified_denominator = 0
        for word, val in counts.items():
            modified_denominator += val ** (3/4)

        unigram_probabilities_modified = {key: val**(3/4) / modified_denominator for key, val in counts.items()}

        self.unigram_probabilities_modified = unigram_probabilities_modified

        return unigram_probabilities_modified

        # self.write_vocabulary()

    # def clean_line(self, line):
    #     """
    #     The function takes a line from the text file as a string,
    #     removes all the punctuation and digits from it and returns
    #     all words in the cleaned line as a list
    #
    #     :param      line:  The line
    #     :type       line:  str
    #     """
    #     #
    #     # REPLACE WITH YOUR CODE HERE
    #     #
    #     return []
    #
    #
    # def text_gen(self):
    #     """
    #     A generator function providing one cleaned line at a time
    #
    #     This function reads every file from the source files line by
    #     line and returns a special kind of iterator, called
    #     generator, returning one cleaned line a time.
    #
    #     If you are unfamiliar with Python's generators, please read
    #     more following these links:
    #     - https://docs.python.org/3/howto/functional.html#generators
    #     - https://wiki.python.org/moin/Generators
    #     """
    #     for fname in self.__sources:
    #         with open(fname, encoding='utf8', errors='ignore') as f:
    #             for line in f:
    #                 yield self.clean_line(line)


    def get_context(self, sent, i):
        """
        Returns the context of the word `sent[i]` as a list of word indices
        
        :param      sent:  The sentence
        :type       sent:  list
        :param      i:     Index of the focus word in the sentence
        :type       i:     int
        """
        #
        # REPLACE WITH YOUR CODE
        #
        left_side = sent[max(i - self.__lws, 0): i]
        right_side = sent[i + 1: min(i + self.__rws, len(sent))]

        return [self.index_mapper[word] for word in left_side + right_side]


    def skipgram_data(self):
        """
        A function preparing data for a skipgram word2vec model in 3 stages:
        1) Build the maps between words and indexes and vice versa
        2) Calculate the unigram distribution and corrected unigram distribution
           (the latter according to Mikolov's article)
        3) Return a tuple containing two lists:
            a) list of focus words
            b) list of respective context words
        """
        self.build_vocabulary()
        self.calculate_unigram_distributions()
        self.calculate_modified_unigram_distributions()

        words, contexts = [], []

        indexword = {}
        wordtoindex = {}
        ctr = 0
        for text_line in self.text_gen():
            for idx, one_word in enumerate(text_line):
                words.append(self.index_mapper[one_word])
                word_context = self.get_context(sent=text_line, i=idx)
                contexts.append(word_context)
                indexword[ctr] = one_word
                wordtoindex[one_word] = ctr
                ctr += 1
        self.indexword = indexword
        return words, contexts, indexword, wordtoindex


    def sigmoid(self, x):
        """
        Computes a sigmoid function
        """
        return 1 / (1 + np.exp(-x))


    def negative_sampling(self, number, xb, pos, unigram_probability=True):
        """
        Sample a `number` of negatives examples with the words in `xb` and `pos` words being
        in the taboo list, i.e. those should be replaced if sampled.
        
        :param      number:     The number of negative examples to be sampled
        :type       number:     int
        :param      xb:         The index of the current focus word
        :type       xb:         int
        :param      pos:        The index of the current positive example
        :type       pos:        int
        """
        #
        # REPLACE WITH YOUR CODE
        #
        if unigram_probability:
            probabilities = [self.unigram_probabilities[x]
                             for idx, x in enumerate(self.unigram_probabilities.keys()) if self.index_mapper[x] not in [xb, *pos]]
        else:
            probabilities = [self.unigram_probabilities_modified[x]
                             for idx, x in enumerate(self.unigram_probabilities_modified.keys()) if self.index_mapper[x] not in [xb, *pos]]

        indexes = [x for x in self.index_mapper.values() if x not in [xb, *pos]]

        chosen_indexes = random.choices(indexes, weights=probabilities, k=number)

        return chosen_indexes


    # def train(self):
    #     """
    #     Performs the training of the word2vec skip-gram model
    #     """
    #     x, t, indexword, wordindex = self.skipgram_data()
    #     N = len(x)
    #
    #     self.__i2w = indexword
    #     self.__w2i = wordindex
    #     self.__V = N
    #     print("Dataset contains {} datapoints".format(N))
    #
    #     # REPLACE WITH YOUR RANDOM INITIALIZATION
    #     self.__W = np.random.rand(N, self.__H)
    #     self.__U = np.random.rand(N, self.__H)
    #
    #     # target_words, context_words_lst = self.skipgram_data()
    #
    #     wordsprocessed = 0
    #     lr = self.__lr
    #     for ep in range(self.__epochs):
    #         numbers = list(range(0, N))
    #         # Shuffle the list
    #         random.shuffle(numbers)
    #         for i_ in tqdm(range(N)):
    #             i = numbers[i_]
    #             word = target_words[i]
    #             context_words = context_words_lst[i]
    #             if not context_words:
    #                 continue
    #             negative_sampled_words = self.negative_sampling(number=self.__nsample, xb=word, pos=context_words)
    #
    #             gradient_target_word = np.zeros(self.__H)
    #
    #             target_context_words = self.__U[context_words]
    #             target_context_words_negative = self.__U[negative_sampled_words]
    #
    #             forward_pass_target = self.sigmoid(target_context_words @ self.__W[word]) - 1
    #             gradient_forward_ = forward_pass_target.reshape(-1, 1) * self.__W[word]
    #             self.__U[context_words] -= lr * gradient_forward_
    #
    #             gradient_target_word += np.mean(forward_pass_target[:, np.newaxis] * target_context_words, axis=0)
    #
    #             forward_pass_negative = self.sigmoid(target_context_words_negative @ self.__W[word])
    #             gradient_forward_negative = forward_pass_negative.reshape(-1, 1) * self.__W[word]
    #
    #             gradient_target_word += np.mean(forward_pass_negative[:, np.newaxis] * self.__W[word], axis=0)
    #             self.__U[negative_sampled_words] -= lr * gradient_forward_negative
    #
    #             self.__W[word] -= lr * gradient_target_word
    #
    #             wordsprocessed += 1
    #
    #             if self.__use_lr_scheduling:
    #                 lr = self.get_lr(lr=lr, wordsprocessed=wordsprocessed, totalwords=N, epochs=self.__epochs)

    def train(self):
        """
        Performs the training of the word2vec skip-gram model
        """
        x, t, indexword, wordindex = self.skipgram_data()
        N = len(x)

        self.__i2w = indexword
        self.__w2i = wordindex
        self.__V = N

        print("Dataset contains {} datapoints".format(N))

        # Initialisation with uniform distribution
        self.__W = np.random.uniform(size=(N, self.__H))
        self.__U = np.random.uniform(size=(N, self.__H))


        learning_rate = self.__lr

        for ep in range(self.__epochs):
            for i in tqdm(range(N)):

                if self.__use_lr_scheduling:
                    learning_rate = self.get_lr(lr=learning_rate, wordsprocessed=(ep + 1) * (i + 1),
                                                totalwords=N, epochs=self.__epochs)
                context_indices = t[i]

                for context_index in context_indices:
                    predicted_prob = self.sigmoid(self.__U[context_index].T.dot(self.__W[i]))
                    self.__W[i] -= learning_rate * self.__U[context_index].dot(predicted_prob - 1)
                    self.__U[context_index] -= learning_rate * self.__W[i].dot(predicted_prob - 1)

                negative_samples = self.negative_sampling(self.__nsample, i, context_indices)

                for negative_sample_index in negative_samples:
                    predicted_prob = self.sigmoid(self.__U[negative_sample_index].T.dot(self.__W[i]))
                    self.__W[i] -= learning_rate * self.__U[negative_sample_index].dot(predicted_prob)
                    self.__U[negative_sample_index] -= learning_rate * self.__W[i].dot(predicted_prob)

    def get_lr(self, lr, wordsprocessed, totalwords, epochs):
        if lr < self.__lr * 0.0001:
            return lr
        return self.__lr * (1 - (wordsprocessed / (totalwords + epochs + 1)))

    def find_nearest(self, words, metric):
        """
        Function returning k nearest neighbors with distances for each word in `words`
        
        We suggest using nearest neighbors implementation from scikit-learn 
        (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html). Check
        carefully their documentation regarding the parameters passed to the algorithm.
    
        To describe how the function operates, imagine you want to find 5 nearest neighbors for the words
        "Harry" and "Potter" using some distance metric `m`. 
        For that you would need to call `self.find_nearest(["Harry", "Potter"], k=5, metric='m')`.
        The output of the function would then be the following list of lists of tuples (LLT)
        (all words and distances are just example values):
    
        [[('Harry', 0.0), ('Hagrid', 0.07), ('Snape', 0.08), ('Dumbledore', 0.08), ('Hermione', 0.09)],
         [('Potter', 0.0), ('quickly', 0.21), ('asked', 0.22), ('lied', 0.23), ('okay', 0.24)]]
        
        The i-th element of the LLT would correspond to k nearest neighbors for the i-th word in the `words`
        list, provided as an argument. Each tuple contains a word and a similarity/distance metric.
        The tuples are sorted either by descending similarity or by ascending distance.
        
        :param      words:   Words for the nearest neighbors to be found
        :type       words:   list
        :param      metric:  The similarity/distance metric
        :type       metric:  string
        """
        #
        # REPLACE WITH YOUR CODE
        #
        neighborsmodel = NearestNeighbors(metric=metric)
        neighborsmodel.fit(self.__W)

        test_X = np.array(list(self.__W[self.index_mapper[word]] for word in words))
        # test_X = np.array([self.index_mapper[idx] for idx in indexes])

        kneighbors = neighborsmodel.kneighbors(X=test_X, n_neighbors=5, return_distance=True)

        # words = [list(self._RandomIndexing__cv.keys())[idx] for idx in kneighbors[1]]
        # lens = kneighbors[0]
        tbret = []
        for idx, word in enumerate(words):
            words = [self.indexword[idx] for idx in kneighbors[1][idx]]
            lens = kneighbors[1][idx]
            one_entry = [(word, len_) for word, len_ in zip(words, lens)]
            tbret.append(one_entry)

        return tbret

    def write_to_file(self):
        """
        Write the model to a file `w2v.txt`
        """
       # try:
        with open("w2v.txt", 'w') as f:
            # to store target word matrix
            W = self._Word2Vec__W
            f.write("{} {}\n".format(self.__V, self.__H))
            for i, w in enumerate(self.indexword.values()):
                f.write(w + " " + " ".join(map(lambda x: "{0:.6f}".format(x), W[i, :])) + "\n")


    @classmethod
    def load(cls, fname):
        """
        Load the word2vec model from a file `fname`
        """
        w2v = None
        try:
            with open(fname, 'r') as f:
                V, H = (int(a) for a in next(f).split())
                w2v = cls([], dimension=H)

                W, i2w, w2i = np.zeros((V, H)), [], {}
                for i, line in enumerate(f):
                    parts = line.split()
                    word = parts[0].strip()
                    w2i[word] = i
                    W[i] = list(map(float, parts[1:]))
                    i2w.append(word)

                w2v.init_params(W, w2i, i2w)
        except:
            print("Error: failing to load the model to the file")
        return w2v


    def interact(self):
        """
        Interactive mode allowing a user to enter a number of space-separated words and
        get nearest 5 nearest neighbors for every word in the vector space
        """
        print("PRESS q FOR EXIT")
        text = input('> ')
        while text != 'q':
            text = text.split()
            neighbors = self.find_nearest(text, 'cosine')

            for w, n in zip(text, neighbors):
                print("Neighbors for {}: {}".format(w, n))
            text = input('> ')


    def train_and_persist(self):
        """
        Main function call to train word embeddings and being able to input
        example words interactively
        """
        self.train()
        self.write_to_file()
        self.interact()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='word2vec embeddings toolkit')
    parser.add_argument('-t', '--text', default='harry_potter_1.txt',
                        help='Comma-separated source text files to be trained on')
    parser.add_argument('-s', '--save', default='w2v.txt', help='Filename where word vectors are saved')
    parser.add_argument('-d', '--dimension', default=50, help='Dimensionality of word vectors')
    parser.add_argument('-ws', '--window-size', default=2, help='Context window size')
    parser.add_argument('-neg', '--negative_sample', default=10, help='Number of negative samples')
    parser.add_argument('-lr', '--learning-rate', default=0.025, help='Initial learning rate')
    parser.add_argument('-e', '--epochs', default=3, help='Number of epochs')
    parser.add_argument('-uc', '--use-corrected', action='store_true', default=True,
                        help="""An indicator of whether to use a corrected unigram distribution
                                for negative sampling""")
    parser.add_argument('-ulrs', '--use-learning-rate-scheduling', action='store_true', default=True,
                        help="An indicator of whether using the learning rate scheduling")
    args = parser.parse_args()

    if os.path.exists(args.save):
        w2v = Word2Vec.load(args.save)
        # Word2Vec(
        #     args.text.split(','), dimension=args.dimension, window_size=args.window_size,
        #     nsample=args.negative_sample, learning_rate=args.learning_rate, epochs=args.epochs,
        #     use_corrected=args.use_corrected, use_lr_scheduling=args.use_learning_rate_scheduling
        # ).build_vocabulary()
        if w2v:
            w2v.interact()
    else:
        w2v = Word2Vec(
            args.text.split(','), dimension=args.dimension, window_size=args.window_size,
            nsample=args.negative_sample, learning_rate=args.learning_rate, epochs=args.epochs,
            use_corrected=args.use_corrected, use_lr_scheduling=args.use_learning_rate_scheduling
        )
        w2v.train_and_persist()
