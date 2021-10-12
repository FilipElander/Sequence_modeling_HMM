from Key import Key
import math
import numpy as np
import codecs
import argparse



class ViterbiBigramDecoder(object):
    """
    This class implements Viterbi decoding using bigram probabilities in order
    to correct keystroke errors.
    """
    def init_a(self, filename):
        """
        Reads the bigram probabilities (the 'A' matrix) from a file.
        """
        with codecs.open(filename, 'r', 'utf-8') as f:
            for line in f:
                i, j, d = [func(x) for func, x in zip([int, int, float], line.strip().split(' '))]
                self.a[i][j] = d


    # ------------------------------------------------------


    def init_b(self):
        """
        Initializes the observation probabilities (the 'B' matrix).
        """
        for i in range(Key.NUMBER_OF_CHARS):
            cs = Key.neighbour[i]

            # Initialize all log-probabilities to some small value.
            for j in range(Key.NUMBER_OF_CHARS):
                self.b[i][j] = -float("inf")

            # All neighbouring keys are assigned the probability 0.1
            for j in range(len(cs)):
                self.b[i][Key.char_to_index(cs[j])] = math.log( 0.1 )

            # The remainder of the probability mass is given to the correct key.
            self.b[i][i] = math.log((10 - len(cs))/10.0)


    # ------------------------------------------------------



    def viterbi(self, s):
        """
        Performs the Viterbi decoding and returns the most likely
        string.
        """
        # First turn chars to integers, so that 'a' is represented by 0,
        # 'b' by 1, and so on.
        index = [Key.char_to_index(x) for x in s]
        #index är hela textstycket översatt från bokstäver till vilken siffra den har och space = 26
        #print(self.b)

        # The Viterbi matrices
        self.v = np.zeros((len(s), Key.NUMBER_OF_CHARS)) # class float 64bits
        self.v[:,:] = -float("inf")  # exp(-inf) = 0 (fyller viterbi matrisen med P(exp) nollor
        self.backptr = np.zeros((len(s) + 1, Key.NUMBER_OF_CHARS), dtype='int') # class int 64 bit

        # Initialization
        self.backptr[0,:] = Key.START_END
        self.v[0,:] = self.a[Key.START_END,:] + self.b[index[0],:]

        # YOUR CODE HERE
        # Induction/recursion step

        for step in range(1,len(index)):
            for state in range(Key.NUMBER_OF_CHARS):
                holder = np.zeros(Key.NUMBER_OF_CHARS)

                for preV in range(Key.NUMBER_OF_CHARS):
                    holder[preV] = self.v[step-1][preV] +  self.a[preV][state] + self.b[state][index[step]]

                self.v[step][state] = max(holder)
                self.backptr[step+1][state] = np.argmax(holder)


        # REPLACE THE LINE BELOW WITH YOUR CODE

        res = ''


        pekad = np.argmax(self.v[len(index)-1])
        for steg in range(len(index),0,-1):
            bokstav = Key.index_to_char(pekad)
            res = bokstav + res
            pekad = self.backptr[steg][pekad]


        return res


    # ------------------------------------------------------



    def __init__(self, filename=None):
        """
        Constructor: Initializes the A and B matrices.
        """
        # The trellis used for Viterbi decoding. The first index is the time step.
        self.v = None

        # The bigram stats.
        self.a = np.zeros((Key.NUMBER_OF_CHARS, Key.NUMBER_OF_CHARS))

        # The observation matrix.
        self.b = np.zeros((Key.NUMBER_OF_CHARS, Key.NUMBER_OF_CHARS))

        # Pointers to retrieve the topmost hypothesis.
        backptr = None

        if filename: self.init_a(filename)
        self.init_b()



    # ------------------------------------------------------


def main():

    parser = argparse.ArgumentParser(description='ViterbiBigramDecoder')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', '-f', type=str, help='decode the contents of a file')
    group.add_argument('--string', '-s', type=str, help='decode a string')
    parser.add_argument('--probs', '-p', type=str,  required=True, help='bigram probabilities file')


    arguments = parser.parse_args()

    if arguments.file:
        with codecs.open(arguments.file, 'r', 'utf-8') as f:
            s1 = f.read().replace('\n', '')
    elif arguments.string:
        s1 = arguments.string

    # Give the filename of the bigram probabilities as a command line argument
    d = ViterbiBigramDecoder(arguments.probs)

    # Append an extra "END" symbol to the input string, to indicate end of sentence.
    result = d.viterbi(s1 + Key.index_to_char(Key.START_END))
    print(result)



if __name__ == "__main__":
    main()
