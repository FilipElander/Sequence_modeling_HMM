from Key import Key
import math
import numpy as np
import codecs
import argparse


class ViterbiTrigramDecoder(object):
    """
    This class implements Viterbi decoding using trigram probabilities in order
    to correct keystroke errors.
    """
    def init_a(self, filename):
        """
        Reads the trigram probabilities (the 'A' matrix) from a file.
        """
        with codecs.open(filename, 'r', 'utf-8') as f:
            for line in f:
                i, j, k, d = [func(x) for func, x in zip([int, int, int, float], line.strip().split(' '))]
                self.a[i][j][k] = d


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
            self.b[i][i] = np.log((10 - len(cs))/10.0)


    # ------------------------------------------------------



    def viterbi(self, s):
        """
        Performs the Viterbi decoding and returns the most likely
        string.
        """
        # First turn chars to integers, so that 'a' is represented by 0,
        # 'b' by 1, and so on.
        index = [Key.char_to_index(x) for x in s]

        # The Viterbi matrices
        self.v = np.zeros((len(s), Key.NUMBER_OF_CHARS, Key.NUMBER_OF_CHARS), dtype='double') #'double'
        self.v[:,:,:] = -float("inf")
        self.backptr = np.zeros((len(s), Key.NUMBER_OF_CHARS, Key.NUMBER_OF_CHARS), dtype='int')

        # initierar första bokstaven som följer dubbel start_end

        # initierar andra bokstaven som alltid har en start_end


        # Initialization

        # YOUR CODE HERE
        self.v[0,Key.START_END,:] = self.a[Key.START_END,Key.START_END,:] + self.b[index[0],:]
        self.backptr[0,Key.START_END,:] = Key.START_END

        # Induction step

        # YOUR CODE HERE

         #tillstånd
        for t in range(1,len(s)): #bokstav i följd
            for k in range(Key.NUMBER_OF_CHARS): # tillstånd
                for j in range(Key.NUMBER_OF_CHARS): # en innan (raden i t-1)
                    holder = np.zeros(Key.NUMBER_OF_CHARS)
                    holder[:] = -float("inf")
                    for i in range(Key.NUMBER_OF_CHARS): # två innan (raden i v(t-1))
                        holder[i] = self.v[t-1,i,j] + self.a[i,j,k] + self.b[k,index[t]]
                    self.v[t,j,k] = max(holder)
                    self.backptr[t,j,k] = np.argmax(holder)



        res = ''

        start =np.amax(self.v[len(s)-1])
        print(start)
        #print(np.where(self.v[len(s)-1] == start ))
        rad2col,pekad = np.where(self.v[len(s)-1] == start)
        rad2col = int(rad2col)
        for steg in range(len(index)-1,-1,-1):
            bokstav = Key.index_to_char(pekad)
            res = bokstav + res
            val2row = self.backptr[steg,rad2col,pekad]
            pekad = rad2col
            rad2col = val2row

        
        return res





    # ------------------------------------------------------



    def __init__(self, filename=None):
        """
        Constructor: Initializes the A and B matrices.
        """
        # The trellis used for Viterbi decoding. The first index is the time step.
        self.v = None

        # The trigram stats.
        self.a = np.zeros((Key.NUMBER_OF_CHARS, Key.NUMBER_OF_CHARS, Key.NUMBER_OF_CHARS), dtype='double')

        # The observation matrix.
        self.b = np.zeros((Key.NUMBER_OF_CHARS, Key.NUMBER_OF_CHARS), dtype='double')

        # Pointers to retrieve the topmost hypothesis.
        backptr = None

        if filename: self.init_a(filename)
        self.init_b()



    # ------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description='ViterbiTrigram decoder')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', '-f', type=str, help='decode the contents of a file')
    group.add_argument('--string', '-s', type=str, help='decode a string')
    parser.add_argument('--probs', '-p', type=str,  required=True, help='trigram probabilities file')


    arguments = parser.parse_args()

    if arguments.file:
        with codecs.open(arguments.file, 'r', 'utf-8') as f:
            s1 = f.read().replace('\n', '')
    elif arguments.string:
        s1 = arguments.string

    # Give the filename of the trigram probabilities as a command line argument
    d = ViterbiTrigramDecoder(arguments.probs)

    # Append two extra "END" symbols to the input string, to indicate end of sentence.
    result = d.viterbi(s1 + Key.index_to_char(Key.START_END) + Key.index_to_char(Key.START_END))
    print(result)
   

if __name__ == "__main__":
    main()
