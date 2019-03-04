import numpy as np
import tensorflow as tf
import copy
import time

class PolyFiller():
    ''' initilises polyphonic model and can generate compositions with specific length
    and specific density/register constraints. It can also fill the next N notes of an
    input/given composition and return the new density/register values '''

    def __init__(self, melody_bin, harmony_options_bin, gct_completion):
        print('Initialising PolyFiller')
        # retrieve saved data
        d = np.load('CM_NN_VL/saved_data/training_data.npz') # CM_NN_VL/ is necessary
        # keep a standard seed
        # self.seed = np.zeros( (128,16) ).reshape( (1, 16, 128) )
        # self.seed_2D = np.zeros( (128 , 16) )
        # cross out gct pitches that belong to the melody
        # print('Before: ', gct_completion)
        # for i in range( melody_bin.shape[1] ):
        #     tmp_idx = np.nonzero( melody_bin[:,i] )[0][0]
        #     gct_completion[ tmp_idx%12, i ] = 0
        # print('After: ', gct_completion)
        # assign inputs
        self.melody_bin = melody_bin
        self.harmony_options_bin = harmony_options_bin
        self.gct_completion = gct_completion
        self.all_matrices = d['all_matrices']
        self.seed = d['seed']
        # in the "current" position of the seed, append the current melody note
        self.seed[0,15,:128] = self.melody_bin[:,0]
        # test batch generation example
        self.max_len = 16
        # composition length: to add an additional seed with bidirectional LSTM?
        self.composition_length = harmony_options_bin.shape[1]
        self.batch_size = 320
        self.step = 1
        self.input_rows = self.all_matrices.shape[0]
        self.output_rows = 128
        self.num_units = [128, 64]
        self.learning_rate = 0.001
        self.epochs = 5000
        self.temperature = 0.5
        # make initial piano roll matrix
        self.matrix = copy.deepcopy( self.melody_bin )
        # the following is to be placed with bidirectional LSTM
        # self.matrix = np.hstack( (self.matrix, self.seed) )
        # initialise model
        tf.reset_default_graph()
        self.x = tf.placeholder("float", [None, self.max_len, self.input_rows])
        self.y = tf.placeholder("float", [None, self.output_rows])
        self.weight = tf.Variable(tf.random_normal([self.num_units[-1], self.output_rows]))
        self.bias = tf.Variable(tf.random_normal([self.output_rows]))
        self.prediction = self.rnn(self.x, self.weight, self.bias, self.input_rows)
        self.dist = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.prediction, labels=self.y)
        self.cost = tf.reduce_mean(self.dist)
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, 'CM_NN_VL/saved_model/file.ckpt')
        # keep the predictions matrix - same size as matrix but with 0s
        # in the place of existing notes and prob values to all others
        # self.predictions = np.zeros( (self.matrix.shape[0], self.matrix.shape[1]) )
    # end constructor
    def run_NN_VL(self):
        # run first "bass note" round
        # self.fill_bass_VL()
        # then run mid notes
        self.fill_mid_VL()
    # end run_NN_VL
    def fill_bass_VL(self):
        small_value = -1000000000
        tmpMat = copy.deepcopy(self.seed)
        for i in range(self.matrix.shape[1]):
            # roll tmpMat according to matrix
            if i > 0:
                # remove_fist_char = self.seed[:,1:,:]
                remove_fist_char = tmpMat[:,1:,:]
                new_input = np.append( self.matrix[:,i] , self.matrix[:,i-1] )
                tmpMat = np.append(remove_fist_char, np.reshape(new_input, [1, 1, self.input_rows]), axis=1)
            # make next prediction
            predicted = self.sess.run([self.prediction], feed_dict = {self.x:tmpMat})
            # currate predictions
            predicted = np.asarray(predicted[0]).astype('float64')[0]
            # get the next note from the predictions
            # zero-out notes that already exist
            predicted[ self.matrix[:,i]>0 ] = small_value
            # zero-out notes that disagree with GCT pcs
            predicted[ self.harmony_options_bin[:,i] == 0 ] = small_value
            # get element that corresponds to the maximum index
            tmp_idx = np.argmax(predicted)
            # print('predicted: ', predicted)
            self.matrix[tmp_idx, i] = 1
            # remove this pc from harmony_options_bin
            tmp_pc = tmp_idx%12
            for j in range(12):
                if tmp_pc+12*j < 128:
                    self.harmony_options_bin[ tmp_pc+12*j , i ] = small_value
            # cross out respective gct completion check
            self.gct_completion[ tmp_idx%12 , i ] = 0
    # end fill_bass_VL
    def fill_mid_VL(self):
        small_value = -100000000
        tmpMat = copy.deepcopy(self.seed)
        for i in range(self.matrix.shape[1]):
            # roll tmpMat according to matrix
            if i > 0:
                # remove_fist_char = self.seed[:,1:,:]
                remove_fist_char = tmpMat[:,1:,:]
                new_input = np.append( self.matrix[:,i] , self.matrix[:,i-1] )
                tmpMat = np.append(remove_fist_char, np.reshape(new_input, [1, 1, self.input_rows]), axis=1)
            # make next prediction
            predicted = self.sess.run([self.prediction], feed_dict = {self.x:tmpMat})
            # currate predictions
            predicted = np.asarray(predicted[0]).astype('float64')[0]
            # get the next note from the predictions
            # zero-out notes that already exist
            predicted[ self.matrix[:,i]>0 ] = small_value
            # zero-out notes that disagree with GCT pcs
            predicted[ self.harmony_options_bin[:,i] == 0 ] = small_value
            # we need to do this until there are no more gct_completion demands
            while np.sum( self.gct_completion[:,i] ) > 0 :
                # get element that corresponds to the maximum index
                tmp_idx = np.argmax(predicted)
                # print('predicted: ', predicted)
                self.matrix[tmp_idx, i] = 1
                # kill this probability
                predicted[ tmp_idx ] = small_value
                # and also kill this pc from harmony_options_bin
                tmp_pc = tmp_idx%12
                for j in range(12):
                    if tmp_pc+12*j < 128:
                        predicted[ tmp_pc+12*j ] = small_value
                # cross out respective gct completion check
                self.gct_completion[ tmp_idx%12 , i ] = 0
            # finally, remove melody note
            tmp_mel = np.max( np.nonzero( self.matrix[:,i] ) )
            self.matrix[tmp_mel,i] = 0
    # end fill_mid_VL
    def fill_notes_in_matrix(self, matrix_in=[], num_notes=1):
        ' samples num_notes in given matrix_in with given density and register values '
        ' returns new matrix and new density and register values '
        if matrix_in:
            self.matrix = matrix_in
        else:
            self.matrix = np.zeros( (self.output_rows, self.composition_length) )
        for i in range(num_notes):
            self.fill_single_note()
    # end fill_notes_in_matrix
    def fill_single_note(self):
        ' fills the next most probable note in self.matrix '
        # initially, update entire predictions matrix
        self.update_predictions()
        # find maximum element
        r = np.where( self.predictions == np.max(self.predictions) )
        # place note
        self.matrix[r[0][0], r[1][0]] = 1
        '''
        # sampling approach
        # sharpen large values
        self.predictions = np.power(self.predictions, 15)
        self.predictions = self.predictions/np.sum(self.predictions)
        # make predictions a 1D array
        tmpPredictions = np.reshape(self.predictions, ( self.predictions.size ))
        selection = np.random.multinomial(1, tmpPredictions, size=1)
        selection = np.reshape(selection, (self.predictions.shape[0], self.predictions.shape[1]))
        # find maximum element
        r = np.where( selection == np.max(selection) )
        # place note
        self.matrix[ r[0][0] , r[1][0] ] = 1
        '''
        # cross out respective gct completion check
        self.gct_completion[ r[0][0]%12 , r[1][0] ] = 0
        # cross out harmonic options that correspond to this pc
        for i in range(11):
            if 12*i + r[0][0]%12 < 128:
                self.harmony_options_bin[ 12*i + r[0][0]%12 , r[1][0] ] = 0
    # end fill_single_note
    def update_predictions(self):
        ' runs from seed to the end of matrix -1column and updates all predictions '
        # scanning from seed to matrix
        # composition = np.array(self.seed[0,:,2:]).transpose()
        tmpMat = copy.deepcopy(self.seed)
        # for each matrix column, do prediction
        for i in range(self.matrix.shape[1]):
            # roll tmpMat according to matrix
            if i > 0:
                # remove_fist_char = self.seed[:,1:,:]
                remove_fist_char = tmpMat[:,1:,:]
                new_input = self.matrix[:,i]
                tmpMat = np.append(remove_fist_char, np.reshape(new_input, [1, 1, self.input_rows]), axis=1)
            # make next prediction
            predicted = self.sess.run([self.prediction], feed_dict = {self.x:tmpMat})
            # currate predictions
            predicted = np.asarray(predicted[0]).astype('float64')[0]
            # fit prediction to predictions matrix
            self.predictions[:,i] = predicted
        self.prediction_to_cdf()
        # zero-out notes that already exist
        self.predictions[ self.matrix>0 ] = 0
        # zero-out notes that disagree with GCT pcs
        self.predictions[ self.harmony_options_bin == 0 ] = 0
    # end update_predictions
    def rnn(self, x, weight, bias, input_rows):
        '''
        define rnn cell and prediction
        '''
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, input_rows])
        x = tf.split(x, self.max_len, 0)
        
        cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n) for n in self.num_units]
        stacked_rnn_cell = tf.contrib.rnn.MultiRNNCell(cells)
        outputs, states = tf.contrib.rnn.static_rnn(stacked_rnn_cell, x, dtype=tf.float32)
        prediction = tf.matmul(outputs[-1], self.weight) + self.bias
        return prediction
    # end rnn
    def prediction_to_cdf(self):
        ' converts predictions array to CDF '
        if np.sum(self.predictions) != 0:
            self.predictions = self.predictions/np.sum(self.predictions)