
import numpy as np
import scipy
import scipy.io
import time
from matplotlib import pyplot as plt


class Neural_Network:
    def __init__(self, hid_layer_size, hidden_layer_num, filename, input_size, lr, batch_size=100):
        np.random.seed(int(time.time()))
        self.learning_rate = lr
        # input_sizexdx...xdx1
        self.hidden_layer_size = hid_layer_size
        self.num_hidden_layers = hidden_layer_num
        self.first_weight = np.zeros((input_size, hid_layer_size))
        self.input_size = input_size
        row = 0
        for r in self.first_weight:
            col = 0
            for element in r:
                self.first_weight[row, col] = np.random.normal(0, np.sqrt(2 / input_size))
                col = col + 1

            row = row + 1

        self.first_bias = np.zeros((hid_layer_size, 1))
        self.first_bias_gradient = self.first_bias
        self.first_weight_gradient = self.first_weight
        self.first_layer_output = []
        self.first_layer_z = []
        self.hidden_layers_weight = []
        self.hidden_weight_gradients = []
        self.hidden_layers_bias = []
        self.hidden_bias_gradients = []
        for i in range(hidden_layer_num):
            to_add = np.zeros((hid_layer_size, hid_layer_size))
            for j in range(hid_layer_size):
                for k in range(hid_layer_size):
                    to_add[j, k] = np.random.normal(0, np.sqrt(2 / len(to_add[j])), 1)
            self.hidden_layers_weight.append(to_add)
            self.hidden_weight_gradients.append(to_add)
            tmp = np.zeros((hid_layer_size, 1))
            self.hidden_layers_bias.append(tmp)
            self.hidden_bias_gradients.append(tmp)
        self.hidden_layer_outputs = []
        self.hidden_layer_zs = []
        self.last_weight = np.zeros((hid_layer_size, 1))
        for i in range(hid_layer_size):
            self.last_weight[i] = np.sqrt(2 / hid_layer_size)
        self.last_weight_gradient = self.last_weight
        self.last_bias = np.zeros((1, 1))
        self.last_bias_gradient = self.last_bias
        self.last_out = []
        self.last_z = []
        # memory for back prop
        self.deltas = [] * (hidden_layer_num + 2)

        self.batch_size = batch_size

        self.train_dat = []
        self.y = []
        self.test_xs = []
        self.test_ys = []
        self.import_data(filename)

    def element_wise_sigmoid(self, xin):
        to_ret = np.zeros(np.shape(xin))

        for i in range(np.shape(xin)[0]):
            for j in range(np.shape(xin)[1]):
                to_ret[i][j] = (1) / (1 + np.exp(-xin[i][j]))

        return to_ret

    def element_wise_diff(self, xin):
        # assume input vector
        to_ret = np.zeros(np.shape(xin))
        for i in range(np.shape(xin)[0]):
            for j in range(np.shape(xin)[1]):
                to_ret[i][j] = np.exp(xin[i][j]) / ((1 + np.exp(xin[i][j])) ** 2)
        return to_ret

    def import_data(self, filename):
        self.x = scipy.io.loadmat(filename)

        self.train_dat = np.append(self.x.get("train0"), self.x.get("train1"), 0)
        # self.train_dat = np.reshape(self.train_dat,(12665,784))
        y_ones = np.zeros((6742, 1))
        for i in range(len(y_ones)):
            y_ones[i] = 1
        self.y = np.append(np.zeros((5923, 1)), y_ones)
        self.y = np.reshape(self.y, (12665, 1))
        all_train = np.append(self.train_dat, self.y, 1)
        # all_train = np.reshape(all_train,(12665,785))
        np.random.shuffle(all_train)

        self.y = []
        self.train_dat = []
        for row in all_train:
            self.train_dat.append(row[:-1])
            self.y.append(row[-1])

        test_0_x = self.x.get("test0")
        test_1_x = self.x.get("test1")
        self.test_xs = np.append(test_0_x, test_1_x, 0)
        test_ys = np.zeros((np.shape(test_1_x)[0], 1))
        for count in range(len(test_ys)):
            test_ys[count] = 1
        test_ys = np.append(test_ys, np.zeros((np.shape(test_0_x)[0], 1)))
        self.test_ys = np.zeros((np.shape(test_ys)[0], 1))
        np.reshape(test_ys, (np.shape(test_ys)[0], 1))
        self.test_ys = test_ys

    def perform_iteration(self, debug=True):
        # select b indices
        indices = []
        for i in range(self.batch_size):
            indices.append(np.random.randint(0, len(self.train_dat) - 1))
        # reset the back prop memory
        self.last_weight_gradient = []
        self.last_bias_gradient = []
        self.hidden_weight_gradients = []
        self.hidden_bias_gradients = []
        for i in range(self.num_hidden_layers):
            self.hidden_weight_gradients.append([])
            self.hidden_bias_gradients.append([])
        self.first_weight_gradient = []
        self.first_bias_gradient = []
        # reset the forward prop memory
        self.first_layer_z = []
        self.first_layer_output = []
        self.hidden_layer_zs = []
        self.hidden_layer_outputs = []
        for i in range(self.num_hidden_layers):
            self.hidden_layer_zs.append([])
            self.hidden_layer_outputs.append([])
        self.last_z = []
        self.last_out = []
        count = 0
        if (debug):
            print("performing forward propagation...")
        for i in indices:
            self.perform_forward_prop_iteration(indices, count)
            count = count + 1
        if debug: print("done")
        # fill in back propagation memory
        # fill in deltas
        if debug: print("filling in deltas...")
        self.deltas = []
        for i in range(self.num_hidden_layers + 2):
            self.deltas.append([])
        # self.deltas = np.zeros((self.num_hidden_layers+2,1))#[[]*(self.num_hidden_layers+2)]
        to_app = []
        count = 0
        for i in indices:
            to_app.append(self.compute_dLdf(i, count) * self.element_wise_diff(self.last_z[count]))
            count = count + 1
        self.deltas[-1].append(to_app)
        to_app = []
        for i in range(self.batch_size):
            tmp = self.hidden_layer_zs[-1][i]
            to_diff = self.element_wise_diff(tmp)
            second = np.matmul(self.hidden_layers_weight[-1], to_diff)
            to_app.append(self.deltas[-1][0][i] * second)
        self.deltas[-2].append(to_app)

        for i in range(self.num_hidden_layers - 1):
            to_app = []
            for j in range(self.batch_size):
                to_app.append(self.deltas[-2 - i][0][j] * np.matmul(self.hidden_layers_weight[-2 - i],
                                                                    self.element_wise_diff(
                                                                        self.hidden_layer_zs[-2 - i][j])))
            self.deltas[-i - 3].append(to_app)
        # fill in last (first) delta
        to_app = []
        for j in range(self.batch_size):
            to_app.append(self.deltas[1][0][j] * np.matmul(self.hidden_layers_weight[0],
                                                           self.element_wise_diff(self.hidden_layer_zs[0][j])))
        self.deltas[0].append(to_app)
        if debug:
            print("done.")
            print("filling in gradients wrt weights and biases...")
        # fill in gradients with respect to weights and biases
        self.last_weight_gradient = np.multiply(self.deltas[-1][0], self.hidden_layer_outputs[-1][0])
        self.last_bias_gradient = self.deltas[-1][0]
        inter = []
        if self.num_hidden_layers > 1:
            for i in range(self.num_hidden_layers):
                inter = []
                for row in range(self.batch_size):
                    delta = self.deltas[-2 - i][0][row]
                    output = self.hidden_layer_outputs[-i][row]
                    inter.append(np.outer(delta, np.transpose(output)))
                self.hidden_weight_gradients[-1 - i] = inter
        elif self.num_hidden_layers == 1:
            for row in range(len(self.deltas[-2][0])):
                inter.append(np.outer(self.deltas[-2][0][row], self.first_layer_output[row]))
            # inter = np.outer(np.transpose(self.deltas[-2][0]),np.transpose(self.first_layer_output))
            self.hidden_weight_gradients[-1] = inter
        self.hidden_bias_gradients[-1] = self.deltas[-2][0]
        # for i in range(self.num_hidden_layers-1):
        #     inter = []
        #     for row in range(len(self.deltas[-3-i][0])):
        #         inter.append(np.outer(self.deltas[-3-i][0][row],self.hidden_layer_outputs[-3-i][row]))
        #     self.hidden_bias_gradients[-2-i] = self.deltas[-3-i][0]
        to_app = []
        for i in range(self.batch_size):
            d = self.train_dat[indices[i]]
            delt = self.deltas[0][0][i]
            delt = np.reshape(delt, (self.hidden_layer_size, 1))
            res = np.outer(d, np.transpose(delt))
            to_app.append(res)
            # if i == 0:
            #     to_app = res
            # else:
            #     to_app = np.add(to_app,res)
        self.first_weight_gradient = (to_app)
        self.first_bias_gradient = self.deltas[0][0]
        if debug:
            print("done.")
            # update weights
            print("updating weights...")
        inter1 = np.sum(self.first_weight_gradient[0], 0)
        second = np.multiply(self.learning_rate, np.transpose(inter1))
        third = np.transpose(np.divide(second, self.batch_size))
        self.first_weight = self.first_weight - third
        inter = np.sum(self.first_bias_gradient, 0)
        second = np.divide(np.multiply(self.learning_rate, inter), self.batch_size)
        self.first_bias = self.first_bias - second
        for i in range(self.num_hidden_layers):
            inter = np.sum(self.hidden_weight_gradients[i], 0)
            second = np.divide(np.multiply(self.learning_rate, inter), self.batch_size)
            self.hidden_layers_weight[i] = self.hidden_layers_weight[i] - second
            inter = np.sum(self.hidden_bias_gradients[i], 0)
            second = np.divide(np.multiply(self.learning_rate, inter), self.batch_size)
            self.hidden_layers_bias[i] = self.hidden_layers_bias[i] - second
        self.last_weight = self.last_weight - self.learning_rate * np.sum(self.last_weight_gradient,
                                                                          0) / self.batch_size
        self.last_bias = self.last_bias - self.learning_rate * np.sum(self.last_bias_gradient, 0) / self.batch_size
        if debug: print("done.")
        return

    def compute_dLdf(self, index_in_dat, iteration):
        to_ret = (1 + np.exp(self.last_z[iteration])) * (1 - 2 * self.y[index_in_dat])
        return to_ret

    def perform_forward_prop_iteration(self, indices, count):
        # performs all forward prop with respect to one observation
        curx = self.train_dat[indices[count]]
        intermed = np.matmul(np.transpose(self.first_weight), curx)
        intermed = np.reshape(intermed, (10, 1))
        second_inter = np.add(intermed, self.first_bias)
        self.first_layer_z.append(second_inter)
        self.first_layer_output.append(self.element_wise_sigmoid(self.first_layer_z[-1]))

        intermed = np.matmul(self.hidden_layers_weight[0], self.first_layer_output[-1])
        intermed = np.reshape(intermed, (self.hidden_layer_size, 1))
        self.hidden_layer_zs[0].append(np.add(intermed, self.hidden_layers_bias[0]))
        to_app = self.element_wise_sigmoid(self.hidden_layer_zs[0][-1])
        to_app = np.reshape(to_app, (self.hidden_layer_size, 1))
        self.hidden_layer_outputs[0].append(to_app)
        for i in range(self.num_hidden_layers - 1):
            intermed = []
            intermed = np.matmul(self.hidden_layers_weight[i + 1], self.hidden_layer_outputs[i][-1])
            intermed = np.reshape(intermed, (self.hidden_layer_size, 1))
            self.hidden_layer_zs[i + 1].append(np.add(intermed, self.hidden_layers_bias[i + 1]))
            app = self.element_wise_sigmoid(self.hidden_layer_zs[i + 1][-1])
            app = np.reshape(app, (self.hidden_layer_size, 1))
            self.hidden_layer_outputs[i + 1].append(app)
        int = np.matmul(np.transpose(self.last_weight), self.hidden_layer_outputs[0][-1])
        int = np.reshape(int, (1, 1))
        sec = np.add(int, self.last_bias)
        self.last_z.append(sec)
        self.last_out.append(self.element_wise_sigmoid(self.last_z[-1]))
        return

    def test_on_single_data(self, dat):
        dat = np.reshape(dat, (784, 1))
        inter = np.matmul(np.transpose(self.first_weight), dat)
        second = np.add(inter, self.first_bias)
        firstz = second

        first_output = (self.element_wise_sigmoid(firstz))
        hid_layer_zs = []
        hid_layer_outs = []
        for i in range(self.num_hidden_layers):
            hid_layer_zs.append([])
            hid_layer_outs.append([])

        hid_layer_zs[0] = (np.matmul(self.hidden_layers_weight[0], first_output) + self.hidden_layers_bias[0])
        hid_layer_outs[0] = (self.element_wise_sigmoid(hid_layer_zs[0]))
        for i in range(self.num_hidden_layers - 1):
            hid_layer_zs[i + 1] = (
                        np.matmul(self.hidden_layers_weight[i + 1], hid_layer_outs[i]) + self.hidden_layers_bias[i + 1])
            hid_layer_outs[i + 1] = (self.element_wise_sigmoid(hid_layer_zs[i + 1]))
        finalz = np.add(np.matmul(np.transpose(self.last_weight), hid_layer_outs[0]), self.last_bias)
        finalout = (self.element_wise_sigmoid(finalz))
        return finalout

    def run_test_on_test_data(self):
        count = 0
        num_right = 0
        for row in self.test_xs:
            pred = self.test_on_single_data(row)
            if pred >= .5 and self.test_ys[count] == 1:
                num_right = num_right + 1
            elif pred < .5 and self.test_ys[count] == 0:
                num_right = num_right + 1
            count = count + 1
        return float(num_right / len(self.test_ys))

    def run_test_on_train_data(self):
        count = 0
        num_right = 0
        for row in self.train_dat:
            pred = self.test_on_single_data(row)
            if pred >= .5 and self.y[count] == 1:
                num_right = num_right + 1
            elif pred < .5 and self.y[count] == 0:
                num_right = num_right + 1
            count = count + 1
        return float(num_right / len(self.y))

    def __equals__(self, other):
        if np.array_equal(self.first_weight, other.first_weight) and np.array_equal(self.hidden_layers_weight,
                                                                                    other.hidden_layers_weight) and np.array_equal(
                self.last_weight, other.last_weight):
            return True
        return False
