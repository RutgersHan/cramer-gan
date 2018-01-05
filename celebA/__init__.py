import numpy as np

data_dir = '/home/hanzhang/Data/celebA_total/celebA_processed'


class TrainDataSampler(object):
    def __init__(self):
        self.data = np.load(data_dir + '/celebA_64_train.npy')
        self.data_num = self.data.shape[0]
        self.count = self.data_num

    def next_batch(self, batch_size):
        if self.count + batch_size >= self.data_num:
            self.count = 0
            ids = np.random.permutation(self.data_num)
            self.data = self.data[ids]

        data_output = self.data[self.count:self.count+batch_size]
        self.count += batch_size
        return data_output

    def __call__(self, batch_size):
        return self.next_batch(batch_size)


class ValDataSampler(object):
    def __init__(self):
        self.data = np.load(data_dir + '/celebA_64_valid.npy')
    def cramer_gan_batch(self, batch_size):
        test_img = self.data[0: batch_size].copy()
        new_img1 = self.data[0: 8]
        new_img1 = np.tile(new_img1, (3, 1, 1, 1))
        new_img2 = self.data[8: 16]
        new_img2 = np.tile(new_img2, (3, 1, 1, 1))
        new_img = np.concatenate((new_img1, new_img2), axis=0)
        test_img[0:48] = new_img
        return test_img

    def __call__(self, batch_size):
        return self.cramer_gan_batch(batch_size)





class NoiseSampler(object):
    def __call__(self, batch_size, z_dim):
        return np.random.uniform(-1.0, 1.0, [batch_size, z_dim])