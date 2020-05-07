import numpy as np
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.getcwd())
from custom_balltree import BallTree, brute_force_nn


def guassian_hash_generator(W, data_size):
    a = np.random.normal(size=(data_size,))
    b = np.random.uniform(low=0., high=W)
    return lambda q: int((np.dot(a,q)+b)/float(W))


class LSH(object):
    def __init__(self, data, hash_fn_generator, hashes_per_table, num_tables):
        self.hash_fn_generator = hash_fn_generator
        self.data = data
        num_samples = self.data.shape[0]
        self.hashes_per_table = hashes_per_table
        self.num_tables = num_tables
        self.hash_tables = np.zeros((num_samples, hashes_per_table, num_tables)) 
        self.hash_fns = {}  # (index_in_table, table)

        self._build()

    def _build(self):
        self._generate_hash_functions()
        for k, sample in enumerate(self.data):
            self.hash_tables[k,:,:] = self._hash_datum(sample) 

    def _generate_hash_functions(self):
        for i in range(self.num_tables):
            for j in range(self.hashes_per_table):
                self.hash_fns[(j,i)] = self.hash_fn_generator()

    def _hash_datum(self, datum):
        datum_hash = np.zeros((self.hashes_per_table, self.num_tables))
        for i in range(self.num_tables):
            for j in range(self.hashes_per_table):
                datum_hash[j,i] = self.hash_fns[(j,i)](datum) 
        return datum_hash

    def query(self, q):
        q = np.array(q, dtype=float).reshape(-1)
        assert q.size == self.data.shape[1]
        q_hash = self._hash_datum(q)

        temp = np.any(np.all(np.equal(q_hash, self.hash_tables), axis=1), axis=-1)
        canidates_indices = np.argwhere(temp).flatten()
        return self.data[canidates_indices]

def test_lsh(N=10000, D=2, HPT=100, T=10):
    rseed = np.random.randint(10000)
    np.random.seed(rseed)
    X = np.random.random((N, D))

    hash_fn_gen = lambda: guassian_hash_generator(0.2, D)
    lsh = LSH(X, hash_fn_gen, HPT, T)
    q = np.ones((1,D))*0.5
    nn_lsh_canidates = lsh.query(q)

    if len(nn_lsh_canidates) > 0:
        nn_lsh, _ = brute_force_nn(q, nn_lsh_canidates)
    else:
        nn_lsh = np.zeros((1,D))
    
    nn_brute, _ = brute_force_nn(q,X)
    print('{} nn canidates'.format(len(nn_lsh_canidates), nn_lsh))
    print('distance to brute force nn {}'.format(np.linalg.norm(nn_brute - nn_lsh)))
    

if __name__ == '__main__':
    test_lsh()




