from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.getcwd())
from custom_balltree import BallTree, brute_force_nn
from lsh import LSH, guassian_hash_generator
from ball_tree_lsh import BallTreeLSH 
from mnist import MNIST


def load_data():
    mndata = MNIST('/home/jaredb/cse291/datasets/mnist')
    images, labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()
    data = np.array(images[:1000]).astype(float)
    q = np.array(test_images)[0,:].astype(float)
    return data, q


def print_bt_stats(bt):
    leaf_nodes = [node for node in bt.nodes if node.is_leaf]
    mean_point_in_leaf = np.mean([len(node.data_index.flatten()) for node in leaf_nodes])

    print('{} samples, {} total nodes, {} leaf nodes, {} avg pts per leaf'.format(
            bt.n_samples,
            len(bt.nodes),
            len(leaf_nodes),
            mean_point_in_leaf,
        )
    )


if __name__ == "__main__":
    data, q = load_data()
    
    # brute force
    nn_brute, nn_brute_dist = brute_force_nn(q, data)

    # ball tree
    bt = BallTree(data, 2000)
    print_bt_stats(bt)
    nn_bt, nn_bt_dist = bt.query(q)

    # LSH
    hash_fn_gen = lambda: guassian_hash_generator(150, data.shape[1])
    lsh = LSH(data, hash_fn_gen, 3, 100)
    nn_lsh_canidates = lsh.query(q)
    print('{} lsh canidates'.format(len(nn_lsh_canidates)))
    if len(nn_lsh_canidates) > 0:
        nn_lsh, _ = brute_force_nn(q, nn_lsh_canidates)
    else:
        nn_lsh  = np.zeros((data.shape[1],))

    # Ball tree LSH
    bt_lsh = BallTreeLSH(bt, lsh)
    nn_bt_lsh, _  = bt_lsh.query(q)

    # compare_results
    _, nn_lsh_to_q = brute_force_nn(nn_lsh, q)
    _, nn_bt_to_q = brute_force_nn(nn_bt, q)
    _, nn_bt_lsh_to_q = brute_force_nn(nn_bt_lsh, q)

    print(nn_lsh_to_q/nn_brute_dist, nn_bt_to_q/nn_brute_dist, nn_bt_lsh_to_q/nn_brute_dist)

