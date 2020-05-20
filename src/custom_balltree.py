from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import attr
from collections import deque
import sys
import os

sys.path.append(os.getcwd())
from utils import(
    brute_force_nn,
    calculate_centroid,
    calculate_max_radius,
    calculate_min_radius,
    find_dimension_with_greatest_spread
)

@attr.s
class Node(object):
    index = attr.ib(type=int)
    centroid = attr.ib(type=np.ndarray)
    radius = attr.ib(type=float)
    is_root = attr.ib(type=bool)
    is_leaf = attr.ib(type=bool)
    parent_node = attr.ib()
    child_1_node = attr.ib()
    child_2_node = attr.ib()
    data_index = attr.ib(type=np.ndarray)  # 1-d numpy array of sample indexes that are in node

    def __eq__(self, other):
        return other.index == self.index

    def __str__(self):
        return ('Node: \nindex {}, is_root {}, is_leaf {}, num_samples {}'
                .format(self.index, self.is_root, self.is_leaf, len(self.data_index)))


class BallTree(object):
    def __init__(self, data, max_leaf_samples):
        self.data = data
        self.max_leaf_samples = max_leaf_samples

        # data[sample index, feature index]
        self.n_samples = self.data.shape[0]
        self.n_features = self.data.shape[1]

        self.nodes = [] 

        # build tree recursively from top-down
        self.root_node = self._recursive_build(
            i_node=0,
            data_index=np.arange(self.n_samples),
            parent_node=None,
        ) 

    def _recursive_build(self, i_node, data_index, parent_node):
        """ Build Node with node index i_node looking at data
        from data[idx_start:idx_end]
        :param i_node int: current node index
        :param idx_start: start data index
        :param idx_end: end data index
        """
        # build node
        data_in_node = self.data[data_index]
        node_centroid = calculate_centroid(data_in_node)
        node_radius = calculate_max_radius(node_centroid, data_in_node)

        node = Node(
            index=i_node,
            centroid=node_centroid,
            radius=node_radius,
            parent_node=parent_node,
            is_root=parent_node is None,
            is_leaf=False,
            child_1_node=None,
            child_2_node=None,
            data_index=data_index,
        )
        self.nodes.append(node)

        if len(data_index) <= self.max_leaf_samples:
            # leaf node
            node.is_leaf = True
        else:
            # internal node
            node.is_leaf = False
            data_index_child_1, data_index_child_2 = self._partition_indices(data_index)

            node.child_1_node = self._recursive_build(2 * i_node + 1, data_index_child_1, node)
            node.child_2_node = self._recursive_build(2 * i_node + 2, data_index_child_2, node)

        return node

    def _reset_query(self):
        self.nn_estimate = np.zeros(self.n_features)
        self.nn_distance = np.inf 
        self.leaves_explored = 0
        self.distance_calcs = 0

    def query_top_down(self, q, performance_limit=np.inf):
        """ Standard ball tree query mechansim. Start from root node
        and work your way down the tree to a leafe not, then backtrack.
        :param q np.ndarray: query point
        :param performance_limit int: 
        :returns (float,float): nearest neighbor data point and distance to
            nearest neighbor
         """
        self._reset_query()
        if q.size != self.n_features:
            raise ValueError("query data dimension must "
                             "match training data dimension")

        self._query_recursive(q, self.root_node)

        return self.nn_estimate

    def query_bottom_up(self, q, start_nodes, performance_limit=np.inf):
        self._reset_query()

        # self.distance_calcs += len(start_nodes)
        start_nodes.sort(key=lambda node: np.linalg.norm(node.centroid - q))

        node_deque = deque(start_nodes)

        # parrallel bottom up search of ball tree
        while node_deque and self.distance_calcs < performance_limit:
            node = node_deque.popleft()

            if not node.is_root:
                self._query_recursive(q, node.parent_node)
                node_deque.append(node.parent_node)

        return self.nn_estimate

    def _query_recursive(self, q, node):
        #------------------------------------------------------------
        # Case 1: query point is outside node radius:
        #         trim it from the query
        
        self.distance_calcs += 1
        if np.linalg.norm(q - node.centroid) - node.radius >= self.nn_distance:
            pass 

        #------------------------------------------------------------
        # Case 2: this is a leaf node.  Update set of nearby points
        elif node.is_leaf:
            self.leaves_explored += 1
            data_in_leaf = self.data[node.data_index]
            self.distance_calcs += len(data_in_leaf)
            nn_canidate, min_radius = calculate_min_radius(q, data_in_leaf)
            if min_radius < self.nn_distance:
                self.nn_distance = min_radius
                self.nn_estimate = nn_canidate 

        #------------------------------------------------------------
        # Case 3: Node is not a leaf.  Recursively query subnodes
        #         starting with the closest
        else:
            self.distance_calcs += 2
            distance_to_child_1_centroid = \
                np.linalg.norm(q - node.child_1_node.centroid)
            
            distance_to_child_2_centroid = \
                np.linalg.norm(q - node.child_2_node.centroid)

            # recursively query subnodes
            if distance_to_child_1_centroid <= distance_to_child_2_centroid: 
                self._query_recursive(q, node.child_1_node)
                self._query_recursive(q, node.child_2_node)
            else:
                self._query_recursive(q, node.child_2_node)
                self._query_recursive(q, node.child_1_node)


    def _partition_indices(self, data_index):
        """ Find dimension with largest spread and split dataset """

        data_in_node = self.data[data_index]

        split_dim = find_dimension_with_greatest_spread(data_in_node)
        median = np.median(data_in_node[:,split_dim], axis=0)

        data_index_eq_median = np.intersect1d(
            np.argwhere(self.data[:, split_dim] == median),
            data_index
        )

        data_index_child_1 = np.intersect1d(
            np.argwhere(self.data[:, split_dim] > median).flatten(), 
            data_index
        )
        
        data_index_child_2 = np.intersect1d(
            np.argwhere(self.data[:, split_dim] < median).flatten(), 
            data_index
        )
        data_index_child_1 = np.append(
            data_index_child_1,
            data_index_eq_median[:int(len(data_index_eq_median)/2.)]
        )

        data_index_child_2 = np.append(
            data_index_child_2,
            data_index_eq_median[int(len(data_index_eq_median)/2.):]
        )

        assert set(data_index_child_1).union(set(data_index_child_2)) == set(data_index)

        return data_index_child_1, data_index_child_2


def plot_ball(ax, node, data, color='k'):
    circle = plt.Circle(node.centroid, node.radius, color=color, fill=False)
    ax.add_artist(circle)
    plt.scatter(node.centroid[0], node.centroid[1], c=color, marker='x')
    plt.scatter(data[:,0], data[:,1], c=color)


def test_tree(N=1000, D=784, R=0):
    rseed = np.random.randint(10000)
    print("-------------------------------------------------------")
    print("1-NN of {} points in {} dimensions".format(N, D))
    print("random seed = {0}".format(rseed))
    np.random.seed(rseed)
    X = np.random.random((N, D))
    bt = BallTree(X, max_leaf_radius=R)
    q = np.ones((D,)) * 0.5
    nn, nn_dist = bt.query(q)
    nn_actual, nn_actual_dist = brute_force_nn(q, X)
    print('Ball tree NN {}, same as brute force->{}'.format(nn, all(nn==nn_actual)))

if __name__ == '__main__':
    test_tree()
