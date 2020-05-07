from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import attr

@attr.s
class Node(object):
    index = attr.ib(type=int)
    centroid = attr.ib(type=np.ndarray)
    radius = attr.ib(type=float)
    is_root = attr.ib(type=float)
    is_leaf = attr.ib(type=float)
    parent_node = attr.ib()
    child_1_node = attr.ib()
    child_2_node = attr.ib()
    data_index = attr.ib(type=np.ndarray)  # 1-d numpy array of sample indexes that are in node


class BallTree(object):
    def __init__(self, data, max_leaf_radius):
        # todo add configurable distance metrix
        self.data = data
        self.max_leaf_radius = max_leaf_radius

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
        node_centroid = self.calculate_centroid(data_in_node)
        _, node_radius = self._calculate_max_radius(node_centroid, data_in_node)

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

        # recursive stop condition
        if node.radius <= self.max_leaf_radius:
            node.is_leaf = True
            return node

        # split node and recursively construct child nodes.
        node.is_leaf = False
        data_index_child_1, data_index_child_2 = self._partition_indices(data_index)

        node.child_1_node = self._recursive_build(2 * i_node + 1, data_index_child_1, node)
        node.child_2_node = self._recursive_build(2 * i_node + 2, data_index_child_2, node)

        return node

    @staticmethod
    def calculate_centroid(data):
        return np.mean(data, axis=0)

    @staticmethod 
    def find_dimension_with_greatest_spread(data):
        return np.argmax(np.max(data, axis=0) - np.min(data, axis=0))

    @staticmethod
    def _calculate_radii(centroid, data):
        diff = np.tile(centroid, (data.shape[0], 1)) - data  # TODO make more efficient
        radii = np.linalg.norm(diff, axis=1)
        return radii 

    def _calculate_max_radius(self, centroid, data):
        radii = self._calculate_radii(centroid, data)
        max_radius_index = np.argmax(radii)
        max_radius = radii[max_radius_index]
        return data[max_radius_index], max_radius

    def _calculate_min_radius(self, centroid, data):
        radii = self._calculate_radii(centroid, data)
        min_radius_index = np.argmin(radii)
        min_radius = radii[min_radius_index]
        return data[min_radius_index], min_radius

    def query(self, q):
        q = np.asarray(q, dtype=float).reshape(1,-1)
        self.nn_estimate = np.zeros(q.shape)
        self.nn_distance = np.inf 

        if q.size != self.n_features:
            raise ValueError("query data dimension must "
                             "match training data dimension")

        self._query_recursive(q, self.root_node)

        return self.nn_estimate, self.nn_distance

    def _query_recursive(self, q, node):
        #------------------------------------------------------------
        # Case 1: query point is outside node radius:
        #         trim it from the query
        if np.linalg.norm(q - node.centroid) - node.radius >= self.nn_distance:
            pass 

        #------------------------------------------------------------
        # Case 2: this is a leaf node.  Update set of nearby points
        elif node.is_leaf:
            data_in_leaf = self.data[node.data_index]
            nn_canidate, min_radius = self._calculate_min_radius(q, data_in_leaf)
            if min_radius < self.nn_distance:
                self.nn_distance = min_radius
                self.nn_estimate = nn_canidate 

        #------------------------------------------------------------
        # Case 3: Node is not a leaf.  Recursively query subnodes
        #         starting with the closest
        else:
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
        # Find the split dimension, ie dimension with largest spread
        data_in_node = self.data[data_index]

        split_dim = self.find_dimension_with_greatest_spread(data_in_node)
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

def brute_force_nn(q, data):
    diff = np.tile(q, (data.shape[0], 1)) - data
    radii = np.linalg.norm(diff, axis=1)
    min_radius_index = np.argmin(radii)
    min_radius = radii[min_radius_index]
    return data[min_radius_index], min_radius

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
