import numpy as np
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.getcwd())
from custom_balltree import BallTree, brute_force_nn
from lsh import LSH, guassian_hash_generator
import itertools
import operator


class BallTreeLSH(object):
    def __init__(self, balltree, lsh):
        self.balltree = balltree
        self.lsh = lsh
        self.bucket_to_balls = {} # (hashed_sample, table) -> [leaf_node,]

        self._build_bucket_to_balls()

    def _build_bucket_to_balls(self):

        # get all leaf nodes
        leaf_nodes = [node for node in self.balltree.nodes if node.is_leaf]

        # associate the hash bucket of a sample with ball tree leaf nodes
        for leaf_node in leaf_nodes:
            for hash_table in range(self.lsh.num_tables):
                for hashed_datum in self.lsh.hash_tables[leaf_node.data_index]:
                    hashed_datum = hashed_datum[:,hash_table]
                    bucket_key = (hash_table, hash(str(hashed_datum)))

                    # a bucket can be associated with many possible balls
                    if self.bucket_to_balls.get(bucket_key) is None:
                        self.bucket_to_balls[bucket_key] = [leaf_node]
                    else:
                        self.bucket_to_balls[bucket_key].append(leaf_node) 

    def query(self, q, performance_limit=np.inf):
        canidate_balls = []
        q = np.array(q, dtype=float).reshape(-1)

        # hash query using LSH
        q_hash = self.lsh._hash_datum(q)  # shape (hashes_per_table, table)

        # generate canidate balls
        for table in range(q_hash.shape[-1]):
            balls_in_bucket = self.bucket_to_balls.get((table, hash(str(q_hash[:,table]))))
            if balls_in_bucket is not None:
                canidate_balls += balls_in_bucket

        if len(canidate_balls) > 0:
            print('num of canidate balls', len(canidate_balls))
            print('data in all canidate balls', sum([len(node.data_index) for node in canidate_balls]))

            nn_estimate = self.balltree.query_bottom_up(
                q=q, 
                start_nodes=canidate_balls, 
                performance_limit=performance_limit
            )
            return nn_estimate
        
        else:
            print('No canidate balls generated')
            return None





