import numpy as np
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.getcwd())
from custom_balltree import BallTree, brute_force_nn
from lsh import LSH, guassian_hash_generator
import itertools
import operator

def most_common(L):
  # get an iterable of (item, iterable) pairs
  SL = sorted((x, i) for i, x in enumerate(L))
  # print 'SL:', SL
  groups = itertools.groupby(SL, key=operator.itemgetter(0))
  # auxiliary function to get "quality" for an item
  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    # print 'item %r, count %r, minind %r' % (item, count, min_index)
    return count, -min_index
  # pick the highest-count/earliest item
  return max(groups, key=_auxfun)[0]


class BallTreeLSH(object):
    def __init__(self, bt, lsh):
        self.bt = bt
        self.lsh = lsh
        self.bucket_to_ball = {} # (hashed_sample, table) -> leaf_node

        self._build_bucket_to_ball()

    def _build_bucket_to_ball(self):
        leaf_nodes = [node for node in self.bt.nodes if node.is_leaf]

        for leaf_node in leaf_nodes:
            for table in range(self.lsh.num_tables):
                for sample in self.lsh.hash_tables[leaf_node.data_index]:
                    sample = sample[:,table]
                    self.bucket_to_ball[(table, hash(str(sample)))] = leaf_node

    def query(self, q):
        canidate_balls = []
        q = np.array(q, dtype=float).reshape(-1)
        q_hash = self.lsh._hash_datum(q)

        for table in range(q_hash.shape[-1]):
            canidate_ball = self.bucket_to_ball.get((table, hash(str(q_hash[:,table]))))
            if canidate_ball is not None:
                canidate_balls.append(canidate_ball)

        if len(canidate_balls) > 0:
            best_canidate = most_common(canidate_balls) 
            print('canidate balls', len(canidate_balls))
            print('dist to best_canidate ball centrod', np.linalg.norm(best_canidate.centroid-q))
            print(best_canidate.index, best_canidate.is_leaf, len(best_canidate.data_index))

            nn_estimate = self.bt.query(q, start_node=best_canidate)
            return nn_estimate
        
        else:
            return None





