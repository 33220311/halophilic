# Copyright 2022 Rostlab
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import re

import numpy as np


def get_clusters(hval_mat, hval_threshold, sequences=None):
    assert hval_mat.shape[0] == hval_mat.shape[1]

    # node pairs above threshold
    pairs = (hval_mat > hval_threshold)
    # sort nodes by number of pairs (from many to few)
    nodes = np.argsort(-np.sum(pairs, axis=1), kind='stable')

    # if sequences are passed, sort by annotated scores
    if sequences:
        assert len(sequences) == nodes.shape[0]

        s_pat = re.compile(r'::score=(\d+);')
        scores = np.zeros(nodes.shape[0], dtype=np.int64)

        for idx, (header, sequence) in enumerate(sequences):
            match = s_pat.search(header)

            if match:
                scores[idx] = int(match.group(1))

        # sort nodes by annotated scores (from high to low)
        nodes = nodes[np.argsort(-scores[nodes], kind='stable')]

    blocked_nodes = set()
    representatives = list()

    # find cluster representatives
    for node in nodes:
        if node in blocked_nodes:
            continue

        representatives.append(node)

        blocked_nodes.update(np.nonzero(pairs[node])[0])

    del pairs
    del blocked_nodes

    groups = {representative: [] for representative in representatives}
    best_representatives = np.argmax(hval_mat[representatives], axis=0)

    # generate unsorted clusters
    for node in nodes:
        if node in groups:
            continue

        # add to best representative
        groups[representatives[best_representatives[node]]].append(node)

    del nodes
    del best_representatives

    clusters = []

    # generate final sorted clusters
    for representative in representatives:
        members = groups[representative]
        members_hvals = hval_mat[representative]

        # sort members by HSSP value (from high to low)
        members.sort(key=lambda m: members_hvals[m], reverse=True)

        clusters.append((representative, members))

    # sort clusters by size (from big to small)
    clusters.sort(key=lambda c: len(c[1]), reverse=True)

    return clusters
