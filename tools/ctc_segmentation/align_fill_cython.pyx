# Copyright 2020, Technische Universität München; Dominik Winkelbauer, Ludwig Kürzinger
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

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

# This code was adapted from
# https://github.com/cornerfarmer/ctc_segmentation/blob/master/align_fill.pyx


import numpy as np
cimport numpy as np

def cython_fill_table(
    np.ndarray[np.float32_t, ndim=2] table,
    np.ndarray[np.float32_t, ndim=2] log_probs,
    np.ndarray[np.int_t, ndim=2] ground_truth,
    np.ndarray[np.int_t, ndim=1] offsets,
    np.ndarray[np.int_t, ndim=1] utt_begin_indices,
    int blank,
    float argskip_prob):

    """
    Fills
    :param table:
    :param log_probs:
    :param ground_truth:
    :param offsets:
    :param utt_begin_indices:
    :param blank:
    :param argskip_prob:
    :return:
    """
    cdef int t
    cdef int offset = 0
    cdef float mean_offset
    cdef int offset_sum = 0
    cdef int lower_offset
    cdef int higher_offset
    cdef float switch_prob, stay_prob, skip_prob
    cdef float prob_max = -1000000000
    cdef float lastMax
    cdef int lastArgMax
    cdef np.ndarray[np.int_t, ndim=1] cur_offset = np.zeros([ground_truth.shape[1]], np.int) - 1
    cdef float max_log_prob
    cdef float p
    cdef int s
    
    # Compute the mean offset between two window positions
    mean_offset = (log_probs.shape[0] - table.shape[0]) / float(table.shape[1])
    lower_offset = int(mean_offset)
    higher_offset = lower_offset + 1

    table[0, 0] = 0
    for c in range(table.shape[1]):
        if c > 0:
            # Compute next window offset
            offset = min(max(0, lastArgMax - table.shape[0] // 2), min(higher_offset, (log_probs.shape[0] - table.shape[0]) - offset_sum))

            # Compute relative offset to previous columns
            for s in range(ground_truth.shape[1] - 1):
                cur_offset[s + 1] = cur_offset[s] + offset
            cur_offset[0] = offset

            # Apply offset and move window one step further
            offset_sum += offset

        # Log offset
        offsets[c] = offset_sum
        lastArgMax = -1
        lastMax = 0
        
        # Go through all rows of the current column
        for t in range((1 if c == 0 else 0), table.shape[0]):
            # Compute max switch probability
            switch_prob = prob_max
            max_log_prob = prob_max
            for s in range(ground_truth.shape[1]):
                if ground_truth[c, s] != -1:
                    if t >= table.shape[0] - (cur_offset[s] - 1) or t - 1 + cur_offset[s] < 0 or c == 0:
                        p = prob_max
                    else:
                        table_row = t - 1 + cur_offset[s]
                        table_col = c - (s + 1)
                        table_value = table[table_row, table_col]

                        log_prob_value = log_probs[t + offset_sum, ground_truth[c, s]]
                        p = table_value + log_prob_value
                    switch_prob = max(switch_prob, p)
                    max_log_prob = max(max_log_prob, log_probs[t + offset_sum, ground_truth[c, s]])

            # Compute stay probability
            if t - 1 < 0:
                stay_prob = prob_max 
            elif c == 0:
                stay_prob = 0
            else:
                stay_prob = table[t - 1, c] + max(log_probs[t + offset_sum, blank], max_log_prob)

            # Use max of stay and switch prob
            table[t, c] = max(switch_prob, stay_prob)
                 
            # Remember the row with the max prob
            if lastArgMax == -1 or lastMax < table[t, c]:
                lastMax = table[t, c]
                lastArgMax = t

    # Return cell index with max prob in last column
    c = table.shape[1] - 1
    t = table[:, c].argmax()
    return t, c

