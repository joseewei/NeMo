# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================

import numpy as np
import torch
import os

from nemo.collections.nlp.utils.callback_utils import get_classification_report, plot_confusion_matrix, tensor2list
from nemo.utils import logging

__all__ = ['eval_iter_callback', 'eval_epochs_done_callback']


def eval_iter_callback(tensors, global_vars):
    GLOBAL_KEYS = ['punct_labels', 'capit_labels', 'punct_preds', 'capit_preds']
    for key in GLOBAL_KEYS:
        if key not in global_vars:
            global_vars[key] = []

    output = {}
    for k, v in tensors.items():
        name = k.split('~~~')
        if len(name) > 1:
            if name[0] == 'logits':
                if 'Capitalization' in k:
                    output['capit_logits'] = torch.cat(v)
                elif 'Punctuation' in k:
                    output['punct_logits'] = torch.cat(v)
            else:
                output[name[0]] = torch.cat(v)

    subtokens_mask = output['subtokens_mask'] > 0.5
    punct_preds = torch.argmax(output['punct_logits'], axis=-1)
    global_vars['punct_preds'].extend(tensor2list(punct_preds[subtokens_mask]))
    global_vars['capit_preds'].extend(tensor2list(torch.argmax(output['capit_logits'], axis=-1)[subtokens_mask]))
    global_vars['punct_labels'].extend(tensor2list(output['punct_labels'][subtokens_mask]))
    global_vars['capit_labels'].extend(tensor2list(output['capit_labels'][subtokens_mask]))
  
    if 'part_sent_logits' in output:
        if 'part_sent_preds' not in global_vars:
            global_vars['part_sent_preds'] = []
            global_vars['part_sent_labels'] = []
            global_vars['punct_corr_preds'] = []
            global_vars['punct_corr_labels'] = []

        num_examples = output['punct_logits'].shape[0]
        part_sent_preds = tensor2list(torch.argmax(output['part_sent_logits'], -1))

        for i in range(num_examples):
            punct_pred = tensor2list(punct_preds[i, :][(output['subtokens_mask'] > 0.5)[i, :]])
            # if the sentence is predicated to be partial, sent the punct of the last word to pad_label 'O'
            if part_sent_preds[i] == 1:
                punct_pred[-1] = 0
                print('corrected')

            global_vars['punct_corr_preds'].extend(punct_pred)
        global_vars['punct_corr_labels'].extend(tensor2list(output['punct_labels'][subtokens_mask]))
        global_vars['part_sent_labels'].extend(tensor2list(output['part_sent_labels']))
        global_vars['part_sent_preds'].extend(part_sent_preds)


def _get_result_dict(tag, class_report):
    results = {}
    for label in class_report:
        if label != 'accuracy':
            label_name = label[: label.index('(label id') - 1] if 'label id' in label else label
            results[tag + 'F1 ' + label_name] = round(class_report[label]['f1-score'] * 100, 2)
            results[tag + 'PR ' + label_name] = round(class_report[label]['precision'] * 100, 2)
            results[tag + 'R ' + label_name] = round(class_report[label]['recall'] * 100, 2)
        else:
            results[tag + 'Acc'] = round(class_report[label] * 100, 2)
    return results


def eval_epochs_done_callback(
    global_vars, punct_label_ids, capit_label_ids, part_sent_label_ids=None, work_dir=None, graph_fold=None, normalize_cm=True
):
    '''
    Args:
      graph_fold (str): path to output folder
      normalize_cm (bool): flag to indicate whether to
        normalize confusion matrix
    '''
    results = {}
    punct_class_report = _eval_epochs_done_callback('punct', global_vars, punct_label_ids, work_dir, graph_fold, normalize_cm)
    results.update(_get_result_dict('p', punct_class_report))

    if 'punct_corr_preds' in global_vars:
        punct_class_report = _eval_epochs_done_callback('punct_corr', global_vars, punct_label_ids, work_dir, graph_fold, normalize_cm)
        results.update(_get_result_dict('p', punct_class_report))

    capit_class_report = _eval_epochs_done_callback('capit', global_vars, capit_label_ids, work_dir, graph_fold, normalize_cm)
    results.update(_get_result_dict('c', capit_class_report))
   
    if 'part_sent_preds' in global_vars:
        part_sent_labels = np.asarray(global_vars['part_sent_labels'])
        part_sent_preds = np.asarray(global_vars['part_sent_preds'])
        part_sent_class_report = _eval_epochs_done_callback(
            'part_sent', global_vars, part_sent_label_ids, work_dir, graph_fold, normalize_cm
        )
        results.update(_get_result_dict('t', part_sent_class_report))
        part_sent_acc = np.mean(part_sent_labels == part_sent_preds)
        logging.info(f'Partial sent task accuracy: {part_sent_acc}')
        results['Part_sent_acc'] = round(part_sent_acc * 100, 2)
    logging.info(f'results: {results}')
    return results


def _eval_epochs_done_callback(task_name, global_vars, label_ids, work_dir=None, graph_fold=None, normalize_cm=True):
    labels = np.array(global_vars[task_name + '_labels'])
    preds = np.array(global_vars[task_name + '_preds'])
  
    if work_dir is not None:
        with open(os.path.join(work_dir, task_name + '_labels_preds.txt'), 'w') as f:
            f.write(' '.join(list(map(str, labels))))
            f.write('\n')
            f.write(' '.join(list(map(str, preds))))
            
        logging.info(f'labels and preds are saved at {work_dir}')

    # calculate and plot confusion_matrix
    if graph_fold:
        plot_confusion_matrix(labels, preds, graph_fold, label_ids, normalize=normalize_cm, prefix=task_name)

    logging.info(f'{get_classification_report(labels, preds, label_ids)}')
    return get_classification_report(labels, preds, label_ids, output_dict=True)
