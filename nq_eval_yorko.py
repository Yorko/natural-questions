# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
r"""Evaluation script for TensorFlow 2.0 Question Answering,
    adapted by yorko from nq_eval, Natural Questions

  https://ai.google.com/research/NaturalQuestions
  https://www.kaggle.com/c/tensorflow2-question-answering/

  ------------------------------------------------------------------------------

  Example usage:

  nq_eval_yorko --gold_path=<path-to-gold-files> --predictions_path=<path_to_json>

  This will compute both the official F1 scores as well as recall@precision
  tables for both long and short answers. Note that R@P are only meaningful
  if your model populates the score fields of the prediction JSON format.

  gold_path should point to the five way annotated dev data in the
  original download format (gzipped jsonlines).

  predictions_path should point to a json file containing the predictions in
  the format given below.

  ------------------------------------------------------------------------------

  Prediction format:

  {'predictions': [
    {
      'example_id': -2226525965842375672,
      'long_answer': {
        'start_byte': 62657, 'end_byte': 64776,
        'start_token': 391, 'end_token': 604
      },
      'long_answer_score': 13.5,
      'short_answers': [
        {'start_byte': 64206, 'end_byte': 64280,
         'start_token': 555, 'end_token': 560}, ...],
      'short_answers_score': 26.4,
      'yes_no_answer': 'NONE'
    }, ... ]
  }
"""

from collections import OrderedDict
from absl import app
from absl import flags
import eval_utils as util

flags.DEFINE_string(
    'gold_path', None, 'Path to the gzip JSON data. For '
                       'multiple files, should be a glob '
                       'pattern (e.g. "/path/to/files-*"')
flags.DEFINE_string('predictions_path', None, 'Path to prediction JSON.')
flags.DEFINE_integer('num_threads', 10, 'Number of threads for reading.')
flags.DEFINE_float('score_thres_long', -100, 'Score threshold for long answers')
flags.DEFINE_float('score_thres_short', -100, 'Score threshold for short answers')

FLAGS = flags.FLAGS


def safe_divide(x, y):
    """Compute x / y, but return 0 if y is zero."""
    if y == 0:
        return 0
    else:
        return x / y


def score_long_answer(gold_label_list, pred_label, score_thres):
    """Scores a long answer as correct or not.

    1) First decide if there is a gold long answer with LONG_NO_NULL_THRESHOLD.
    2) The prediction will get a match if:
       a. There is a gold long answer.
       b. The prediction span match exactly with *one* of the non-null gold
          long answer span.

    Args:
      gold_label_list: A list of NQLabel, could be None.
      pred_label: A single NQLabel, could be None.
      score_thres; score threshold

    Returns:
      gold_has_answer, pred_has_answer, is_correct, score
    """
    gold_has_answer = util.gold_has_long_answer(gold_label_list)

    is_correct = False
    score = pred_label.long_score

    pred_has_answer = pred_label and (
        not pred_label.long_answer_span.is_null_span()) and score >= score_thres

    # Both sides are non-null spans.
    if gold_has_answer and pred_has_answer:
        for gold_label in gold_label_list:
            # while the voting results indicate there is an long answer, each
            # annotator might still say there is no long answer.
            if gold_label.long_answer_span.is_null_span():
                continue

            if util.nonnull_span_equal(gold_label.long_answer_span,
                                       pred_label.long_answer_span):
                is_correct = True
                break

    return gold_has_answer, pred_has_answer, is_correct, score


def score_short_answer(gold_label_list, pred_label, score_thres):
    """Scores a short answer as correct or not.

    1) First decide if there is a gold short answer with SHORT_NO_NULL_THRESHOLD.
    2) The prediction will get a match if:
       a. There is a gold short answer.
       b. The prediction span *set* match exactly with *one* of the non-null gold
          short answer span *set*.

    Args:
      gold_label_list: A list of NQLabel.
      pred_label: A single NQLabel.
      score_thres: score threshold

    Returns:
      gold_has_answer, pred_has_answer, is_correct, score
    """

    # There is a gold short answer if gold_label_list not empty and non null
    # answers is over the threshold (sum over annotators).
    gold_has_answer = util.gold_has_short_answer(gold_label_list)

    is_correct = False
    score = pred_label.short_score

    # There is a pred long answer if pred_label is not empty and short answer
    # set is not empty.
    pred_has_answer = pred_label and (
            (not util.is_null_span_list(pred_label.short_answer_span_list)) or
            pred_label.yes_no_answer != 'none') and score >= score_thres

    # Both sides have short answers, which contains yes/no questions.
    if gold_has_answer and pred_has_answer:
        if pred_label.yes_no_answer != 'none':  # System thinks its y/n questions.
            for gold_label in gold_label_list:
                if pred_label.yes_no_answer == gold_label.yes_no_answer:
                    is_correct = True
                    break
        else:
            for gold_label in gold_label_list:
                if util.span_set_equal(gold_label.short_answer_span_list,
                                       pred_label.short_answer_span_list):
                    is_correct = True
                    break

    return gold_has_answer, pred_has_answer, is_correct, score


def score_answers(gold_annotation_dict, pred_dict,
                  score_thres_long, score_thres_short):
    """Scores all answers for all documents.

    Args:
      gold_annotation_dict: a dict from example id to list of NQLabels.
      pred_dict: a dict from example id to list of NQLabels.
      score_thres_long: score threshold for long answers
      score_thres_short: score threshold for short answers

    Returns:
      long_answer_stats: List of scores for long answers.
      short_answer_stats: List of scores for short answers.
    """
    gold_id_set = set(gold_annotation_dict.keys())
    pred_id_set = set(pred_dict.keys())

    if gold_id_set.symmetric_difference(pred_id_set):
        raise ValueError('ERROR: the example ids in gold annotations and example '
                         'ids in the prediction are not equal.')

    long_answer_stats = []
    short_answer_stats = []

    for example_id in gold_id_set:
        gold = gold_annotation_dict[example_id]
        pred = pred_dict[example_id]

        long_answer_stats.append(score_long_answer(gold, pred, score_thres_long))
        short_answer_stats.append(score_short_answer(gold, pred, score_thres_short))

    # use the 'score' column, which is last
    long_answer_stats.sort(key=lambda x: x[-1], reverse=True)
    short_answer_stats.sort(key=lambda x: x[-1], reverse=True)

    return long_answer_stats, short_answer_stats


def compute_f1(answer_stats, prefix=''):
    """Computes F1, precision, recall for a list of answer scores.

    Args:
      answer_stats: List of per-example scores.
      prefix (''): Prefix to prepend to score dictionary.

    Returns:
      Dictionary mapping string names to scores.
    """

    has_gold, has_pred, is_correct, _ = list(zip(*answer_stats))
    precision = safe_divide(sum(is_correct), sum(has_pred))
    recall = safe_divide(sum(is_correct), sum(has_gold))
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return OrderedDict({
        prefix + 'n': len(answer_stats),
        prefix + 'f1': f1,
        prefix + 'precision': precision,
        prefix + 'recall': recall
    })


def compute_final_f1(long_answer_stats, short_answer_stats):
    """Computes overall F1 given long and short answers, ignoring scores.

    Note: this assumes that the answers have been thresholded.

    Arguments:
       long_answer_stats: List of long answer scores.
       short_answer_stats: List of short answer scores.

    Returns:
       Dictionary of name (string) -> score.
    """
    scores = compute_f1(long_answer_stats, prefix='long-answer-')
    scores.update(compute_f1(short_answer_stats, prefix='short-answer-'))
    scores.update(compute_f1(long_answer_stats + short_answer_stats, prefix='all-answer-'))
    return scores


def main(_):
    nq_gold_dict = util.read_annotation(FLAGS.gold_path, n_threads=FLAGS.num_threads)

    nq_pred_dict = util.read_prediction_json(FLAGS.predictions_path)

    long_answer_stats, short_answer_stats = score_answers(nq_gold_dict,
                                                          nq_pred_dict,
                                                          score_thres_long=FLAGS.score_thres_long,
                                                          score_thres_short=FLAGS.score_thres_short)

    # reporting results
    print('*' * 20)

    scores = compute_final_f1(long_answer_stats, short_answer_stats)
    print('*' * 20)
    print('SCORES (n={}):'.format(
        scores['long-answer-n']))
    print('              F1     /  P      /  R')
    print('Long answer  {: >7.2%} / {: >7.2%} / {: >7.2%}'.format(
        scores['long-answer-f1'], scores['long-answer-precision'],
        scores['long-answer-recall']))
    print('Short answer {: >7.2%} / {: >7.2%} / {: >7.2%}'.format(
        scores['short-answer-f1'], scores['short-answer-precision'],
        scores['short-answer-recall']))
    print('All answers  {: >7.2%} / {: >7.2%} / {: >7.2%}'.format(
        scores['all-answer-f1'], scores['all-answer-precision'],
        scores['all-answer-recall']))


if __name__ == '__main__':
    flags.mark_flag_as_required('gold_path')
    flags.mark_flag_as_required('predictions_path')
    app.run(main)
