"""Coordinates the complete prediction pipeline

All prediction steps have to be added to the prediction pipeline.
"""

import os
from shutil import copyfile
from multiprocessing import cpu_count, Pool
from time import time
import argparse
from evaluation import Evaluator
import preprocessing as pre
import cnn
import postprocessing as post


# List contains every prediction step that is executed in order to produce
# the final prediction
prediction_steps = [
    pre.clean_map,
    pre.find_threshold,
    pre.normalize_map,
    cnn.predict_with_module,
    post.build_backbone_trace,
    post.helix_refinement
]


def run_predictions(input_path, output_path, thresholds_file, num_skip, check_existing):
    """Creates thread pool which will concurrently run the prediction for every
    protein map in the 'input_path'

    Parameters
    ----------
    input_path: str
        Path of the input directory where the different protein directories are
        located

    output_path: str
        Path of the folder where all generated files will be stored

    thresholds_file: str
        Path of the JSON file which contains the threshold values for the input
        files

    num_skip: int
        The number of prediction steps that should be skipped

    check_existing: bool
        If set prediction steps are only executed if their results are not
        existing in the output path yet
    """
    emdb_ids = filter(lambda d: os.path.isdir(input_path + d), os.listdir(input_path))
    pipeline = PredictionPipeline(input_path,
                                  output_path,
                                  thresholds_file,
                                  num_skip,
                                  check_existing,
                                  prediction_steps)

    start_time = time()
    pool = Pool(min(cpu_count(), len(emdb_ids)))
    results = pool.map(pipeline.run, emdb_ids)

    # Filter 'None' results
    results = filter(lambda r: r is not None, results)

    evaluator = Evaluator(input_path)
    for prediction_result in results:
        evaluator.evaluate(prediction_result)

    evaluator.create_report(output_path, time() - start_time)


class PredictionPipeline:

    def __init__(self, input_path, output_path, thresholds_file, num_skip, check_existing, prediction_steps):
        self.input_path = input_path
        self.output_path = output_path
        self.thresholds_file = thresholds_file
        self.num_skip = num_skip
        self.check_existing = check_existing
        self.prediction_steps = prediction_steps

    def run(self, emdb_id):
        # Directory that contains paths to all relevant files. This will be
        # updated with every prediction step
        paths = self._make_paths(emdb_id)

        start_time = time()
        for prediction_step in self.prediction_steps:
            paths['output'] = self.output_path + emdb_id + '/' + prediction_step.__name__.split('.')[0] + '/'
            os.makedirs(paths['output'], exist_ok=True)

            prediction_step.update_paths(paths)
            if self.num_skip > 0 or (self.check_existing and not files_exist(paths)):
                self.num_skip -= 1
            else:
                prediction_step.execute(paths)

        if os.path.isfile(paths['traces_refined']):
            copyfile(paths['traces_refined'], self.output_path + emdb_id + '/' + emdb_id + '.pdb')

        return PredictionResult(emdb_id, paths['traces_refined'], paths['ground_truth'], time() - start_time)

    def _make_paths(self, emdb_id):
        mrc_file = get_file(self.input_path + emdb_id, ['mrc', 'map'])
        gt_file = get_file(self.input_path + emdb_id, ['pdb', 'ent'])
        # Directory that contains paths to all relevant files. This will be
        # updated with every prediction step
        paths = {
            'input': self.input_path + emdb_id + '/' + mrc_file,
            'ground_truth': self.input_path + emdb_id + '/' + gt_file,
        }

        if self.thresholds_file is not None:
            paths['thresholds_file'] = self.thresholds_file

        return paths


class PredictionResult:

    def __init__(self, emdb_id, predicted_file, gt_file, execution_time):
        emdb_id = emdb_id
        predicted_file = predicted_file
        gt_file = gt_file
        execution_time = execution_time


def files_exist(paths):
    """Checks if all files specified in the 'paths' dict exist"""
    for path in paths.values():
        if not os.path.isdir(path) and not os.path.isfile(path):
            return False

    return True


def get_file(path, allowed_extensions):
    """Returns file in path with allowed extension"""
    return next(f for f in os.listdir(path) if f.split('.')[-1] in allowed_extensions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cα Backbone Prediction from High Resolution CryoEM Data')
    parser.add_argument('input', type=str, help='Folder containing protein maps')
    parser.add_argument('output', type=str, help='Folder where prediction results will be stored')
    parser.add_argument('-t', '--thresholds', metavar='Thresholds', type=str,
                        help='JSON file which contains the thresholds')
    parser.add_argument('-s', '--skip', metavar='N', type=int, nargs=1, default=0,
                        help='Number of prediction steps that should be skipped')
    parser.add_argument('-c', '--check_existing', action='store_const', const=True, default=False,
                        help='Check if results already exists and if so skip prediction step')

    args = parser.parse_args()

    args.input += '/' if args.input[-1] != '/' else ''
    args.output += '/' if args.output[-1] != '/' else ''

    run_predictions(args.input, args.output, args.thresholds, args.skip, args.check_existing)
