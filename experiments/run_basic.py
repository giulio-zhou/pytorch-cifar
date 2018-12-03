import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import main


def get_output_prefix(args, trial, seed):
    if args.lr_sched:
        return "{}_{}_{}_{}_{}_0.0_{}_trial{}_seed{}".format(args.sb_strategy,
                                                                args.dataset,
                                                                args.net,
                                                                args.sampling_min,
                                                                args.batch_size,
                                                                args.decay,
                                                                trial,
                                                                seed)
    else:
        return  "{}_{}_{}_{}_{}_{}_{}_trial{}_seed{}".format(args.sb_strategy,
                                                                args.dataset,
                                                                args.net,
                                                                args.sampling_min,
                                                                args.batch_size,
                                                                args.lr,
                                                                args.decay,
                                                                trial,
                                                                seed)

def get_output_file(args, trial, seed):
    prefix = get_output_prefix(args, trial, seed)
    return "{}_v2".format(prefix)

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def run_experiment(args):

    # Directory management
    base_directory = "/proj/BigLearning/ahjiang/output/"
    dataset_directory = os.path.join(base_directory, args.dataset)
    output_directory = os.path.join(dataset_directory, args.experiment_name)
    pickle_directory = os.path.join(output_directory, "pickles")
    make_dir(dataset_directory)
    make_dir(output_directory)
    make_dir(pickle_directory)

    # Random seed management
    num_seeds = 3
    seeds = range(0, num_seeds * 10, 10)

    for seed in seeds:
        output_file = get_output_file(args, 1, seed)
        args.pickle_prefix = get_output_prefix(args, 1, seed)

        # Capture stdout to output file and run experiment
        stdout_ = sys.stdout
        output_path = os.path.join(output_directory, output_file)
        print("Writing results to {}".format(output_path))
        sys.stdout = open(output_path, 'w')
        main.main(args)
        sys.stdout = stdout_

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch SB Training')
    main.set_experiment_default_args(parser)

    # Experiment args
    parser.add_argument('--experiment-name', help='experiment name')

    args = parser.parse_args()

    run_experiment(args)

