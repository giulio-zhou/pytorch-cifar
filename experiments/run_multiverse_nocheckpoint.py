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

def run_experiment(args, experiment_name):

    # Directory management
    base_directory = "/proj/BigLearning/ahjiang/output/"
    dataset_directory = os.path.join(base_directory, args.dataset)
    output_directory = os.path.join(dataset_directory, experiment_name)
    pickle_directory = os.path.join(output_directory, "pickles")
    make_dir(dataset_directory)
    make_dir(output_directory)
    make_dir(pickle_directory)

    # Random seed management
    num_seeds = 1
    seeds = [i + 1337 for i in range(0, num_seeds * 10, 10)]

    for trial, seed in enumerate(seeds):
        output_file = get_output_file(args, trial+1, seed)
        args.pickle_prefix = get_output_prefix(args, trial+1, seed)

        # Capture stdout to output file and run experiment
        stdout_ = sys.stdout
        output_path = os.path.join(output_directory, output_file)
        print("Writing results to {}".format(output_path))
        sys.stdout = open(output_path, 'w')
        main.main(args)
        sys.stdout.flush()
        sys.stdout = stdout_

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch SB Training')
    main.set_experiment_default_args(parser)

    # New experiment-specific args
    parser.add_argument('--experiment-prefix', help='experiment prefix')
    parser.add_argument('--num-extra-epochs', type=int, help='number of epochs to perform experiment for')
    args = parser.parse_args()

    checkpoint_epochs = [-1]

    args.max_num_backprops = 50000 * args.num_extra_epochs
    print("Overriding max num backprops to {}".format(args.max_num_backprops))

    args.sb_start_epoch = 0
    print("Overriding sb start epoch to {}".format(args.sb_start_epoch))

    args.augment = True
    print("Overriding augment to {}".format(args.augment))

    selectivities = [1, 0.4, 0.6, 0.8]
    #selectivities = [0.2, 1, 0.4, 0.6, 0.8]

    for epoch in checkpoint_epochs:
        experiment_name = "{}_epoch{}".format(args.experiment_prefix, 
                                              epoch)
        for selectivity in selectivities:
            args.sampling_min = selectivity
            print("Overriding sampling_min to {}".format(args.sampling_min))
            run_experiment(args, experiment_name)

