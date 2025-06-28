import argparse
import pickle
import numpy as np
import torch
from mpi4py import MPI

from ml import dataset
from ml import training
from ml.model import RecurrentModel


def main():
    # create and parse arguments
    parser = argparse.ArgumentParser(description='train recurrent model to add with each carry function, under specified semanticity')
    parser.add_argument('-b', '--base', type=int, required=True,
                        help='Specified base')
    parser.add_argument('-N', '--num_digits', type=int, required=False, default=6,
                        help='maximum number of digits to generalize to (default: 6)')
    parser.add_argument('-m', '--model', type=str, required=False, default='RNN',
                        help='model type (RNN, GRU, or LSTM; default: RNN)')
    parser.add_argument('-e', '--epochs', type=int, required=False, default=2500,
                        help='number of training epochs (default: 2500)')
    parser.add_argument('-t', '--trials', type=int, required=False, default=10,
                        help='number of training trials (default: 10)')
    parser.add_argument('-p', '--parallel', action='store_true', default=False,
                        help='run in parallel with MPI (default: False)')
    parser.add_argument('-d', '--directory', type=str, required=False, default='pickles/learning_metrics',
                        help='directory of pickled output')
    args = parser.parse_args()

    # initialize MPI if parallel
    if args.parallel:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    # get carry tables
    with open('pickles/carry_tables/all_tables_d1_b2-6.pickle', 'rb') as f:
        all_tables = pickle.load(f)
    tables = all_tables[args.base]

    # specify torch device (set to GPU if available), set number of CPU workers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initialize learning rate, hidden dim
    lrs = {'RNN': 0.005, 'GRU': 0.05, 'LSTM': 0.05}
    lr = lrs[args.model]
    hidden_dim = 2*args.base if args.model == 'RNN' else args.base

    # initialize metrics storage
    all_learning_metrics = {}

    # train model for each table
    for dc, table in tables.items():
        # check if table is a single value carry table, and find corresponding unit
        if not len(np.unique(table)) == 2:
            continue
        else:
            unit = np.unique(table)[1]

        # initialize learning metrics
        avg_ood_accs = np.zeros(args.num_digits - 2)

        # evaluate model multiple times, average metrics
        for trial in range(args.trials):
            if args.parallel and (trial % size != rank):
                continue

            # initialize OOD metrics
            ood_accs = []

            # train model
            model = RecurrentModel(args.base, hidden_dim, args.model).to(device)
            training_dataloader, testing_dataloader = dataset.prepare(
                b=args.base, depth=6, table=table, semanticity=True, unit=unit,
                batch_size=32, split_type='OOD', split_depth=3, sample=True
            )
            training.train(model, training_dataloader, device, epochs=args.epochs, lr=lr)

            # test on training data
            acc = training.test(model, training_dataloader, device=device, return_accuracy=True)
            ood_accs.append(acc)

            # OOD test up to d digits
            for d in range(4, args.num_digits + 1):
                # prepare dataloaders
                training_dataloader, testing_dataloader = dataset.prepare(
                    b=args.base, depth=d, table=table, semanticity=True, unit=unit,
                    batch_size=32, split_type='OOD', split_depth=3, sample=True
                )

                # test model
                acc = training.test(model, testing_dataloader, device=device, return_accuracy=True)
                ood_accs.append(acc)
                
            # average OOD accuracies
            if args.parallel:
                if rank == 0:
                    all_ood_accs = comm.gather(ood_accs, root=0)
                    for ood_accs in all_ood_accs:
                        avg_ood_accs += np.array(ood_accs) / args.trials
            else:
                avg_ood_accs += np.array(ood_accs) / args.trials

        # add to local (if parallel) or all learning metrics
        all_learning_metrics[dc] = avg_ood_accs
        print(f'completed trials for table:\n{table}\n')

    # pickle all learning metrics
    with open(f'{args.directory}/learning_metrics{args.base}_ood{args.num_digits}_semantic_{args.model}_{args.trials}trials.pickle', 'wb') as f:
        pickle.dump(all_learning_metrics, f)


if __name__ == '__main__':
    main()