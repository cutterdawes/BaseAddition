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
    parser = argparse.ArgumentParser(description='train recurrent model to add with each carry function')
    parser.add_argument('-b', '--base', type=int, required=True,
                        help='specified base')
    parser.add_argument('-m', '--model', type=str, required=False, default='RNN',
                        help='model type (RNN, GRU, or LSTM; default: RNN)')
    parser.add_argument('-e', '--epochs', type=int, required=False, default=2500,
                        help='number of training epochs (default: 2500)')
    parser.add_argument('-t', '--trials', type=int, required=False, default=10,
                        help='number of training trials (default: 10)')
    parser.add_argument('-p', '--parallel', action='store_true', default=False,
                        help='run in parallel with MPI (default: False)')
    parser.add_argument('-d', '--directory', type=str, required=False, default='pickles/learning_metrics',
                        help='directory of pickled output (default: pickles/learning_metrics)')
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

    # specify torch device (set to GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initialize learning metrics, learning rate, hidden dim
    if args.parallel:
        local_learning_metrics = {}
    else:
        all_learning_metrics = {}
    lrs = {'RNN': 0.005, 'GRU': 0.05, 'LSTM': 0.05}
    lr = lrs[args.model]
    hidden_dim = 2*args.base if args.model == 'RNN' else args.base

    # train model on each table
    for i, (dc, table) in enumerate(tables.items()):
        if args.parallel and (i % size != rank):
            continue
        
        # initialize learning metrics
        avg_losses = np.zeros(int(args.epochs / 10))
        avg_training_accs = np.zeros(int(args.epochs / 10))
        avg_testing_accs = np.zeros(int(args.epochs / 10))

        # evaluate model multiple times, average metrics
        for _ in range(args.trials):
            
            # initialize model and dataloaders
            model = RecurrentModel(args.base, hidden_dim, args.model).to(device)
            training_dataloader, testing_dataloader = dataset.prepare(
                b=args.base, depth=6, table=table, batch_size=32, split_type='OOD', split_depth=3, sample=True
            )

            # evaluate model and store output
            losses, training_accs, testing_accs = training.eval(
                model, training_dataloader, testing_dataloader, device, epochs=args.epochs, lr=lr, print_loss_and_acc=False
            )
            avg_losses += (np.array(losses) / args.trials)
            avg_training_accs += (np.array(training_accs) / args.trials)
            avg_testing_accs += (np.array(testing_accs) / args.trials)
            
        # add to local (if parallel) or all learning metrics
        learning_metrics = {'loss': avg_losses, 'training_acc': avg_training_accs, 'testing_acc': avg_testing_accs}
        if args.parallel:
            local_learning_metrics[dc] = learning_metrics
            print(f'Rank {rank}: completed trials for table:\n{table}\n')
        else:
            all_learning_metrics[dc] = learning_metrics
            print(f'completed trials for table:\n{table}\n')

    # if parallel, gather all local learning metrics
    if args.parallel:
        all_local_learning_metrics = comm.gather(local_learning_metrics, root=0)
        # if root, combine and save results
        if rank == 0:
            all_learning_metrics = {}
            for local_learning_metrics in all_local_learning_metrics:
                all_learning_metrics.update(local_learning_metrics)

    # pickle all learning metrics
    with open(f'{args.directory}/learning_metrics{args.base}_{args.model}_{args.trials}trials.pickle', 'wb') as f:
        pickle.dump(all_learning_metrics, f)


if __name__ == '__main__':
    main()