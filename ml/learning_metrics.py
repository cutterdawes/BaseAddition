import argparse
import pickle
import numpy as np
import addition_data
import torch
import addition_eval
from models import RNN, GRU, LSTM


def main():
    # create and parse arguments
    parser = argparse.ArgumentParser(description='compute the valid carry tables for specified base')
    parser.add_argument('-b', '--base', type=int, required=True,
                        help='specified base')
    parser.add_argument('-m', '--model', type=str, required=False, default='RNN',
                        help='model type (RNN, GRU, or LSTM; default: RNN)')
    parser.add_argument('-e', '--epochs', type=int, required=False, default=2500,
                        help='number of training epochs (default: 2500)')
    parser.add_argument('-t', '--trials', type=int, required=False, default=10,
                        help='number of training trials (default: 10)')
    parser.add_argument('-w', '--workers', type=int, required=False, default=0,
                        help='number of CPU workers for data preparation (default: 0)')
    parser.add_argument('-d', '--directory', type=str, required=False, default='../pickles/learning_metrics',
                        help='directory of pickled output (default: ../pickles/learning_metrics)')
    args = parser.parse_args()

    # get carry tables
    with open('../pickles/carry_tables/all_tables_d1_b2-6.pickle', 'rb') as f:
        all_tables = pickle.load(f)
    tables = all_tables[args.base]
    
    # specify torch device (set to GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # train model for each table
    all_learning_metrics = {}
    models = {'RNN': RNN, 'GRU': GRU, 'LSTM': LSTM}
    lrs = {'RNN': 0.01, 'GRU': 0.05, 'LSTM': 0.05}
    for dc, table in tables.items():
            
        # initialize learning metrics
        avg_losses = np.zeros(int(args.epochs / 10))
        avg_training_accs = np.zeros(int(args.epochs / 10))
        avg_testing_accs = np.zeros(int(args.epochs / 10))

        # evaluate model multiple times, average metrics
        for _ in range(args.trials):

            # initialize model and dataloaders
            model = models[args.model](args.base, 1).to(device)
            training_dataloader, testing_dataloader = addition_data.prepare(
                b=args.base, depth=6, table=table, batch_size=64, split_type='OOD', split_depth=3, sample=True, num_workers=args.workers
            )

            # evaluate model and store output
            lr = lrs[args.model]
            losses, training_accs, testing_accs = addition_eval.eval(
                model, training_dataloader, testing_dataloader, device, epochs=args.epochs, lr=lr, print_loss_and_acc=False
            )
            avg_losses += (np.array(losses) / args.trials)
            avg_training_accs += (np.array(training_accs) / args.trials)
            avg_testing_accs += (np.array(testing_accs) / args.trials)
        
        # add to learning metrics
        learning_metrics = {'loss': avg_losses, 'training_acc': avg_training_accs, 'testing_acc': avg_testing_accs}
        all_learning_metrics[dc] = learning_metrics

        # print progress
        print(f'completed trials for table:\n{table}\n')

    # pickle all learning metrics
    with open(f'{args.directory}/learning_metrics{args.base}_{args.model}_{args.trials}trials.pickle', 'wb') as f:
        pickle.dump(all_learning_metrics, f)


if __name__ == '__main__':
    main()