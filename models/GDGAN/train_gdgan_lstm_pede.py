from __future__ import division, print_function

import argparse
import datetime
import pickle
import time

import torch.optim as optim
from sklearn.cluster import DBSCAN
from sknetwork.topology import get_connected_components
from torch.optim import lr_scheduler

from models.DANTE.F1_calc import group_correctness
from models.WavenetNRI.data_utils import *
from models.WavenetNRI.nri_pede import get_groups_from_ids
from models.WavenetNRI.utils import *
from models_gdgan import *


def train_generator():
    loss_train = []
    mse_train = []
    loss_val = []
    mse_val = []
    sc_train = []
    sc_val = []
    generator.train()
    training_indices = np.arange(len(examples_train))
    np.random.shuffle(training_indices)

    optimizer.zero_grad()
    idx_count = 0
    accumulation_steps = min(args.batch_size, len(examples_train))
    fake_examples = []  # to store fake examples
    hiddens = []  # to store hidden states

    for idx in training_indices:
        example = examples_train[idx]
        # add batch size
        example = example.unsqueeze(0)
        # shape: [1, n_atoms, n_timesteps, n_in]
        n_atoms = example.size(1)  # get number of atoms
        n_timesteps = example.size(2)
        T_obs = int(n_timesteps / 2)
        rel_rec, rel_send = create_edgeNode_relation(n_atoms, self_loops=False)
        rel_rec_t, rel_send_t = create_edgeNode_relation(T_obs, self_loops=True)

        if args.cuda:
            example = example.cuda()
            rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
            rel_rec_t, rel_send_t = rel_rec_t.cuda(), rel_send_t.cuda()

        example = example.float()

        # generate random noise
        noise = torch.randn(1, n_atoms, args.n_noise)
        if args.cuda:
            noise = noise.cuda()

        # generate fake examples and hidden states
        x_fake, hs = generator(example, noise, rel_rec, rel_send, rel_rec_t, rel_send_t)
        # x_fake: [1, n_atoms, n_timesteps, n_in]
        # hs: [1, n_atoms, n_h*pred_timesteps]

        # compute L1 norm of hidden state hs
        loss_sc = args.sc_weight * (torch.norm(hs, p=1, dim=-1).sum()) / (hs.size(0) * hs.size(1))
        sc_train.append(loss_sc.item())

        # compute reconstruction error
        out = x_fake[:, :, T_obs + 1:, :]
        target = example[:, :, T_obs + 1:, :]
        loss_nll = nll_gaussian(out, target, args.var)
        loss_mse = F.mse_loss(out, target)

        loss = loss_nll + loss_sc
        loss = loss / accumulation_steps
        loss.backward()

        idx_count += 1

        if idx_count % args.batch_size == 0 or idx_count == len(examples_train):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            accumulation_steps = min(args.batch_size, len(examples_train) - idx_count)

        mse_train.append(loss_mse.item())
        loss_train.append(loss_nll.item())
        sc_train.append(loss_sc.item())

    generator.eval()

    valid_indices = np.arange(len(examples_valid))

    with torch.no_grad():
        for idx in valid_indices:
            example = examples_valid[idx]
            # add batch size
            example = example.unsqueeze(0)
            # shape: [1, n_atoms, n_timesteps, n_in]
            n_atoms = example.size(1)  # get number of atoms
            n_timesteps = example.size(2)  # get timesteps
            T_obs = int(n_timesteps / 2)
            rel_rec, rel_send = create_edgeNode_relation(n_atoms, self_loops=False)
            rel_rec_t, rel_send_t = create_edgeNode_relation(T_obs, self_loops=True)

            if args.cuda:
                example = example.cuda()
                rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
                rel_rec_t, rel_send_t = rel_rec_t.cuda(), rel_send_t.cuda()

            example = example.float()
            # generate random noise
            noise = torch.randn(1, n_atoms, args.n_noise)
            if args.cuda:
                noise = noise.cuda()
            # generate fake examples and hidden states
            x_fake, hs = generator(example, noise, rel_rec, rel_send, rel_rec_t, rel_send_t)
            # x_fake: [1, n_atoms, n_timesteps, n_in]
            # hs: [1, n_atoms, n_h*T_pred]
            # compute L1 norm of hidden state hs
            loss_sc = args.sc_weight * (torch.norm(hs, p=1, dim=-1).sum()) / (hs.size(0) * hs.size(1))
            sc_val.append(loss_sc.item())

            # compute reconstruction error
            out = x_fake[:, :, T_obs + 1:, :]
            target = example[:, :, T_obs + 1:, :]
            loss_nll = nll_gaussian(out, target, args.var)
            loss_mse = F.mse_loss(out, target)

            loss_val.append(loss_nll.item())
            mse_val.append(loss_mse.item())

    return np.mean(loss_train), np.mean(mse_train), np.mean(sc_train), np.mean(loss_val), np.mean(mse_val), np.mean(
        sc_val)


def test_generator():
    loss_test = []
    mse_test = []
    sc_test = []

    generator = torch.load(generator_file)
    generator.eval()

    test_indices = np.arange(len(examples_test))

    with torch.no_grad():
        for idx in test_indices:
            example = examples_test[idx]
            example = example.unsqueeze(0)
            # shape: [1, n_atoms, n_timesteps, n_in]
            n_atoms = example.size(1)  # get number of atoms
            n_timesteps = example.size(2)  # get timesteps
            T_obs = int(n_timesteps / 2)
            rel_rec, rel_send = create_edgeNode_relation(n_atoms, self_loops=False)
            rel_rec_t, rel_send_t = create_edgeNode_relation(T_obs, self_loops=True)

            if args.cuda:
                example = example.cuda()
                rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
                rel_rec_t, rel_send_t = rel_rec_t.cuda(), rel_send_t.cuda()

            example = example.float()
            # generate random noise
            noise = torch.randn(1, n_atoms, args.n_noise)
            if args.cuda:
                noise = noise.cuda()
            # generate fake examples and hidden states
            # generate fake examples and hidden states
            x_fake, hs = generator(example, noise, rel_rec, rel_send, rel_rec_t, rel_send_t)
            # x_fake: [1, n_atoms, n_timesteps, n_in]
            # hs: [1, n_atoms, n_h*T_pred]
            # compute L1 norm of hidden state hs
            loss_sc = args.sc_weight * (torch.norm(hs, p=1, dim=-1).sum()) / (hs.size(0) * hs.size(1))
            sc_test.append(loss_sc.item())

            # compute reconstruction error
            out = x_fake[:, :, T_obs + 1:, :]
            target = example[:, :, T_obs + 1:, :]
            loss_nll = nll_gaussian(out, target, args.var)
            loss_mse = F.mse_loss(out, target)

            loss_test.append(loss_nll.item())
            mse_test.append(loss_mse.item())

        print("loss_test: ", np.mean(loss_test),
              "mse_test: ", np.mean(mse_test),
              "sc_test: ", np.mean(sc_test))

    return np.mean(loss_test), np.mean(mse_test), np.mean(sc_test)


def train(epoch, best_val_loss):
    loss_train, mse_train, sc_train, loss_val, mse_val, sc_val = train_generator()
    print("Epoch: {:04d}".format(epoch + 1),
          "loss_train: {:.10f}".format(loss_train),
          "mse_train: {:.10f}".format(mse_train),
          "sc_train: {:.10f}".format(sc_train),
          "loss_val: {:.10f}".format(loss_val),
          "mse_val: {:.10f}".format(mse_val),
          "sc_val: {:.10f}".format(sc_val))
    if config['save_folder'] and loss_val < best_val_loss:
        torch.save(generator, generator_file)
        print("Best model so far, saving...")
        print("Epoch: {:04d}".format(epoch + 1),
              "loss_train: {:.10f}".format(loss_train),
              "mse_train: {:.10f}".format(mse_train),
              "sc_train: {:.10f}".format(sc_train),
              "loss_val: {:.10f}".format(loss_val),
              "mse_val: {:.10f}".format(mse_val),
              "sc_val: {:.10f}".format(sc_val),
              file=log)
        log.flush()

    return loss_train, mse_train, sc_train, loss_val, mse_val, sc_val


def test_gmitre():
    generator = torch.load(generator_file)
    generator.eval()

    precision_all = []
    recall_all = []
    F1_all = []
    precision_one_all = []
    recall_one_all = []
    F1_one_all = []
    precision_two_thirds_all = []
    recall_two_thirds_all = []
    F1_two_thirds_all = []

    test_indices = np.arange(len(examples_test))

    with torch.no_grad():
        for idx in test_indices:
            example = examples_test[idx]
            label = labels_test[idx]
            example = example.unsqueeze(0)
            # [1, n_atoms, n_timesteps, n_in]
            batch_size = example.size(0)
            n_atoms = example.size(1)
            n_timesteps = example.size(2)
            T_obs = int(n_timesteps / 2)
            rel_rec, rel_send = create_edgeNode_relation(n_atoms, self_loops=False)
            rel_rec_t, rel_send_t = create_edgeNode_relation(T_obs, self_loops=True)
            example = example.float()

            label = torch.diag_embed(label)  # shape: [n_edges, n_edges]
            label = label.float()
            label_converted = torch.matmul(rel_send.t(),
                                           torch.matmul(label, rel_rec))
            label_converted = label_converted.cpu().detach().numpy()
            # shape: [n_atoms, n_atoms]

            if label_converted.sum() == 0:
                gID = list(range(label_converted.shape[1]))
            else:
                gID = list(get_connected_components(label_converted))

            if args.cuda:
                example = example.cuda()
                rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
                rel_rec_t, rel_send_t = rel_rec_t.cuda(), rel_send_t.cuda()

            # generate random noise
            noise = torch.randn(batch_size, n_atoms, args.n_noise)
            if args.cuda:
                noise = noise.cuda()

            # generate fake examples and hidden states
            x_fake, hs = generator(example, noise, rel_rec, rel_send, rel_rec_t, rel_send_t)
            # x_fake: [1, n_atoms, n_timesteps, n_in]
            # hs: [1, n_atoms, n_h*pred_timesteps]
            hs = hs.squeeze(0)
            # shape: [n_atoms, n_h*pred_timesteps]
            hs_numpy = hs.cpu().detach().numpy()
            hidden = hs_numpy

            clustering = DBSCAN(min_samples=1).fit(hidden)
            predicted_labels = clustering.labels_
            recall, precision, F1 = compute_groupMitre_labels(gID, predicted_labels)
            predicted_groups = get_groups_from_ids(predicted_labels)
            true_groups = get_groups_from_ids(gID)
            _, _, _, precision_one, recall_one = group_correctness(predicted_groups, true_groups, 1)
            _, _, _, precision_two_thirds, recall_two_thirds = group_correctness(predicted_groups, true_groups, 2 / 3)
            if precision_one * recall_one == 0:
                f1_one = 0
            else:
                f1_one = float(2) * precision_one * recall_one / (precision_one + recall_one)
            if precision_two_thirds * recall_two_thirds == 0:
                f1_two_thirds = 0
            else:
                f1_two_thirds = float(2) * precision_two_thirds * recall_two_thirds / (
                        precision_two_thirds + recall_two_thirds)

            recall_all.append(recall)
            precision_all.append(precision)
            F1_all.append(F1)
            precision_two_thirds_all.append(precision_two_thirds)
            recall_two_thirds_all.append(recall_two_thirds)
            F1_two_thirds_all.append(f1_two_thirds)
            precision_one_all.append(precision_one)
            recall_one_all.append(recall_one)
            F1_one_all.append(f1_one)

        average_recall = np.mean(recall_all)
        average_precision = np.mean(precision_all)
        average_F1 = np.mean(F1_all)

        average_one_recall = np.mean(recall_one_all)
        average_one_precision = np.mean(precision_one_all)
        average_one_F1 = np.mean(F1_one_all)

        average_two_thirds_recall = np.mean(recall_two_thirds_all)
        average_two_thirds_precision = np.mean(precision_two_thirds_all)
        average_two_thirds_F1 = np.mean(F1_two_thirds_all)

    print("Average recall: ", average_recall)
    print("Average precision: ", average_precision)
    print("Average F1: ", average_F1)

    print("Average T=1 recall: ", average_one_recall)
    print("Average T=1 precision: ", average_one_precision)
    print("Average T=1 F1: ", average_one_F1)

    print("Average T=2/3 recall: ", average_two_thirds_recall)
    print("Average T=2/3 precision: ", average_two_thirds_precision)
    print("Average T=2/3 F1: ", average_two_thirds_F1)

    print("Average recall: {:.10f}".format(average_recall),
          "Average precision: {:.10f}".format(average_precision),
          "Average F1: {:.10f}".format(average_F1),
          file=log)

    print("Average T=1 recall: {:.10f}".format(average_one_recall),
          "Average T=1 precision: {:.10f}".format(average_one_precision),
          "Average T=1 F1: {:.10f}".format(average_one_F1),
          file=log)

    print("Average T=2/3 recall: {:.10f}".format(average_two_thirds_recall),
          "Average T=2/3 precision: {:.10f}".format(average_two_thirds_precision),
          "Average T=2/3 F1: {:.10f}".format(average_two_thirds_F1),
          file=log)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument("--no-seed", action="store_true", default=False,
                        help="don't use seed")

    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train.')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Number of samples per batch.')
    parser.add_argument("--lr", type=float, default=5e-3,
                        help="Initial Learning Rate.")

    parser.add_argument("--n-in", type=int, default=2, help="Input dimensions.")
    parser.add_argument('--n-emb', type=int, default=16, help='Dimensions of Embedding')
    parser.add_argument("--n-hid", type=int, default=32,
                        help="Dimensions of hidden states.")
    parser.add_argument("--n-noise", type=int, default=4,
                        help="Dimensions of noise.")
    parser.add_argument("--split", type=str, default="0",
                        help="Split of the dataset.")
    parser.add_argument('--load-folder', type=str, default='',
                        help='Where to load the trained model if finetunning. ' +
                             'Leave empty to train from scratch')
    parser.add_argument('--timesteps', type=int, default=15,
                        help='The number of time steps per sample.')

    parser.add_argument('--lr-decay', type=int, default=200,
                        help='After how epochs to decay LR by a factor of gamma.')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='LR decay factor.')
    parser.add_argument("--sc-weight", type=float, default=0.2,
                        help="Sparse Constraint Weight.")

    parser.add_argument('--var', type=float, default=0.1,
                        help='Output variance.')

    parser.add_argument("--teaching", action="store_true", default=True,
                        help="Whether use teaching force.")
    parser.add_argument('-c', '--config', type=str, default="./config/gdgan_pede.yml")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)
    config = read_yaml(args.config)

    if not args.no_seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    # Save model and meta-data
    if config['save_folder']:
        exp_counter = 0
        now = datetime.datetime.now()
        timestamp = now.isoformat()
        save_folder = "{}/{}/fold_{}/{}".format(config['save_folder'], config['suffix'], args.split, args.seed)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        meta_file = os.path.join(save_folder, 'metadata.pkl')
        generator_file = os.path.join(save_folder, 'generator.pt')
        log_file = os.path.join(save_folder, 'log.txt')
        log = open(log_file, 'w')
        pickle.dump({'args': args}, open(meta_file, "wb"))

    else:
        print("WARNING: No save_folder provided!" +
              "Testing (within this script) will throw an error.")

    # Load data
    data_folder = '{}/fold_{}'.format(config['dataset_folder'], args.split)

    with open(os.path.join(data_folder, "tensors_train.pkl"), 'rb') as f:
        examples_train = pickle.load(f)
    with open(os.path.join(data_folder, "labels_train.pkl"), 'rb') as f:
        labels_train = pickle.load(f)
    with open(os.path.join(data_folder, "tensors_valid.pkl"), 'rb') as f:
        examples_valid = pickle.load(f)
    with open(os.path.join(data_folder, "labels_valid.pkl"), 'rb') as f:
        labels_valid = pickle.load(f)
    with open(os.path.join(data_folder, "tensors_test.pkl"), 'rb') as f:
        examples_test = pickle.load(f)
    with open(os.path.join(data_folder, "labels_test.pkl"), 'rb') as f:
        labels_test = pickle.load(f)

    # examples_train = [torch.from_numpy(example) for example in examples_train]
    # examples_valid = [torch.from_numpy(example) for example in examples_valid]
    # examples_test = [torch.from_numpy(example) for example in examples_test]

    generator = LSTMGenerator(args.n_in, args.n_emb, args.n_hid, args.n_noise)

    if args.cuda:
        generator = generator.cuda()

    optimizer = optim.Adam(list(generator.parameters()), lr=args.lr)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr, gamma=args.gamma)

    # Train model
    t_total = time.time()
    best_val_loss = np.inf
    best_epoch = 0
    for epoch in range(args.epochs):
        loss_train, mse_train, sc_train, loss_val, mse_val, sc_val = train(epoch, best_val_loss)
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            best_epoch = epoch + 1

    print("Optimization Finished!")
    print("Best Epoch: {:04d}".format(best_epoch))
    if config['save_folder']:
        print("Best Epoch: {:04d}".format(best_epoch), file=log)
        log.flush()

    test_generator()

    test_gmitre()
