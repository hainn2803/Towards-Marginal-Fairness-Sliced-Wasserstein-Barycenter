import argparse
import os
from swae_classification.models.mnist_classification import MNIST_Classification
from swae_classification.trainer import SWAEBatchTrainer
from swae_classification.distributions import *
import torch.optim as optim
import torchvision.utils as vutils
from dataloader.dataloader import *
from utils import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def main():
    # train args
    parser = argparse.ArgumentParser(description='Sliced Wasserstein Autoencoder PyTorch')
    parser.add_argument('--dataset', default='mnist', help='dataset name')
    parser.add_argument('--num-classes', type=int, default=10, help='number of classes')
    parser.add_argument('--datadir', default='/input/', help='path to dataset')
    parser.add_argument('--outdir', default='/output/', help='directory to output images and model checkpoints')

    parser.add_argument('--batch-size', type=int, default=500, metavar='BS',
                        help='input batch size for training (default: 500)')
    parser.add_argument('--batch-size-test', type=int, default=500, metavar='BST',
                        help='input batch size for evaluating (default: 500)')

    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.0005)')

    parser.add_argument('--weight_swd', type=float, default=1,
                        help='weight of swd (default: 1)')
    parser.add_argument('--weight_fsw', type=float, default=1,
                        help='weight of fsw (default: 1)')

    parser.add_argument('--method', type=str, default='FEFBSW', metavar='MED',
                        help='method (default: FEFBSW)')
    parser.add_argument('--num-projections', type=int, default=10000, metavar='NP',
                        help='number of projections (default: 500)')
    parser.add_argument('--embedding-size', type=int, default=48, metavar='ES',
                        help='embedding latent space (default: 48)')

    parser.add_argument('--alpha', type=float, default=0.9, metavar='A',
                        help='RMSprop alpha/rho (default: 0.9)')
    parser.add_argument('--beta1', type=float, default=0.9, metavar='B1',
                        help='Adam beta1 (default: 0.9)')
    parser.add_argument('--beta2', type=float, default=0.999, metavar='B2',
                        help='Adam beta2 (default: 0.999)')

    parser.add_argument('--distribution', type=str, default='circle', metavar='DIST',
                        help='Latent Distribution (default: circle)')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer (default: adam)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-workers', type=int, default=8, metavar='N',
                        help='number of dataloader workers if device is CPU (default: 8)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')

    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='number of batches to log training status (default: 10)')
    parser.add_argument('--saved-model-interval', type=int, default=100, metavar='N',
                        help='number of epochs to save training artifacts (default: 1)')
                        
    parser.add_argument('--lambda-obsw', type=float, default=1.0, metavar='OBSW',
                        help='hyper-parameter of OBSW method')
    args = parser.parse_args()
    # create output directory

    if args.method == "OBSW":
        args.method = f"OBSW_{args.lambda_obsw}"

    args.outdir = os.path.join(args.outdir, args.dataset)
    args.outdir = os.path.join(args.outdir, f"seed_{args.seed}")
    args.outdir = os.path.join(args.outdir, f"lr_{args.lr}")
    args.outdir = os.path.join(args.outdir, f"fsw_{args.weight_fsw}")
    args.outdir = os.path.join(args.outdir, args.method)

    args.datadir = os.path.join(args.datadir, args.dataset)

    outdir_checkpoint = os.path.join(args.outdir, "checkpoint")
    outdir_convergence = os.path.join(args.outdir, "convergence")
    outdir_latent = os.path.join(args.outdir, "latent")

    os.makedirs(args.datadir, exist_ok=True)
    os.makedirs(outdir_checkpoint, exist_ok=True)
    os.makedirs(outdir_convergence, exist_ok=True)
    os.makedirs(outdir_latent, exist_ok=True)

    # determine device and device dep. args
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    # set random seed
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    print(f"Method: {args.method} \n")
    if args.optimizer == 'rmsprop':
        print(
            'batch size {}\nepochs {}\nRMSprop lr {} alpha {}\ndistribution {}\nusing device {}\nseed set to {}\n'.format(
                args.batch_size, args.epochs, args.lr, args.alpha, args.distribution, device.type, args.seed
            ))
    else:
        print(
            'batch size {}\nepochs {}\n{}: lr {} betas {}/{}\ndistribution {}\nusing device {}\nseed set to {}\n'.format(
                args.batch_size, args.epochs, args.optimizer,
                args.lr, args.beta1, args.beta2, args.distribution,
                device.type, args.seed
            ))

    # build train and test set data loaders
    if args.dataset == 'mnist':
        data_loader = MNISTDataLoader(data_dir=args.datadir, train_batch_size=args.batch_size, test_batch_size=args.batch_size_test)
        train_loader, test_loader = data_loader.create_dataloader()
        model = MNIST_Classification().to(device)
    elif args.dataset == 'mnist_lt':
        data_loader = MNISTLTDataLoader(data_dir=args.datadir, train_batch_size=args.batch_size, test_batch_size=args.batch_size_test)
        train_loader, test_loader = data_loader.create_dataloader()
        model = MNIST_Classification().to(device)
    else:
        raise NotImplementedError

    # create optimizer
    if args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=args.alpha)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    elif args.optimizer == 'adamax':
        optimizer = optim.Adamax(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    elif args.optimizer == 'adamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

    if args.dataset == 'mnist' or args.dataset == 'mnist_lt':
        if args.distribution == 'circle':
            distribution_fn = rand_cirlce2d
        elif args.distribution == 'ring':
            distribution_fn = rand_ring2d
        else:
            distribution_fn = rand_uniform2d
    else:
        if args.distribution == 'uniform':
            distribution_fn = rand(args.embedding_size)
        elif args.distribution == 'normal':
            distribution_fn = randn(args.embedding_size)
        else:
            raise ('distribution {} not supported'.format(args.distribution))

    # create batch sliced_wasserstein autoencoder trainer
    trainer = SWAEBatchTrainer(autoencoder=model,
                               optimizer=optimizer,
                               distribution_fn=distribution_fn,
                               num_classes=data_loader.num_classes,
                               num_projections=args.num_projections,
                               weight_swd=args.weight_swd,
                               weight_fsw=args.weight_fsw,
                               device=device,
                               method=args.method,
                               lambda_obsw=args.lambda_obsw)

    list_loss = list()

    METHOD_NAME = {
        "EFBSW": "es-MFSWB",
        "FBSW": "us-MFSWB",
        "lowerboundFBSW": "s-MFSWB",
        "OBSW_0.1": "MFSWB $\lambda = 0.1$",
        "OBSW_1.0": "MFSWB $\lambda = 1.0$",
        "OBSW_10.0": "MFSWB $\lambda = 10.0$",
        "BSW": "USWB",
        "None": "SWAE"
    }

    for epoch in range(args.epochs):
        print('training...')
        model.train()

        for batch_idx, (x, y) in enumerate(train_loader, start=0):
            batch = trainer.train_on_batch(x, y)

            # for i in range(data_loader.num_classes):
            #     print(i, torch.sum(y == i))

            if (batch_idx + 1) % args.log_interval == 0:
                print('Train Epoch: {} ({:.2f}%) [{}/{}]\tLoss: {:.6f}'.format(
                    epoch + 1, float(epoch + 1) / (args.epochs) * 100.,
                    (batch_idx + 1), len(train_loader),
                    batch['loss'].item()))
        model.eval()

        with torch.no_grad():

            if (epoch + 1) % args.saved_model_interval == 0 or (epoch + 1) == args.epochs:

                test_encode, test_targets, test_loss = list(), list(), 0.0
                total_corrects, num_instances = torch.zeros(data_loader.num_classes), torch.zeros(data_loader.num_classes)

                for test_batch_idx, (x_test, y_test) in enumerate(test_loader, start=0):
                    test_evals = trainer.test_on_batch(x_test, y_test)
                    corrects = test_evals['y_pred'].to('cpu') == y_test.to('cpu')

                    for i in range(data_loader.num_classes):
                        total_corrects[i] += torch.sum(corrects[y_test == i])
                        num_instances[i] += torch.sum(y_test == i)
                        # print(i, total_corrects[i], num_instances[i])

                    test_encode.append(test_evals['z_latent'].detach())
                    test_loss += test_evals['loss'].item()
                    test_targets.append(y_test)

                cls_accuracy = total_corrects / num_instances
                accuracy = torch.sum(total_corrects) / torch.sum(num_instances)
                cls_acc = dict()
                for i in range(data_loader.num_classes):
                    cls_acc[i] = cls_accuracy[i]

                test_loss /= len(test_loader)
                list_loss.append(test_loss)
                test_encode, test_targets = torch.cat(test_encode), torch.cat(test_targets)
                test_encode, test_targets = test_encode.cpu().numpy(), test_targets.cpu().numpy()

                # Replace the print statements with the following code
                with open(f"{args.outdir}/output.txt", "a") as f:  # Open the file in append mode
                    f.write(f"On Epoch {epoch + 1}, evaluate on test set: \n")
                    f.write(f"Accuracy on each class: {cls_acc}, total accuracy: {accuracy}\n")
                    f.write(f"Total accuracy: {accuracy}\n")
                    f.write(f"Fairness degreee of accuracy: {compute_fairness(cls_accuracy)}\n")
                    f.write(f"Averaging Distance degreee of accuracy: {compute_averaging_distance(cls_accuracy)}\n")

                print(f"On Epoch {epoch + 1}, evaluate on test set:")
                print(f"Accuracy on each class: {cls_acc}, total accuracy: {accuracy}")
                print(f"Total accuracy: {accuracy}")
                print(f"Fairness degree of accuracy: {compute_fairness(cls_accuracy)}")
                print(f"Averaging Distance degree of accuracy: {compute_averaging_distance(cls_accuracy)}")

                print(f"Shape of test dataset to plot: {test_encode.shape}, {test_targets.shape}")

                if test_encode.shape[1] >= 2:
                    tsne = TSNE(n_components=2, random_state=args.seed)
                    test_encode = tsne.fit_transform(test_encode)

                # plot
                plt.figure(figsize=(10, 10))
                classes = np.unique(test_targets)
                colors = plt.cm.Spectral(np.linspace(0, 1, len(classes)))
                for i, class_label in enumerate(classes):
                    plt.scatter(test_encode[test_targets == class_label, 0],
                                test_encode[test_targets == class_label, 1],
                                c=[colors[i]],
                                cmap=plt.cm.Spectral,
                                label=class_label)

                plt.legend()
                # plt.rc('text', usetex=True)
                # title = f'{METHOD_NAME[args.method]}' + " F={:.3f}, W={:.3f}".format(F, W)
                # title = f'{args.method}' + " F={:.3f}, W={:.3f}".format(F, W)
                title = f'{args.method}'
                plt.title(title)
                plt.savefig('{}/epoch_{}_test_latent.pdf'.format(outdir_latent, epoch))
                plt.close()

                e = epoch + 1
                outdir_end = os.path.join(outdir_checkpoint, f"epoch_{e}")
                imagesdir_epoch = os.path.join(outdir_end, "images")
                chkptdir_epoch = os.path.join(outdir_end, "model")

                os.makedirs(imagesdir_epoch, exist_ok=True)
                os.makedirs(chkptdir_epoch, exist_ok=True)
                torch.save(model.state_dict(), '{}/{}_{}.pth'.format(chkptdir_epoch, args.dataset, args.method))

    plot_convergence(range(1, len(list_loss) + 1), list_loss, 'Test loss',
                     f'In testing loss convergence plot of {args.method}',
                     f"{outdir_convergence}/test_loss_convergence.png")


if __name__ == '__main__':
    main()
