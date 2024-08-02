import argparse
import matplotlib as mpl
import os
mpl.use('Agg')
from swae.models.mnist import MNISTAutoencoder
from swae.models.cifar10 import CIFAR10Autoencoder
from swae.models.stl10 import STL10Autoencoder
from swae.distributions import *
from evaluate.eval_ws import *
from evaluate.eval_fid import *
from dataloader.dataloader import *
from utils import *


def main():
    # train args
    parser = argparse.ArgumentParser(description='Sliced Wasserstein Autoencoder PyTorch')
    parser.add_argument('--dataset', default='mnist', help='dataset name')
    parser.add_argument('--num-classes', type=int, default=10, help='number of classes')
    parser.add_argument('--datadir', default='data', help='path to dataset')
    parser.add_argument('--outdir', default='result', help='directory to output images and model checkpoints')
    parser.add_argument('--imagedir', default='images', help='path to generative image directory')
    parser.add_argument('--statdir', default='stats', help='path to statistic directory')
    parser.add_argument('--dims', type=int, default=2048, metavar='ES',
                        help='InceptionV3 layer')

    parser.add_argument('--batch-size', type=int, default=500, metavar='BS',
                        help='input batch size for training (default: 500)')
    parser.add_argument('--batch-size-test', type=int, default=500, metavar='BST',
                        help='input batch size for evaluating (default: 500)')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--weight_swd', type=float, default=1,
                        help='weight of swd (default: 1)')
    parser.add_argument('--weight_fsw', type=float, default=1,
                        help='weight of fsw (default: 1)')

    parser.add_argument('--method', type=str, default='EFBSW', metavar='MED',
                        help='method (default: EFBSW)')
    parser.add_argument('--num-projections', type=int, default=10000, metavar='NP',
                        help='number of projections (default: 500)')

    parser.add_argument('--embedding-size', type=int, default=48, metavar='ES',
                        help='embedding latent space (default: 48)')
    parser.add_argument('--distribution', type=str, default='circle', metavar='DIST',
                        help='Latent Distribution (default: circle)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-workers', type=int, default=8, metavar='N',
                        help='number of dataloader workers if device is CPU (default: 8)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')

    parser.add_argument('--lambda-obsw', type=float, default=1, metavar='OBSW',
                        help='hyper-parameter of OBSW method')

    parser.add_argument('--checkpoint-period', type=int, default=300, metavar='S',
                        help='checkpoint period (100, 200, 300)')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    # set random seed
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        
    if args.method == "OBSW":
        args.method = f"OBSW_{args.lambda_obsw}"

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    args.outdir = os.path.join(args.outdir, args.dataset)
    args.outdir = os.path.join(args.outdir, f"seed_{args.seed}")
    args.outdir = os.path.join(args.outdir, f"lr_{args.lr}")
    args.outdir = os.path.join(args.outdir, f"fsw_{args.weight_fsw}")
    args.outdir = os.path.join(args.outdir, args.method)
    outdir_checkpoint = os.path.join(args.outdir, "checkpoint")
    args.datadir = os.path.join(args.datadir, args.dataset)

    if args.dataset != "mnist":

        args.stat_ground_truth = os.path.join(args.statdir, args.dataset)

        args.imagedir = os.path.join(args.imagedir, args.dataset)
        args.imagedir = os.path.join(args.imagedir, f"seed_{args.seed}")
        args.imagedir = os.path.join(args.imagedir, f"lr_{args.lr}")
        args.imagedir = os.path.join(args.imagedir, f"fsw_{args.weight_fsw}")
        args.imagedir = os.path.join(args.imagedir, args.method)
        args.gen_dir = os.path.join(args.imagedir, "gen")

        args.statdir = os.path.join(args.statdir, args.dataset)
        args.statdir = os.path.join(args.statdir, f"seed_{args.seed}")
        args.statdir = os.path.join(args.statdir, f"lr_{args.lr}")
        args.statdir = os.path.join(args.statdir, f"fsw_{args.weight_fsw}")
        args.statdir = os.path.join(args.statdir, args.method)
        args.stat_gen_dir = os.path.join(args.statdir, "gen")

        os.makedirs(args.imagedir, exist_ok=True)
        os.makedirs(args.gen_dir, exist_ok=True)

        os.makedirs(args.statdir, exist_ok=True)
        os.makedirs(args.stat_gen_dir, exist_ok=True)


    if args.dataset == 'mnist':
        data_loader = MNISTDataLoader(data_dir=args.datadir, train_batch_size=args.batch_size, test_batch_size=args.batch_size_test)
        train_loader, test_loader = data_loader.create_dataloader()
        model = MNISTAutoencoder().to(device)
    elif args.dataset == 'cifar10':
        data_loader = CIFAR10DataLoader(data_dir=args.datadir, train_batch_size=args.batch_size, test_batch_size=args.batch_size_test)
        train_loader, test_loader = data_loader.create_dataloader()
        model = CIFAR10Autoencoder(embedding_dim=args.embedding_size).to(device)
    elif args.dataset == 'stl10':
        data_loader = STL10DataLoader(data_dir=args.datadir, train_batch_size=args.batch_size, test_batch_size=args.batch_size_test)
        train_loader, test_loader = data_loader.create_dataloader()
        model = STL10Autoencoder(embedding_dim=args.embedding_size).to(device)
    else:
        raise NotImplementedError

    if args.dataset == 'mnist':
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

    with torch.no_grad():

        pretrained_model_path = f"{outdir_checkpoint}/epoch_{args.checkpoint_period}/model/{args.dataset}_{args.method}.pth"
        check_path = os.path.isfile(pretrained_model_path)
        print(f"Check if pretrained model path {pretrained_model_path} exit or not: {check_path}")
        assert os.path.isfile(pretrained_model_path) is True, f"not exist {pretrained_model_path}"
        output_file = f'{args.outdir}/evaluate_epoch_{args.checkpoint_period}_{args.method}.log'
        if device == "cpu":
            model.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(pretrained_model_path))

            if args.dataset == "mnist":
                RL, LP, WG, F, AD = ultimate_evaluation(args=args,
                                                        model=model,
                                                        test_loader=test_loader,
                                                        prior_distribution=distribution_fn,
                                                        device=device)
            else:
                RL, LP, WG, F, AD = ultimate_evaluate_fid(args=args,
                                                          model=model,
                                                          test_loader=test_loader,
                                                          prior_distribution=distribution_fn,
                                                          device=device)
        with open(output_file, 'a') as f:
            f.write(f"Evaluating pretrained model: {pretrained_model_path}:\n")
            f.write(f" +) Reconstruction loss: {RL}\n")
            f.write(f" +) Wasserstein distance between generated and real images: {WG}\n")
            f.write(f" +) Wasserstein distance between posterior and prior distribution: {LP}\n")
            f.write(f" +) Fairness: {F}\n")
            f.write(f" +) Averaging distance: {AD}\n")
            f.write("\n")


if __name__ == "__main__":
    main()
