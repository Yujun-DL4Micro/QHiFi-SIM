import torch
from torch.utils.data import DataLoader
from augmented_dataset import AugmentedDataset
from argparse import ArgumentParser
from train import train
from test_model import test
import random
import numpy as np
import os


def build_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="QHiFi", help="selecting the training model")
    parser.add_argument("--resume", type=str, default="", help="Resume path (default: none)")
    parser.add_argument('--experiment_name', type=str, default='test')
    parser.add_argument("--checkpoint_for_test", type=str, default="./checkpoint/checkpoint_qHiFi.pth.tar", help="Resume path (default: none)")
    parser.add_argument("--datapath", type=str, default='../dataset_QHiFi/MT/', help="data path")
    parser.add_argument("--datalist", type=str, default='../dataset_QHiFi/MT/train_list.txt', help="train list")
    parser.add_argument("--evallist", type=str, default='../dataset_QHiFi/MT/eval_list.txt', help="eval list")
    parser.add_argument("--testlist", type=str, default='../dataset_QHiFi/MT/test_list.txt', help="test list")
    
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument("--patch_size", type=int, default=256, help="Training patch size")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--nEpochs", type=int, default=150, help="Number of epochs to train")

    parser.add_argument('--weight', type=float, default=0.58, help='weight')
    parser.add_argument('--loss_corrected_l1', type=float, default=0.36,
                        help='weight for corrected image L1 loss')
    parser.add_argument('--loss_corrected_ssim', type=float, default=0.1,
                        help='weight for corrected image SSIM loss')

    parser.add_argument("--results_path", type=str, default='./test_outs/.../')
    parser.add_argument("--output_path", type=str, default='Eval_results/results/')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoint/', help="data path")
 
    parser.add_argument("--validate_interval", type=int, default=10)
    parser.add_argument("--sample_interval", type=int, default=10)
    parser.add_argument("--validate_num", type=int, default=100)
    parser.add_argument("--sample_num", type=int, default=3)
    parser.add_argument("--model_save_frequency", type=int, default=1)
    
    parser.add_argument("--train_or_test", type=str, default='train')

    parser.add_argument('--disable_ssfe', action='store_true', help='Disable SR structural feature extraction branch')
    parser.add_argument('--disable_wife', action='store_true',
                        help='Disable WF intensity difference feature extraction branch')
    parser.add_argument('--disable_channel_attn', action='store_true', help='Disable channel attention (FCAB)')
    parser.add_argument('--disable_spatial_attn', action='store_true', help='Disable spatial attention (SAB)')
    parser.add_argument('--disable_identity', action='store_true', help='Disable global identity skip connection')
    parser.add_argument('--disable_ssim_loss', action='store_true', help='Disable SSIM term in loss function (only L1)')
    parser.add_argument('--disable_loss2', action='store_true', help='Disable structural fidelity loss (Loss2)')

    parser.add_argument("--psf_size", type=int, default=31,
                        help="PSF spatial size after cropping (odd, <=31)")


    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")


    args = parser.parse_args()

    args.weight = 0.5
    args.loss_corrected_l1 = 0.1
    args.loss_corrected_ssim = 0.1

    if args.experiment_name != "default":
        args.checkpoint_path = os.path.join(args.checkpoint_path, args.experiment_name)
        args.output_path = os.path.join(args.output_path, args.experiment_name)

    return args


def main(args):
    """
    Main function to execute the training or testing pipeline.

    Args:
        args (argparse.Namespace): Parsed command-line arguments from build_args().
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.train_or_test == 'train':
        train_data = AugmentedDataset(
            args.datapath,
            args.datalist,
            mode='train',
            augment=True,
            psf_size = args.psf_size
        )

        eval_data = AugmentedDataset(
            args.datapath,
            args.evallist,
            mode='eval',
            augment=False,
            psf_size = args.psf_size
        )

        print(f"Total training patches: {len(train_data)}")
        print(f"Total validation patches: {len(eval_data)}")

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        eval_loader = DataLoader(eval_data, batch_size=1, shuffle=False)
        train(args, train_loader, eval_loader, device)

    elif args.train_or_test == 'test':
        test_data = AugmentedDataset(
            args.datapath,
            args.testlist,
            mode='test',
            augment=False,
            psf_size = args.psf_size
        )

        print(f"Total testing patches: {len(test_data)}")

        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
        test(args, test_loader, device)
    else:
        print("Error: wrong args for train_or_test!")
        exit()


if __name__ == "__main__":
    args = build_args()
    main(args)
    print("over!")


