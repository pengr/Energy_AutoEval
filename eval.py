import argparse
import logging
import torch.backends.cudnn as cudnn
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import timm
import time
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, \
    DistilBertForSequenceClassification

import models.chenyaofo as chenyaofo_models
from models.chenyaofo import *     # require-must for load torchvision model
import models.torchvision as torchvision_models
from models.torchvision import *   # require-must for load torchvision model
from datasets import *
from utils import *
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
## plot figures
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter


parser = argparse.ArgumentParser(description='fit a regressor to predict the unseen test accuracy')
# Data options
parser.add_argument('--data-dir', default='/data/datasets/', help='path to dataset')
parser.add_argument('--dataset', default='CIFAR-10', help='dataset name', choices=['MNIST', 'CIFAR-10', 'CIFAR-100',
                    'ImageNet', 'ImageNet-200', 'Tiny-ImageNet-200', 'MNLI', "living17", "entity13", "entity30", "nonliving26"])
parser.add_argument('--num-classes', default=10, type=int, help='Number of classes')
parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers')
# Model options
chenyaofo_model_names = sorted(name for name in chenyaofo_models.__dict__ if name.islower() and not name.startswith("__") and callable(chenyaofo_models.__dict__[name]))
torchvision_model_names = sorted(name for name in torchvision_models.__dict__ if not name.startswith("__") and callable(torchvision_models.__dict__[name]))
timm_pretrained_model_names = timm.list_models(pretrained=True)
timm_unpretrained_model_names = timm.list_models(pretrained=False)
bert_model_names = ['bert-base-uncased', 'roberta-base', 'distilbert-base-uncased']
model_names = chenyaofo_model_names + torchvision_model_names + timm_unpretrained_model_names + timm_pretrained_model_names + bert_model_names
parser.add_argument('-a', '--arch', required=True, type=str, choices=model_names, help="the model used to run this script")
parser.add_argument('--pretrained', action='store_true', default=False, help='Use the pretrained model')
parser.add_argument('--epoch', type=int, default=1, help='choose which epoch to eval')
parser.add_argument('--steps', type=int, default=0, help='choose which steps to eval')
parser.add_argument('--bs', '--batch-size', default=256, type=int, help='batch size.')
parser.add_argument('--seed', type=int, default=1, help='seed for initializing training.')
parser.add_argument('--max-token-length', default=512, type=int, help='Max token length used for BERT models.')
parser.add_argument('--save-dir', default='/data/checkpoints/energy_autoeval/', help='path to save checkpoints')
parser.add_argument('--ckpt-dir', default="/data/checkpoints/energy_autoeval/checkpoint_xxx.pth", type=str, required=True,
                    help='checkpoint path (default: <save-dir>/checkpoint_xxx.pth')
# Method-specific options
parser.add_argument('--score', type=str, default='EMD', help='energy variants: EMD,EMD1,AVG')
parser.add_argument('--T', default=1., type=float, help='temperature: energy')
parser.add_argument('--T-MAE', default=1., type=float, help='temperature to modify MAE')
parser.add_argument('--colorId', default=0, type=int, metavar='N', help='the palette color of correlation scatterplot')


def visualize_corr(args, energies, accuracies):
    num_points = len(energies)
    if args.dataset == 'CIFAR-10' or args.dataset == 'CIFAR-100' or args.dataset == 'ImageNet':
        num_class = 19
    elif args.dataset == 'Tiny-ImageNet-200':
        num_class = 15
    elif args.dataset == 'MNLI':
        num_class = 12
    # Scatter plot with different markers
    markers_dict = ['v', 's', 'p', 'h', 'o', 'v', 's', 'p', 'h', 'o', 'v', 's', 'p', 'h', 'o', 'v', 's', 'p', 'h']
    colors_dict = np.linspace(0, 1, num_class)
    # Set up the figure and axis
    plt.figure(figsize=(6, 6))
    cmap = plt.get_cmap('RdYlBu')

    if args.dataset == 'CIFAR-10' or args.dataset == 'CIFAR-100' or args.dataset == 'ImageNet':
        label_dict = ["fog", "frost", "motion_blur", "brightness", "zoom_blur", "snow", "defocus_blur", "glass_blur", \
                      "gaussian_noise", "shot_noise", "impulse_noise", "contrast", "elastic_transform", "pixelate", \
                      "jpeg_compression", "speckle_noise", "spatter", "gaussian_blur", "saturate"]
        for i in range(0, num_points, 5):
            marker_index = i // 5
            color_value = colors_dict[marker_index]
            plt.scatter(energies[i:i + 5], accuracies[i:i + 5], color=cmap(color_value), edgecolors='black',
                        linewidths=0.5, label=label_dict[marker_index], marker=markers_dict[marker_index])
    elif args.dataset == 'Tiny-ImageNet-200':
        label_dict = ["fog", "frost", "motion_blur", "brightness", "zoom_blur", "snow", "defocus_blur", "glass_blur", \
                      "gaussian_noise", "shot_noise", "impulse_noise", "contrast", "elastic_transform", "pixelate", \
                      "jpeg_compression"]
        for i in range(0, num_points, 5):
            marker_index = i // 5
            color_value = colors_dict[marker_index]
            plt.scatter(energies[i:i + 5], accuracies[i:i + 5], color=cmap(color_value), edgecolors='black',
                        linewidths=0.5, label=label_dict[marker_index], marker=markers_dict[marker_index])
    elif args.dataset == 'MNLI':
        label_dict = ["MNLI-M", "MNLI-MM", "SNLI", "BREAK-NLI", "HANS", "SNLI-HRAD",
                      "STRESS-L", "STRESS-S", "STRESS-O", "STRESS-A", "SICK", "EQUATE-NAT"]
        for i in range(0, num_points):
            marker_index = i
            color_value = colors_dict[marker_index]
            plt.scatter(energies[i:i + 1], accuracies[i:i + 1], color=cmap(color_value), edgecolors='black',
                        linewidths=0.5, label=label_dict[marker_index], marker=markers_dict[marker_index], s=50)

    # 添加图例，放在图表的右侧
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    # Add light grid lines
    plt.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
    # Customize the plot
    plt.title("MDE  v.s.  Accuracy", fontsize=24)
    plt.xlabel("MDE", fontsize=24)
    plt.ylabel("Accuracy", fontsize=24)
    plt.xticks(fontsize=16)  # 设置 x 轴刻度线的字体大小为 16
    plt.yticks(fontsize=16)  # 设置 y 轴刻度线的字体大小为 16
    # 拟合回归直线
    slope, intercept, r_value, p_value, std_err = stats.linregress(energies, accuracies)
    regression_line = slope * energies + intercept
    plt.plot(energies, regression_line, color='royalblue', alpha=0.5, linewidth=3, label=f'Regression Line')
    # Calculate Spearman and Pearson correlation coefficients
    spearman_corr, _ = stats.spearmanr(energies, accuracies)
    pearson_corr, _ = stats.pearsonr(energies, accuracies)
    kendall_corr, _ = stats.kendalltau(energies, accuracies)
    slr = LinearRegression()
    slr.fit(energies.reshape(-1, 1), accuracies.reshape(-1, 1))
    r2 = slr.score(energies.reshape(-1, 1), accuracies.reshape(-1, 1))
    # Add correlation coefficients to the plot
    corr_text = f" R²={r2:.3f} \n r={pearson_corr:.3f} \n "+ chr(961)+f"={spearman_corr:.3f} "
    plt.annotate(corr_text, xy=(0.05, 0.95), xycoords="axes fraction", fontsize=16, ha='left', va='top',
                 bbox=dict(boxstyle='round,pad=0.5', edgecolor='gray', facecolor='white', alpha=0.6))
    # Move the legend outside the plot
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.pop(-1)
    plt.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    if args.dataset == 'MNLI':
        plt.savefig(f'{args.save_dir}/correlation_{args.dataset}_{args.arch}_T{args.T}_epoch{args.epoch}_steps{args.steps}.pdf',
            bbox_inches='tight', pad_inches=0.0)
    elif args.dataset == 'TinyImageNet-200' and args.arch == 'VGG19_bn':
        plt.savefig(f'{args.save_dir}/correlation_{args.dataset}_{args.arch}_T{args.T}_epoch{args.epoch}_steps{args.steps}.pdf',
            bbox_inches='tight', pad_inches=0.0)
    else:
        plt.savefig(f'{args.save_dir}/correlation_{args.dataset}_{args.arch}_T{args.T}_epoch{args.epoch}.pdf',
                    bbox_inches='tight', pad_inches=0.0)


def load_model(args):
    if args.arch in chenyaofo_model_names:    ## CIFAR10, CIFAR100
        if args.pretrained:
            model = torch.hub.load("chenyaofo/pytorch-cifar-models", args.arch, pretrained=args.pretrained)
        else:
            model = eval(args.arch)()
    if args.arch in torchvision_model_names:   ## TinyImageNet200
        model = eval(args.arch)(base_model=args.arch, pretrained=args.pretrained, num_classes=args.num_classes)
    if args.arch in timm_unpretrained_model_names + timm_pretrained_model_names:  ## ImageNet
        while True:
            try:
                # assert args.pretrained, 'pretrained is True'
                # model = timm.create_model(args.arch, pretrained=True)
                model = timm.create_model(args.arch, pretrained=False)
                state_dict = torch.load(args.ckpt_dir)
                model.load_state_dict(state_dict)
                break
            except Exception:
                continue
    if args.arch in bert_model_names:  ## MNLI
        while True:
            try:
                if args.arch == 'bert-base-uncased':
                    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=args.num_classes)
                elif args.arch == 'roberta-base':
                    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=args.num_classes)
                elif args.arch == 'distilbert-base-uncased':
                    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=args.num_classes)
                break
            except Exception:
                continue
    return model


def test(args, i, model, test_loader, device, T=1):
    losses, accs = 0.0, 0.0
    criterion = torch.nn.CrossEntropyLoss()
    energies = []
    model.eval()

    with torch.no_grad():
        ## exclusive to BERT models
        if args.arch in bert_model_names:
            for batch in tqdm(test_loader):
                if args.arch == 'bert-base-uncased':
                    input_ids, attn_mask, token_type_ids, label = batch['input_ids'].to(device), batch['attn_mask'].to(device), \
                                                                  batch['token_type_ids'].to(device), batch['label'].to(device)
                    loss, output = model(input_ids,
                                         token_type_ids=token_type_ids,
                                         attention_mask=attn_mask,
                                         labels=label).values()
                else:
                    input_ids, attn_mask, label = batch['input_ids'].to(device), batch['attn_mask'].to(device), \
                                                  batch['label'].to(device)
                    loss, output = model(input_ids,
                                         attention_mask=attn_mask,
                                         labels=label).values()
                ## accuracy, loss
                acc = multi_acc(output, label)
                accs += acc.item() * label.size(0)
                losses += loss.item() * label.size(0)
                ## energy
                energy = -T * (torch.logsumexp(output / T, dim=1))
                energies.append(energy.detach().cpu())
        else:
            for data, target in tqdm(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                ## accuracy, loss
                acc = accuracy(output, target)
                accs += acc[0].item() * target.size(0)
                losses += loss.item() * target.size(0)
                ## energy
                energy = -T * (torch.logsumexp(output / T, dim=1))
                energies.append(energy.detach().cpu())

        ## stack all energy
        energies = torch.cat(energies, 0)
        # avg energy-based meta-distribution, avg accuracy, avg loss
        avg_energies = 0
        if args.score == 'EMD':
            avg_energies = torch.log_softmax(energies, dim=0).mean()
            avg_energies = torch.log(-avg_energies).item()
        elif args.score == 'EMD1':
            avg_energies = torch.log_softmax(energies, dim=0).mean()
            avg_energies = -avg_energies.item()
        elif args.score == 'AVG':
            avg_energies = energies.mean()
            avg_energies = -avg_energies.item()
        avg_accs = accs / len(test_loader.dataset)
        avg_losses = losses / len(test_loader.dataset)

    logging.info(f"%s Dataset%d: Test Energy: %.2f, Test ACC: %.2f, Test loss: %.2f" \
                 % (args.dataset, i, avg_energies, avg_accs, avg_losses))
    return avg_energies, avg_accs


def main():
    args = parser.parse_args()

    ## fix seed
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        cudnn.benchmark = True
        if args.seed is not None:
            cudnn.deterministic = True
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Make save directory and log file
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.isdir(args.save_dir):
        raise Exception('%s is not a dir' % args.save_dir)
    if args.arch in bert_model_names:
        logging.basicConfig(filename=os.path.join(args.save_dir, f'{args.dataset}_{args.arch}_T{args.T}_epoch{args.epoch}_steps{args.steps}.log'), level=logging.INFO)
    elif args.arch == 'VGG19_bn' and args.dataset == 'Tiny-ImageNet-200':
        logging.basicConfig(filename=os.path.join(args.save_dir, f'{args.dataset}_{args.arch}_T{args.T}_epoch{args.epoch}_steps{args.steps}.log'), level=logging.INFO)
    else:
        logging.basicConfig(filename=os.path.join(args.save_dir, f'{args.dataset}_{args.arch}_T{args.T}_epoch{args.epoch}.log'), level=logging.INFO)

    ## load data
    logging.info(f'==> Preparing data on {args.dataset}..')
    _, _, _, testloaders = get_data(args.data_dir, args.dataset, args.bs, args.arch, args.workers,
                                    train=False, max_token_length=args.max_token_length)

    ## build Model
    if args.arch in bert_model_names:
        logging.info(f'==> Building model on {args.dataset}_{args.arch}_T{args.T}_epoch{args.epoch}_steps{args.steps}..')
    elif args.arch == 'VGG19_bn' and args.dataset == 'Tiny-ImageNet-200':
        logging.info(f'==> Building model on {args.dataset}_{args.arch}_T{args.T}_epoch{args.epoch}_steps{args.steps}..')
    else:
        logging.info(f'==> Building model on {args.dataset}_{args.arch}_T{args.T}_epoch{args.epoch}..')
    assert args.arch in model_names, "the model need to be chosen from model_names"
    model = load_model(args)

    # Resume model training or reload a trainsed model if needed
    reload_epoch, reload_steps = 0, 0
    if args.arch in bert_model_names:
        args.ckpt_dir = str('{}checkpoint_{:04d}_{:04d}.pth'.format(args.ckpt_dir, args.epoch, args.steps))
        if not os.path.exists(args.ckpt_dir):
            print("=> no checkpoint found at '{}'".format(args.ckpt_dir))
        else:
            if os.path.isfile(args.ckpt_dir):
                checkpoint = torch.load(args.ckpt_dir)
                reload_epoch, reload_steps = checkpoint['epoch'], checkpoint['steps']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> Reloaded checkpoint '{}' (epoch {}, steps {})".format(args.ckpt_dir, reload_epoch, reload_steps))
    elif args.arch == 'VGG19_bn' and args.dataset == 'Tiny-ImageNet-200':
        args.ckpt_dir = str('{}checkpoint_{:04d}_{:04d}.pth'.format(args.ckpt_dir, args.epoch, args.steps))
        if not os.path.exists(args.ckpt_dir):
            print("=> no checkpoint found at '{}'".format(args.ckpt_dir))
        else:
            if os.path.isfile(args.ckpt_dir):
                checkpoint = torch.load(args.ckpt_dir)
                reload_epoch, reload_steps = checkpoint['epoch'], checkpoint['steps']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> Reloaded checkpoint '{}' (epoch {}, steps {})".format(args.ckpt_dir, reload_epoch, reload_steps))
    # elif args.arch == 'ImageNet':
    #     if not os.path.exists(args.ckpt_dir):
    #         print("=> no checkpoint found at '{}'".format(args.ckpt_dir))
    #     else:
    #         if os.path.isfile(args.ckpt_dir):
    #             checkpoint = torch.load(args.ckpt_dir)
    #             model.load_state_dict(checkpoint['state_dict'])
    #             print("=> Reloaded checkpoint '{}' ".format(args.ckpt_dir))
    else:
        args.ckpt_dir = str('{}checkpoint_{:04d}.pth'.format(args.ckpt_dir, args.epoch))
        if not os.path.exists(args.ckpt_dir):
            print("=> no checkpoint found at '{}'".format(args.ckpt_dir))
        else:
            if os.path.isfile(args.ckpt_dir):
                checkpoint = torch.load(args.ckpt_dir)
                reload_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> Reloaded checkpoint '{}' (epoch {})".format(args.ckpt_dir, reload_epoch))
    model.to(device)
    model.eval()

    ## use multiple gpu
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    ## Collect the <energy, acc> pairs of meta-set
    logging.info(f"==> Collecting on meta-set of {args.dataset}...")

    if not os.path.exists(f'{args.save_dir}/accuracies_{args.dataset}_{args.arch}_T{args.T}_epoch{args.epoch}_steps{args.steps}.npy') and \
        not os.path.exists(f'{args.save_dir}/energies_{args.dataset}_{args.arch}_T{args.T}_epoch{args.epoch}.npy'):
        energies, accuracies = [], []
        for i, testloader in enumerate(testloaders[-1]):
            energy, acc = test(args, i, model, testloader, device, args.T)
            energies.append(energy)
            accuracies.append(acc)
        energies, accuracies = np.array(energies), np.array(accuracies)
        if args.arch in bert_model_names:
            np.save(f'{args.save_dir}/accuracies_{args.dataset}_{args.arch}_T{args.T}_epoch{args.epoch}_steps{args.steps}.npy', accuracies)
            np.save(f'{args.save_dir}/energies_{args.dataset}_{args.arch}_T{args.T}_epoch{args.epoch}_steps{args.steps}.npy', energies)
        elif args.arch == 'VGG19_bn' and args.dataset == 'Tiny-ImageNet-200':
            np.save(f'{args.save_dir}/accuracies_{args.dataset}_{args.arch}_T{args.T}_epoch{args.epoch}_steps{args.steps}.npy', accuracies)
            np.save(f'{args.save_dir}/energies_{args.dataset}_{args.arch}_T{args.T}_epoch{args.epoch}_steps{args.steps}.npy', energies)
        else:
            np.save(f'{args.save_dir}/accuracies_{args.dataset}_{args.arch}_T{args.T}_epoch{args.epoch}.npy', accuracies)
            np.save(f'{args.save_dir}/energies_{args.dataset}_{args.arch}_T{args.T}_epoch{args.epoch}.npy', energies)

    logging.info(f"\n==> Regressing on meta-set of {args.dataset}...")
    if args.arch in bert_model_names:
        energies = np.load(f'{args.save_dir}/energies_{args.dataset}_{args.arch}_T{args.T}_epoch{args.epoch}_steps{args.steps}.npy')
        accuracies = np.load(f'{args.save_dir}/accuracies_{args.dataset}_{args.arch}_T{args.T}_epoch{args.epoch}_steps{args.steps}.npy')
    elif args.arch == 'VGG19_bn' and args.dataset == 'Tiny-ImageNet-200':
        energies = np.load(f'{args.save_dir}/energies_{args.dataset}_{args.arch}_T{args.T}_epoch{args.epoch}_steps{args.steps}.npy')
        accuracies = np.load(f'{args.save_dir}/accuracies_{args.dataset}_{args.arch}_T{args.T}_epoch{args.epoch}_steps{args.steps}.npy')
    else:
        energies = np.load(f'{args.save_dir}/energies_{args.dataset}_{args.arch}_T{args.T}_epoch{args.epoch}.npy')
        accuracies = np.load(f'{args.save_dir}/accuracies_{args.dataset}_{args.arch}_T{args.T}_epoch{args.epoch}.npy')

    # Visualize correlation scatterplot
    visualize_corr(args, energies, accuracies)

    # Statistic Spearman, Pearson, Kendall's correlation
    rho, pval = stats.spearmanr(energies, accuracies)
    logging.info(f'Spearman\'s rank correlation-rho %.3f' %(rho))
    logging.info(f'Spearman\'s rank correlation-pval %.3f' %(pval))
    rho, pval = stats.pearsonr(energies, accuracies)
    logging.info(f'Pearsons correlation-rho %.3f' %(rho))
    logging.info(f'Pearsons correlation-pval %.3f' %(pval))
    rho, pval = stats.kendalltau(energies, accuracies)
    logging.info(f'Kendall\'s rank correlation-rho %.3f' %(rho))
    logging.info(f'Kendall\'s rank correlation-pval %.3f' %(pval))

    ## Fit linear and robust linear regressor to statistic R2
    slr = LinearRegression()
    slr.fit(energies.reshape(-1, 1), accuracies.reshape(-1, 1))
    R2 = slr.score(energies.reshape(-1, 1), accuracies.reshape(-1, 1))
    logging.info(f'Linear coefficient of determination-R2 %.3f' %(R2))

    robust_reg = HuberRegressor()
    robust_reg.fit(energies.reshape(-1, 1), accuracies.reshape(-1))
    robust_R2 = robust_reg.score(energies.reshape(-1, 1), accuracies.reshape(-1, 1))
    logging.info(f'Robust linear coefficient of determination-robust_R2 %.3f\n' %(robust_R2))

    # Predict each unseen test accuracy and MAE
    logging.info(f"==> Evaluating on unseen test sets of {args.dataset}...")
    for i, testloader in enumerate(testloaders[:-1]):
        test_energy, test_acc = test(args, i, model, testloader, device, args.T_MAE)
        test_energy, test_acc = np.array(test_energy), np.array(test_acc)
    
        ## Linear regressor
        pred = slr.predict(test_energy.reshape(-1, 1))
        MAE = mean_absolute_error(pred, test_acc.reshape(-1, 1))
        logging.info('Linear regressor: %s Unseen Testset%d: True Energy %.2f, True Acc: %.2f, Pred Acc: %.2f, MAE: %.2f' \
                     % (args.dataset, i, test_energy, test_acc, pred, MAE))
    
        # ## Robust linear regressor
        # robust_pred = robust_reg.predict(test_energy.reshape(-1, 1))
        # robust_MAE = mean_absolute_error(robust_pred, test_acc.reshape(-1, 1))
        # logging.info(f'Robust linear regressor: %s Unseen Testset%d: True Energy %.2f, True Acc: %.2f, Pred Acc: %.2f, MAE: %.2f\n' \
        #              % (args.dataset, i, test_energy, test_acc, robust_pred, robust_MAE))


if __name__ == "__main__":
    main()