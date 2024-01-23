import argparse
import logging
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import timm
import time
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, \
    DistilBertForSequenceClassification, get_linear_schedule_with_warmup

import models.chenyaofo as chenyaofo_models
from models.chenyaofo import *   # require-must for load torchvision model
import models.torchvision as torchvision_models
from models.torchvision import *   # require-must for load torchvision model
from datasets import *
from utils import *
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

parser = argparse.ArgumentParser(description='train a image classifier to be evaluated')
## Data options
parser.add_argument('--data-dir', default='/data/datasets/', help='path to dataset')
parser.add_argument('--dataset', default='CIFAR-10', help='dataset name', choices=['MNIST', 'CIFAR-10', 'CIFAR-100',
                    'ImageNet', 'ImageNet-200', 'Tiny-ImageNet-200', 'MNLI', "living17", "entity13", "entity30", "nonliving26"])
parser.add_argument('--num-classes', default=10, type=int, help='number of classes')
parser.add_argument('-j', '--workers', default=8, type=int, help='number of data loading workers')
## Model options
chenyaofo_model_names = sorted(name for name in chenyaofo_models.__dict__ if name.islower() and not name.startswith("__") and callable(chenyaofo_models.__dict__[name]))
torchvision_model_names = sorted(name for name in torchvision_models.__dict__ if not name.startswith("__") and callable(torchvision_models.__dict__[name]))
timm_pretrained_model_names = timm.list_models(pretrained=True)
timm_unpretrained_model_names = timm.list_models(pretrained=False)
bert_model_names = ['bert-base-uncased', 'roberta-base', 'distilbert-base-uncased']
model_names = chenyaofo_model_names + torchvision_model_names + timm_unpretrained_model_names + timm_pretrained_model_names + bert_model_names
parser.add_argument('-a', '--arch', required=True, type=str, choices=model_names, help="the model used to run this script")
parser.add_argument('--pretrained', action='store_true', default=False, help='Use the pretrained model')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--bs', '--batch-size', default=256, type=int, help='batch size.')
parser.add_argument('--seed', type=int, default=1, help='seed for initializing training.')
parser.add_argument('--max-token-length', default=512, type=int, help='Max token length used for BERT models.')
parser.add_argument('--warmup-rate', type=float, default=0., help='The proportion of steps for the warmup phase')
parser.add_argument('--save-dir', default='/data/checkpoints/energy_autoeval/', help='path to save checkpoints')
parser.add_argument('--ckpt-dir', default="/data/checkpoints/energy_autoeval/checkpoint_xxx.pth", type=str,
                    help='checkpoint path to use (default: <save-dir>/checkpoint_xxx.pth')
## Optimization options
parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam', 'Adadelta', 'AdamW'], help='optimizer name')
parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, help='initial learning rate', dest="lr")
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--nesterov', action='store_true', default=False, help='nesterov flag for SGD')
parser.add_argument('-wd', '--weight-decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--scheduler', default='CosineAnnealingLR_projnorm',
                    choices=[None, 'CosineAnnealingLR', 'StepLR', 'WarmupLinear'], help='scheduler name')
parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
# Method-specific options
parser.add_argument('--score', type=str, default='EMD', help='energy variants: EMD,EMD1,AVG')
parser.add_argument('--T', default=1., type=float, help='temperature: energy')


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
                assert args.pretrained, 'pretrained is True'
                model = timm.create_model(args.arch, pretrained=True)
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


def load_optimizer(args, model):
    if args.optimizer == "AdamW":
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=args.lr)
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=args.nesterov, weight_decay=args.weight_decay)
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    return optimizer


def load_scheduler(args, optimizer, trainloader):
    if args.scheduler == "WarmupLinear" and args.arch in bert_model_names:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(
            args.warmup_rate * (len(trainloader) * args.epochs)), num_training_steps=(len(trainloader) * args.epochs))
    if args.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    return scheduler


def train_bert(args, model, train_loader, test_loader, epoch, optimizer, scheduler, criterion, device, steps):
    losses, accs = 0.0, 0.0
    energies = []
    model.train()

    ## BERT models
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
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
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        steps +=1

        ## accuracy, loss
        acc = multi_acc(output, label)
        accs += acc.item() * label.size(0)
        losses += loss.item() * label.size(0)
        ## energy
        energy = -args.T * (torch.logsumexp(output / args.T, dim=1))
        energies.append(energy.detach().cpu())

        ## store a ckpt file every 1000 iterations
        if steps % 1000 == 0:
            ## stack all energy
            energies_steps = torch.cat(energies, 0)
            # avg energy-based meta-distribution, avg accuracy, avg loss
            avg_energies_steps = 0
            if args.score == 'EMD':
                avg_energies_steps = torch.log_softmax(energies_steps, dim=0).mean()
                avg_energies_steps = torch.log(-avg_energies_steps).item()
            avg_accs_steps = accs / len(train_loader.dataset)
            avg_losses_steps = losses / len(train_loader.dataset)

            logging.info(f"Epoch %d: Steps: %d, Train Energy: %.2f, Train ACC: %.2f, Train loss: %.2f, Lr: %f" \
                         % (epoch, steps, avg_energies_steps, avg_accs_steps, avg_losses_steps,
                            scheduler.get_last_lr()[0] if scheduler is not None else args.lr))
            checkpoint_name = 'checkpoint_{:04d}_{:04d}.pth'.format(epoch, steps)
            energy, acc = test(args, model, test_loader, criterion, device)
            save_checkpoint({
                'epoch': epoch,
                'steps': steps,
                'arch': args.arch,
                'energy': energy,
                'acc': acc,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=False, filename=os.path.join(args.save_dir, checkpoint_name))

    ## stack all energy
    energies = torch.cat(energies, 0)
    # avg energy-based meta-distribution, avg accuracy, avg loss
    avg_energies = 0
    if args.score == 'EMD':
        avg_energies = torch.log_softmax(energies, dim=0).mean()
        avg_energies = torch.log(-avg_energies).item()
    avg_accs = accs / len(train_loader.dataset)
    avg_losses = losses / len(train_loader.dataset)

    logging.info(f"Epoch %d: Train Energy: %.2f, Train ACC: %.2f, Train loss: %.2f, Lr: %f" \
            %(epoch, avg_energies, avg_accs, avg_losses, scheduler.get_last_lr()[0] if scheduler is not None else args.lr))
    return steps

# 100000 image
# batch = 64
# 设置每200个batch step一次
def train_vgg19_bn(args, model, train_loader, test_loader, epoch, optimizer, scheduler, criterion, device, steps):
    losses, accs = 0.0, 0.0
    energies = []
    model.train()
    for data, target in tqdm(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        steps += 1
        ## accuracy, loss
        acc = accuracy(output, target)
        accs += acc[0].item() * target.size(0)
        losses += loss.item() * target.size(0)
        ## energy
        energy = -args.T * (torch.logsumexp(output / args.T, dim=1))
        energies.append(energy.detach().cpu())

        ## store a ckpt file every 100 iterations
        if steps % 200 == 0:
            ## stack all energy
            energies_steps = torch.cat(energies, 0)
            # avg energy-based meta-distribution, avg accuracy, avg loss
            avg_energies_steps = 0
            if args.score == 'EMD':
                avg_energies_steps = torch.log_softmax(energies_steps, dim=0).mean()
                # avg_energies_steps = torch.log(-avg_energies_steps).item()
                avg_energies_steps = avg_energies_steps.item()
            avg_accs_steps = accs / (len(data) * steps)
            avg_losses_steps = losses / (len(data) * steps)

            logging.info(f"Epoch %d: Steps: %d, Train Energy: %.2f, Train ACC: %.2f, Train loss: %.2f, Lr: %f" \
                         % (epoch, steps, avg_energies_steps, avg_accs_steps, avg_losses_steps,
                            scheduler.get_last_lr()[0] if scheduler is not None else args.lr))
            checkpoint_name = 'checkpoint_{:04d}_{:04d}.pth'.format(epoch, steps)
            # energy, acc = test(args, model, test_loader, criterion, device)
            save_checkpoint({
                'epoch': epoch,
                'steps': steps,
                'arch': args.arch,
                # 'energy': energy,
                # 'acc': acc,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=False, filename=os.path.join(args.save_dir, checkpoint_name))
    if scheduler is not None:
        scheduler.step()
    ## stack all energy
    energies = torch.cat(energies, 0)
    # avg energy-based meta-distribution, avg accuracy, avg loss
    avg_energies = 0
    if args.score == 'EMD':
        avg_energies = torch.log_softmax(energies, dim=0).mean()
        # avg_energies = torch.log(-avg_energies).item()
        avg_energies = avg_energies.item()
    avg_accs = accs / len(train_loader.dataset)
    avg_losses = losses / len(train_loader.dataset)

    logging.info(f"Epoch %d: Train Energy: %.2f, Train ACC: %.2f, Train loss: %.2f, Lr: %f" \
            %(epoch, avg_energies, avg_accs, avg_losses, scheduler.get_last_lr()[0] if scheduler is not None else args.lr))
    return steps

def train(args, model, train_loader, epoch, optimizer, scheduler, criterion, device):
    losses, accs = 0.0, 0.0
    energies = []
    model.train()

    for data, target in tqdm(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        ## accuracy, loss
        acc = accuracy(output, target)
        accs += acc[0].item() * target.size(0)
        losses += loss.item() * target.size(0)
        ## energy
        energy = -args.T * (torch.logsumexp(output / args.T, dim=1))
        energies.append(energy.detach().cpu())

    if scheduler is not None:
        scheduler.step()

    ## stack all energy
    energies = torch.cat(energies, 0)
    # avg energy-based meta-distribution, avg accuracy, avg loss
    avg_energies = 0
    if args.score == 'EMD':
        avg_energies = torch.log_softmax(energies, dim=0).mean()
        avg_energies = torch.log(-avg_energies).item()
    avg_accs = accs / len(train_loader.dataset)
    avg_losses = losses / len(train_loader.dataset)

    logging.info(f"Epoch %d: Train Energy: %.2f, Train ACC: %.2f, Train loss: %.2f, Lr: %f" \
            %(epoch, avg_energies, avg_accs, avg_losses, scheduler.get_last_lr()[0] if scheduler is not None else args.lr))


def test(args, model, test_loader, criterion, device):
    losses, accs = 0.0, 0.0
    energies = []
    model.eval()

    with torch.no_grad():
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
                energy = -args.T * (torch.logsumexp(output / args.T, dim=1))
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
                energy = -args.T * (torch.logsumexp(output / args.T, dim=1))
                energies.append(energy.detach().cpu())

        ## stack all energy
        energies = torch.cat(energies, 0)
        # avg energy-based meta-distribution, avg accuracy, avg loss
        avg_energies = 0
        if args.score == 'EMD':
            avg_energies = torch.log_softmax(energies, dim=0).mean()
            avg_energies = torch.log(-avg_energies).item()
        avg_accs = accs / len(test_loader.dataset)
        avg_losses = losses / len(test_loader.dataset)

    logging.info(f"Test: Test Energy: %.2f, Test ACC: %.2f, Test loss: %.2f" %(avg_energies, avg_accs, avg_losses))
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

    ## Make save directory and log file
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.isdir(args.save_dir):
        raise Exception('%s is not a dir' % args.save_dir)
    logging.basicConfig(filename=os.path.join(args.save_dir, f'{args.dataset}_{args.arch}_T{args.T}.log'), level=logging.INFO)

    ## load data
    logging.info(f'==> Preparing data on {args.dataset}..')
    _, trainloader, _, testloaders = get_data(args.data_dir, args.dataset, args.bs, args.arch, args.workers,
                                              train=True, max_token_length=args.max_token_length)

    ## build model
    logging.info(f'==> Building model on {args.dataset}_{args.arch}_T{args.T}..')
    assert args.arch in model_names, "the model need to be chosen from model_names"
    model = load_model(args)

    # Resume model training or reload a trainsed model if needed
    reload_epoch, reload_steps = 0, 0
    if not os.path.exists(args.ckpt_dir):
        print("=> no checkpoint found at '{}'".format(args.ckpt_dir))
    else:
        if os.path.isfile(args.ckpt_dir):
            checkpoint = torch.load(args.ckpt_dir)
            reload_epoch = checkpoint['epoch']
            if args.arch in bert_model_names:
                reload_steps = checkpoint['steps']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> Reloaded checkpoint '{}' (epoch {}, steps {})".format(args.ckpt_dir, reload_epoch, reload_steps))
    model.to(device)
    model.eval()
    logging.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # load optimizer, scheduler, loss function
    optimizer = load_optimizer(args, model)
    scheduler = load_scheduler(args, optimizer, trainloader)
    criterion = torch.nn.CrossEntropyLoss()

    ## use multiple gpu
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    logging.info(f"Training model from epoch {reload_epoch} to epoch {args.epochs}.")
    best_acc = 0.0
    if args.arch in bert_model_names:
        for epoch in range(reload_epoch, args.epochs):
            reload_steps = train_bert(args, model, trainloader, testloaders[-1][0], epoch, optimizer, scheduler, criterion, device, reload_steps)
            energy, acc = test(args, model, testloaders[-1][0], criterion, device)
            ## Save model
            is_best = acc > best_acc
            if is_best:
                best_steps, best_epoch, best_acc = reload_steps, epoch, acc
            checkpoint_name = 'checkpoint_{:04d}_{:04d}.pth'.format(epoch, reload_steps)
            save_checkpoint({
                'epoch': epoch,
                'steps': reload_steps,
                'arch': args.arch,
                'energy': energy,
                'acc': acc,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=is_best, filename=os.path.join(args.save_dir, checkpoint_name))
        logging.info("Training has finished")
        logging.info(f'Best classification accuracy {best_acc} at epoch {best_epoch} step {best_steps}')
    elif args.arch == 'VGG19_bn' and args.dataset == 'Tiny-ImageNet-200':
        for epoch in range(reload_epoch, args.epochs):
            reload_steps = train_vgg19_bn(args, model, trainloader, testloaders[0], epoch, optimizer, scheduler, criterion, device, reload_steps)
            energy, acc = test(args, model, testloaders[0], criterion, device)
            ## Save model
            is_best = acc > best_acc
            if is_best:
                best_steps, best_epoch, best_acc = reload_steps, epoch, acc
            checkpoint_name = 'checkpoint_{:04d}_{:04d}.pth'.format(epoch, reload_steps)
            save_checkpoint({
                'epoch': epoch,
                'steps': reload_steps,
                'arch': args.arch,
                'energy': energy,
                'acc': acc,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=is_best, filename=os.path.join(args.save_dir, checkpoint_name))
        logging.info("Training has finished")
        logging.info(f'Best classification accuracy {best_acc} at epoch {best_epoch} step {best_steps}')
    else:
        for epoch in range(reload_epoch, args.epochs):
            train(args, model, trainloader, epoch, optimizer, scheduler, criterion, device)
            energy, acc = test(args, model, testloaders[0], criterion, device)
            ## Save model
            is_best = acc > best_acc
            if is_best:
                best_epoch, best_acc = epoch, acc
            checkpoint_name = 'checkpoint_{:04d}.pth'.format(epoch)
            save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'energy': energy,
                'acc': acc,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=is_best, filename=os.path.join(args.save_dir, checkpoint_name))
        logging.info("Training has finished")
        logging.info(f'Best classification accuracy {best_acc} at epoch {best_epoch}')


if __name__ == "__main__":
    main()