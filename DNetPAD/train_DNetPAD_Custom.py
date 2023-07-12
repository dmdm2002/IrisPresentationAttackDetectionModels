import os
import argparse
import torchvision.models as models
import tqdm
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from DNetPAD.CustomDataset import CustomDataset

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Description of all argument
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32)
parser.add_argument('--MaxEpochs', type=int, default=30)
parser.add_argument('--datasetPath', required=False, default='Z:/2nd_paper/dataset/ND/ROI/Ablation/NestedUVC_DualAttention_Parallel_Fourier/2-fold', type=str)
parser.add_argument('--TestdatasetPath', required=False, default='Z:/2nd_paper/dataset/ND/ROI/Ablation/NestedUVC_DualAttention_Parallel_Fourier/2-fold', type=str)
parser.add_argument('--outputPath', required=False, default='Z:/2nd_paper/backup/Compare/Detectors/DNet-PAD/NestedUVC_DualAttention_Parallel_Fourier/Notcross-fold/2-fold/try_1/', type=str)
parser.add_argument('--method', default='DesNet121', type=str)
parser.add_argument('--nClasses', default=2, type=int)

device = 'cuda'
args = parser.parse_args()

# Model
model = models.densenet121(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, args.nClasses)
model = model.to(device)

os.makedirs(f'{args.outputPath}/log', exist_ok=True)
os.makedirs(f'{args.outputPath}/ckp', exist_ok=True)


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ]
)
labels = ['fake', 'live']
tr_dataset = CustomDataset(args.datasetPath, 'B', labels, transform)
te_dataset = CustomDataset(args.TestdatasetPath, 'A', labels, transform)

lr = 0.005
optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-6, momentum=0.9)
lr_sched = optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)
criterion = nn.CrossEntropyLoss()

tr_loader = data.DataLoader(dataset=tr_dataset, batch_size=args.batchSize, shuffle=True)
te_loader = data.DataLoader(dataset=te_dataset, batch_size=args.batchSize, shuffle=False)

summary = SummaryWriter(f'{args.outputPath}/log')

best_score = {'epoch': 0, 'acc': 0, 'loss': 0}
history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
for ep in range(args.MaxEpochs):
    train_loss, train_acc = 0, 0
    test_loss, test_acc = 0, 0

    model.train()
    for _, (item, label) in enumerate(tqdm.tqdm(tr_loader, desc=f'[Train {ep}/{args.MaxEpochs}]')):
        item = item.to(device)
        label = label.to(device)

        logits = model(item)
        loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += (logits.argmax(1) == label).type(torch.float).sum().item()

    with torch.no_grad():
        model.eval()
        for _, (item, label) in enumerate(tqdm.tqdm(te_loader, desc=f'[Test {ep}/{args.MaxEpochs}]')):
            item = item.to(device)
            label = label.to(device)

            logits = model(item)
            loss = criterion(logits, label)

            test_loss += loss.item()
            test_acc += (logits.argmax(1) == label).type(torch.float).sum().item()

    train_loss = train_loss / len(tr_loader)
    train_acc = train_acc / len(tr_loader.dataset)

    test_loss = test_loss / len(te_loader)
    test_acc = test_acc / len(te_loader.dataset)

    if best_score['acc'] <= test_acc:
        best_score['acc'] = test_acc
        best_score['epoch'] = ep
        best_score['loss'] = test_loss

    print('\n')
    print('-------------------------------------------------------------------')
    print(f"Epoch: {ep}/{args.MaxEpochs}")
    print(f"Train acc: {train_acc} | Train loss: {train_loss}")
    print(f"Test acc: {test_acc} | Test loss: {test_loss}")
    print('-------------------------------------------------------------------')
    print(f"Best acc epoch: {best_score['epoch']}")
    print(f"Best acc: {best_score['acc']} | Best loss: {best_score['loss']}")
    print('-------------------------------------------------------------------')

    summary.add_scalar('Train/acc', train_loss, ep)
    summary.add_scalar('Train/loss', train_acc, ep)
    summary.add_scalar('Test/acc', test_acc, ep)
    summary.add_scalar('Test/loss', test_loss, ep)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optim_adamw_state_dict": optimizer.state_dict(),
            "epoch": ep,
        },
        os.path.join(f"{args.outputPath}/ckp", f"{ep}.pth"),
    )
