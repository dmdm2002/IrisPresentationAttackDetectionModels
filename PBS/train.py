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

from PBS.Loss import PixWiseBCELoss
from PBS.CustomDataset import CustomDataset
from PBS.Model.PBSModel import PBSModel
from PBS.Model.APBSModel import APBSModel

# Description of all argument
device = 'cuda:0'
# Model
model = APBSModel()
model = model.to(device)

outputPath = 'Z:/2nd_paper/backup/Compare/Detectors/A-PBS/2-fold/try_0/'
datasetPath = 'Z:/2nd_paper/dataset/ND/Full/Compare/NestedUVC_DualAttention_Parallel_Fourier/2-fold'
MaxEpochs = 20

os.makedirs(f'{outputPath}/log', exist_ok=True)
os.makedirs(f'{outputPath}/ckp', exist_ok=True)


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

labels = ['fake', 'live']
tr_dataset = CustomDataset(datasetPath, 'B', labels, transform)
te_dataset = CustomDataset(datasetPath, 'A', labels, transform)

lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
loss_fn = PixWiseBCELoss().to(device)

tr_loader = data.DataLoader(dataset=tr_dataset, batch_size=16, shuffle=True)
te_loader = data.DataLoader(dataset=te_dataset, batch_size=16, shuffle=False)

summary = SummaryWriter(f'{outputPath}/log')

best_score = {'epoch': 0, 'acc': 0, 'loss': 0}
history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
for ep in range(MaxEpochs):
    train_loss, train_acc = 0, 0
    test_loss, test_acc = 0, 0

    model.train()
    for _, (item, mask, label) in enumerate(tqdm.tqdm(tr_loader, desc=f'[Train {ep}/{MaxEpochs}]')):
        item = item.to(device)
        mask = mask.to(device)
        label = label.to(device)

        mask_logits, binary_logits = model(item)
        loss = loss_fn(mask_logits, binary_logits, mask, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        probs = binary_logits >= torch.FloatTensor([0.5]).to(device)
        train_acc += (probs == label).sum().item()

    with torch.no_grad():
        model.eval()
        for _, (item, mask, label) in enumerate(tqdm.tqdm(te_loader, desc=f'[Test {ep}/{MaxEpochs}]')):
            item = item.to(device)
            mask = mask.to(device)
            label = label.to(device)

            mask_logits, binary_logits = model(item)
            loss = loss_fn(mask_logits, binary_logits, mask, label)

            test_loss += loss.item()

            probs = binary_logits >= torch.FloatTensor([0.5]).to(device)
            test_acc += (probs == label).sum().item()

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
        os.path.join(f"{outputPath}/ckp", f"{ep}.pth"),
    )
