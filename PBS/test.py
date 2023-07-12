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

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Description of all argument
device = 'cuda:0'
# Model
model = PBSModel()
# model = model.to(device)

outputPath = 'Z:/2nd_paper/backup/Compare/Detectors/PBS/2-fold/try_0/'
datasetPath = 'Z:/2nd_paper/dataset/ND/Full/Compare/NestedUVC_DualAttention_Parallel_Fourier/Attack/Gamma'
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
te_dataset = CustomDataset(datasetPath, 'A/gamma_0.8', labels, transform)

lr = 1e-4
loss_fn = PixWiseBCELoss().to(device)
te_loader = data.DataLoader(dataset=te_dataset, batch_size=32, shuffle=False)

checkpoint = torch.load(f'{outputPath}/ckp/2.pth', map_location=device)
model = model.to(device)
model.load_state_dict(checkpoint["model_state_dict"])

test_loss, test_acc = 0, 0
with torch.no_grad():
    model.eval()
    for _, (item, mask, label) in enumerate(tqdm.tqdm(te_loader, desc=f'[Test]')):
        item = item.to(device)
        mask = mask.to(device)
        label = label.to(device)

        mask_logits, binary_logits = model(item)
        loss = loss_fn(mask_logits, binary_logits, mask, label)

        test_loss += loss.item()

        probs = binary_logits >= torch.FloatTensor([0.5]).to(device)
        test_acc += (probs == label).sum().item()

test_loss = test_loss / len(te_loader)
test_acc = test_acc / len(te_loader.dataset)

print('\n')
print('-------------------------------------------------------------------')
print(f"Test acc: {test_acc} | Test loss: {test_loss}")
print('-------------------------------------------------------------------')