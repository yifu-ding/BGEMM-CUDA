import torch
import torch.nn as nn
from bgemm_linear import *

import torch
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, Dataset
import pdb
import random
import os
import time
import argparse

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def train(model, train_loader, optimizer, criterion, device, dtype=torch.float):
    losses = []
    # ensure model is in training mode
    model.train()    
    
    for i, data in enumerate(train_loader, 0):        
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()   
        # forward pass
        output = model(inputs.to(dtype))
        loss = criterion(output, target.unsqueeze(1))
        # import pdb; pdb.set_trace()
        
        # backward pass + run optimizer to update weights
        loss.backward()
        optimizer.step()
    
        # keep track of loss value
        losses.append(loss.data.cpu().numpy()) 

    return losses


def test(model, test_loader, device, dtype=torch.float):    
    # ensure model is in eval mode
    model.eval() 
    y_true = []
    y_pred = []
    # st = time.time()
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            output_orig = model(inputs.to(dtype))
            # run the output through sigmoid
            output = torch.sigmoid(output_orig)  
            # compare against a threshold of 0.5 to generate 0/1
            pred = (output.detach().cpu().numpy() > 0.5) * 1
            target = target.cpu().to(dtype)
            y_true.extend(target.tolist()) 
            y_pred.extend(pred.reshape(-1).tolist())
    # ed = time.time()
    # print("test time: %.5f" % ((float)(ed-st) / 60.0))
    return accuracy_score(y_true, y_pred)


def display_loss_plot(losses, title="Training loss", xlabel="Iterations", ylabel="Loss"):
    # 'losses' can be any list
    x_axis = [i for i in range(len(losses))]
    plt.plot(x_axis,losses)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.show()
    plt.savefig("%s_x=%s_y=%s.pdf" % (title, xlabel, ylabel))
    plt.cla()
    plt.clf()
    
def get_preqnt_dataset(data_dir: str, train: bool):
    unsw_nb15_data = np.load(data_dir + "/unsw_nb15_binarized.npz")
    if train:
        partition = "train"
    else:
        partition = "test"
    part_data = unsw_nb15_data[partition].astype(np.float16)
    part_data = torch.from_numpy(part_data)
    # part_data_in = part_data[:, :-1]
    sample_num = int(part_data.shape[0]/32)*32
    part_data_in = part_data[:sample_num, :512]-0.5
    part_data_out = part_data[:sample_num, -1]
    return TensorDataset(part_data_in, part_data_out)

class FC_half2float(nn.Linear):

    def __init__(self, in_channels, out_channels, bias=False):
        super(FC_half2float, self).__init__(in_channels, out_channels, bias)

    def forward(self, x):
        out = nn.functional.linear(x.float(), self.weight.float())
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out


class MLP(nn.Module):
    def __init__(self, Linear=BGEMMLinear, dtype=torch.half, bias=True,
                 input_size=512, hidden1=128, hidden2=128, hidden3=128, num_classes=1):
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(input_size, hidden1, bias=bias)
        self.lin2 = Linear(hidden1, hidden2, bias=bias)
        self.lin3 = Linear(hidden2, hidden3, bias=bias)
        # self.lin2.weight = nn.Parameter()
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.bn3 = nn.BatchNorm1d(hidden3)
        
        self.fc = FC_half2float(hidden3, num_classes, bias=True)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = self.lin1(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.relu(x)
        
        x = self.lin2(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.relu(x)
        
        x = self.lin3(x)
        x = self.bn3(x)
        x = self.dropout(x)
        x = self.relu(x)
        
        x = self.fc(x)
        return x
    
def main():
    # Setting seeds for reproducibility
    seed_all(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("model", default="bgemm_elastic", type=str, choices=['bgemm_elastic', 'bgemm_bireal', 'nn_elastic', 'bnn_fp16', 'fp16'],)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Target device: " + str(device))
    assert device == torch.device("cuda"), "cannot use bgemm_linear without cuda."

    if args.model == 'bgemm_elastic':
        linear, dtype = BGEMMLinear_elastic_signed, torch.float # torch.half
    elif args.model == 'nn_elastic':
        linear, dtype = NNLinear_elastic_signed, torch.float # torch.half
    elif args.model == 'bgemm_bireal':
        linear, dtype = BGEMMLinear_bireal, torch.float # torch.half
    elif args.model == 'bnn_fp16':
        linear, dtype = BNNLinear, torch.float # torch.half
    elif args.model == "fp16":
        linear, dtype = nn.Linear, torch.float
    else:
        linear, dtype = BGEMMLinear_elastic_signed, torch.float # torch.half
    print("use linear: %s " % linear)
        
    mlp = MLP(Linear=linear, dtype=dtype, bias=True)
    mlp = mlp.to(device)
    
    # get dataset 
    train_quantized_dataset = get_preqnt_dataset(data_dir=".", train=True)
    test_quantized_dataset = get_preqnt_dataset(data_dir=".", train=False)

    print("Samples in each set: train = %d, test = %s" % (len(train_quantized_dataset), len(test_quantized_dataset))) 
    print("Shape of one input sample: " +  str(train_quantized_dataset[0][0].shape))

    # set up dataloader
    batch_size = 256
    train_quantized_loader = DataLoader(train_quantized_dataset, batch_size=batch_size, shuffle=True)
    test_quantized_loader = DataLoader(test_quantized_dataset, batch_size=batch_size, shuffle=False)    
    
    
    # define training settings
    num_epochs = 20
    lr = 0.001
    # loss criterion and optimizer
    criterion = nn.BCEWithLogitsLoss().to(device)
    all_parameters = list(mlp.named_parameters())
    scaling_factors = ['sw', 'sa', 'zpw', 'zpa']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in all_parameters if (any(nd in n for nd in scaling_factors))], 'lr': 1e-4},
        {'params': [p for n, p in all_parameters if (not any(nd in n for nd in scaling_factors))]},
    ]
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr, betas=(0.9, 0.999))

    # Setting seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    running_loss = []
    running_test_acc = []
    t = trange(num_epochs, desc="Training loss", leave=True)
    
    st = time.time()
    for epoch in t:
        loss_epoch = train(mlp, train_quantized_loader, optimizer, criterion, device, dtype=dtype)
        test_acc = test(mlp, test_quantized_loader, device, dtype=dtype)
        t.set_description("Training loss = %f test accuracy = %f" % (np.mean(loss_epoch), test_acc))
        t.refresh() # to show immediately the update           
        running_loss.append(loss_epoch)
        running_test_acc.append(test_acc)
    
    ed = time.time()
    print("Total training time: %.5f s" % ((float)(ed-st)))
    # %matplotlib inline

    loss_per_epoch = [np.mean(loss_per_epoch) for loss_per_epoch in running_loss]
    display_loss_plot(loss_per_epoch)

    acc_per_epoch = [np.mean(acc_per_epoch) for acc_per_epoch in running_test_acc]
    display_loss_plot(acc_per_epoch, title="Test accuracy", ylabel="Accuracy [%]")

    test(mlp, test_quantized_loader, device, dtype=dtype)
    # torch.save(mlp.state_dict(), "state_dict_self-trained.pth")
    
    
main()