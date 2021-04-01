"""
In this example we will be using the probabilistic attacker on spiking input.

1. Download MNIST and generate spiking version (N(euromorphic)-MNIST). 
N-MNIST has elements that are frames of a short sequence, where the MNIST digit is slightly varied.
A DVS (Dynamic Vision Sensor) usually returns events (like spikes in the retina). These spikes
have different polarity (boolean, 0 or 1). Polarity 1 means that a spike was generated
by a sharp increase in intensity and polarity 0 vice versa.

2. Train standard CNN on MNIST

3. Use SINABS to convert to spiking version 

4. Attack:
We interpret the spikes of an image as Bernoulli-distributed random variables.
If only the spiking image is given, e.g. [0,1,0,0,1], then these events are interpreted
as probabilities that the corresponding neuron emits a spike.

If the two images I_t and I_{t+1} are given, from which the events are generated, we can
interpret sigmoid(|I_t - I_{t+1}| / T) as the spiking probabilities. This way, we leverage
the fact that we have the information about the images that generated these events.

The attacker (e.g. PGD) now tries to iteratively change the spiking probabilities in some epsilon ball.
The network is evaluated on every time-step and we use monte carlo sampling to approximate the gradient.
We use the reparameterization trick for Bernoulli variables in order to compute the gradient of the
probabilities.

Alg.:
Given: Network f, Image X [T,32,32]
Return: Attacking probabilities P [T,32,32]
P <- X # Initialize the probabilities with the spikes in the image
P <- P + eps # Add some random initial noise to the probabilities to avoid sampling X the whole time
for i in range(N_{attack steps}):
    g <- [0] # Initialize gradient to zero
    for j in range(N_{MC}):
        e <- U[0,1] # Sample from standard uniform distribution
        X_j <- g(e,P,T) # Apply reparameterization trick using sampled random, current probabilities, and temperature 
        g <- g + 1/N_{MC} * grad( loss(y,round(X_j)) ) w.r.t. X_j # Compute gradient of loss under sampled input
    g_transform <- arg max_{v, |v|_p <= 1} vTg # This is just sign(g) if we are in l-infinity or g/norm2(g) if we are in l2
    P <- Project(P + alpha * g_transform) # Step of projected gradient ascent (PGA)
return P
"""

import pathlib
import urllib.request
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from videofig import videofig
from sinabs.from_torch import _from_model
from sinabs.utils import normalize_weights
from sinabs.network import Network as SinabsNetwork
from aermanager.parsers import parse_nmnist
from aermanager.datasets import FramesDataset, SpikeTrainDataset
from aermanager.dataset_generator import gen_dataset_from_folders
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from cleverhans.torch.utils import clip_eta
from cleverhans.torch.utils import optimize_linear
from cleverhans.torch.attacks.fast_gradient_method import _fast_gradient_method_grad

torch.manual_seed(42)

def get_prediction(net, data, mode="prob"):
    net.reset_states()
    if mode == "prob":
        output = net(data)
    elif mode == "non_prob":
        output = net.forward_np(data)
    else:
        raise Exception
    output = output.sum(axis=0)
    pred = output.argmax()
    return pred

def get_test_acc(net, dataloader, limit = -1):
    acc = []
    for data, target in dataloader:
        data = data[0].to(device)
        data[data > 1] = 1.0
        pred = get_prediction(net, data)
        correct = pred.item() == target.item()
        acc.append(correct)
        if len(acc) > limit:
            break
    return sum(acc)/len(acc)*100

path = pathlib.Path("./data/")
path.mkdir(parents=True, exist_ok=True)
def load_n_extract(lab, url):
    if not (path / f"N-MNIST/{lab}.zip").exists():
        urllib.request.urlretrieve(url, path / f"N-MNIST/{lab}.zip")
        with zipfile.ZipFile(path / f"N-MNIST/{lab}.zip", 'r') as f:
            f.extractall(path / "N-MNIST/")
load_n_extract("Test", "https://www.dropbox.com/sh/tg2ljlbmtzygrag/AADSKgJ2CjaBWh75HnTNZyhca/Test.zip?dl=1")
load_n_extract("Train", "https://www.dropbox.com/sh/tg2ljlbmtzygrag/AABlMOuR15ugeOxMCX0Pvoxga/Train.zip?dl=1")

def gen_ds(lab):
    if not (path / f"dataset300/N-MNIST/{lab}/").exists():
        gen_dataset_from_folders(
            source_path=path / f"N-MNIST/{lab}", 
            destination_path=path / f"dataset300/N-MNIST/{lab}/",
            pattern="*.bin",
            spike_count=300,
            parser=parse_nmnist)
gen_ds("Test")
gen_ds("Train")

ann = nn.Sequential(
    nn.Conv2d(2, 20, 5, 1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2,2),
    nn.Conv2d(20, 32, 5, 1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2,2),
    nn.Conv2d(32, 128, 3, 1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2,2),
    nn.Flatten(),
    nn.Linear(128, 500, bias=False),
    nn.ReLU(),
    nn.Linear(500, 10, bias=False),
)

device = "cuda" if torch.cuda.is_available() else "cpu"
ann = ann.to(device)

if not (path / "mnist_params300.pt").exists():
    dataset_train = FramesDataset(
        path / "dataset300/N-MNIST/Train/",
        transform=np.float32,
        target_transform=int)
    dataloader_train = DataLoader(dataset_train, shuffle=True, num_workers=4, batch_size=128)
    optim = torch.optim.Adam(ann.parameters(), lr=1e-3)
    n_epochs = 1
    for n in range(n_epochs):
        for data, target in pbar:
            data, target = data.to(device), target.to(device) # GPU
            output = ann(data)
            optim.zero_grad()
            loss = F.cross_entropy(output, target)
            loss.backward()
            optim.step()
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
    torch.save(ann.state_dict(), "./data/mnist_params300.pt")
else:
    ann.load_state_dict(torch.load(path / "mnist_params300.pt"))

dataset_test_frames = FramesDataset(
    path / "dataset300/N-MNIST/Test/",
    transform=np.float32, # must turn frame data into floats
    target_transform=int)
dataloader_test_frames = DataLoader(dataset_test_frames, shuffle=False, num_workers=4, batch_size=64)
batch = None
for data, target in dataloader_test_frames:
    batch = data
    break
normalize_weights(ann.cpu(), 
                    torch.tensor(batch).float(),
                    output_layers=['1','4','7','11'],
                    param_layers=['0','3','6','10','12'])

# Create spiking model
input_shape = (2, 34, 34)
spk_model = _from_model(ann, input_shape=input_shape, add_spiking_output=True)

def reparameterization_bernoulli(P, temperature):
    eps = 1e-20 # Avoid -inf
    rand_unif = torch.rand(P.size())
    X = torch.sigmoid((torch.log(rand_unif+eps)-torch.log(1-rand_unif+eps)+torch.log(P+eps)-torch.log(1-P+eps))/temperature)
    return X

class ProbNetwork(SinabsNetwork):
    def __init__(
        self,
        model,
        spk_model,
        input_shape,
        synops = False,
        temperature = 0.01
    ):
        self.temperature = temperature
        super().__init__(model, spk_model, input_shape, synops)

    def forward(self, P):
        X = reparameterization_bernoulli(P, self.temperature)
        return super().forward(X)

    def forward_np(self, X):
        return super().forward(X)

prob_net = ProbNetwork(
        ann,
        spk_model,
        input_shape=input_shape
    )
prob_net.spiking_model[0].weight.data *= 7

dataset_test_spiketrains = SpikeTrainDataset(
    path / "dataset300/N-MNIST/Test/",
    transform=np.float32,
    target_transform=int,
    dt=1000)
dataloader_test_spiketrains = DataLoader(dataset_test_spiketrains, shuffle=True, num_workers=4, batch_size=1)

for data, target in dataloader_test_spiketrains:
    P0 = data
    break

P0 = P0[0].to(device)
P0[P0 > 1] = 1.0
model_pred = get_prediction(prob_net, P0)

# Attack
N_pgd = 30
N_MC = 10
eps = 0.05
eps_iter = 0.01
rand_minmax = 0.01
norm = np.inf

eta = torch.zeros_like(P0).uniform_(-rand_minmax, rand_minmax) # Calculate initial perturbation
eta = clip_eta(eta, norm, eps) # Clip initial perturbation
P_adv = P0 + eta
P_adv = torch.clamp(P_adv, 0.0, 1.0) # Clip for probabilities

def loss_fn(spike_out, target):
    outputs = torch.reshape(torch.max(spike_out,axis=0)[0], (1,10))
    target = torch.tensor([target])
    return F.cross_entropy(outputs, target)

def get_grad(prob_net, P_adv, eps_iter, norm, model_pred, loss_fn):
    prob_net.reset_states()
    g = _fast_gradient_method_grad(
            model_fn=prob_net,
            x=P_adv,
            eps=eps_iter,
            norm=norm,
            y=model_pred,
            loss_fn=loss_fn)
    return g

def get_mc_P_adv(prob_net, P_adv, eps_iter, norm, loss_fn, N_MC):
    g = 0.0
    for j in range(N_MC):
        g_j = get_grad(prob_net, P_adv, eps_iter, norm, model_pred, loss_fn)
        g += 1/N_MC * g_j
    eta = optimize_linear(g, eps, norm)    
    P_adv = P_adv + eta    
    return P_adv

for i in range(N_pgd):
    print(i,"/",N_pgd)
    P_adv = get_mc_P_adv(prob_net, P_adv, eps_iter, norm, loss_fn, N_MC)
    eta = P_adv - P0
    eta = clip_eta(eta, norm, eps)
    P_adv = P0 + eta
    P_adv = torch.clamp(P_adv, 0.0, 1.0)

# Evaluate the network N_MC times
print("Original prediction",get_prediction(prob_net, P0))
for i in range(N_MC):
    model_pred = get_prediction(prob_net, P_adv, "prob")
    print("Trial",i,model_pred)

# # Evaluate the resulting data
N_rows = N_cols = 4

class Redraw(object):
    def __init__(self, data, pred):
        self.initialized = False
        self.data = data # [T,32,32]
        self.pred = pred
        self.f0 = 0
        self.max = self.data.size(0)

    def draw(self, f, ax):
        X = self.data[int(self.f0 % self.max)]
        if not self.initialized:
            ax.set_title(f"Pred {str(float(self.pred))}")
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_yticks([])
            ax.set_xticks([])
            self.im = ax.imshow(self.data[0])
            self.initialized = True
        else:
            self.im.set_data(X)
        self.f0 += 1

def redraw_fn(f, axes):
    for i in range(len(redraw_fn.sub)):
        redraw_fn.sub[i].draw(f, axes[i])

data = []
for i in range(N_rows * N_cols):
    image = reparameterization_bernoulli(P_adv, temperature=prob_net.temperature)
    assert ((image >= 0.0) & (image <= 1.0)).all()
    pred = get_prediction(prob_net, torch.round(image), "non_prob")
    data.append((torch.sum(image, 1),pred))

redraw_fn.sub = [Redraw(el[0],el[1]) for el in data]

videofig(
    num_frames=100,
    play_fps=50,
    redraw_func=redraw_fn, 
    grid_specs={'nrows': N_rows, 'ncols': N_cols})