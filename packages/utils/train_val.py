import torch
import warnings
from gradient_descent_the_ultimate_optimizer import gdtuo

def train(model, loss_fn, optimizer, num_epochs, trainloader, device):
    warnings.filterwarnings("ignore")

    print(f"\n***Beginning Training***")

    # use gdtuo.ModuleWrapper to allow nn.Module be optimized by hyperoptimizers
    mw = gdtuo.ModuleWrapper(model, optimizer=optimizer)
    mw.initialize()

    for epoch in range(1, num_epochs+1):
        running_loss = 0.0
        for index, (features_, labels_) in enumerate(trainloader):
            mw.begin() # call this before each step, enables gradient tracking on desired params
            features, labels = torch.reshape(features_, (-1, 28 * 28)).to(device), labels_.to(device)
            pred = mw.forward(features)
            loss = loss_fn(pred, labels)
            mw.zero_grad()
            loss.backward(create_graph=True) # important! use create_graph=True
            mw.step()
            running_loss += loss.item() * features_.size(0)
        train_loss = running_loss / len(trainloader.dataset)
        print(f"\tEpoch[{epoch}/{num_epochs}]: Training Loss = {train_loss:.5f}", flush=True)
        
    print(f"***Training Complete***")