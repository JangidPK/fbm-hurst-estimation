

import copy
import torch
from torch.utils.data import DataLoader
import numpy as np

def train_model(model, 
                train_ds, val_ds, 
                epochs: int = 100, 
                batch_size: int = 64,
                 lr: float = 1e-3, 
                 weight_decay: float = 1e-5, 
                 patience: int = 10,
                 device: str | None = None, 
                 verbose: bool = True):
    
    """
    Train a sequence regression model 

    Returns
    -------
    model : the model with best validation-loss weights loaded
    history : dict with "train_loss" and "val_loss" lists

    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using: ", device)
    model.to(device)



    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)




    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()



    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0



    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum, n_train = 0.0, 0

        for xb, yb in train_loader:

            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * xb.size(0)
            n_train += xb.size(0)

        train_loss = train_loss_sum / n_train

        model.eval()
        val_loss_sum, n_val = 0.0, 0
        with torch.no_grad():

            for xb, yb in val_loader:

                xb, yb = xb.to(device), yb.to(device)

                preds = model(xb)
                loss = criterion(preds, yb)

                val_loss_sum += loss.item() * xb.size(0)
                n_val += xb.size(0)

        val_loss = val_loss_sum / n_val

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
            print(f"Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0

        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (best val_loss={best_val_loss:.5f})")
                break

    model.load_state_dict(best_state)
    return model, history


# predictionss
@torch.no_grad()
def predict(model, 
            dataset, 
            batch_size: int = 128, 
            device: str | None = None):
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds, targets = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        p = model(xb).cpu().numpy()
        preds.append(p)
        targets.append(yb.numpy())

    
    return np.concatenate(preds), np.concatenate(targets)
