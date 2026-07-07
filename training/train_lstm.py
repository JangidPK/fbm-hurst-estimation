
import numpy as np
import torch


from data.generator import FractionalBrownianMotionGenerator
from data.dataset import FBmDataset, train_val_test_split
from models.lstm import LSTMHurstRegressor
from training.trainer import train_model, predict
from evaluation.metrics import summarize


def main(n_samples=2000, 
         T=256, 
         epochs=60, 
         seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("Generating dataset...")

    gen = FractionalBrownianMotionGenerator(T=T, 
                                            
                                            method="davies-harte", seed=seed)

    X, y = gen.generate_dataset(n_samples=n_samples, H_range=(0.1, 0.9))

    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = train_val_test_split(X, y)

    train_ds = FBmDataset(X_tr, y_tr, normalize="zscore")
    val_ds = FBmDataset(X_val, y_val, normalize="zscore")
    test_ds = FBmDataset(X_te, y_te, normalize="zscore")

    model = LSTMHurstRegressor(hidden_size=128, num_layers=2, dropout=0.2)

    print("Training LSTM...")
    model, history = train_model(model, train_ds, val_ds, epochs=epochs, lr=1e-3)

    preds, targets = predict(model, test_ds)
    metrics = summarize(targets, preds)
    print("Test metrics:", metrics)

    return model, history, (preds, targets)


if __name__ == "__main__":
    main()
