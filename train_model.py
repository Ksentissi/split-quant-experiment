"""
Trains the SplitCNN once on CIFAR-10 and saves a checkpoint. This is the
ONLY training run: the exact same weights and the exact same cut layer are
reused for every compression method / bit rate / replica count tested in
run_experiment.py (Control: do not retrain differently per method).
"""
import argparse
import time

import torch
import torch.nn as nn

from data_utils import get_train_loader, get_test_loader, get_device
from model import SplitCNN


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="split_cnn_cifar10.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader = get_train_loader(args.data_dir, args.batch_size)
    test_loader = get_test_loader(args.data_dir)

    model = SplitCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            running_loss += loss.item() * x.size(0)
        sched.step()
        train_loss = running_loss / len(train_loader.dataset)
        test_acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch+1}/{args.epochs}  loss={train_loss:.4f}  test_acc={test_acc:.4f}  "
              f"time={time.time()-t0:.1f}s")

    final_acc = evaluate(model, test_loader, device)
    print(f"Final clean test accuracy (uncompressed): {final_acc:.4f}")
    torch.save({"state_dict": model.state_dict(), "test_acc": final_acc}, args.out)
    print(f"Saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
