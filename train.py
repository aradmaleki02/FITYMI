from torch import nn
from torch.optim import SGD
from tqdm import tqdm
import eval

from datasets.datasets_train import get_full_train_loader


def freeze_model(model, layers=6):
    for i in range(layers):
        for p in model.transformer.encoder.layer[i].parameters():
            p.requires_grad = False


def train_model(args, model, device, train_loader=None):
    # freeze_model(model)
    optimizer = SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model.train()
    loss_funct = nn.CrossEntropyLoss()
    if train_loader is None:
        full_train_loader = get_full_train_loader(args)
    else:
        full_train_loader = train_loader
    for epoch in range(args.epochs):
        total_num, total_loss = 0, 0
        train_bar = tqdm(full_train_loader)
        for (x, y) in train_bar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = loss_funct(logits, y)
            loss.backward()
            optimizer.step()
            total_num += x.size(0)
            total_loss += loss.item() * x.size(0)
            train_bar.set_description(f'Training Epoch : {epoch}, Loss: {total_loss / total_num:.6f}')
        print("=" * 100)
        # eval.evaluate_model(args, model, device)
    return model
