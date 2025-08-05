# src/train_utils.py

import os
import torch
from tqdm import tqdm

best_acc = 0  # reținut global pentru salvarea celui mai bun model


def train(net, trainloader, optimizer, scheduler, criterion, epoch, device):
    print("\n🟩 Training - Epoca", epoch + 1)
    net.train()
    train_loss = 0
    total = 0
    total_correct = 0

    for inputs, targets in tqdm(trainloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == targets).sum().item()
        total += targets.size(0)

    acc = 100.0 * total_correct / total
    avg_loss = train_loss / len(trainloader)
    print(f"✅ Train Epoca [{epoch + 1}]  Loss: {avg_loss:.4f}  Accuracy: {acc:.2f}%")


def test(net, testloader, optimizer, criterion, epoch, device, results_txt="results", model_name="model_best"):
    global best_acc
    print("\n🟦 Validare - Epoca", epoch + 1)
    net.eval()
    test_loss = 0
    total = 0
    total_correct = 0

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, desc="Validation"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == targets).sum().item()
            total += targets.size(0)

    acc = 100.0 * total_correct / total
    avg_loss = test_loss / len(testloader)

    print(f"\n📊 Validation Epoca [{epoch + 1}]  Loss: {avg_loss:.4f}  Accuracy: {acc:.2f}%")

    # Salvează log
    with open(results_txt + ".txt", "a") as f:
        f.write(f"Validation Epoch {epoch + 1}\tLoss: {avg_loss:.4f}\tAcc@1: {acc:.2f}%\n")

    # Salvează cel mai bun model
    if acc > best_acc:
        print(f"💾 Saving new best model with Accuracy: {acc:.2f}%")
        state = {
            'model_state_dict': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }

        os.makedirs('checkpoint', exist_ok=True)
        torch.save(state, f'checkpoint/{model_name}.pt')
        best_acc = acc

    return best_acc
