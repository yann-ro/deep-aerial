import numpy as np
from time import time
import torch
from model.metrics.segmentation import mIoU, pixel_accuracy
from tqdm.notebook import tqdm
from torchvision import transforms as T
import matplotlib.pyplot as plt
from model.metrics import plot_acc, plot_loss, plot_score

class SegModel:
    def __init__(self, model, device) -> None:
        self.model = model
        self.device = device
        self.history = {}

    def fit(
        self, epochs, model, train_loader, val_loader, criterion, optimizer, scheduler
    ):
        torch.cuda.empty_cache()
        train_losses = []
        test_losses = []
        val_iou = []
        val_acc = []
        train_iou = []
        train_acc = []
        lrs = []
        min_loss = np.inf

        model.to(self.device)
        fit_time = time()

        for e in range(epochs):
            since = time()
            running_loss = 0
            iou_score = 0
            accuracy = 0

            # training loop
            model.train()

            for _, data in enumerate(tqdm(train_loader)):

                # training phase
                image_tiles, mask_tiles = data

                image = image_tiles.to(self.device)
                mask = mask_tiles.to(self.device)

                # forward
                output = model(image)
                loss = criterion(output, mask)

                # evaluation metrics
                iou_score += mIoU(output, mask)
                accuracy += pixel_accuracy(output, mask)

                # backward
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # step the learning rate
                lrs.append(self.get_lr(optimizer))
                scheduler.step()

                running_loss += loss.item()

            model.eval()
            test_loss = 0
            test_accuracy = 0
            val_iou_score = 0
            # validation loop
            with torch.no_grad():
                for _, data in enumerate(tqdm(val_loader)):
                    # reshape to 9 patches from single image, delete batch size
                    image_tiles, mask_tiles = data

                    image = image_tiles.to(self.device)
                    mask = mask_tiles.to(self.device)
                    output = model(image)
                    # evaluation metrics
                    val_iou_score += mIoU(output, mask)
                    test_accuracy += pixel_accuracy(output, mask)
                    # loss
                    loss = criterion(output, mask)
                    test_loss += loss.item()

            # calculatio mean for each batch
            train_losses.append(running_loss / len(train_loader))
            test_losses.append(test_loss / len(val_loader))

            if min_loss > (test_loss / len(val_loader)):
                print(
                    f"Loss Decreasing.. {min_loss:.3f} >> {(test_loss / len(val_loader)):.3f} "
                )
                min_loss = test_loss / len(val_loader)

            if (test_loss / len(val_loader)) > min_loss:
                min_loss = test_loss / len(val_loader)
                print("Loss Not Decrease")

            # iou
            val_iou.append(val_iou_score / len(val_loader))
            train_iou.append(iou_score / len(train_loader))
            train_acc.append(accuracy / len(train_loader))
            val_acc.append(test_accuracy / len(val_loader))
            print(
                f"Epoch:{e + 1}/{epochs}..",
                f"Train Loss: {running_loss / len(train_loader):.3f}..",
                f"Val Loss: {test_loss / len(val_loader):.3f}..",
                f"Train mIoU:{iou_score / len(train_loader):.3f}..",
                f"Val mIoU: {val_iou_score / len(val_loader):.3f}..",
                f"Train Acc:{accuracy / len(train_loader):.3f}..",
                f"Val Acc:{test_accuracy / len(val_loader):.3f}..",
                f"Time: {(time() - since) / 60:.2f}m",
            )

        self.history = {
            "train_loss": train_losses,
            "val_loss": test_losses,
            "train_miou": train_iou,
            "val_miou": val_iou,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "lrs": lrs,
        }
        print(f"Total time: {(time() - fit_time) / 60:.2f} min")

        return self.history

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    def predict_image_mask_pixel(
        self, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ):
        self.model.eval()
        t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        image = t(image)
        self.model.to(self.device)
        image = image.to(self.device)
        mask = mask.to(self.device)
        with torch.no_grad():

            image = image.unsqueeze(0)
            mask = mask.unsqueeze(0)

            output = self.model(image)
            acc = pixel_accuracy(output, mask)
            masked = torch.argmax(output, dim=1)
            masked = masked.cpu().squeeze(0)
        return masked, acc

    def predict_image_mask_miou(
        self, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ):
        self.model.eval()
        t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        image = t(image)
        self.model.to(self.device)
        image = image.to(self.device)
        mask = mask.to(self.device)

        with torch.no_grad():

            image = image.unsqueeze(0)
            mask = mask.unsqueeze(0)

            output = self.model(image)
            score = mIoU(output, mask)
            masked = torch.argmax(output, dim=1)
            masked = masked.cpu().squeeze(0)
        return masked, score

    def miou_score(self, test_set):
        score_iou = []
        for i in tqdm(range(len(test_set))):
            img, mask = test_set[i]
            pred_mask, score = self.predict_image_mask_miou(img, mask)
            score_iou.append(score)
        
        print(f"Test Set mIoU {np.mean(score_iou):.4f}")
        return score_iou

    def pixel_acc(self, test_set):
        accuracy = []
        for i in tqdm(range(len(test_set))):
            img, mask = test_set[i]
            pred_mask, acc = self.predict_image_mask_pixel(img, mask)
            accuracy.append(acc)
        
        print(f"Test Set Pixel Accuracy {np.mean(accuracy):.4f}")
        return accuracy
    
    def plot_history(self):
        plt.figure(figsize=(20,5))
        
        plt.subplot(1,3,1)
        plot_loss(self.history)
        
        plt.subplot(1,3,2)
        plot_score(self.history)
        
        plt.subplot(1,3,3)
        plot_acc(self.history)
        plt.show()
