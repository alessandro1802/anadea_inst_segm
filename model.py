import os
import numpy as np
from contextlib import redirect_stdout

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from pycocotools.cocoeval import COCOeval

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
import torchmetrics

import pytorch_lightning as pl


class SegmentationModule(pl.LightningModule):
    def __init__(self, batch_size, num_classes, height, width):
        super().__init__()
        self.width = width
        self.height = height
        self.batch_size = batch_size

        # Backbone MobileNetV2
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features
        # Freeze layers in the backbone for fine-tuning
        for param in self.backbone.parameters():
            param.requires_grad = False
        # A new classifier for segmentation task
        self.conv1 = nn.Conv2d(1280, 512, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(512, num_classes, kernel_size=1)
        # Ensure the new classifier layers require gradients
        for param in self.conv1.parameters():
            param.requires_grad = True
        for param in self.conv2.parameters():
            param.requires_grad = True

        # Loss function
        self.dice_loss = self.dice_loss_fn
        # Metrics
        self.iou_metric = torchmetrics.JaccardIndex(num_classes=num_classes, task="binary")
        self.iou = 0.0
        self.map = 0.0

    def forward(self, x):
        # Backbone forward pass
        x = self.backbone(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        # Upscale the output using bilinear interpolation to match the target mask size
        x = F.interpolate(x, size=(self.height, self.width), mode='bilinear', align_corners=False)
        return x

    def loss_step(self, batch, batch_idx):
        images, masks = batch
        images = torch.stack(images)
        masks = torch.stack(masks)
        # Forward pass
        output = self.forward(images)
        # Compute Loss
        loss_dice = self.dice_loss(output, masks)
        return loss_dice, output, masks

    def training_step(self, batch, batch_idx):
        total_loss, output, masks = self.loss_step(batch, batch_idx)
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, batch_size=self.batch_size, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, output, masks = self.loss_step(batch, batch_idx)
        pred_mask = output
        self.iou = self.iou_metric(pred_mask, masks)
        self.map = self.compute_map(pred_mask, masks)
        self.log("val_loss", total_loss, on_step=True, on_epoch=True, batch_size=self.batch_size, prog_bar=True)
        return total_loss

    def test_step(self, batch, batch_idx):
        total_loss, output, masks = self.loss_step(batch, batch_idx)
        pred_mask = output
        self.iou = self.iou_metric(pred_mask, masks)
        self.map = self.compute_map(pred_mask, masks)
        self.log("test_loss", total_loss, on_step=True, on_epoch=True, batch_size=self.batch_size, prog_bar=True)
        return total_loss

    def compute_map(self, preds, masks):
        coco_preds = []
        coco_gts = []
        coco_images = []
        for i in range(preds.size(0)):
            image_id = i  # Using the batch index as a unique image ID
            pred = preds[i]
            gt = masks[i]
            # Convert masks to binary format (0 and 1)
            pred_binary = (pred.argmax(dim=0).cpu().numpy()).astype(np.uint8)  # Converting to binary mask
            gt_binary = (gt.argmax(dim=0).cpu().numpy()).astype(np.uint8)
            # Encode the binary masks in COCO RLE format
            pred_rle = maskUtils.encode(np.asfortranarray(pred_binary))
            gt_rle = maskUtils.encode(np.asfortranarray(gt_binary))
            # Add prediction and ground truth entries
            coco_preds.append({
                'image_id': image_id,
                'category_id': 1,  # Assuming a single class (e.g., cell)
                'segmentation': pred_rle,
                'score': 1.0,  # Add a dummy confidence score
                'id': i,
                'area': np.sum(pred_binary)  # Calculate the area as the sum of non-zero pixels in the binary mask
            })
            coco_gts.append({
                'image_id': image_id,
                'category_id': 1,
                'segmentation': gt_rle,
                'iscrowd': 0,  # Add iscrowd field, set to 0 for individual instances
                'id': i,
                'area': np.sum(gt_binary)  # Calculate the area as the sum of non-zero pixels in the binary mask
            })
            # Add image metadata (required by COCOeval)
            coco_images.append({
                'id': image_id,
                'height': gt.shape[1],  # Height of the mask
                'width': gt.shape[2]  # Width of the mask
            })
        # Create COCO-style datasets with annotations and images
        coco_dt = COCO()
        coco_gt = COCO()
        coco_dt.dataset['annotations'] = coco_preds
        coco_gt.dataset['annotations'] = coco_gts
        # Add the required categories and images metadata
        category = {'id': 1, 'name': 'cell'}  # Define category info
        coco_dt.dataset['categories'] = [category]
        coco_gt.dataset['categories'] = [category]
        coco_dt.dataset['images'] = coco_images
        coco_gt.dataset['images'] = coco_images
        with open(os.devnull, 'w') as fnull:
            with redirect_stdout(fnull):
                coco_dt.createIndex()
                coco_gt.createIndex()
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
        with open(os.devnull, 'w') as fnull:
            with redirect_stdout(fnull):
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
        return coco_eval.stats[0]  # mAP score

    def on_validation_epoch_end(self):
        self.log("val_IoU", self.iou, on_epoch=True, batch_size=self.batch_size)
        self.iou = 0.0
        self.log("val_MaP", self.map, on_epoch=True, batch_size=self.batch_size)
        self.map = 0.0

    def on_test_epoch_end(self):
        self.log("val_IoU", self.iou, on_epoch=True, batch_size=self.batch_size)
        self.iou = 0.0
        self.log("val_MaP", self.map, on_epoch=True, batch_size=self.batch_size)
        self.map = 0.0

    def dice_loss_fn(self, preds, targets, smooth=1e-6):
        """Compute the Dice coefficient loss for multi-class segmentation."""
        preds = torch.sigmoid(preds)  # Shape: (B, 2, H, W)
        preds_flat = preds.view(preds.size(0), preds.size(1), -1)  # Shape: (B, 2, H*W)
        targets_flat = targets.view(targets.size(0), targets.size(1), -1)  # Shape: (B, 2, H*W)
        intersection = torch.sum(preds_flat * targets_flat, dim=2)
        union = torch.sum(preds_flat, dim=2) + torch.sum(targets_flat, dim=2)
        dice_scores = (2. * intersection + smooth) / (union + smooth)
        return 1 - torch.mean(dice_scores)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]
