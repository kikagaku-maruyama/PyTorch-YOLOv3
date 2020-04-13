from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

import warnings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)
    
    warnings.filterwarnings('ignore')

    # tensorboard logger を使用
    #logger = Logger("logs")

    # 結果を格納するリストを用意
    mAPs = []
    train_losses_tmp = []
    train_losses = []
    val_losses = []

    # device の指定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # output, checkpoints フォルダの作成
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # data config の取得
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # モデルを読み込み、重みを初期化
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # チェックポイント、事前学習済み重みの読み込み
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # dataloader 作成
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    # 最適化手法の選択
    optimizer = torch.optim.Adam(model.parameters())

    # ログ表示する指標
    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    # 学習ループ
    for epoch in range(opt.epochs):

        # 訓練モード
        model.train()

        start_time = time.time()

        # mini-batch ループ
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            
            batches_done = len(dataloader) * epoch + batch_i

            # cuda.FloatTensor に変換
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            # 推論 & ロス算出 
            # output; bbox, conf, cls
            loss, outputs = model(imgs, targets)
            loss.backward()

            train_losses_tmp.append(loss)

            # 累積勾配
            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            # ログ表示
            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            # yolo layer ごとに各種指標を表示（ミニバッチ単位）
            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                #tensorboard_log = []
                #for j, yolo in enumerate(model.yolo_layers):
                #    for name, metric in yolo.metrics.items():
                #        if name != "grid_size":
                #            tensorboard_log += [(f"{name}_{j+1}", metric)]
                #tensorboard_log += [("loss", loss.item())]
                #logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)
        
        avg_train_loss = torch.tensor(train_losses_tmp).mean()
        train_losses.append(avg_train_loss)
        
        # 検証データへの准電波
        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class, avg_val_loss = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )

            # val_loss をエポック単位で保存
            val_losses.append(avg_val_loss)


            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]

            #tensorboard logger に追加
            #logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

            # mAP をリストに追加
            mAPs.append(AP.mean())


        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)

    # 学習結果の表示
    import pandas as pd
    import matplotlib.pyplot as plt

    # mAP
    mAPs = torch.tensor(mAPs).numpy()
    result = pd.DataFrame({"mAP" : mAPs})
    result.plot()
    plt.savefig("mAP.png")

    # losses
    train_losses = torch.tensor(train_losses).numpy()
    val_losses = torch.tensor(val_losses).numpy()

    result = pd.DataFrame({"train_loss": train_losses, "val_loss": val_losses})
    result.plot()
    plt.savefig("losses.png")

