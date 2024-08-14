import os
import time
import json
import torch
import random
import warnings
import numpy as np
import setproctitle
import pandas as pd
from tqdm import tqdm
import torchnet as tnt
from torch.utils.data import DataLoader
from utils.dataset import SICKLE_Dataset
from utils import utae_utils, model_utils
from utils.weight_init import weight_init
from utils.metric import get_metrics, RMSELoss
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings("ignore")
setproctitle.setproctitle("miras_crop_yield_beginning")

class CFG:
    model='utae'
    encoder_widths=[64, 128]
    decoder_widths=[32, 128]
    out_conv=[32, 16]
    str_conv_k=4
    str_conv_s=2
    str_conv_p=1
    agg_mode='att_group'
    encoder_norm='group'
    n_head=16
    d_model=256
    d_k=4
    device="cuda" if torch.cuda.is_available() else "cpu"
    num_workers=20
    seed=42
    epochs=10
    batch_size=32
    lr=0.1
    num_classes=1
    ignore_index=-999
    pad_value=0
    padding_mode='reflect'
    resume=''
    run_id=''
    task="crop_yield"
    data_dir="./sickle_toy_dataset"
    use_augmentation=True,
    actual_season=False

def checkpoint(log, config):
    with open(
            os.path.join("logs", "trainlog.json"), "w"
    ) as outfile:
        json.dump(log, outfile, indent=4)

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print("Can not use deterministic algorithm. Error: ", e)
    print(f"> SEEDING DONE {seed}")

def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(c, device) for c in x]

def iterate(model,
            data_loader,
            criterion,
            optimizer=None,
            scheduler=None,
            mode="train",
            epoch=1,
            task="crop_yield",
            device=None,
            CFG=None):
    loss_meter = tnt.meter.AverageValueMeter()
    predictions = None
    targets = None
    pid_masks = None
    t_start = time.time()
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=mode)
    for i, batch in pbar:
        if device is not None:
            batch = recursive_todevice(batch, device)
        data, masks = batch
        plot_mask = masks["plot_mask"]
        masks = masks[task]
        masks = masks.float()
        if mode != "train":
            with torch.inference_mode():
                y_pred = model(data)
        else:
            optimizer.zero_grad()
            y_pred = model(data)
        if task=="crop_yield": 
            loss = criterion(y_pred, masks, plot_mask)
        if mode == "train":
            loss.backward()
            optimizer.step()
        if predictions is None:
            predictions = y_pred
            targets = masks
            pid_masks = plot_mask
        else:
            predictions = torch.cat([predictions, y_pred], dim=0)
            targets = torch.cat([targets, masks], dim=0)
            pid_masks = torch.cat([pid_masks, plot_mask], dim=0)
        loss_meter.add(loss.item())
        mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        pbar.set_postfix(
            Loss=f"{loss.item():0.4f}",
            gpu_mem=f"{mem:0.2f} GB"
        )
    if scheduler is not None and epoch < 3 * CFG.epochs // 4:
        scheduler.step()

    t_end = time.time()
    total_time = t_end - t_start
    print("Epoch time: {:.1f}s".format(total_time))
    metrics = get_metrics(predictions, targets, pid_masks, ignore_index=CFG.ignore_index, task=task)
    return loss_meter.value()[0], metrics

def main():
    set_seed(CFG.seed)
    satellite_metadata = {
        "S1": {
            "bands": ['VV', 'VH'],
            "rgb_bands": [0, 1, 0],
            "mask_res": 10,
            "img_size": (32, 32),
        }
    }
    CFG.satellites = {"S1": satellite_metadata["S1"]}
    CFG.primary_sat = "S1"
    CFG.img_size = satellite_metadata["S1"]["img_size"]
    device = torch.device(CFG.device)
    data_dir = CFG.data_dir
    df = pd.read_csv(os.path.join(data_dir, "sickle_dataset_tabular.csv"))
    df = df[df.YIELD > 0].reset_index(drop=True)
    train_df = df[df.SPLIT == "train"].reset_index(drop=True)
    val_df = df[df.SPLIT == "val"].reset_index(drop=True)
    test_df = df[df.SPLIT == "test"].reset_index(drop=True)

    dt_args = dict(
        data_dir=data_dir,
        satellites=CFG.satellites,
        ignore_index=CFG.ignore_index,
        transform=CFG.use_augmentation,
        actual_season=CFG.actual_season
    )

    dt_train = SICKLE_Dataset(df=train_df, phase="train", **dt_args)
    dt_args["transform"] = False
    dt_val = SICKLE_Dataset(df=val_df, **dt_args)
    dt_test = SICKLE_Dataset(df=test_df, **dt_args)

    collate_fn = lambda x: utae_utils.pad_collate(x, pad_value=CFG.pad_value)

    train_loader = DataLoader(
        dt_train,
        batch_size=CFG.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=CFG.num_workers,
    )
    val_loader = DataLoader(
        dt_val,
        batch_size=CFG.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=CFG.num_workers,
    )
    test_loader = DataLoader(
        dt_test,
        batch_size=CFG.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=CFG.num_workers,
    )

    batch_data, masks = next(iter(train_loader))
    (samples, dates) = batch_data["S1"]
    print("Samples Shape", samples.shape, "Masks Shape", masks["crop_type"].shape)
    print("dates", dates[0])
    print("Train {}, Val {}, Test {}".format(len(dt_train), len(dt_val), len(dt_test)))
    model = model_utils.Fusion_model(CFG)
    model.apply(weight_init)
    model = model.to(device)
    CFG.N_params = utae_utils.get_ntrainparams(model)
    print("TOTAL TRAINABLE PARAMETERS :", CFG.N_params)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
    criterion = RMSELoss(ignore_index=CFG.ignore_index)
    scheduler = CosineAnnealingLR(optimizer, T_max=3 * CFG.epochs // 4, eta_min=1e-4)


    trainlog = {}
    best_metric = torch.inf
    for epoch in range(1, CFG.epochs + 1):
        print("EPOCH {}/{}".format(epoch, CFG.epochs))
        model.train()
        train_loss, train_metrics = iterate(
            model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            mode="train",
            device=device,
            epoch=epoch,
            task=CFG.task,
            CFG=CFG,
        )
        print("Validation . . .")
        model.eval()
        val_loss, val_metrics = iterate(
            model,
            data_loader=val_loader,
            criterion=criterion,
            mode="val",
            device=device,
            task=CFG.task,
            CFG=CFG,
        )
        lr = optimizer.param_groups[0]["lr"]
        train_rmse, train_mae, train_mape = train_metrics
        val_rmse, val_mae, val_mape = val_metrics
        deciding_metric = val_mae
        print(f"Val RMSE: {val_rmse:0.4f} | Val MAE: {val_mae:0.4f} | Val MAPE: {val_mape:0.4f}")
        trainlog[epoch] = {
            "train_loss": train_loss,
            "train_rmse": train_rmse.item(),
            "train_mae": train_mae.item(),
            "train_mape": train_mape.item(),
            "val_loss": val_loss,
            "val_rmse": val_rmse.item(),
            "val_mae": val_mae.item(),
            "val_mape": val_mape.item(),
            "lr": lr, 
        }
        checkpoint(trainlog, CFG)
        save_dict = {
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "model": model.state_dict()
        }
        if deciding_metric < best_metric:
            print(f"Valid Score Improved ({best_metric:0.4f} ---> {deciding_metric:0.4f})")
            best_metric = deciding_metric
            torch.save(save_dict, "checkpoint_best.pth")
        torch.save(save_dict, "checkpoint_last.pth")

main()