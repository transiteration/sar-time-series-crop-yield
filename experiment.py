import os
import json
import torch
import random
import argparse
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

def checkpoint(log, config):
    with open(
            os.path.join(CFG.log_dir, CFG.model_name + "_trainlog.json"), "w"
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

def recursive_to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_to_device(v, device) for k, v in x.items()}
    else:
        return [recursive_to_device(c, device) for c in x]

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn,
               device: torch.device,
               task: str,
               CFG):
    model.train()
    loss_meter = tnt.meter.AverageValueMeter()
    predictions = None
    targets = None
    pid_masks = None
    
    for i, batch in enumerate(dataloader):
        batch = recursive_to_device(batch, device)
        data, masks = batch
        plot_mask = masks["plot_mask"]
        masks = masks[task].float()

        optimizer.zero_grad()
        y_pred = model(data)
        loss = loss_fn(y_pred, masks, plot_mask)
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

    metrics = get_metrics(predictions, targets, pid_masks, ignore_index=CFG.ignore_index, task=task)
    return loss_meter.value()[0], metrics

def eval_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn,
              device: torch.device,
              task: str,
              CFG):
    model.eval()
    loss_meter = tnt.meter.AverageValueMeter()
    predictions = None
    targets = None
    pid_masks = None

    with torch.inference_mode():
        for i, batch in enumerate(dataloader):
            batch = recursive_to_device(batch, device)
            data, masks = batch
            plot_mask = masks["plot_mask"]
            masks = masks[task].float()

            y_pred = model(data)
            loss = loss_fn(y_pred, masks, plot_mask)

            if predictions is None:
                predictions = y_pred
                targets = masks
                pid_masks = plot_mask
            else:
                predictions = torch.cat([predictions, y_pred], dim=0)
                targets = torch.cat([targets, masks], dim=0)
                pid_masks = torch.cat([pid_masks, plot_mask], dim=0)

            loss_meter.add(loss.item())

    metrics = get_metrics(predictions, targets, pid_masks, ignore_index=CFG.ignore_index, task=task)
    return loss_meter.value()[0], metrics

def train_loop(CFG) -> None:
    set_seed(CFG.seed)
    os.makedirs(CFG.save_dir, exist_ok=True)
    os.makedirs(CFG.log_dir, exist_ok=True)

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

    train_dataset = SICKLE_Dataset(df=train_df, phase="train", **dt_args)
    dt_args["transform"] = False
    val_dataset = SICKLE_Dataset(df=val_df, **dt_args)

    collate_fn = lambda x: utae_utils.pad_collate(x, pad_value=CFG.pad_value)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=CFG.num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=CFG.num_workers,
    )

    model = model_utils.Fusion_model(CFG)
    model.apply(weight_init)
    model.to(device)

    CFG.N_params = utae_utils.get_ntrainparams(model)
    print("TOTAL TRAINABLE PARAMETERS :", CFG.N_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
    criterion = RMSELoss(ignore_index=CFG.ignore_index)
    scheduler = CosineAnnealingLR(optimizer, T_max=3 * CFG.epochs // 4, eta_min=1e-4)

    trainlog = {}
    best_metric = torch.inf

    pbar = tqdm(range(1, CFG.epochs + 1), total=CFG.epochs, desc="Training Loop")

    for epoch in pbar:
        print(f"EPOCH {epoch}/{CFG.epochs}")

        train_loss, train_metrics = train_step(model=model,
                                               dataloader=train_dataloader,
                                               optimizer=optimizer,
                                               loss_fn=criterion,
                                               device=device,
                                               task=CFG.task,
                                               CFG=CFG)

        val_loss, val_metrics = eval_step(model=model,
                                          dataloader=val_dataloader,
                                          loss_fn=criterion,
                                          device=device,
                                          task=CFG.task,
                                          CFG=CFG)

        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        train_rmse, train_mae, train_mape = train_metrics
        val_rmse, val_mae, val_mape = val_metrics
        deciding_metric = val_mae

        gpu_mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        pbar.set_postfix(Val_RMSE=f"{val_rmse:0.4f}", Val_MAE=f"{val_mae:0.4f}", 
                         LR=f"{lr:0.6f}", GPU_Mem=f"{gpu_mem:0.2f} GB")

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
            torch.save(save_dict, os.path.join(CFG.save_dir, CFG.model_name) + "_best.pth")
        torch.save(save_dict, os.path.join(CFG.save_dir, CFG.model_name) + "_last.pth")

    print("Training complete.")

if __name__ == "__main__":        
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="utae",
        type=str,
        help="Type of architecture to use. Can be one of: (utae/unet3d/fpn/convlstm/convgru/uconvlstm/buconvlstm)",
    )
    parser.add_argument("--encoder_widths", default="[64,128]", type=str)
    parser.add_argument("--decoder_widths", default="[32,128]", type=str)
    parser.add_argument("--out_conv", default="[32, 16]")
    parser.add_argument("--str_conv_k", default=4, type=int)
    parser.add_argument("--str_conv_s", default=2, type=int)
    parser.add_argument("--str_conv_p", default=1, type=int)
    parser.add_argument("--agg_mode", default="att_group", type=str)
    parser.add_argument("--encoder_norm", default="group", type=str)
    parser.add_argument("--n_head", default=16, type=int)
    parser.add_argument("--d_model", default=256, type=int)
    parser.add_argument("--d_k", default=4, type=int)
    parser.add_argument(
        "--device",
        default= "cuda" if torch.cuda.is_available() else "cpu",
        type=str,
        help="Name of device to use for tensor computations (cuda/cpu)",
    )
    parser.add_argument(
        "--num_workers", default=10, type=int, help="Number of data loading workers"
    )
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs per fold")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--lr", default=1e-1, type=float, help="Learning rate")
    parser.add_argument("--num_classes", default=1, type=int)
    parser.add_argument("--ignore_index", default=-999, type=int)
    parser.add_argument("--pad_value", default=0, type=float)
    parser.add_argument("--padding_mode", default="reflect", type=str)
    parser.add_argument('--task', type=str, default="crop_yield", help="Available Tasks are crop_yield")
    parser.add_argument('--actual_season', default=True, help="whether to consider actual season or not.")
    parser.add_argument('--data_dir', type=str, default="./sickle_toy_dataset")
    parser.add_argument('--use_augmentation', type=bool, default=True)
    parser.add_argument('--log_dir', type=str, default="./logs")
    parser.add_argument('--save_dir', type=str, default="./models")
    parser.add_argument('--model_name', type=str, default="sample")
    parser.set_defaults(cache=False)
    CFG = parser.parse_args() 
    list_args = ["encoder_widths", "decoder_widths", "out_conv"]
    for k, v in vars(CFG).items():
        if k in list_args and v is not None:
            v = v.replace("[", "")
            v = v.replace("]", "")
            try:
                CFG.__setattr__(k, list(map(int, v.split(","))))
            except:
                CFG.__setattr__(k, list(map(str, v.split(","))))

    train_loop(CFG)
