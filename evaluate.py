import os
import torch
import argparse
import warnings
import numpy as np
import setproctitle
import pandas as pd
from torch.utils.data import DataLoader
from utils.dataset import SICKLE_Dataset
from utils import utae_utils, model_utils
from utils.metric import RMSELoss
import matplotlib.pyplot as plt
from experiment import set_seed, eval_step, recursive_to_device

warnings.filterwarnings("ignore")
setproctitle.setproctitle("miras_crop_yield_evaluation")

# Evaluation Function
def eval_process(CFG) -> None:
    set_seed(CFG.seed)
    device = torch.device(CFG.device)
    os.makedirs(CFG.eval_dir, exist_ok=True)

    # Satellite Metadata Setup
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
    
    # Data Loading
    df = pd.read_csv(os.path.join(CFG.data_dir, "sickle_dataset_tabular.csv"))
    df = df[df.YIELD > 0].reset_index(drop=True)
    test_df = df[df.SPLIT == "val"].reset_index(drop=True)

    dt_args = dict(
        data_dir=CFG.data_dir,
        satellites=CFG.satellites,
        ignore_index=CFG.ignore_index,
        transform=CFG.use_augmentation,
        actual_season=CFG.actual_season
    )
    test_dataset = SICKLE_Dataset(df=test_df, **dt_args)
    collate_fn = lambda x: utae_utils.pad_collate(x, pad_value=CFG.pad_value)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=CFG.num_workers,
    )
    model = model_utils.Fusion_model(CFG)
    model.to(device)
    checkpoint = torch.load(os.path.join(CFG.models_dir, CFG.model_name + ".pth"))
    model.load_state_dict(checkpoint["model"])
    loss_fn = RMSELoss(ignore_index=CFG.ignore_index)

    # _, val_metrics = eval_step(model=model,
    #                                 dataloader=val_dataloader,
    #                                 loss_fn=loss_fn,
    #                                 device=device,
    #                                 CFG=CFG)
        
    # val_rmse, val_mae, val_mape = val_metrics
    # print(f"Val RMSE: {val_rmse:0.4f} | Val MAE: {val_mae:0.4f} | Val MAPE: {val_mape:0.4f}")
    model.eval()
    log = True
    with torch.inference_mode():
        batch = next(iter(test_dataloader))
        batch = recursive_to_device(batch, device)
        data, masks = batch
        plot_mask = masks["plot_mask"]
        masks = masks[CFG.task].float()
        y_pred = model(data)
        if log:
            y_pred = y_pred.squeeze(dim=1)
            images, dates = data["S1"]
            images, dates, gt_masks, pred_masks = \
                images.cpu().numpy(), dates.cpu().numpy(), \
                masks.cpu().numpy(), y_pred.cpu().numpy()        
            gt_masks[gt_masks == -999] = -1
            gt_masks[gt_masks < -1] = 0
            i = 0
            for image, date, gt_mask, pred_mask in \
                zip(images, dates, gt_masks, pred_masks):
                image = image[len(date[date != 0]) - 1]
                image = image[CFG.satellites["S1"]["rgb_bands"]].transpose(1, 2, 0)
                image = ((image - np.min(image)) / (np.max(image) - np.min(image)))
                # pred_mask_whole = generate_heatmap(copy.deepcopy(pred_mask))
                # pred_mask[gt_mask == -1] = -1
                plt.imsave(f"eval_results/{i}_S1.png", image)
                plt.imsave(f"eval_results/{i}_gt.png", gt_mask)
                plt.imsave(f"eval_results/{i}_pred_mask_whole.png", pred_mask)
                i += 1

if __name__ == "__main__":        
    parser = argparse.ArgumentParser(description="Training configuration")
    parser.add_argument(
        "--model",
        default="utae",
        type=str,
        help="Type of architecture to use. Can be one of: (utae/unet3d/fpn/convlstm/convgru/uconvlstm/buconvlstm)",
    )
    parser.add_argument('--model_name', type=str, required=True, help="Name of the model")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        type=str,
        help="Name of device to use for tensor computations (cuda/cpu)",
    )
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("--lr", default=1e-1, type=float, help="Learning rate")
    parser.add_argument("--num_classes", default=1, type=int, help="Number of output classes")
    parser.add_argument("--ignore_index", default=-999, type=int, help="Index to ignore during training")
    parser.add_argument("--out_conv", default=[32, 16], nargs='+', type=int, help="Output convolutional layers")
    parser.add_argument("--pad_value", default=0, type=float, help="Padding value for the input data")
    parser.add_argument("--num_workers", default=10, type=int, help="Number of data loading workers")
    parser.add_argument('--data_dir', type=str, default="/home/thankscarbon/Documents/miras/sickle_dataset", help="Directory for the dataset")
    parser.add_argument('--use_augmentation', type=bool, default=False, help="Whether to use data augmentation or not")
    parser.add_argument('--task', type=str, default="crop_yield", 
                        help="Available Tasks: sowing_date, transplanting_date, harvesting_date, crop_yield")
    parser.add_argument('--actual_season', default=False, action='store_true', help="Whether to consider actual season or not")
    parser.add_argument('--models_dir', type=str, default="./models", help="Directory to save models")
    parser.add_argument('--eval_dir', type=str, default="./eval_results", help="Directory to evaluate model")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")
    parser.set_defaults(cache=False)
    CFG = parser.parse_args()     
    eval_process(CFG)
