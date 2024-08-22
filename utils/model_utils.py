from models_builder import utae, pastis_unet3d, convlstm, convgru, fpn
import torch.nn as nn
import torch


class Build_model(nn.Module):
    def __init__(self, CFG):
        super(Build_model, self).__init__()
        self.CFG = CFG
        self.sat = list(CFG.satellites.keys())[0]
        self.model = self.get_model(self.sat)

    def forward(self, data):
        (images, dates) = data[self.sat]
        y_pred = self.model(images, batch_positions=dates)
        return y_pred
    
    def get_model(self,sat):
        config = self.CFG
        input_dim = len(config.satellites[sat]["bands"])
        if config.model == "utae":
            model = utae.UTAE(
                input_dim=input_dim,
                encoder_widths=[64, 128],
                decoder_widths=[32, 128],
                out_conv=[32, 16],
                str_conv_k=4,
                str_conv_s=2,
                str_conv_p=1,
                agg_mode="att_group",
                encoder_norm="group",
                n_head=16,
                d_model=256,
                d_k=4,
                encoder=False,
                return_maps=False,
                pad_value=config.pad_value,
                padding_mode="reflect",
            )
        elif config.model == "unet3d":
            model = pastis_unet3d.UNet3D(
                in_channel=input_dim, n_classes=config.out_conv[-1], pad_value=config.pad_value
            )
        elif config.model == "fpn":
            model = fpn.FPNConvLSTM(
                input_dim=input_dim,
                num_classes=config.out_conv[-1],
                inconv=[32, 64],
                n_levels=4,
                n_channels=64,
                hidden_size=88,
                input_shape=config.img_size,
                mid_conv=True,
                pad_value=config.pad_value,
            )
        elif config.model == "convlstm":
            model = convlstm.ConvLSTM_Seg(
                num_classes=config.out_conv[-1],
                input_size=config.img_size,
                input_dim=input_dim,
                kernel_size=(3, 3),
                hidden_dim=160,
            )
        elif config.model == "convgru":
            model = convgru.ConvGRU_Seg(
                num_classes=config.out_conv[-1],
                input_size=config.img_size,
                input_dim=input_dim,
                kernel_size=(3, 3),
                hidden_dim=180,
            )
        else:
            raise NotImplementedError
        return model


class Fusion_model(Build_model):
    def __init__(self, CFG):
        super(Fusion_model, self).__init__(CFG)
        self.CFG = CFG
        self.models = nn.ModuleDict()
        for satellite in CFG.satellites.keys():
            model = self.get_model(sat=satellite)
            self.models[satellite] = model
        self.conv_final = nn.Conv2d(len(self.CFG.satellites.keys()) * self.CFG.out_conv[-1], CFG.num_classes,kernel_size=3, stride=1, padding=1)

    def forward(self, data):
        y_preds = {}
        for satellite in self.CFG.satellites.keys():
            (images, dates) = data[satellite]
            model = self.models[satellite]
            y_preds[satellite] = model(images, batch_positions=dates)
        y_pred = torch.cat(list(y_preds.values()), dim=1)
        y_pred = self.conv_final(y_pred)
        return y_pred