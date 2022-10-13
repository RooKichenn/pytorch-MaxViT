# --------------------------------------------------------
# MaxViT
# Written by ZeChen Wu
# --------------------------------------------------------
from .max_vit import MaxViT


def build_model(config):
    model_type = config.MODEL.TYPE
    print(f"Creating model: {model_type}")
    if model_type == "MaxViT":
        model = eval(model_type)(
            in_channels=config.MODEL.MAX_VIT.IN_CHANS,
            depths=config.MODEL.MAX_VIT.DEPTHS,
            channels=config.MODEL.MAX_VIT.CHANNELS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.MAX_VIT.EMBED_DIM,
            num_heads=config.MODEL.MAX_VIT.DIM_HEAD,
            grid_window_size=config.MODEL.MAX_VIT.WINDOW_SIZE,
            global_pool=config.MODEL.MAX_VIT.POOL,
            drop=config.MODEL.DROP_RATE,
            drop_path=config.MODEL.DROP_PATH_RATE,
        )
    return model
