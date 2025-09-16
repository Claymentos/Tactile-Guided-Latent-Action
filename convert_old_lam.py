import torch
from omegaconf import OmegaConf
import argparse


def update_config(args):
    model_config_path = "/".join(args.path.split("/")[0:-1]) + "/model_config.yaml"
    model_config = OmegaConf.load(model_config_path)

    model_config.algo.model.feature_extractor = {
        "_target_": "lam.feature_extractor.base.Identity",
        "patch_size": model_config.algo.model.patch_size,
        "model_dim": model_config.algo.model.model_dim,
        "image_size": model_config.algo.model.image_size,
        "channels": model_config.algo.model.decoder.channels
    }

    del model_config.algo.model.encoder.channels
    del model_config.algo.model.decoder.channels
    del model_config.algo.model.image_size
    del model_config.algo.model.patch_size
    del model_config.algo.model.model_dim
    del model_config.algo.model.channels

    OmegaConf.save(model_config, model_config_path + f".{args.keep_embedding}_only")


def main(args):
    update_config(args)

    checkpoint = torch.load(args.path)
    new_state_dict = {}

    for k, v in checkpoint['state_dict'].items():
        new_k = k

        if "model.to_image" in k:
            new_k = k.replace("model.to_image", "model.feature_extractor.to_image")

        if "model.encoder.to_patch_embedding" in k:
            if args.keep_embedding == "encoder":
                new_k = k.replace("model.encoder.to_patch_embedding", "model.feature_extractor.to_patch_embedding")
            else:   
                continue

        if "model.decoder.to_patch_embedding" in k:
            if args.keep_embedding == "decoder":
                new_k = k.replace("model.decoder.to_patch_embedding", "model.feature_extractor.to_patch_embedding")
            else:
                continue
        
        new_state_dict[new_k] = v

    checkpoint['state_dict'] = new_state_dict
    torch.save(checkpoint, args.path + f".{args.keep_embedding}_only")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--keep_embedding", type=str, choices=["encoder", "decoder"], default="decoder")
    args = parser.parse_args()
    main(args)