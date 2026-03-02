from yaml import CLoader as Loader
from yaml import dump, load
import os
MODEL_KEYS = ["seq_len_channel", "patch_len", "patch_stride_len", "revin_affine", "d_model", "dropout", "n_heads", "e_layers", "attention_dropout", "torch_dtype", "value_embedding_bias", "orth_gain", "pos_embed_type", "set_input_mask", "output_attention", "activation"]

class Config:
    def __init__(
        self,
        config_file_path="configs/config.yaml",
        default_config_file_path="configs/default.yaml",
        verbose: bool = False,
    ):
        """
        Class to read and parse the config.yml file
        """
        self.config_file_path = config_file_path
        self.default_config_file_path = default_config_file_path
        self.verbose = verbose

    def parse(self, run_name: str = None, if_override: bool = False):
        with open(self.config_file_path, "rb") as f:
            self.config = load(f, Loader=Loader)

        with open(self.default_config_file_path, "rb") as f:
            default_config = load(f, Loader=Loader)

        for key in default_config.keys():
            if self.config.get(key) is None:
                self.config[key] = default_config[key]
                if self.verbose:
                    print(f"Using default config for {key} : {default_config[key]}")

        if if_override and run_name is not None:
            config_path = os.path.join("results/wandb_configs", f"{run_name}.yaml")
            with open(config_path, "rb") as f:
                override_config = load(f, Loader=Loader)
            for key in MODEL_KEYS:
                if override_config.get(key) is not None:
                    self.config[key] = override_config[key]["value"]
        
        return self.config

    def save_config(self):
        with open(self.config_file_path, "w") as f:
            dump(self.config, f)
            