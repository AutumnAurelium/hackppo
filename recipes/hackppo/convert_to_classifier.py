import sys
from omegaconf import DictConfig

from torchtune import config, training

import torch.nn as nn

from torchtune.training.checkpointing import Checkpointer

from torchtune.training.checkpointing._utils import update_state_dict_for_classifier
from torchtune.rlhf.utils._convert_weights import _REWARD

_REWARD["lm_head.weight"] = "output.weight"

class ClassifierConverter:
    def __init__(self, cfg: DictConfig):
        self.checkpointer = config.instantiate(cfg.checkpointer)
        self.model = config.instantiate(cfg.model)  # This should be your classifier model

        # Load the base model state dict
        state_dict = self.checkpointer.load_checkpoint()
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]

        # Use the built-in classifier conversion utility
        update_state_dict_for_classifier(
            state_dict=state_dict,
            model_named_parameters=self.model.named_parameters(),
            force_override=True  # Force it to use the classifier's output layer
        )

        # Load the converted state dict
        self.model.load_state_dict(state_dict, strict=False)

        # Save the converted checkpoint
        self.checkpointer.save_checkpoint({"model": state_dict}, epoch=0)


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    config.log_config(recipe_name="PPOFullFinetuneRecipeSingleDevice", cfg=cfg)
    ClassifierConverter(cfg=cfg)

if __name__ == "__main__":
    sys.exit(recipe_main())