from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.dataset import ModalityConfig
from gr00t.experiment.data_config import DATA_CONFIG_MAP
import pathlib
from gr00t.data.dataset import (
    LE_ROBOT_MODALITY_FILENAME,
    LeRobotMixtureDataset,
    LeRobotSingleDataset,
    ModalityConfig,
)

import json


single_dataset_path = pathlib.Path("/mnt/data/franka_lerobot_dataset")

def get_modality_keys(dataset_path: pathlib.Path) -> dict[str, list[str]]:
    """
    Get the modality keys from the dataset path.
    Returns a dictionary with modality types as keys and their corresponding modality keys as values,
    maintaining the order: video, state, action, annotation
    """
    modality_path = dataset_path / LE_ROBOT_MODALITY_FILENAME
    with open(modality_path, "r") as f:
        modality_meta = json.load(f)

    # Initialize dictionary with ordered keys
    modality_dict = {}
    for key in modality_meta.keys():
        modality_dict[key] = []
        for modality in modality_meta[key]:
            modality_dict[key].append(f"{key}.{modality}")
    return modality_dict

modality_keys_dict = get_modality_keys(single_dataset_path)
video_modality_keys = modality_keys_dict["video"]
language_modality_keys = modality_keys_dict["annotation"]
state_modality_keys = modality_keys_dict["state"]
action_modality_keys = modality_keys_dict["action"]


# 2. construct modality configs from dataset
modality_configs = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=video_modality_keys,  # we will include all video modalities
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=state_modality_keys,
    ),
    "action": ModalityConfig(
        delta_indices=[0],
        modality_keys=action_modality_keys,
    ),
}

print(modality_keys_dict)
# get the modality configs and transforms
# modality_config = data_config.modality_config()
# transforms = data_config.transform()

# # This is a LeRobotSingleDataset object that loads the data from the given dataset path.
dataset = LeRobotSingleDataset(
    dataset_path=str(single_dataset_path),
    modality_configs=modality_configs,
    transforms=None,  # we can choose to not apply any transforms
    embodiment_tag=EmbodimentTag.GR1, # the embodiment to use
)

# # This is an example of how to access the data.
print(dataset[5])
