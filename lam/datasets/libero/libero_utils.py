import os
import numpy as np

from transformers import AutoModel, AutoTokenizer, logging
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
from libero.libero.benchmark import get_benchmark
from hydra.utils import to_absolute_path
from tqdm import trange

import lam.datasets.libero.file_utils as FileUtils
import lam.datasets.libero.obs_utils as ObsUtils
from lam.datasets.libero_datasets import SequenceDataset

np.set_printoptions(suppress=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def build_dataset(data_prefix,
                  suite_name,
                  benchmark_name, 
                  mode, 
                  seq_len, 
                  frame_stack,
                  shape_meta,
                  n_demos,
                  demo_ids=None,
                  extra_obs_modality=None,
                  obs_seq_len=1, 
                  hdf5_normalize_obs=False,
                  load_obs=True,
                  load_proprio=False,
                  task_embedding_format="clip",
                  episodic=False,
                  pre_compute_task_embeddings=False,
                  dataset_keys=["actions"],
                  ):
    benchmark = get_benchmark(benchmark_name)()
    n_tasks = benchmark.n_tasks

    if demo_ids is not None:
        few_shot_demos = demo_ids
    elif mode == 'fewshot':            
        few_shot_demos = [1, 5, 10, 20, 45]
    else:
        few_shot_demos = None

    few_shot_demos_list = [f"demo_{i}" for i in few_shot_demos] if few_shot_demos is not None else None
    
    manip_datasets = []
    descriptions = []
    obs_modality = {}

    if load_obs:
        obs_modality['rgb'] = list(shape_meta['observation']['rgb'].keys())
        if load_proprio:
            obs_modality['low_dim'] = list(shape_meta['observation']['lowdim'].keys())
        if extra_obs_modality is not None:
            for key in extra_obs_modality:
                obs_modality[key] = obs_modality[key] + extra_obs_modality[key]

        ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_modality})

    for i in trange(n_tasks):
        task_i_dataset = get_dataset(
            dataset_path=os.path.join(
                data_prefix, suite_name, benchmark.get_task_demonstration(i)
            ),
            obs_modality=obs_modality,
            seq_len=seq_len,
            obs_seq_len=obs_seq_len,
            frame_stack=frame_stack,
            hdf5_normalize_obs=hdf5_normalize_obs,
            load_obs=load_obs,
            few_demos = few_shot_demos_list,
            n_demos=n_demos,
            episodic=episodic,
            dataset_keys=dataset_keys,
        )
        task_description = benchmark.get_task(i).language
        descriptions.append(task_description)
        manip_datasets.append(task_i_dataset)

    if pre_compute_task_embeddings:
        task_embs = get_task_embs(task_embedding_format, descriptions)
        benchmark.set_task_embs(task_embs)
        datasets = [
            SequenceVLDataset(ds, emb, i, td) for i,(ds, emb, td) in enumerate(zip(manip_datasets, task_embs, descriptions))
        ]
    else:
        datasets = [
            SequenceVLDataset(ds, None, i, td) for i, (ds, td) in enumerate(zip(manip_datasets, descriptions))
        ]
    n_demos = [data.n_demos for data in datasets]
    n_sequences = [data.total_num_sequences for data in datasets]
    concat_dataset = ConcatDataset(datasets)
    print("\n===================  Benchmark Information  ===================")
    print(f" Name: {benchmark.name}")
    print(f" # Tasks: {n_tasks}")
    print(" # demonstrations: " + " ".join(f"({x})" for x in n_demos))
    print(" # sequences: " + " ".join(f"({x})" for x in n_sequences))
    print("=======================================================================\n")
    return concat_dataset

def get_dataset(
    dataset_path,
    obs_modality,
    seq_len=1,
    obs_seq_len=1,
    frame_stack=1,
    filter_key=None,
    hdf5_cache_mode=None,
    hdf5_normalize_obs=False,
    load_obs=True,
    few_demos=None,
    n_demos=None,
    episodic=False,
    dataset_keys=["actions"],
    ):
    all_obs_keys = []
    for modality_name, modality_list in obs_modality.items():
        all_obs_keys += modality_list
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path, all_obs_keys=all_obs_keys, verbose=False
    )
    seq_len = seq_len
    filter_key = filter_key
    if load_obs:
        obs_keys = shape_meta["all_obs_keys"]
    else:
        obs_keys = []
    dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=obs_keys,
        dataset_keys=dataset_keys,
        load_next_obs=False,
        frame_stack=frame_stack,
        seq_length=seq_len,  # length-10 temporal sequences
        obs_seq_length=obs_seq_len,
        pad_frame_stack=True,
        pad_seq_length=True,  # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=hdf5_cache_mode,  # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr=False,
        hdf5_normalize_obs=hdf5_normalize_obs,
        filter_by_attribute=filter_key,  # can optionally provide a filter key here
        few_demos=few_demos,
        n_demos=n_demos,
        episodic=episodic
    )
    return dataset

class SequenceVLDataset(Dataset):
    def __init__(self, sequence_dataset, task_emb, task_id, task_description):
        self.sequence_dataset = sequence_dataset
        self.task_description = task_description
        self.task_emb = task_emb
        self.task_id = task_id
        self.n_demos = self.sequence_dataset.n_demos
        self.total_num_sequences = self.sequence_dataset.total_num_sequences

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx):
        return_dict = self.sequence_dataset.__getitem__(idx)
        return_dict["task_description"] = self.task_description
        if self.task_emb is not None:
            return_dict["task_emb"] = self.task_emb
            return_dict["task_id"] = self.task_id        

        return return_dict


def get_task_embs(task_embedding_format, descriptions):
    logging.set_verbosity_error()
    if task_embedding_format == "bert":
        tz = AutoTokenizer.from_pretrained(
            "bert-base-cased", cache_dir=to_absolute_path("./bert")
        )
        model = AutoModel.from_pretrained(
            "bert-base-cased", cache_dir=to_absolute_path("./bert")
        )
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=25,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        masks = tokens["attention_mask"]
        input_ids = tokens["input_ids"]
        task_embs = model(tokens["input_ids"], tokens["attention_mask"])[
            "pooler_output"
        ].detach()
    elif task_embedding_format == "gpt2":
        tz = AutoTokenizer.from_pretrained("gpt2")
        tz.pad_token = tz.eos_token
        model = AutoModel.from_pretrained("gpt2")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=25,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model(**tokens)["last_hidden_state"].detach()[:, -1]
    elif task_embedding_format == "clip":
        tz = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32", clean_up_tokenization_spaces=True)
        model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=25,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model.get_text_features(**tokens).detach()
    elif task_embedding_format == "roberta":
        tz = AutoTokenizer.from_pretrained("roberta-base")
        tz.pad_token = tz.eos_token
        model = AutoModel.from_pretrained("roberta-base")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=25,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model(**tokens)["pooler_output"].detach()
    
    return task_embs

