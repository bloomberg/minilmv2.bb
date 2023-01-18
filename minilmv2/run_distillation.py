"""Copyright 2022 Bloomberg Finance L.P.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Contains code to perform distillation over a large corpus (Wikipedia + bookcorpus) using MiniLMv2.
"""

import json
import logging
import os
import sys
from ast import literal_eval

import pkg_resources
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .data_utils import get_tokenized_datasets
from .minilmv2 import MiniLM
from .parsers import get_data_parser, get_model_parser, split_args_by_parser

logger = logging.getLogger(__name__)


def _get_args():
    data_parser = get_data_parser()
    model_parser = get_model_parser()
    training_parser = HfArgumentParser((TrainingArguments))

    parsers = {
        "data_params": data_parser,
        "model_params": model_parser,
        "training_params": training_parser,
    }
    params = split_args_by_parser(sys.argv[1:], parsers)
    params["training_params"].label_names = ["start_positions", "end_positions"]
    params["training_params"].local_rank = _get_rank() if _is_distributed() else -1
    hf_training_args = TrainingArguments(**vars(params["training_params"]))
    data_args = params["data_params"]
    model_args = params["model_params"]
    return data_args, model_args, hf_training_args


def _is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _get_rank():
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def main():
    """Main entry point for running distillation."""
    data_args, model_args, hf_training_args = _get_args()

    data_args.train_config = json.loads(
        pkg_resources.resource_string(__name__, data_args.train_config)
    )
    data_args.val_config = (
        json.loads(pkg_resources.resource_string(__name__, data_args.val_config))
        if data_args.val_config
        else None
    )

    # Set seed before initializing model.
    set_seed(hf_training_args.seed)

    input_model_dir = model_args.input_model_dir

    checkpoint_dir = model_args.checkpoint_dir
    tokenizer_dir = (
        model_args.tokenizer_dir if model_args.tokenizer_dir else input_model_dir
    )

    # Teacher
    logger.info("Loading Teacher from pretrained model")
    teacher = AutoModel.from_pretrained(input_model_dir)
    logger.info("Loaded Teacher")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, use_fast=True, max_length=data_args.max_seq_len
    )

    # Student
    logger.info("Initializing student model")
    student_config = AutoConfig.from_pretrained(input_model_dir)
    student_config.hidden_size = model_args.student_hidden_size
    student_config.num_hidden_layers = model_args.student_num_layers
    student_config.num_attention_heads = model_args.student_attention_heads

    logger.info("Student Configuration")
    logger.info(student_config)
    student = AutoModel.from_config(student_config)

    logger.info("Initializing MiniLMv2")
    # Note: change the hyperparameter minilm_relations in line if you need
    # The format is {(relation id1, relation id2): weight}
    # Relation ids are denoted as 1: Query, 2: Key, 3: Value
    minilm_relations = literal_eval(model_args.minilm_relations)
    logger.info(f"MiniLM relations: {minilm_relations}")
    distiller = MiniLM(
        teacher=teacher,
        student=student,
        L=model_args.L,
        M=model_args.student_num_layers,
        relations=minilm_relations,
        A_r=model_args.num_relation_heads,
    )

    if checkpoint_dir is not None:
        logger.info("Loading model from checkpoint")
        distiller_state_dict = torch.load(checkpoint_dir + "/pytorch_model.bin")
        distiller.load_state_dict(distiller_state_dict)
        logger.info("Loaded checkpoint")

    logger.info("Initializing Train Dataset")
    train_dataset, val_dataset = get_tokenized_datasets(
        data_args, tokenizer, {"padding": "do_not_pad"}
    )

    if _is_distributed():
        print("Process is distributed")
        # To avoid deadlocks on the tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    is_global_primary = _get_rank() == 0
    if is_global_primary:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    # Set-up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_global_primary else logging.WARN,
    )

    logger.info(f"Data Arguments: {vars(data_args)}")
    logger.info(f"Model Arguments: {vars(model_args)}")
    logger.info(f"Training Arguments: {vars(hf_training_args)}")

    logger.info("Initializing Trainer")

    logger.info(
        f"Process rank: {hf_training_args.local_rank}, device: {hf_training_args.device}, n_gpu: {hf_training_args.n_gpu}"
        + f"distributed training: {bool(hf_training_args.local_rank != -1)}, 16-bits training: {hf_training_args.fp16}"
        + f"16-bits optimization level {hf_training_args.fp16_opt_level}"
    )

    print(f"HF Training Args : {hf_training_args}")
    print(f"os.getenv('RANK') = {os.getenv('RANK')}")
    print(f"os.getenv('LOCAL_RANK') = {os.getenv('LOCAL_RANK')}")
    print(f"os.getenv('WORLD_SIZE') = {os.getenv('WORLD_SIZE')}")
    trainer = Trainer(
        model=distiller,
        args=hf_training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=transformers.DataCollatorWithPadding(
            tokenizer, padding="longest"
        ),
    )

    logger.info("Training Model")
    trainer.train(resume_from_checkpoint=checkpoint_dir)

    print("---- DONE -----")


if __name__ == "__main__":
    main()
