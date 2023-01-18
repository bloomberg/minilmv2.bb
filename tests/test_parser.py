"""
Copyright 2022 Bloomberg Finance L.P.
 
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import shlex
from argparse import Namespace

from transformers import HfArgumentParser, TrainingArguments

from minilmv2.parsers import *


def test_parsers():
    data_parser = get_data_parser()
    model_parser = get_model_parser()
    training_parser = HfArgumentParser((TrainingArguments))
    group_parsers = {
        "data_params": data_parser,
        "model_params": model_parser,
        "training_params": training_parser,
    }

    cl = "data_params \
    --train_config ./train_config.json \
    training_params \
      --learning_rate 6e-4 \
      --adam_epsilon 1e-6 \
      --adam_beta1 0.9 \
      --adam_beta2 0.999 \
      --weight_decay 0.01 \
      --max_steps 400000 \
      --save_steps 50000 \
      --logging_steps 1000 \
      --warmup_steps 4000 \
    --ddp_find_unused_parameters  true\
    --output_dir ./out\
      --warmup_steps 25 \
    model_params \
     --input_model_dir ./model/bert-base-uncased/ \
      --student_hidden_size 384 \
      --student_num_layers 6 \
      --student_attention_heads 12 \
      --L 12 \
      --num_relation_heads 48"
    args = shlex.split(cl)
    grouped_params = split_args_by_parser(args, group_parsers)
    assert grouped_params["data_params"] == Namespace(
        max_seq_len=512,
        train_config="./train_config.json",
        val_config=None,
    )
    assert grouped_params["model_params"] == Namespace(
        L=12,
        checkpoint_dir=None,
        input_model_dir="./model/bert-base-uncased/",
        num_relation_heads=48,
        student_attention_heads=12,
        student_hidden_size=384,
        student_num_layers=6,
        tokenizer_dir=None,
        minilm_relations="{(1, 1): 1, (2, 2): 1, (3, 3): 1}",
    )
    assert "training_params" in grouped_params


if __name__ == "__main__":
    test_parsers()
    print("all passed")
