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

import os
import shlex
import tempfile

from transformers import BertTokenizer

from minilmv2.data_utils import get_tokenized_datasets
from minilmv2.parsers import get_data_parser


def test_data_utils():
    with tempfile.TemporaryDirectory() as dirname:
        os.makedirs(f"{dirname}/data/csv")
        os.makedirs(f"{dirname}/data/json")
        os.makedirs(f"{dirname}/tokenizer")
        with open(f"{dirname}/data/csv/file1.csv", "w") as f:
            f.write("text\nThis is an example sentence")
        with open(f"{dirname}/data/csv/file2.csv", "w") as f:
            f.write("text\nThis is another example sentence")
        with open(f"{dirname}/tokenizer/vocab.txt", "w") as f:
            f.write("[PAD]\ndummy1\ndummy2")
        config = {
            "urls": [f"{dirname}/data/csv/file1.csv", f"{dirname}/data/csv/file2.csv"],
            "format": "csv",
            "column_names": "text",
        }

        cl = "--train_config train_config.json"  # dummy config file
        data_args = get_data_parser().parse_args(shlex.split(cl))
        data_args.train_config = (
            config  # setting the config directly instead of writing & reading from file
        )

        tokenizer = BertTokenizer.from_pretrained(f"{dirname}/tokenizer/")
        train_set, val_set = get_tokenized_datasets(data_args, tokenizer)
        examples = [_ for _ in train_set]
        assert len(examples) == 2
        assert "input_ids" in examples[0]
        assert "token_type_ids" in examples[0]
        assert "attention_mask" in examples[0]


if __name__ == "__main__":
    test_data_utils()
    print("all passed")
