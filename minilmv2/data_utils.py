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

Tools for streaming and working with large datasets.
"""

import logging
import os
from typing import List, Union

from datasets import DownloadConfig, disable_caching, load_dataset
from datasets.data_files import DataFilesDict, DataFilesList, Url

logger = logging.getLogger(__name__)


def get_data_files_dict(urls: str) -> Union[DataFilesDict, List[str]]:
    """Returns a data files dict object with the given URLs.

    Args:
        urls: List of files to download

    Returns:
        data_files_dict: Dict conforming to the processed format in HF datasets so that no other preprocessing / validation is performed by the library.
    """
    disable_caching()
    data_file_list = [Url(u) for u in urls]
    data_files_dict = DataFilesDict()
    data_files_dict["train"] = DataFilesList(data_file_list, origin_metadata=None)  # type: ignore
    return data_files_dict


def prepare_dataset(tokenizer, config, max_seq_len, tokenization_args):
    """Prepare dataset from list of files.

    Args:
        tokenizer: Tokenizer to apply on the files.
        config: Configuration json for the data files.
        max_seq_len: Maximum sequence length.
        tokenization_args: Addtional arguments to be passed to tokenizer's call function. Truncation and max_length are set by default.

    Returns:
        dataset: HuggingFace dataset object.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    download_config = DownloadConfig(max_retries=50)
    data_format = config.get("format", "text")
    data_files_dict = get_data_files_dict(config["urls"])
    dataset = load_dataset(
        data_format,
        data_files=data_files_dict,
        download_config=download_config,
        streaming=True,
    )["train"]
    columns = config.get("columns", "text").split(",")

    def tokenize_fn(examples):
        text = ["\n".join(e) for e in zip(*[examples[c] for c in columns])]
        return tokenizer(
            text, truncation=True, max_length=max_seq_len, **tokenization_args
        )

    return dataset.map(tokenize_fn, batched=True).with_format("torch")


def get_tokenized_datasets(data_args, tokenizer, tokenization_args=None):
    """Get the tokenized train and dev datasets.

    Args:
        data_args: Arguments from data parser.
        tokenizer: Tokenizer to apply on the datasets.
        tokenization_args: Addtional arguments to be passed to tokenizer's call function. Truncation and max_length are set by default.
    Returns:`
        Tuple of train and val tokenized datasets.
    """
    if not tokenization_args:
        tokenization_args = {}
    train_tokenized_dataset = prepare_dataset(
        tokenizer,
        data_args.train_config,
        data_args.max_seq_len,
        tokenization_args,
    )
    val_tokenized_dataset = (
        prepare_dataset(
            tokenizer,
            data_args.val_config,
            data_args.max_seq_len,
            tokenization_args,
        )
        if data_args.val_config
        else None
    )

    return train_tokenized_dataset, val_tokenized_dataset
