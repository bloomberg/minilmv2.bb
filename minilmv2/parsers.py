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

Argument parsers for distillation.
"""

import argparse


def get_data_parser():
    """Returns argument parser for data-related arguments.

    Args:

    Returns:
        Parser which can be used for data arguments
    """
    parser = argparse.ArgumentParser(
        epilog="Parameters for specifying data paths and configurations"
    )
    parser.add_argument(
        "--train_config",
        type=str,
        required=True,
        help="Configuration file for training dataset. Must be an array of {dir : directory, format: (txt|json|csv), columns : Comma separated list of column names (for csv) or field names (for json) containing the text. Content of the columns will be concatenated with a new line separator.} ",
    )
    parser.add_argument(
        "--val_config",
        type=str,
        required=False,
        help="Configuration file for validation dataset. Must be an array of {dir : directory, format: (txt|json|csv), columns : Comma separated list of column names (for csv) or field names (for json) containing the text. Content of the columns will be concatenated with a new line separator.} ",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="Max Sequence Length considered for an input",
    )
    return parser


def get_model_parser():
    """Returns argument parser for model-related arguments.

    Args:

    Returns:
        Parser which can be used for model arguments
    """
    parser = argparse.ArgumentParser(
        epilog="Parameters for specifying data paths and configurations"
    )
    parser.add_argument(
        "--input_model_dir",
        type=str,
        required=True,
        help="The directory from which the pre-trained model is loaded.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=False,
        help="Checkpoint directory to resume training from",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        required=False,
        help="The directory from which the pre-trained model's tokenizer is loaded. If not provided, 'input_model_dir' will be used.",
    )
    parser.add_argument(
        "--student_hidden_size",
        type=int,
        required=True,
        help="Student model hidden size.",
    )
    parser.add_argument(
        "--student_num_layers",
        type=int,
        required=True,
        help="Student model hidden layer number.",
    )
    parser.add_argument(
        "--student_attention_heads",
        type=int,
        required=True,
        help="Student model attention head number.",
    )
    parser.add_argument(
        "--L", type=int, required=True, help="Teacher's layer to distill from."
    )
    parser.add_argument(
        "--num_relation_heads",
        type=int,
        required=True,
        help="Number of relation heads in MiniLM.",
    )
    parser.add_argument(
        "--minilm_relations",
        type=str,
        required=False,
        default="{(1, 1): 1, (2, 2): 1, (3, 3): 1}",
        help="Relations to use and their weights. The format is {(relation id1, relation id2): weight}. Relation ids are denoted as 1: Query, 2: Key, 3: Value",
    )
    return parser


def split_args_by_parser(args, parsers):
    """Splits a list of argument strings into groups, and parses each group with the specified parser.

    Args:
        args: list of arguments strings (e.g., system.argv).
        parsers: dictionary mapping group names to parsers. The group names are expected to precede the arguments for that group.

    Returns:
        parsed_args: Dictionary mapping group name to ArgumentParser Namespace.
    """
    parser_names = parsers.keys()
    parser_name_locs = sorted(
        map(
            lambda parser_name: (
                parser_name,
                args.index(parser_name) if parser_name in args else -1,
            ),
            parser_names,
        ),
        key=lambda x: x[1],
    )
    assert all(
        filter(lambda x: x[1] != -1, parser_name_locs)
    ), f"Required keys {parser_names} must be present"
    parsed_args = {}
    for idx, parser_name_loc in enumerate(parser_name_locs):
        parser_name, loc = parser_name_loc
        end_loc = (
            len(args)
            if idx == len(parser_name_locs) - 1
            else parser_name_locs[idx + 1][1]
        )
        parsed_args[parser_name] = parsers[parser_name].parse_args(
            args[loc + 1 : end_loc]
        )
    return parsed_args
