# MiniLMv2 implementation

This repository contains an implementation of the [MiniLMv2](https://aclanthology.org/2021.findings-acl.188.pdf) distillation technique. Please refer to the original paper cited below for more details about the method.

This code builds on the [Hugging Face Transformers](https://github.com/huggingface/transformers) library.

## Menu

- [Quick Start](#quick-start)
- [Hyperparameters](#hyperparameters)
- [Citation](#citation)
- [Results](#results)
- [License](#license)
- [Code of Conduct](#code-of-conduct)
- [Security Vulnerability Reporting](#security-vulnerability-reporting)

## Quick Start

1. Collect the data which the teacher was trained on (e.g., for bert_base: [Wikipedia](https://huggingface.co/datasets/wikipedia) and [Bookcorpus](https://huggingface.co/datasets/bookcorpus)).
2. Update `minilmv2/train_config.json` to point to your data. The path can be a local path or a publicly accessible HTTP URL.
3. In `run.sh`, update the input_model_dir to point to your teacher model (loadable with [AutoModel.from_pretrained](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html#transformers.AutoConfig.from_pretrained)) and make any necessary changes to launch a distributed job. It is recommended to run this on multiple GPUs.
4. Make any necessary changes to save the student model after training (e.g., saving student state dict).
3. Start the distillation by running `run.sh`.

## Hyperparameters

For distillation, the `run.sh` script contains the hyperparameters used to distill our models and can be modified for other settings as required. The `run_eval.sh` script contains the evaluation hyperparameters and we hypertune our models over different learning rates (`1e-5` - `4e-5`) and num epochs (`3`,`5`,`10`).

## Citation
The MiniLMV2 technique was originally proposed by:
```
  @inproceedings{wang-etal-2021-minilmv2,
    title = "{M}ini{LM}v2: Multi-Head Self-Attention Relation Distillation for Compressing Pretrained Transformers",
    author = "Wang, Wenhui  and
      Bao, Hangbo  and
      Huang, Shaohan  and
      Dong, Li  and
      Wei, Furu",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.188",
    doi = "10.18653/v1/2021.findings-acl.188",
    pages = "2140--2151",
}
```

## Results

The table below summarizes our results for two models (6x384 and 6x768) distilled from bert-base. GLUE results were obtained by running the `hf_run_evaluation.py`, which is a near replica of the `run_glue` [script](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py) from Hugging Face.

|                      | qnli  | qqp   | rte   | sst2  | mnli  | Avg    |
|----------------------|-------|-------|-------|-------|-------|--------|
| MiniLM 6x768 (ours)  | 89.05 | 90.47 | 60.65 | 91.63 | 82.92 |  82.94 |
| MiniLM 6x384 (ours)  | 89.44 | 90.47 | 63.18 | 91.28 | 82.59 | 83.392 |
| MiniLM 6x768 (paper) |  90.8 |  91.1 |  72.1 |  92.4 |  84.2 |  86.12 |
| MiniLM 6x384 (paper) | 90.24 | 90.51 | 66.43 | 91.17 | 82.91 |  84.25 |

## License
Copyright 2022 Bloomberg Finance L.P.  Licensed under the Apache License, Version 2.0. 

## Code of Conduct

This project has adopted a [Code of Conduct](https://github.com/bloomberg/.github/blob/master/CODE_OF_CONDUCT.md).
If you have any concerns about the Code, or behavior which you have experienced in the project, please
contact us at opensource@bloomberg.net.

## Security Vulnerability Reporting

If you believe you have identified a security vulnerability in this project, please send an email to the project
team at opensource@bloomberg.net that details the suspected issue and any methods you've found to reproduce it.

Please do NOT open an issue in the GitHub repository, as we'd prefer to keep vulnerability reports private until
we've had an opportunity to review and address them.
