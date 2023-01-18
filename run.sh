# Copyright 2022 Bloomberg Finance L.P.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

CMD="python -m minilmv2.run_distillation -- "
# Set seed to ensure reproducibility. Set to -1 for no seed
SEED=42

ARGS="data_params \
    --train_config ./train_config.json \
    training_params \
      --per_device_train_batch_size 256 \
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
    --seed=${SEED}\
    model_params \
     --input_model_dir ./model/bert-base-uncased/ \
      --student_hidden_size 384 \
      --student_num_layers 6 \
      --student_attention_heads 12 \
      --L 12 \
      --num_relation_heads 48\
"

$CMD $ARGS
