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

CMD="python -m minilmv2.hf_run_evaluation "
# Set seed to ensure reproducibility. Set to -1 for no seed
SEED=42
# Change the task_name to train on other datasets
TASK_NAME="sst2"
for num_epochs in 3 5 10
do
    for lr in 0.00002 0.00003 0.00004 0.00001 0.000015 
    do
        ARGS="--model_name_or_path <path_to_student_model> \
  --task_name ${TASK_NAME} \
  --do_train \
  --do_eval \
  --eval_steps 1000000000 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --seed=${SEED} \
  --evaluation_strategy epoch \
  --max_seq_length 128 \
  --save_steps 1000000000 \
  --per_device_train_batch_size 32 \
  --learning_rate $lr \
  --num_train_epochs $num_epochs \
  --output_dir ./out/${TASK_NAME}/$num_epochs/${lr/./_}/ \
  --overwrite_output_dir
"
        echo "$CMD $ARGS"
        $CMD $ARGS
    done
done
