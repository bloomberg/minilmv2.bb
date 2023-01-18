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

import copy
from ast import literal_eval

import pytest
import torch
from transformers import BertConfig, BertModel

from minilmv2.minilmv2 import MiniLM


@pytest.fixture
def distiller():
    return create_minilm_distiller()


def create_minilm_distiller():
    teacher = BertModel(BertConfig())
    student = BertModel(BertConfig(num_hidden_layers=4))
    minilm_distiller = MiniLM(
        teacher=teacher,
        student=student,
        L=12,
        M=4,
        relations={(1, 1): 1, (2, 2): 1, (3, 3): 1},
        A_r=48,
    )
    return minilm_distiller


def have_same_weights(model1, model2):
    """Return True if two models have the same weights and False if not."""
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def test_relation_tensors(distiller):
    """Test the function that reshape query/key/value tensors for multiple relation heads."""
    batch_size = 8
    seq_length = 128
    hidden_size = 768
    num_relation_heads = 48
    relation_head_size = hidden_size // num_relation_heads  # 16

    query = torch.rand(batch_size, seq_length, hidden_size)  # (8, 128, 768)
    assert list(query.shape) == [8, 128, 768]

    relation_query = distiller._transpose_for_scores_relation(query, relation_head_size)
    assert list(relation_query.shape) == [
        8,
        48,
        128,
        16,
    ]  # (batch_size, relation_head_number, seq_length, relation_head_size)


def test_weights(distiller):
    """Verify that the student model weights get updated while the teacher model stay unchanged."""
    teacher_copy = copy.deepcopy(distiller.teacher)
    student_copy = copy.deepcopy(distiller.student)

    # The copies should have same weights with teacher and student initially
    assert have_same_weights(teacher_copy, distiller.teacher)
    assert have_same_weights(student_copy, distiller.student)

    optimizer = torch.optim.AdamW(distiller.parameters(), lr=1e-5)
    inputs = {
        "input_ids": torch.LongTensor(
            [[101, 7592, 2026, 3899, 102], [101, 1045, 2031, 3899, 102]]
        ),
        "token_type_ids": torch.LongTensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
        "attention_mask": torch.LongTensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
    }
    num_epochs = 10
    assert torch.is_grad_enabled(), "this test requires grad to be enabled"
    for i in range(num_epochs):
        optimizer.zero_grad()
        loss = distiller.forward(**inputs)[0]
        loss.backward()
        optimizer.step()

    # After distillation, teacher's weights should stay unchanged
    # while student's weights should be updated
    assert have_same_weights(teacher_copy, distiller.teacher)
    assert not have_same_weights(student_copy, distiller.student)


def test_teacher_in_eval(distiller):
    distiller.train()
    assert not distiller.teacher.training


if __name__ == "__main__":
    distiller = create_minilm_distiller()
    test_relation_tensors(distiller)
    test_weights(distiller)
    test_teacher_in_eval(distiller)
    print("all passed")
