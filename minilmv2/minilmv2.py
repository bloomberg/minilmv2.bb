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

Implementation of MiniLMv2 distillation technique.
"""

import logging
import math
from typing import Dict, Tuple

import torch
from torch import nn


class MiniLM(nn.Module):
    """MiniLMv2 model.

    Arguments:
        teacher(nn.Module): the teacher model
        student(nn.Module): the student model
        L(int): Lth layer of the teacher model to distill. For an L-layer model, L can be 1, ..., L
                Note that as MiniLMv2 paper suggests, L may not be the last layer
        M(int): Number of layers of the student model, which is also the layer of the student model
                used in distillation. For an m-layer model, M can be 1, ..., m
        relations(Dict[Tuple[int, int], float]): A dictionary of self-attention relation pairs and weights with
                key = (Teacher, Student), value = weight.
                e.g. {(1,1): 1/3, (2,2): 1/3, (3,3): 1/3} means considering relations for
                Query-Query, Key-Key, and Value-Value in the loss with weight 1/3 for each
        A_r(int): Number of relation heads
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        L: int,
        M: int,
        relations: Dict[Tuple[int, int], float],
        A_r: int,
    ):
        """Initialize a MiniLMv2 model."""
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.kl_loss_fn = torch.nn.KLDivLoss(reduction="sum")
        self.teacher.eval()
        self.student.train()
        self.L = L
        self.M = M
        self.relations = relations  # note: we implement relation weights in formula (5) as hyperparameters
        self.A_r = A_r

        # Make sure not updating teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        logging.warning(
            "Setting teacher model to eval mode and disabling gradient update for MiniLM training. "
            "You must manually reset it to train mode and enable gradient update if you wish to continue updating the teacher after distillation."
        )

    def _get_relation_vectors(self, self_attn, prev_hidden, relation_head_size: int):
        """Get query, key, and value of relation heads of the last attention layer.

        The vectors' shape will be (batch_size, relation_head_number, seq_length, relation_head_size).
        """
        q = self._transpose_for_scores_relation(
            self_attn.query(prev_hidden), relation_head_size
        )
        k = self._transpose_for_scores_relation(
            self_attn.key(prev_hidden), relation_head_size
        )
        v = self._transpose_for_scores_relation(
            self_attn.value(prev_hidden), relation_head_size
        )
        return q, k, v

    def _transpose_for_scores_relation(self, x: torch.Tensor, relation_head_size: int):
        """Adapted from BertSelfAttention.get_transposed_attns().

        Arguments:
            x (Tensor): a vector (query, key, or value) of shape (batch_size, seq_length, hidden_size)
            relation_head_size (int): relation head size
        Return:
            x_relation (Tensor): a vector (query, key, or value) of shape
                                (batch_size, relation_head_number, seq_length, relation_head_size)
        """
        new_x_shape = [*x.size()[:-1], self.A_r, relation_head_size]
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def _get_kl_loss(
        self, rel_T: torch.Tensor, rel_S: torch.Tensor, attention_mask: torch.Tensor
    ):
        """Compute KL divergence loss of teacher and student on one relation.

        This function is a vectorized version of formula (6) in the MiniLM paper.
        The paper does not handle batching and attention mask.

        Arguments:
            rel_T: a self attention relation of the teacher (batch_size, A_r, seq_len, seq_len)
            rel_S: a self attention relation of the student (batch_size, A_r, seq_len, seq_len)
            attention_mask: attention mask of a batch of input
        """
        # Note: rel_T is the target and rel_S is the input of KL Div loss for KLDivLoss(), before softmax.
        # KLDivLoss() needs log of inputs (rel_S)
        # Reference:
        # (1) torch source: https://github.com/pytorch/pytorch/blob/7cc029cb75c292e93d168e117e46a681ace02e79/aten/src/ATen/native/Loss.cpp#L71
        # (2) wikipedia: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
        loss = 0.0
        batch_size = attention_mask.shape[0]
        seq_lengths = attention_mask.sum(-1).tolist()
        for b in range(batch_size):
            cur_seq_len = seq_lengths[b]  # current sequence length
            # While we kind of get the same values from output.attentions from BertModel, it seems to do a weird thing by
            # applying dropout post softmax. The paper's calculations do not apply this
            R_L_T = torch.nn.Softmax(dim=-1)(rel_T[b, :, :cur_seq_len, :cur_seq_len])
            R_M_S = torch.nn.functional.log_softmax(
                rel_S[b, :, :cur_seq_len, :cur_seq_len], dim=-1
            )  # KL DIV loss needs log, so do log_softmax
            loss += self.kl_loss_fn(
                R_M_S.reshape(-1, cur_seq_len), R_L_T.reshape(-1, cur_seq_len)
            ) / (
                self.A_r * cur_seq_len
            )  # normalize by relation head num and seq length
        loss /= batch_size  # normalize by batch_size as well
        return loss

    def train(self, mode=True):
        """Override the train method to define specific behavior.

        Arguments:
            mode: Boolean indicating whether to train or eval
        """
        super().train(mode)
        self.teacher.eval()

    def forward(self, input_ids, token_type_ids, attention_mask):
        """Run a forward pass over the input. Return a tuple of one element, which is the MiniLM loss.

        Note: the return value is a tuple since HuggingFace trainer uses outputs[0] as loss.

        Arguments:
            input_ids: input_id tokens for a batch.
            token_type_ids: token_type_ids (indicating sentence_id) for a batch.
            attention_mask: Attention mask (indicating which tokens are pad tokens) for a batcb.
        """
        inputs = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }
        teacher_outs = self.teacher(
            **inputs, output_hidden_states=True, output_attentions=True
        )
        student_outs = self.student(
            **inputs, output_hidden_states=True, output_attentions=True
        )

        L = self.L  # layer to distill from in teacher (can be any teacher layer)
        M = self.M  # layer to distill to in student (last layer)

        d_h_T = self.teacher.config.hidden_size  # teacher's hidden size
        d_h_S = self.student.config.hidden_size  # student's hidden size
        d_r_T = d_h_T // self.A_r  # teacher's relation head size
        d_r_S = d_h_S // self.A_r  # student's relation head size

        # hidden_states contains L+1 elements for the teacher and M+1 elements for the student,
        # since the first is embedding
        # To calculate query, key, and value for the last attention layer, we get the hidden states
        # of the second last layer (L+1 -2 = L - 1)
        hidden_L_1_T = teacher_outs.hidden_states[L - 1]
        hidden_M_1_S = student_outs.hidden_states[M - 1]

        # Get relation vectors (query, key, value) of the shape (batch_size, A_r, seq_len, d_r) based on Figure 1
        relation_vectors_T = self._get_relation_vectors(
            self.teacher.encoder.layer[L - 1].attention.self, hidden_L_1_T, d_r_T
        )
        relation_vectors_S = self._get_relation_vectors(
            self.student.encoder.layer[M - 1].attention.self, hidden_M_1_S, d_r_S
        )

        loss = 0  # total loss of all types of relations
        for relation_pair, weight in self.relations.items():
            # Calculate loss for each pairs of relations
            # 1-> Query, 2-> Key, 3-> Value.
            # relation pair of (1,2) indicates to compute QK for teacher and student and apply loss on it
            m, n = relation_pair  # m and n are 1-indexed

            # Formula (7) and (8)
            A_L_T_scaleddot = torch.matmul(
                relation_vectors_T[m - 1], relation_vectors_T[n - 1].transpose(-1, -2)
            ) / math.sqrt(
                d_r_T
            )  # (batch_size, A_r, seq_len, seq_len)
            A_M_S_scaleddot = torch.matmul(
                relation_vectors_S[m - 1], relation_vectors_S[n - 1].transpose(-1, -2)
            ) / math.sqrt(d_r_S)

            # Compute relaiton loss (Formula (6))
            l_relation = self._get_kl_loss(
                A_L_T_scaleddot.detach(), A_M_S_scaleddot, inputs["attention_mask"]
            )

            # Aggregate losses (Formula (5))
            loss += weight * l_relation

        return (loss,)
