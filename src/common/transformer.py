import logging
import math
import torch
from torch import nn
import random


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


BertLayerNorm = torch.nn.LayerNorm


class BertAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, attention_probs_dropout_prob, name="default"):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_head_size = hidden_size // num_heads
        self.name = name

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size + 1e-5)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        attention_probs_mean = attention_probs.mean([0, 1, 2])
        # import random
        # if random.random() < 0.05:
        #     print(f"{self.name} attention_probs_mean = {attention_probs_mean}")

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertAttOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(BertAttOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SelfattLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, attention_probs_dropout_prob, hidden_dropout_prob, name="default"):
        super(SelfattLayer, self).__init__()
        self.self = BertAttention(hidden_size, num_heads, attention_probs_dropout_prob, name)
        self.output = BertAttOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class FQFormer(nn.Module):
    def __init__(self, query_num=1, layer_num=2, hidden_size=64, num_heads=8, attention_probs_dropout_prob=0.,
                 hidden_dropout_prob=0.):
        super(FQFormer, self).__init__()

        self.query_num = query_num
        self.layer_num = layer_num
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.queries = nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.empty(1, self.query_num, hidden_size), gain=1.0),
            requires_grad=True
        )

        self.layers = nn.ModuleList(
            [SelfattLayer(hidden_size, num_heads, attention_probs_dropout_prob, hidden_dropout_prob, name=f"layer{i}")
             for i in range(self.layer_num)]
        )

    def forward(self, x, attention_mask=None):
        B, L, D = x.size()
        queries = self.queries.repeat([B, 1, 1])

        x = torch.cat([queries, x], 1)
        for layer in range(self.layer_num):
            x = self.layers[layer](x, attention_mask)
        out = x
        return out