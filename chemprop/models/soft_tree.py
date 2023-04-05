from typing import List, Tuple, Optional, Callable
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import numpy as np

from chemprop.nn_utils import get_activation_function
from chemprop.args import TrainArgs


class SoftTreeReadout(nn.Module,ABC):
    """Readout networks that use soft trees instead of FFNN layers."""

    def __init__(
        self,
        args: TrainArgs,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float,
        activation: str,
        dataset_type: str,
        **kwargs
    ):
        super().__init__()

        self.num_trees = args.number_of_trees
        self.layer_before_trees = args.layer_before_trees
        self.soft_tree_mode = args.soft_tree_mode
        self.soft_tree_response_mode = args.soft_tree_response_mode
        self.comparison_function = self.get_comparison_function(args.soft_tree_comparison_function)
        self.use_temperature = args.soft_tree_use_temperature
        self.feature_choice_function = self.get_feature_choice_function(args.soft_tree_feature_function)
        self.device = args.device

        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size
        self.dataset_type = dataset_type
        self.activation = get_activation_function(activation)

        if self.layer_before_trees:
            self.input_size = hidden_size
            self.initial_layer = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(input_dim, hidden_size),
                self.activation,
            )  # [batch, features]
        else:
            self.input_size = input_dim
            self.initial_layer = nn.Identity()
        
        self.feature_dropout = nn.Dropout(self.dropout)

        self.response = nn.Parameter(
            torch.zeros([1, self.num_trees, 2**self.num_layers]),
            requires_grad=True,
        ) # [batch=1, trees, 2**depth]

        self.feature_weights = self.create_feature_weights()
        # [trees, depth, features]

        self.feature_thresholds = self.create_feature_thresholds()
        # [trees, depth, batch=1]

        self.log_temperatures = self.create_log_temperature()
        # [trees, depth, batch=1]

        # binary codes for mapping between 1-hot vectors and bin indices, as written by Popov et al.
        with torch.no_grad():
            indices = torch.arange(2 ** self.num_layers)
            offsets = 2 ** torch.arange(self.num_layers)
            bin_codes = (indices.view(1, -1) // offsets.view(-1, 1) % 2).to(torch.float32)
            bin_codes_1hot = torch.stack([bin_codes, 1.0 - bin_codes], dim=-1)
            self.bin_codes_1hot = bin_codes_1hot.to(device=self.device)
            # ^-- [depth, 2 ** depth, 2]

    def set_response(self, new_response: torch.Tensor):
        with torch.no_grad():
            self.response[...] = new_response
    
    @abstractmethod
    def create_feature_weights(self) -> nn.Parameter:
        pass

    @abstractmethod
    def create_feature_thresholds(self) -> nn.Parameter:
        pass

    @abstractmethod
    def create_log_temperature(self) -> nn.Parameter:
        pass
    
    def get_feature_choice_function(self, feature_choice_function) -> Callable:
        feature_choices = {
            "softmax": nn.Softmax(dim=2),
            "sigmoid": nn.Sigmoid(),
            # "entmax": "not implemented",
            "linear": nn.Identity(),
        }
        return feature_choices[feature_choice_function]
    
    def get_comparison_function(self, comparison_function) -> Callable:
        comparisons = {
            "softmax": nn.Softmax(dim=3),
            "sigmoid": nn.Sigmoid(),
            # "entmax": "not implemented"
        }
        return comparisons[comparison_function]



class ObliviousTreeReadout(SoftTreeReadout):
    """Readout based on the Neural Oblivious Decision Ensembles method of Popov et al."""

    def create_feature_weights(self) -> nn.Parameter:
        return nn.Parameter(torch.zeros([self.num_trees, self.num_layers, self.input_size]), requires_grad=True)

    def create_feature_thresholds(self) -> nn.Parameter:
        return nn.Parameter(torch.zeros([self.num_trees, self.num_layers, 1]), requires_grad=True)
    
    def create_log_temperature(self) -> nn.Parameter:
        if self.use_temperature:
            return nn.Parameter(torch.zeros([self.num_trees, self.num_layers, 1]), requires_grad=True)
        else:
            return torch.zeros([1], device=self.device)

    def forward(
        self,
        input: torch.Tensor,  # [batch, features]
    ) -> torch.Tensor:
        input = self.initial_layer(input)
        # [batch, features]
        activated_weights = self.feature_choice_function(self.feature_dropout(self.feature_weights))
        # [trees, depth, features]
        weighted_features = torch.matmul(activated_weights, input.T)
        # [trees, depth, batch]
        comparator_logits = (weighted_features - self.feature_thresholds) * torch.exp(self.log_temperatures)
        # [trees, depth, batch]
        comparator_logits = torch.stack([-comparator_logits, comparator_logits], dim=-1)
        # [trees, depth, batch, 2]
        comparators = self.comparison_function(comparator_logits)
        # [trees, depth, batch, 2]
        # codes_1hot has a size [depth, 2 ** depth, 2]
        bin_matches = torch.einsum('tdbs,dcs->btdc', comparators, self.bin_codes_1hot)
        # [batch, trees, depth, 2 ** depth]
        response_weights = torch.prod(bin_matches, dim=-2)
        # [batch, trees, 2 ** depth]
        response = response_weights * self.response
        # [batch, trees, 2 ** depth]
        output = response.sum(dim=[1,2]).unsqueeze(-1)
        # [batch, tasks=1] haven't implemented multitask
        # raise Exception
        return output
    
    def response_weights(
        self,
        input: torch.Tensor,  # [batch, features]
    ) -> torch.Tensor:
        with torch.inference_mode():
            input = self.initial_layer(input)
            # [batch, features]
            activated_weights = self.feature_choice_function(self.feature_dropout(self.feature_weights))
            # [trees, depth, features]
            weighted_features = torch.matmul(activated_weights, input.T)
            # [trees, depth, batch]
            comparator_logits = (weighted_features - self.feature_thresholds) * torch.exp(self.log_temperatures)
            # [trees, depth, batch]
            comparator_logits = torch.stack([-comparator_logits, comparator_logits], dim=-1)
            # [trees, depth, batch, 2]
            comparators = self.comparison_function(comparator_logits)
            # [trees, depth, batch, 2]
            # codes_1hot has a size [depth, 2 ** depth, 2]
            bin_matches = torch.einsum('tdbs,dcs->btdc', comparators, self.bin_codes_1hot)
            # [batch, trees, depth, 2 ** depth]
            response_weights = torch.prod(bin_matches, dim=-2)
            # [batch, trees, 2 ** depth]
            return response_weights.detach().clone().unsqueeze(1) # # [batch, 1, trees, 2 ** depth]

