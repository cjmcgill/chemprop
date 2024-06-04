from typing import List, Union, Tuple

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn

from .mpn import MPN
from .ffn import build_ffn, MultiReadout
from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph
from chemprop.nn_utils import initialize_weights


class MoleculeModel(nn.Module):
    """A :class:`MoleculeModel` is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, args: TrainArgs):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        super(MoleculeModel, self).__init__()

        self.classification = args.dataset_type == "classification"
        self.multiclass = args.dataset_type == "multiclass"
        self.loss_function = args.loss_function
        self.vp = args.vp
        self.vle = args.vle
        self.fugacity_balance = args.fugacity_balance
        self.device = args.device
        self.hidden_size = args.hidden_size
        self.noisy_temperature = args.noisy_temperature

        if hasattr(args, "train_class_sizes"):
            self.train_class_sizes = args.train_class_sizes
        else:
            self.train_class_sizes = None

        # when using cross entropy losses, no sigmoid or softmax during training. But they are needed for mcc loss.
        if self.classification or self.multiclass:
            self.no_training_normalization = args.loss_function in [
                "cross_entropy",
                "binary_cross_entropy",
            ]

        self.is_atom_bond_targets = args.is_atom_bond_targets

        if self.is_atom_bond_targets:
            self.atom_targets, self.bond_targets = args.atom_targets, args.bond_targets
            self.atom_constraints, self.bond_constraints = (
                args.atom_constraints,
                args.bond_constraints,
            )
            self.adding_bond_types = args.adding_bond_types

        self.relative_output_size = 1
        if self.multiclass:
            self.relative_output_size *= args.multiclass_num_classes
        if self.loss_function == "mve":
            self.relative_output_size *= 2  # return means and variances
        if self.loss_function == "dirichlet" and self.classification:
            self.relative_output_size *= (
                2  # return dirichlet parameters for positive and negative class
            )
        if self.loss_function == "evidential":
            self.relative_output_size *= (
                4  # return four evidential parameters: gamma, lambda, alpha, beta
            )

        if self.fugacity_balance is not None:
            if self.vle == "activity":
                self.relative_output_size *= 1/2
            elif self.vle == "wohl":
                self.relative_output_size *= 3/4
        elif self.vle == "basic":
            self.relative_output_size *= 2/3 # gets out y_1 and log10P, but calculates y_2 from it to return three results
        elif self.vle == "activity":
            self.relative_output_size *= 2/3 # uses two activity parameters internally and returns three results
        elif self.vle == "wohl":
            self.relative_output_size *= 1 # uses three function parameters internally and returns three results
        elif self.vp == 'basic':
            self.relative_output_size *= 1 # predicts vp directly
        elif self.vp == 'two_var':
            self.relative_output_size *= 2 # uses two antoine parameters internally and returns one result
        elif self.vp == "antoine":
            self.relative_output_size *= 3 # uses three antoine parameters internally and returns one result
        elif self.vp == "four_var":
            self.relative_output_size *= 4 # uses four antoine parameters internally and returns one result
        elif self.vp == "five_var":
            self.relative_output_size *= 5 # uses five antoine parameters internally and returns one result


        if self.classification or self.vle is not None:
            self.sigmoid = nn.Sigmoid()

        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)

        if self.loss_function in ["mve", "evidential", "dirichlet"]:
            self.softplus = nn.Softplus()

        self.create_encoder(args)
        self.create_ffn(args)

        initialize_weights(self)

    def create_encoder(self, args: TrainArgs) -> None:
        """
        Creates the message passing encoder for the model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        self.encoder = MPN(args)

        if args.checkpoint_frzn is not None:
            if args.freeze_first_only:  # Freeze only the first encoder
                for param in list(self.encoder.encoder.children())[0].parameters():
                    param.requires_grad = False
            else:  # Freeze all encoders
                for param in self.encoder.parameters():
                    param.requires_grad = False

    def create_ffn(self, args: TrainArgs) -> None:
        """
        Creates the feed-forward layers for the model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        self.multiclass = args.dataset_type == "multiclass"
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            if args.reaction_solvent:
                first_linear_dim = args.hidden_size + args.hidden_size_solvent
            else:
                first_linear_dim = args.hidden_size * args.number_of_molecules
            if args.use_input_features:
                first_linear_dim += args.features_size

        if args.atom_descriptors == "descriptor":
            atom_first_linear_dim = first_linear_dim + args.atom_descriptors_size
        else:
            atom_first_linear_dim = first_linear_dim

        if args.bond_descriptors == "descriptor":
            bond_first_linear_dim = first_linear_dim + args.bond_descriptors_size
        else:
            bond_first_linear_dim = first_linear_dim

        # Create FFN layers
        if self.is_atom_bond_targets:
            self.readout = MultiReadout(
                atom_features_size=atom_first_linear_dim,
                bond_features_size=bond_first_linear_dim,
                atom_hidden_size=args.ffn_hidden_size + args.atom_descriptors_size,
                bond_hidden_size=args.ffn_hidden_size + args.bond_descriptors_size,
                num_layers=args.ffn_num_layers,
                output_size=self.relative_output_size,
                dropout=args.dropout,
                activation=args.activation,
                atom_constraints=args.atom_constraints,
                bond_constraints=args.bond_constraints,
                shared_ffn=args.shared_atom_bond_ffn,
                weights_ffn_num_layers=args.weights_ffn_num_layers,
            )
        else:
            self.readout = build_ffn(
                first_linear_dim=atom_first_linear_dim,
                hidden_size=args.ffn_hidden_size,
                num_layers=args.ffn_num_layers,
                output_size=int(np.rint(self.relative_output_size * args.num_tasks)),
                dropout=args.dropout,
                activation=args.activation,
                dataset_type=args.dataset_type,
                spectra_activation=args.spectra_activation,
            )
            if self.vle == "wohl":
                self.wohl_q = build_ffn(
                    first_linear_dim=self.hidden_size + 1, # +1 for temperature only
                    hidden_size=args.ffn_hidden_size,
                    num_layers=args.ffn_num_layers,
                    output_size=1, # q
                    dropout=args.dropout,
                    activation=args.activation,
                    dataset_type=args.dataset_type,
                    spectra_activation=args.spectra_activation,
                )
            vp_output_size_dict = {"basic": 1, "two_var": 2, "antoine": 3, "four_var": 4, "five_var": 5}
            if self.fugacity_balance == "intrinsic_vp":
                self.intrinsic_vp = build_ffn(
                    first_linear_dim=self.hidden_size + 1, # +1 for temperature only
                    hidden_size=args.ffn_hidden_size,
                    num_layers=args.ffn_num_layers,
                    output_size=vp_output_size_dict[args.vp],
                    dropout=args.dropout,
                    activation=args.activation,
                    dataset_type=args.dataset_type,
                    spectra_activation=args.spectra_activation,
                )

        if args.checkpoint_frzn is not None:
            if args.frzn_ffn_layers > 0:
                if self.is_atom_bond_targets:
                    if args.shared_atom_bond_ffn:
                        for param in list(self.readout.atom_ffn_base.parameters())[
                            0 : 2 * args.frzn_ffn_layers
                        ]:
                            param.requires_grad = False
                        for param in list(self.readout.bond_ffn_base.parameters())[
                            0 : 2 * args.frzn_ffn_layers
                        ]:
                            param.requires_grad = False
                    else:
                        for ffn in self.readout.ffn_list:
                            if ffn.constraint:
                                for param in list(ffn.ffn.parameters())[
                                    0 : 2 * args.frzn_ffn_layers
                                ]:
                                    param.requires_grad = False
                            else:
                                for param in list(ffn.ffn_readout.parameters())[
                                    0 : 2 * args.frzn_ffn_layers
                                ]:
                                    param.requires_grad = False
                else:
                    for param in list(self.readout.parameters())[
                        0 : 2 * args.frzn_ffn_layers
                    ]:  # Freeze weights and bias for given number of layers
                        param.requires_grad = False

    def forward_vp(
            self,
            output: torch.Tensor,
            hybrid_model_features_batch: torch.Tensor,
    ):
        """
        Calculates the vapor pressure within the forward function
        """
        temp_batch = hybrid_model_features_batch[:,[2]]
        if self.vp == "basic":
            output = output
        if self.vp == "two_var":
            antoine_a, antoine_b = torch.chunk(output, 2, dim=1)
            output = antoine_a + (antoine_b / temp_batch)
        if self.vp == "antoine":
            antoine_a, antoine_b, antoine_c = torch.chunk(output, 3, dim=1)
            output = antoine_a - (antoine_b / (antoine_c + temp_batch))
        if self.vp == "four_var":
            antoine_a, antoine_b, antoine_c, antoine_d = torch.chunk(output, 4, dim=1)
            output = antoine_a + (antoine_b / temp_batch) + (antoine_c * torch.log(temp_batch)) + (antoine_d * torch.pow(temp_batch, 6))
        if self.vp == "five_var":
            antoine_a, antoine_b, antoine_c, antoine_d, antoine_e = torch.chunk(output, 5, dim=1)
            output = antoine_a + (antoine_b / temp_batch) + (antoine_c * torch.log(temp_batch)) + (antoine_d * torch.pow(temp_batch, antoine_e))
        return output

    def fingerprint(
        self,
        batch: Union[
            List[List[str]],
            List[List[Chem.Mol]],
            List[List[Tuple[Chem.Mol, Chem.Mol]]],
            List[BatchMolGraph],
        ],
        features_batch: List[np.ndarray] = None,
        atom_descriptors_batch: List[np.ndarray] = None,
        atom_features_batch: List[np.ndarray] = None,
        bond_descriptors_batch: List[np.ndarray] = None,
        bond_features_batch: List[np.ndarray] = None,
        fingerprint_type: str = "MPN",
    ) -> torch.Tensor:
        """
        Encodes the latent representations of the input molecules from intermediate stages of the model.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :param atom_features_batch: A list of numpy arrays containing additional atom features.
        :param bond_descriptors_batch: A list of numpy arrays containing additional bond descriptors.
        :param bond_features_batch: A list of numpy arrays containing additional bond features.
        :param fingerprint_type: The choice of which type of latent representation to return as the molecular fingerprint. Currently
                                 supported MPN for the output of the MPNN portion of the model or last_FFN for the input to the final readout layer.
        :return: The latent fingerprint vectors.
        """
        if fingerprint_type == "MPN":
            return self.encoder(
                batch,
                features_batch,
                atom_descriptors_batch,
                atom_features_batch,
                bond_descriptors_batch,
                bond_features_batch,
            )
        elif fingerprint_type == "last_FFN":
            return self.readout[:-1](
                self.encoder(
                    batch,
                    features_batch,
                    atom_descriptors_batch,
                    atom_features_batch,
                    bond_descriptors_batch,
                    bond_features_batch,
                )
            )
        else:
            raise ValueError(f"Unsupported fingerprint type {fingerprint_type}.")

    def forward(
        self,
        batch: Union[
            List[List[str]],
            List[List[Chem.Mol]],
            List[List[Tuple[Chem.Mol, Chem.Mol]]],
            List[BatchMolGraph],
        ],
        features_batch: List[np.ndarray] = None,
        atom_descriptors_batch: List[np.ndarray] = None,
        atom_features_batch: List[np.ndarray] = None,
        bond_descriptors_batch: List[np.ndarray] = None,
        bond_features_batch: List[np.ndarray] = None,
        constraints_batch: List[torch.Tensor] = None,
        bond_types_batch: List[torch.Tensor] = None,
        hybrid_model_features_batch: List[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        Runs the :class:`MoleculeModel` on input.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :param atom_features_batch: A list of numpy arrays containing additional atom features.
        :param bond_descriptors_batch: A list of numpy arrays containing additional bond descriptors.
        :param bond_features_batch: A list of numpy arrays containing additional bond features.
        :param constraints_batch: A list of PyTorch tensors which applies constraint on atomic/bond properties.
        :param bond_types_batch: A list of PyTorch tensors storing bond types of each bond determined by RDKit molecules.
        :return: The output of the :class:`MoleculeModel`, containing a list of property predictions.
        """

        if hybrid_model_features_batch is not None:
            hybrid_model_features_batch = torch.from_numpy(np.array(hybrid_model_features_batch, dtype=np.float64)).float().to(self.device)

        if self.noisy_temperature is not None and self.training:
            features_batch = np.array(features_batch)
            noise_batch = np.random.randn(len(features_batch)) * self.noisy_temperature
            features_batch[:,0] = features_batch[:,0] + noise_batch

        if self.is_atom_bond_targets:
            encodings = self.encoder(
                batch,
                features_batch,
                atom_descriptors_batch,
                atom_features_batch,
                bond_descriptors_batch,
                bond_features_batch,
            )
            output = self.readout(encodings, constraints_batch, bond_types_batch)
        else:
            encodings = self.encoder(
                batch,
                features_batch,
                atom_descriptors_batch,
                atom_features_batch,
                bond_descriptors_batch,
                bond_features_batch,
            )
            output = self.readout(encodings)

            # Extra outputs for VLE models
            if self.vle == "wohl" or self.fugacity_balance == "intrinsic_vp":
                encoding_1 = encodings[:,:self.hidden_size]
                encoding_1 = torch.concatenate([encoding_1, hybrid_model_features_batch[:,[2]]], axis=1) # include T feature at the end
                encoding_2 = encodings[:,self.hidden_size:2*self.hidden_size] # includes features at the end
                encoding_2 = torch.concatenate([encoding_2, hybrid_model_features_batch[:,[2]]], axis=1) # include T feature at the end
                if self.vle == "wohl":
                    q_1 = nn.functional.softplus(self.wohl_q(encoding_1))
                    q_2 = nn.functional.softplus(self.wohl_q(encoding_2))
                if self.fugacity_balance == "intrinsic_vp":
                    vp1_output = self.intrinsic_vp(encoding_1)
                    vp2_output = self.intrinsic_vp(encoding_2)
                    log10p1sat = self.forward_vp(vp1_output, hybrid_model_features_batch)
                    log10p2sat = self.forward_vp(vp2_output, hybrid_model_features_batch)

        # Don't apply sigmoid during training when using BCEWithLogitsLoss
        if (
            self.classification
            and not (self.training and self.no_training_normalization)
            and self.loss_function != "dirichlet"
        ):
            if self.is_atom_bond_targets:
                output = [self.sigmoid(x) for x in output]
            else:
                output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape(
                (output.shape[0], -1, self.num_classes)
            )  # batch size x num targets x num classes per target
            if (
                not (self.training and self.no_training_normalization)
                and self.loss_function != "dirichlet"
            ):
                output = self.multiclass_softmax(
                    output
                )  # to get probabilities during evaluation, but not during training when using CrossEntropyLoss

        # Apply post-processing for VLE models
        if self.vle is not None:
            if self.vle == "basic":
                logity_1, log10P = torch.chunk(output, 2, dim=1)
                y_1 = self.sigmoid(logity_1)
                y_2 = 1 - y_1
                output = torch.cat([y_1, y_2, log10P], axis=1)
            else:  # vle in ["activity", "wohl"] # x1 x2 T P1sat P2sat
                # print(hybrid_model_features_batch.shape)
                # print(hybrid_model_features_batch)
                x_1 = hybrid_model_features_batch[:,[0]]
                x_2 = hybrid_model_features_batch[:,[1]]
                T = hybrid_model_features_batch[:,[2]]
                if self.fugacity_balance != "intrinsic_vp":
                    log10p1sat = hybrid_model_features_batch[:,[3]]
                    log10p2sat = hybrid_model_features_batch[:,[4]]
                if self.vle == "activity":
                    gamma_1 = torch.exp(output[:,[0]])
                    gamma_2 = torch.exp(output[:,[1]])
                else:  # vle == "wohl"
                    # There's a coefficient before the fitted A that you can get using scipy.special.binom(ith-wohl-degree, jth term) where term 0 and i are defined as zero for excess properties
                    # The ith degree of Wohl adds i terms to the expansion (but first and last are zero so really i-2)
                    # all terms in the expansion are of the form gE = Sum A * z**n1 + z**n2 * (N1*q1+N2*q2)
                    # for to get ln(gamma1) * RT = d/dN1 [Sum A * z1**n1 + z2**n2 * (N1*q1+N2*q2)]
                    # and each term is d/dN1 [A * z1**n1 * z2**n2] = A * n1 * z1**(n1-1) * z2**(n2+1) * q1 + A * (1-n2) * z1**n1 * z2**n2 * q1
                    a12, a112, a122 = torch.chunk(output, 3, dim=1)
                    z_1 = q_1 * x_1 / (q_1 * x_1 + q_2 * x_2)
                    z_2 = q_2 * x_2 / (q_1 * x_1 + q_2 * x_2)
                    gamma_1 = torch.exp(
                        2*a12*z_2**2*q_1
                        + 6*a112*z_1*z_2**2*q_1 - 3*a112*z_1**2*z_2*q_1 + 3*a112*z_1**2*z_2*q_1
                        +3*a122*z_2**3*q_1 - 6*a122*z_1*z_2**2*q_1 + 3*a122*z_1*z_2**2*q_1
                    )
                    gamma_2 = torch.exp(
                        2*a12*z_1**2*q_2
                        + 6*a122*z_2*z_1**2*q_2 - 3*a122*z_2**2*z_1*q_2 + 3*a122*z_2**2*z_1*q_2
                        +3*a112*z_1**3*q_2 - 6*a112*z_2*z_1**2*q_2 + 3*a112*z_2*z_1**2*q_2
                    )
                    gamma_1_inf = torch.exp(2*a12*q_1 + 3*a122*q_1)
                if self.fugacity_balance is None:
                    p1sat = 10**log10p1sat
                    p2sat = 10**log10p2sat
                    P1 = p1sat * x_1 * gamma_1
                    P2 = p2sat * x_2 * gamma_2
                    P = P1 + P2
                    y_1 = P1 / P
                    y_2 = P2 / P
                    log10P = torch.log10(P)
                    output = torch.cat([y_1, y_2, log10P], axis=1)
                else: # fugacity_balance == "intrinsic_vp" or "tabulated_vp"
                    # if self.training:
                        output = torch.cat([gamma_1, gamma_2, log10p1sat, log10p2sat], axis=1)
                    # else: # predict mode
                    #     p1sat = 10**log10p1sat
                    #     p2sat = 10**log10p2sat
                    #     P1 = p1sat * x_1 * gamma_1
                    #     P2 = p2sat * x_2 * gamma_2
                    #     P = P1 + P2
                    #     y_1 = P1 / P
                    #     y_2 = P2 / P
                    #     log10P = torch.log10(P)
                    #     output = torch.cat([y_1, y_2, log10P, gamma_1_inf], axis=1)

        # VP
        if self.vp is not None and self.fugacity_balance is None:
            output = self.forward_vp(output, hybrid_model_features_batch)

        # Multi output loss functions
        if self.loss_function == "mve":
            if self.is_atom_bond_targets:
                outputs = []
                for x in output:
                    means, variances = torch.split(x, x.shape[1] // 2, dim=1)
                    variances = self.softplus(variances)
                    outputs.append(torch.cat([means, variances], axis=1))
                return outputs
            else:
                means, variances = torch.split(output, output.shape[1] // 2, dim=1)
                variances = self.softplus(variances)
                output = torch.cat([means, variances], axis=1)
        if self.loss_function == "evidential":
            if self.is_atom_bond_targets:
                outputs = []
                for x in output:
                    means, lambdas, alphas, betas = torch.split(
                        x, x.shape[1] // 4, dim=1
                    )
                    lambdas = self.softplus(lambdas)  # + min_val
                    alphas = (
                        self.softplus(alphas) + 1
                    )  # + min_val # add 1 for numerical contraints of Gamma function
                    betas = self.softplus(betas)  # + min_val
                    outputs.append(torch.cat([means, lambdas, alphas, betas], dim=1))
                return outputs
            else:
                means, lambdas, alphas, betas = torch.split(
                    output, output.shape[1] // 4, dim=1
                )
                lambdas = self.softplus(lambdas)  # + min_val
                alphas = (
                    self.softplus(alphas) + 1
                )  # + min_val # add 1 for numerical contraints of Gamma function
                betas = self.softplus(betas)  # + min_val
                output = torch.cat([means, lambdas, alphas, betas], dim=1)
        if self.loss_function == "dirichlet":
            if self.is_atom_bond_targets:
                outputs = []
                for x in output:
                    outputs.append(nn.functional.softplus(x) + 1)
                return outputs
            else:
                output = nn.functional.softplus(output) + 1

        return output
