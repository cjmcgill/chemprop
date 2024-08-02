from typing import List, Union, Tuple

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn

from .mpn import MPN
from .ffn import build_ffn, binary_equivariant_readout
from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph
from chemprop.nn_utils import initialize_weights
from .vp import forward_vp
from .vle import forward_vle_basic, forward_vle_activity, forward_vle_wohl


class MoleculeModel(nn.Module):
    """A :class:`MoleculeModel` is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, args: TrainArgs):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        super(MoleculeModel, self).__init__()
        self.args = args
        self.is_atom_bond_targets = False
        self.classification = args.dataset_type == "classification"
        self.multiclass = args.dataset_type == "multiclass"
        self.loss_function = args.loss_function
        self.vp = args.vp
        self.vle = args.vle
        self.wohl_order = args.wohl_order
        self.fugacity_balance = args.fugacity_balance
        self.intrinsic_vp = args.intrinsic_vp
        self.vle_inf_dilution = args.vle_inf_dilution
        self.device = args.device
        self.hidden_size = args.hidden_size
        self.noisy_temperature = args.noisy_temperature
        self.sigmoid = nn.functional.sigmoid
        self.softplus = nn.functional.softplus
        self.binary_equivariant = args.binary_equivariant

        self.output_size = args.num_tasks

        if self.vp is not None:
            vp_number_parameters_dict = {"basic": 1, "two_var": 2, "antoine": 3, "four_var": 4, "five_var": 5,
                                         "ambrose4": 4, "ambrose5": 5, "riedel4": 4, "riedel5": 5}
            self.vp_output_size = vp_number_parameters_dict[self.vp]
            if self.vle is None:
                self.output_size = self.vp_output_size

        if self.vle == "basic":
            if self.vle_inf_dilution:
                self.output_size = 4 # y_1, y_2, log10P, gamma1_inf_dilution
            else:
                self.output_size = 3 # y_1, y_2, log10P
        elif self.vle == "activity":
            self.output_size = 2 # gamma_1, gamma_2
        elif self.vle == "wohl":
            wohl_number_parameters_dict = {2: 1, 3: 3, 4: 6, 5: 10}
            self.output_size = wohl_number_parameters_dict[self.wohl_order]

        if self.binary_equivariant:
            if self.vle == "wohl":
                if self.wohl_order == 2: # a12; a112, a122; a1112, a1222, a1122; a11112, a11122, a11222, a12222
                    self.output_equivariant_pairs = []
                    self.features_equivariant_pairs = [] # T
                elif self.wohl_order == 3:
                    self.output_equivariant_pairs = [(1,2)]
                    self.features_equivariant_pairs = []
                elif self.wohl_order == 4:
                    self.output_equivariant_pairs = [(1,2), (3,5)]
                    self.features_equivariant_pairs = []
                elif self.wohl_order == 5:
                    self.output_equivariant_pairs = [(1,2), (3,5), (6,9), (7,8)]
                    self.features_equivariant_pairs = []
                else:
                    raise ValueError(f"Unsupported equivariant method with wohl order {self.wohl_order}.")
            elif self.vle == "activity":
                self.output_equivariant_pairs = [(0,1)]
                self.features_equivariant_pairs = [(0,1)] # x1, x2, T
            elif self.vle == "basic":
                self.output_equivariant_pairs = [(0,1)] # y1, y2, log10P
                self.features_equivariant_pairs = [(0,1), (3,4)] # x1, x2, T, log10P1sat, log10P2sat
            else:
                raise ValueError(f"Unsupported equivariant method with vle {self.vle}.")

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
        first_linear_dim = args.hidden_size * args.number_of_molecules
        if args.use_input_features:
            first_linear_dim += args.features_size
        if self.binary_equivariant:
            first_linear_dim = 2 * args.hidden_size + args.features_size # 2 molecules + features

        # Create FFN layers
        self.readout = build_ffn(
            first_linear_dim=first_linear_dim,
            hidden_size=args.ffn_hidden_size,
            num_layers=args.ffn_num_layers,
            output_size=self.output_size,
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
        if self.intrinsic_vp:
            self.intrinsic_vp = build_ffn(
                first_linear_dim=self.hidden_size + 1, # +1 for temperature only
                hidden_size=args.ffn_hidden_size,
                num_layers=args.ffn_num_layers,
                output_size=self.vp_output_size,
                dropout=args.dropout,
                activation=args.activation,
                dataset_type=args.dataset_type,
                spectra_activation=args.spectra_activation,
            )

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

        # get temperature and x for use in parameterized equations
        features_batch = torch.from_numpy(np.array(features_batch, dtype=np.float64)).float().to(self.device)
        if self.vle == "basic":
            output_temperature_batch = hybrid_model_features_batch[:,[2]]
            input_temperature_batch = features_batch[:,[2]]
            x_1 = hybrid_model_features_batch[:,[0]]
            x_2 = hybrid_model_features_batch[:,[1]]
        elif self.vle is not None:
            output_temperature_batch = hybrid_model_features_batch[:,[2]]
            input_temperature_batch = features_batch[:,[0]]
            x_1 = hybrid_model_features_batch[:,[0]]
            x_2 = hybrid_model_features_batch[:,[1]]
        elif self.vp is not None:
            output_temperature_batch = hybrid_model_features_batch[:,[0]]
            input_temperature_batch = features_batch[:,[0]]
            x_1 = None
            x_2 = None
        else:
            output_temperature_batch = None            
            input_temperature_batch = None
            x_1 = None
            x_2 = None

        # get Tc and Pc
        if self.vp in ["ambrose4", "ambrose5", "riedel4", "riedel5"] and self.vle is None:
            Tc = hybrid_model_features_batch[:,[1]]
            log10Pc = hybrid_model_features_batch[:,[2]]
        elif self.vp in ["ambrose4", "ambrose5", "riedel4", "riedel5"] and self.vle is not None:
            raise NotImplementedError("Ambrose and Riedel equations are not implemented for VLE models.")
        else:
            Tc = None
            log10Pc = None

        if self.noisy_temperature is not None and self.training:
            # noise is applied to the features temperature not the temperature batch
            noise_batch = np.random.randn(len(features_batch)) * self.noisy_temperature
            input_temperature_batch += noise_batch

        # Make the encodings
        encodings = self.encoder(
            batch,
            features_batch,
            atom_descriptors_batch,
            atom_features_batch,
            bond_descriptors_batch,
            bond_features_batch,
        )

        if self.vle == "wohl" or self.intrinsic_vp or self.binary_equivariant:
            encoding_1 = encodings[:,:self.hidden_size] # first molecule
            encoding_2 = encodings[:,self.hidden_size:2*self.hidden_size] # second molecule

        # readout section
        if self.binary_equivariant:
            output = binary_equivariant_readout(encoding_1, encoding_2, input_temperature_batch, self.readout, self.output_equivariant_pairs, self.features_equivariant_pairs)
        else:
            output = self.readout(encodings)

        if self.vle == "wohl":
            q_1 = nn.functional.softplus(self.wohl_q(torch.cat([encoding_1, input_temperature_batch], axis=1)))
            q_2 = nn.functional.softplus(self.wohl_q(torch.cat([encoding_2, input_temperature_batch], axis=1)))

        if self.intrinsic_vp:
            vp1_output = self.intrinsic_vp(torch.cat([encoding_1, input_temperature_batch], axis=1))
            vp2_output = self.intrinsic_vp(torch.cat([encoding_2, input_temperature_batch], axis=1))
            log10p1sat = forward_vp(self.vp, vp1_output, output_temperature_batch)
            log10p2sat = forward_vp(self.vp, vp2_output, output_temperature_batch)
        elif self.vle is not None:
            log10p1sat = hybrid_model_features_batch[:,[3]]
            log10p2sat = hybrid_model_features_batch[:,[4]]
        else:
            log10p1sat = None
            log10p2sat = None

        # VLE models
        if self.vle == "basic":
            output = forward_vle_basic(output, self.vle_inf_dilution)
        elif self.vle is not None:
            if self.vle == "activity":
                gamma_1, gamma_2 = forward_vle_activity(
                    output=output,
                )
            elif self.vle == "wohl":
                gamma_1, gamma_2 = forward_vle_wohl(
                    output=output,
                    wohl_order=self.wohl_order,
                    x_1=x_1,
                    x_2=x_2,
                    q_1=q_1,
                    q_2=q_2,
                )
            else:
                raise ValueError(f"Unsupported VLE model {self.vle}.")
            
            if self.training and self.fugacity_balance:
                output = torch.cat([gamma_1, gamma_2, log10p1sat, log10p2sat], axis=1)
            elif self.training:
                p1sat = 10**log10p1sat
                p2sat = 10**log10p2sat
                P1 = p1sat * x_1 * gamma_1
                P2 = p2sat * x_2 * gamma_2
                P = P1 + P2
                y_1 = P1 / P
                y_2 = P2 / P
                log10P = torch.log10(P)
                output = torch.cat([y_1, y_2, log10P], axis=1)
                if self.vle_inf_dilution:
                    output = torch.cat([output, gamma_1], axis=1)
            else: # not training
                p1sat = 10**log10p1sat
                p2sat = 10**log10p2sat
                P1 = p1sat * x_1 * gamma_1
                P2 = p2sat * x_2 * gamma_2
                P = P1 + P2
                y_1 = P1 / P
                y_2 = P2 / P
                log10P = torch.log10(P)
                output = torch.cat([y_1, y_2, log10P, gamma_1, gamma_2, log10p1sat, log10p2sat], axis=1)

        # VP
        if self.vp is not None and self.vle is None:
            output = forward_vp(self.vp, output, output_temperature_batch, Tc, log10Pc)

        return output
