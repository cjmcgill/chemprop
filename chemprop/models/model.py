from typing import List, Union, Tuple

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn

from .mpn import MPN
from .ffn import build_ffn
from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph
from chemprop.nn_utils import initialize_weights
from .vp import forward_vp
from .vle import forward_vle


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
        self.device = args.device
        self.hidden_size = args.hidden_size
        self.noisy_temperature = args.noisy_temperature
        self.sigmoid = nn.functional.sigmoid
        self.softplus = nn.functional.softplus

        self.relative_output_size = 1

        if self.fugacity_balance is not None:
            if self.vle == "activity":
                self.relative_output_size *= 2/4
            elif self.vle == "wohl":
                if args.wohl_order == 3:
                    self.relative_output_size *= args.wohl_params / 4 #3 parameter for 3rd-order Wohl model
                elif args.wohl_order == 4:
                    self.relative_output_size *= args.wohl_params / 4 #6 parameter for 6th-order Wohl model
                elif args.wohl_order == 5:
                    self.relative_output_size *= args.wohl_params / 4 #10 parameter for 9th-order wohl model
                
        elif self.vle == "basic":
            self.relative_output_size *= 2/3 # gets out y_1 and log10P, but calculates y_2 from it to return three results
        elif self.vle == "activity":
            self.relative_output_size *= 2/3 # uses two activity parameters internally and returns three results
        elif self.vle == "wohl":
            if args.wohl_order == 3:
                self.relative_output_size *= args.wohl_params / 3 #uses three function parameters internally and returns three results
            elif args.wohl_order == 4:
                self.relative_output_size *= args.wohl_params / 3 #for 6th order Wohl
            elif args.wohl_order == 5:
                self.relative_output_size *= args.wohl_params / 3 #for 9th order wohl
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

        # Create FFN layers
        self.readout = build_ffn(
            first_linear_dim=first_linear_dim,
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

        if self.noisy_temperature is not None and self.training:
            # noise is applied to the features temperature not the temperature batch
            noise_batch = np.random.randn(len(features_batch)) * self.noisy_temperature
            input_temperature_batch += noise_batch

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
            encoding_1 = encodings[:,:self.hidden_size] # first molecule
            encoding_1 = torch.concatenate([encoding_1, input_temperature_batch], axis=1) # include T feature at the end
            encoding_2 = encodings[:,self.hidden_size:2*self.hidden_size] # second molecule
            encoding_2 = torch.concatenate([encoding_2, input_temperature_batch], axis=1) # include T feature at the end

        if self.vle == "wohl":
            q_1 = nn.functional.softplus(self.wohl_q(encoding_1))
            q_2 = nn.functional.softplus(self.wohl_q(encoding_2))

        if self.fugacity_balance == "intrinsic_vp":
            vp1_output = self.intrinsic_vp(encoding_1)
            vp2_output = self.intrinsic_vp(encoding_2)
            log10p1sat = forward_vp(self.vp, vp1_output, input_temperature_batch)
            log10p2sat = forward_vp(self.vp, vp2_output, input_temperature_batch)
        elif self.fugacity_balance == "tabulated_vp":
            log10p1sat = hybrid_model_features_batch[:,[3]]
            log10p2sat = hybrid_model_features_batch[:,[4]]
        elif self.vle is not None:
            log10p1sat = hybrid_model_features_batch[:,[3]]
            log10p2sat = hybrid_model_features_batch[:,[4]]
        else:
            log10p1sat = None
            log10p2sat = None

        # VLE models
        if self.vle is not None:
            output = forward_vle(
                vle=self.vle,
                output=output,
                wohl_order=self.wohl_order,
                fugacity_balance=self.fugacity_balance,
                temperature=output_temperature_batch,
                x_1=x_1,
                x_2=x_2,
                q_1=q_1,
                q_2=q_2,
                log10p1sat=log10p1sat,
                log10p2sat=log10p2sat,
            )

        # VP
        if self.vp is not None and self.fugacity_balance is None:
            output = forward_vp(self.vp, output, output_temperature_batch)

        return output
