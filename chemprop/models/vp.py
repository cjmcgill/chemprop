import torch
import torch.nn as nn
import numpy as np


def forward_vp(
    vp: str,
    output: torch.Tensor,
    temperature: torch.Tensor,
    Tc: torch.Tensor = None,
    log10Pc: torch.Tensor = None,
):
    """
    
    """
    if vp == "basic":
        pass
    elif vp == "antoine":
        antoine_a, antoine_b, antoine_c = torch.chunk(output, 3, dim=1)
        antoine_b = nn.functional.softplus(antoine_b)
        output = antoine_a - (antoine_b / torch.abs(antoine_c + temperature))
    elif vp == "four_var":
        antoine_a, antoine_b, antoine_c, antoine_d = torch.chunk(output, 4, dim=1)
        antoine_b = nn.functional.softplus(antoine_b)
        antoine_c = nn.functional.softplus(antoine_c)
        antoine_d = nn.functional.softplus(antoine_d)
        output = antoine_a - (antoine_b / temperature) - (antoine_c * torch.log(temperature)) - (antoine_d * torch.pow(temperature, 6))
    elif vp == "five_var":
        antoine_a, antoine_b, antoine_c, antoine_d, antoine_e = torch.chunk(output, 5, dim=1)
        antoine_b = nn.functional.softplus(antoine_b)
        antoine_c = nn.functional.softplus(antoine_c)
        antoine_d = nn.functional.softplus(antoine_d)
        antoine_e = nn.functional.softplus(antoine_e)
        output = antoine_a - (antoine_b / temperature) - (antoine_c * torch.log(temperature)) - (antoine_d * torch.pow(temperature, antoine_e))
    elif vp == "simplified":
        antoine_a, antoine_b = torch.chunk(output, 2, dim=1)
        antoine_b = nn.functional.softplus(antoine_b)
        output = antoine_a - (antoine_b / temperature)
    elif vp in ["ambrose4","ambrose5"]:
        tau = 1 - temperature / Tc
        if vp == "ambrose4":
            ambrose_a, ambrose_b, ambrose_c, ambrose_d = torch.chunk(output, 4, dim=1)
            log10Pr = (ambrose_a * tau + ambrose_b * tau**1.5 + ambrose_c * tau**2.5 + ambrose_d * tau**5) / (1 - tau)
        elif vp == "ambrose5":
            ambrose_a, ambrose_b, ambrose_c, ambrose_d, ambrose_e = torch.chunk(output, 5, dim=1)
            log10Pr = (ambrose_a * tau + ambrose_b * tau**1.5 + ambrose_c * tau**2.5 + ambrose_d * tau**5 + ambrose_e * tau**6) / (1 - tau)
        output = log10Pc + log10Pr
    elif vp in ["riedel4","riedel5"]:
        Tr = temperature / Tc
        if vp == "riedel4":
            riedel_a, riedel_b, riedel_c, riedel_d = torch.chunk(output, 4, dim=1)
            riedel_b = nn.functional.softplus(riedel_b)
            riedel_c = nn.functional.softplus(riedel_c)
            riedel_d = nn.functional.softplus(riedel_d)
            log10Pr = (riedel_a - riedel_b / Tr - riedel_c * torch.log(Tr) - riedel_d * Tr**6)
        elif vp == "riedel5":
            riedel_a, riedel_b, riedel_c, riedel_d, riedel_e = torch.chunk(output, 5, dim=1)
            riedel_b = nn.functional.softplus(riedel_b)
            riedel_c = nn.functional.softplus(riedel_c)
            riedel_d = nn.functional.softplus(riedel_d)
            riedel_e = nn.functional.softplus(riedel_e)
            log10Pr = (riedel_a - riedel_b / Tr - riedel_c * torch.log(Tr) - riedel_d * Tr**riedel_e)
        output = log10Pc + log10Pr
    return output


def get_vp_parameter_names(
        vp: str,
        molecule_id: int = None,
):
    """
    Get the coefficients for the vapor pressure model. If there are multiple molecules
    in a mixture for vle prediction, the coefficients are suffixed with the molecule ID.
    """
    if vp == "antoine":
        names = ["antoine_a", "antoine_b", "antoine_c"]
    elif vp == "four_var":
        names = ["antoine_a", "antoine_b", "antoine_c", "antoine_d"]
    elif vp == "five_var":
        names = ["antoine_a", "antoine_b", "antoine_c", "antoine_d", "antoine_e"]
    elif vp == "simplified":
        names = ["antoine_a", "antoine_b"]
    elif vp == "ambrose4":
        names = ["ambrose_a", "ambrose_b", "ambrose_c", "ambrose_d"]
    elif vp == "ambrose5":
        names = ["ambrose_a", "ambrose_b", "ambrose_c", "ambrose_d", "ambrose_e"]
    elif vp == "riedel4":
        names = ["riedel_a", "riedel_b", "riedel_c", "riedel_d"]
    elif vp == "riedel5":
        names = ["riedel_a", "riedel_b", "riedel_c", "riedel_d", "riedel_e"]
    else:
        raise NotImplementedError(f"Vapor pressure model {vp} not supported")
    if molecule_id is not None:
        names = [f"{name}_{molecule_id}" for name in names]
    return names


def unscale_vp_parameters(
        parameters: np.ndarray,
        target_scaler,
        hybrid_model_features_scaler,
        vle: str,
        vp: str,
):
    """
    Unscale the vapor pressure parameters.
    """
    if vle is not None:
        mean_p, std_p = target_scaler.means[2], target_scaler.stds[2]
        scale_t = hybrid_model_features_scaler.stds[2]
    else:
        mean_p, std_p = target_scaler.means[0], target_scaler.stds[0]
        scale_t = hybrid_model_features_scaler.stds[0]
    if vp == "antoine":
        antoine_a, antoine_b, antoine_c = np.split(parameters, 3, axis=1)
        antoine_a = std_p * antoine_a - mean_p
        antoine_b = std_p * scale_t * antoine_b
        antoine_c = scale_t * antoine_c
        parameters = np.concatenate([antoine_a, antoine_b, antoine_c], axis=1)
    elif vp == "four_var":
        antoine_a, antoine_b, antoine_c, antoine_d = np.split(parameters, 4, axis=1)
        antoine_b = std_p * scale_t * antoine_b
        antoine_c = scale_t * antoine_c
        antoine_d = std_p * antoine_d / scale_t ** 6
        antoine_a = std_p * antoine_a - antoine_c * np.log10(scale_t) + mean_p
        parameters = np.concatenate([antoine_a, antoine_b, antoine_c, antoine_d], axis=1)
    elif vp == "five_var":
        antoine_a, antoine_b, antoine_c, antoine_d, antoine_e = np.split(parameters, 5, axis=1)
        antoine_b = std_p * scale_t * antoine_b
        antoine_c = scale_t * antoine_c
        antoine_d = std_p * antoine_d / scale_t ** antoine_e
        antoine_a = std_p * antoine_a - antoine_c * np.log10(scale_t) + mean_p
        parameters = np.concatenate([antoine_a, antoine_b, antoine_c, antoine_d, antoine_e], axis=1)
    elif vp == "simplified":
        antoine_a, antoine_b = np.split(parameters, 2, axis=1)
        antoine_a = std_p * antoine_a - mean_p
        antoine_b = std_p * scale_t * antoine_b
        parameters = np.concatenate([antoine_a, antoine_b], axis=1)
    elif vp in ["ambrose4", "ambrose5"]:
        parameters = std_p * parameters
    elif vp in ["riedel4", "riedel5"]:
        parameters = std_p * parameters[:, :5] # if riedel5, don't unscale the riedel_e parameter
    else:
        raise NotImplementedError(f"Vapor pressure model {vp} not supported for unscaling vp parameters")
    return parameters