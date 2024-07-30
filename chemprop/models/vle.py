import torch
import torch.nn as nn
import numpy as np


def forward_vle_basic(
        output: torch.Tensor,
):
    """
    vle output calculation for the basic vle model
    """
    logity_1, log10P = torch.chunk(output, 2, dim=1)
    y_1 = nn.functional.sigmoid(logity_1)
    y_2 = 1 - y_1
    output = torch.cat([y_1, y_2, log10P], axis=1)
    return output


def forward_vle_activity(
        output: torch.Tensor,
        fugacity_balance: str,
        x_1: torch.Tensor,
        x_2: torch.Tensor,
        log10p1sat: torch.Tensor,
        log10p2sat: torch.Tensor,
):
    """
    vle output calculation for the activity coefficient model
    """
    gamma_1 = torch.exp(output[:,[0]])
    gamma_2 = torch.exp(output[:,[1]])
    if fugacity_balance is None:
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
        output = torch.cat([gamma_1, gamma_2, log10p1sat, log10p2sat], axis=1)
    return output


def forward_vle_wohl(
        output: torch.Tensor,
        wohl_order: int,
        fugacity_balance: str,
        x_1: torch.Tensor,
        x_2: torch.Tensor,
        q_1: torch.Tensor,
        q_2: torch.Tensor,
        log10p1sat: torch.Tensor,
        log10p2sat: torch.Tensor,
):
    """
    """
    z_1 = q_1 * x_1 / (q_1 * x_1 + q_2 * x_2)
    z_2 = q_2 * x_2 / (q_1 * x_1 + q_2 * x_2)

    if wohl_order == 3:
        a12, a112, a122 = torch.chunk(output, 3, dim=1)
        gamma_1 = torch.exp(
            2 * a12 * z_2**2 * q_1 +
            6 * a112 * z_1 * z_2**2 * q_1 -
            3 * a122 * z_1 * z_2**2 * q_1 +
            3 * a122 * z_2**3 * q_1
        )
        gamma_2 = torch.exp(
            2 * a12 * z_1**2 * q_2 +
            3 * a112 * z_1**3 * q_2 -
            3 * a112 * z_1**2 * z_2 * q_2 +
            6 * a122 * z_1**2 * z_2 * q_2
        )
    elif wohl_order == 4:
        a12, a112, a122, a1112, a1222, a1122 = torch.chunk(output, 6, dim=1)
        gamma_1 = torch.exp(
            2 * a12 * z_2**2 * q_1 +
            6 * a112 * z_1 * z_2**2 * q_1 -
            3 * a122 * z_1 * z_2**2 * q_1 +
            3 * a122 * z_2**3 * q_1 +
            12 * a1112 * z_1**2 * q_1 * z_2**2 * q_2 +
            4 * a1222 * q_1 * z_2**4 -
            8 * a1222 * z_1 * q_1 * z_2**3 +
            12 * a1122 * z_1 * q_1 * z_2**3 -
            6 * a1122 * z_1**2 * q_1 * z_2**2
        )
        gamma_2 = torch.exp(
            2 * a12 * z_1**2 * q_2 +
            3 * a112 * z_1**3 * q_2 -
            3 * a112 * z_1**2 * z_2 * q_2 +
            6 * a122 * z_1**2 * z_2 * q_2 +
            4 * a1112 * z_1**4 * q_2 -
            8 * a1112 * z_1**3 * z_2 * q_2 +
            12 * a1122 * z_1**2 * z_2**2 * q_2 +
            12 * a1222 * z_1**3 * z_2 * q_2 -
            6 * a1222 * z_1**2 * z_2**2 * q_2
        )
    elif wohl_order == 5:
        a12, a112, a122, a1112, a1222, a1122, a11112, a11122, a11222, a12222 = torch.chunk(output, 10, dim=1)
        gamma_1 = torch.exp(
            2 * a12 * z_2**2 * q_1 +
            6 * a112 * z_1 * z_2**2 * q_1 -
            3 * a122 * z_1 * z_2**2 * q_1 +
            3 * a122 * z_2**3 * q_1 +
            12 * a1112 * z_1**2 * q_1 * z_2**2 * q_2 +
            4 * a1222 * q_1 * z_2**4 -
            8 * a1222 * z_1 * q_1 * z_2**3 +
            12 * a1122 * z_1 * q_1 * z_2**3 -
            6 * a1122 * z_1**2 * q_1 * z_2**2 +
            20 * a11112 * z_1**3 * q_1 * z_2**2 +
            30 * a11122 * z_1**2 * q_1 * z_2**3 -
            10 * a11122 * z_1**3 * q_1 * z_2**2 +
            20 * a11222 * z_1 * q_1 * z_2**4 -
            20 * a11222 * z_1**2 * q_1 * z_2**3 +
            5 * a12222 * q_1 * z_2**5 -
            15 * a12222 * z_1 * q_1 * z_2**4
        )
        gamma_2 = torch.exp(
            2 * a12 * z_1**2 * q_2 +
            3 * a112 * z_1**3 * q_2 -
            3 * a112 * z_1**2 * z_2 * q_2 +
            6 * a122 * z_1**2 * z_2 * q_2 +
            4 * a1112 * z_1**4 * q_2 -
            8 * a1112 * z_1**3 * z_2 * q_2 +
            12 * a1122 * z_1**2 * z_2**2 * q_2 +
            12 * a1222 * z_1**3 * z_2 * q_2 -
            6 * a1222 * z_1**2 * z_2**2 * q_2 +
            5 * a11112 * z_1**5 * q_2 -
            15 * a11112 * z_1**4 * z_2 * q_2 +
            10 * a11122 * z_1**4 * z_2 * q_2 -
            10 * a11122 * z_1**3 * z_2**2 * q_2 +
            30 * a11222 * z_1**2 * z_2**3 * q_2 -
            10 * a11222 * z_1**3 * z_2**2 * q_2 +
            20 * a12222 * z_1**2 * z_2**3 * q_2
        )
    else:
        raise NotImplementedError(f"Wohl order {wohl_order} not supported")

    if fugacity_balance is None:
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
        output = torch.cat([gamma_1, gamma_2, log10p1sat, log10p2sat], axis=1)

    return output