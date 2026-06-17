"""Photon-statistics model helpers used by the demo notebooks.

This module intentionally keeps only the physical-model helpers that are used
by the maintained ``demo_*`` notebooks. Older exploratory QuTiP models were
removed because they were not covered by tests and some depended on undefined
state constructors.

The closed-form one-boson probabilistic-addition helper works without QuTiP.
The coherent-state density-matrix helper requires the optional physics extra:
``pip install .[physics]``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import factorial


def _require_qutip():
    """Return the qutip module or raise a clear optional-dependency error."""
    try:
        import qutip as qt_mod
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "QuTiP is required for this model. Install with: pip install .[physics]"
        ) from exc
    return qt_mod


def test_P_pba(mu: float, p: float, l: float, k: int) -> float:
    """Closed-form probability ``P_k`` for the one-boson-addition demo.

    The parameter ``l`` is preserved for compatibility with the original
    notebooks; the current closed-form expression does not use it.
    """
    return (np.exp(-mu) / factorial(k)) * (
        (mu ** (k - 1) * p * k**2 / (mu + 1)) + (1 - p) * mu**k
    )


test_P_pba.__test__ = False


def state_dm_pba_co(Ns: int, alpha: float, p: float, nad: int = 1):
    """Probabilistically boson-added coherent-state density matrix.

    Each addition step mixes the normalized photon-added state with the
    previous state: ``rho -> p * unit(a.dag() rho a) + (1-p) * rho``.
    """
    qt_mod = _require_qutip()
    a = qt_mod.destroy(Ns)
    rho = qt_mod.coherent_dm(Ns, alpha, method="analytic")

    for _ in range(nad):
        added = a.dag() * rho * a
        rho = p * added.unit() + (1 - p) * rho

    return rho.unit()


def pba_Pk_co(Ns: int, mu: float, p: float, k: int, nad: int = 1) -> float:
    """Photon-number probability ``P_k`` for the coherent PBA demo model."""
    qt_mod = _require_qutip()
    alpha = np.sqrt(mu)
    state = state_dm_pba_co(Ns, alpha, p, nad=nad)
    ket = qt_mod.states.fock(Ns, k)
    return float(np.real(ket.dag() * state * ket))
