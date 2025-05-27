from pypsa import Network
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def patched_get_switchable_as_dense(n, component, attr, snapshots=None, inds=None):
    """
    PyPSA v0.28.0-compatible version of get_switchable_as_dense with safe fallbacks.
    """
    df = n.df(component)
    pnl = n.pnl(component)

    if snapshots is None:
        snapshots = n.snapshots

    index = df.index

    # handle case where attr not in df or pnl
    if attr in pnl:
        varying_i = pnl[attr].columns
    else:
        varying_i = pd.Index([])

    fixed_i = df.index.difference(varying_i)

    if inds is not None:
        index = index.intersection(inds)
        varying_i = varying_i.intersection(inds)
        fixed_i = fixed_i.intersection(inds)

    # default fallback for missing static attr
    if attr in df.columns:
        fixed_vals = df.loc[fixed_i, attr]
    else:
        logger.warning(f"[PATCHED] Attribute '{attr}' not in static component df of {component}; using default value of 1.0")
        fixed_vals = pd.Series(1.0, index=fixed_i)

    static = pd.DataFrame(
        np.repeat([fixed_vals.values], len(snapshots), axis=0),
        index=snapshots,
        columns=fixed_i,
    )

    if attr in pnl:
        varying = pnl[attr].reindex(index=snapshots, columns=varying_i)
    else:
        varying = pd.DataFrame(index=snapshots, columns=varying_i)

    result = pd.concat([static, varying], axis=1)
    result = result.reindex(columns=index)
    result.index.name = "snapshot"
    result.columns.name = component

    return result

# Patch into Network
Network.get_switchable_as_dense = patched_get_switchable_as_dense
print("[INFO] Successfully patched get_switchable_as_dense (PyPSA v0.28.0 safe mode)")