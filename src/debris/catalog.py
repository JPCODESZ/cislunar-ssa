"""Synthetic debris catalog for cislunar space."""

import numpy as np

def generate_debris_catalog(n_objects=100, region='cislunar', seed=None):
    """
    Generate synthetic debris population.
    
    Parameters:
        n_objects: number of debris objects
        region: 'cislunar', 'leo', or 'geo'
        seed: random seed for reproducibility
    Returns:
        dict with keys: id, state (x,y,z,vx,vy,vz), size, epoch
    """
    if seed is not None:
        np.random.seed(seed)
    
    if region == 'cislunar':
        # Between Earth and Moon, roughly
        x = np.random.uniform(-1.5, 1.5, n_objects)
        y = np.random.uniform(-1.5, 1.5, n_objects)
        z = np.random.uniform(-0.5, 0.5, n_objects)
        vx = np.random.normal(0, 0.1, n_objects)
        vy = np.random.normal(0, 0.1, n_objects)
        vz = np.random.normal(0, 0.05, n_objects)
    else:
        raise NotImplementedError(f"Region '{region}' not yet implemented")
    
    sizes = np.random.lognormal(-2, 1, n_objects)  # meters, roughly
    
    catalog = {
        'id': np.arange(n_objects),
        'state': np.column_stack([x, y, z, vx, vy, vz]),
        'size': sizes,
        'epoch': 0.0
    }
    return catalog

def catalog_to_propagator_states(catalog):
    """Convert catalog states to list for propagator."""
    return [state for state in catalog['state']]
