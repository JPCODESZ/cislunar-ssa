"""Conjunction screening between objects."""

import numpy as np

def screen_conjunctions(states1, states2, threshold=0.01):
    """
    Find close approaches between two sets of objects.
    
    Parameters:
        states1, states2: arrays of [x,y,z,vx,vy,vz]
        threshold: distance threshold (non-dimensional)
    Returns:
        list of (i, j, distance) tuples
    """
    conjunctions = []
    for i, s1 in enumerate(states1):
        for j, s2 in enumerate(states2):
            dist = np.sqrt((s1[0]-s2[0])**2 + (s1[1]-s2[1])**2 + (s1[2]-s2[2])**2)
            if dist < threshold:
                conjunctions.append((i, j, dist))
    return conjunctions
