"""
Alg.:
Given: Network f, Image X [T,32,32]
Return: Attacking probabilities P [T,32,32]
P <- X # Initialize the probabilities with the spikes in the image
P <- P + eps # Add some random initial noise to the probabilities to avoid sampling X the whole time
for i in range(N_{attack steps}):
    g <- [0] # Initialize gradient to zero
    for j in range(N_{MC}):
        e <- U[0,1] # Sample from standard uniform distribution
        X_j <- g(e,P,T) # Apply reparameterization trick using sampled random, current probabilities, and temperature 
        g <- g + 1/N_{MC} * grad( loss(y,round(X_j)) ) w.r.t. X_j # Compute gradient of loss under sampled input
    g_transform <- arg max_{v, |v|_p <= 1} vTg # This is just sign(g) if we are in l-infinity or g/norm2(g) if we are in l2
    P <- Project(P + alpha * g_transform) # Step of projected gradient ascent (PGA)
return P
"""

import numpy as np
import torch

def spike_prob():
    
