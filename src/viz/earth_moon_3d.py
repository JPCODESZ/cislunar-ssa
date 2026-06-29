"""3D interactive visualization of cislunar space."""

import numpy as np
import plotly.graph_objects as go

def plot_3d_system(sol=None, L_points=None, title="Cislunar Space"):
    """
    Interactive 3D plot of Earth-Moon system with optional trajectory.
    
    Parameters:
        sol: solve_ivp result (optional)
        L_points: dict of Lagrange points (optional)
        title: plot title
    Returns:
        plotly Figure
    """
    fig = go.Figure()
    
    # Earth
    fig.add_trace(go.Scatter3d(
        x=[-0.012150585609624], y=[0], z=[0],
        mode='markers',
        marker=dict(size=10, color='blue'),
        name='Earth'
    ))
    
    # Moon
    fig.add_trace(go.Scatter3d(
        x=[1 - 0.012150585609624], y=[0], z=[0],
        mode='markers',
        marker=dict(size=5, color='gray'),
        name='Moon'
    ))
    
    # Lagrange points
    if L_points:
        lx, ly, lz = [], [], []
        labels = []
        for name, pos in L_points.items():
            lx.append(pos[0])
            ly.append(pos[1])
            lz.append(pos[2])
            labels.append(name)
        
        fig.add_trace(go.Scatter3d(
            x=lx, y=ly, z=lz,
            mode='markers+text',
            marker=dict(size=4, color='red'),
            text=labels,
            textposition='top center',
            name='Lagrange Points'
        ))
    
    # Trajectory
    if sol:
        fig.add_trace(go.Scatter3d(
            x=sol.y[0], y=sol.y[1], z=sol.y[2],
            mode='lines',
            line=dict(width=2, color='green'),
            name='Trajectory'
        ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z',
            aspectmode='data'
        ),
        width=800,
        height=600
    )
    
    return fig

def plot_debris_cloud(catalog, title="Debris Cloud"):
    """
    3D scatter plot of debris catalog.
    """
    states = catalog['state']
    
    fig = go.Figure(data=[go.Scatter3d(
        x=states[:, 0],
        y=states[:, 1],
        z=states[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=catalog['size'],
            colorscale='Viridis',
            opacity=0.6
        ),
        name='Debris'
    )])
    
    # Earth
    fig.add_trace(go.Scatter3d(
        x=[-0.012150585609624], y=[0], z=[0],
        mode='markers',
        marker=dict(size=10, color='blue'),
        name='Earth'
    ))
    
    # Moon
    fig.add_trace(go.Scatter3d(
        x=[1 - 0.012150585609624], y=[0], z=[0],
        mode='markers',
        marker=dict(size=5, color='gray'),
        name='Moon'
    ))
    
    fig.update_layout(
        title=title,
        scene=dict(aspectmode='data'),
        width=800,
        height=600
    )
    
    return fig
