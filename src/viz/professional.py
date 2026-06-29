"""Professional-grade 3D visualization for cislunar SSA."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Real proportions (non-dimensional units)
EARTH_RADIUS_ND = 6378.137 / 384400.0  # ~0.0166
MOON_RADIUS_ND = 1737.4 / 384400.0   # ~0.0045

def plot_cislunar_system(sol=None, L_points=None, debris=None, 
                         title="Cislunar Space Situational Awareness",
                         show_hill_sphere=True):
    """
    Professional 3D cislunar visualization with proper scaling.
    """
    fig = go.Figure()
    
    # Earth - properly sized, blue with atmosphere glow
    fig.add_trace(go.Scatter3d(
        x=[-0.012150585609624], y=[0], z=[0],
        mode='markers',
        marker=dict(
            size=15,
            color='rgb(30, 100, 200)',
            opacity=0.9,
            line=dict(color='rgb(100, 180, 255)', width=2)
        ),
        name='Earth',
        hovertemplate='Earth<br>x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>'
    ))
    
    # Moon - properly sized, gray
    fig.add_trace(go.Scatter3d(
        x=[1 - 0.012150585609624], y=[0], z=[0],
        mode='markers',
        marker=dict(
            size=8,
            color='rgb(180, 180, 180)',
            opacity=0.9,
            line=dict(color='rgb(220, 220, 220)', width=1)
        ),
        name='Moon',
        hovertemplate='Moon<br>x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>'
    ))
    
    # Lagrange points - labeled, colored by stability
    if L_points:
        lx, ly, lz = [], [], []
        labels = []
        colors = []
        
        for name, pos in L_points.items():
            lx.append(pos[0])
            ly.append(pos[1])
            lz.append(pos[2])
            labels.append(name)
            # L1, L2, L3 are unstable (red), L4, L5 are stable (green)
            colors.append('rgb(255, 80, 80)' if name in ['L1', 'L2', 'L3'] else 'rgb(80, 200, 80)')
        
        fig.add_trace(go.Scatter3d(
            x=lx, y=ly, z=lz,
            mode='markers+text',
            marker=dict(size=6, color=colors, opacity=0.8),
            text=labels,
            textposition='top center',
            textfont=dict(size=10, color='white'),
            name='Lagrange Points',
            hovertemplate='%{text}<br>x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>'
        ))
    
    # Trajectory with gradient color (velocity magnitude)
    if sol:
        # Calculate velocity magnitude for coloring
        v = np.sqrt(sol.y[3]**2 + sol.y[4]**2 + sol.y[5]**2)
        
        fig.add_trace(go.Scatter3d(
            x=sol.y[0], y=sol.y[1], z=sol.y[2],
            mode='lines',
            line=dict(
                width=3,
                color=v,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Velocity', thickness=10)
            ),
            name='Trajectory',
            hovertemplate='t: %{text}<br>v: %{marker.color:.4f}<extra></extra>',
            text=[f't={t:.2f}' for t in sol.t]
        ))
        
        # Start point
        fig.add_trace(go.Scatter3d(
            x=[sol.y[0][0]], y=[sol.y[1][0]], z=[sol.y[2][0]],
            mode='markers',
            marker=dict(size=8, color='green', symbol='diamond'),
            name='Start',
            hovertemplate='Start<br>x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>'
        ))
        
        # End point
        fig.add_trace(go.Scatter3d(
            x=[sol.y[0][-1]], y=[sol.y[1][-1]], z=[sol.y[2][-1]],
            mode='markers',
            marker=dict(size=8, color='red', symbol='x'),
            name='End',
            hovertemplate='End<br>x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>'
        ))
    
    # Debris cloud
    if debris:
        states = debris['state']
        sizes = debris['size']
        
        fig.add_trace(go.Scatter3d(
            x=states[:, 0],
            y=states[:, 1],
            z=states[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=sizes,
                colorscale='Hot',
                opacity=0.4,
                showscale=True,
                colorbar=dict(title='Size (m)', thickness=10, x=0.9)
            ),
            name='Debris',
            hovertemplate='Debris %{text}<br>size: %{marker.color:.3f} m<extra></extra>',
            text=[f'#{i}' for i in range(len(states))]
        ))
    
    # Hill sphere approximation (wireframe sphere around Earth)
    if show_hill_sphere:
        theta = np.linspace(0, 2*np.pi, 50)
        phi = np.linspace(0, np.pi, 25)
        theta, phi = np.meshgrid(theta, phi)
        
        # Hill sphere radius ~0.069 non-dimensional
        r_hill = 0.069
        x_earth = -0.012150585609624
        
        x_hill = x_earth + r_hill * np.sin(phi) * np.cos(theta)
        y_hill = r_hill * np.sin(phi) * np.sin(theta)
        z_hill = r_hill * np.cos(phi)
        
        fig.add_trace(go.Surface(
            x=x_hill, y=y_hill, z=z_hill,
            opacity=0.05,
            colorscale=[[0, 'blue'], [1, 'blue']],
            showscale=False,
            name='Earth Hill Sphere',
            hoverinfo='skip'
        ))
    
    # Professional layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color='white'),
            x=0.5
        ),
        scene=dict(
            xaxis=dict(
                title='x (non-dimensional)',
                gridcolor='rgb(50, 50, 50)',
                backgroundcolor='rgb(10, 10, 10)',
                showbackground=True
            ),
            yaxis=dict(
                title='y (non-dimensional)',
                gridcolor='rgb(50, 50, 50)',
                backgroundcolor='rgb(10, 10, 10)',
                showbackground=True
            ),
            zaxis=dict(
                title='z (non-dimensional)',
                gridcolor='rgb(50, 50, 50)',
                backgroundcolor='rgb(10, 10, 10)',
                showbackground=True
            ),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8)
            ),
            bgcolor='rgb(5, 5, 5)'
        ),
        paper_bgcolor='rgb(5, 5, 5)',
        plot_bgcolor='rgb(5, 5, 5)',
        font=dict(color='white'),
        legend=dict(
            x=0.02, y=0.98,
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='rgb(100,100,100)',
            borderwidth=1
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig

def plot_orbital_elements_comparison():
    """
    Compare different orbital environments (LEO, GEO, Cislunar).
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('LEO', 'GEO', 'Cislunar'),
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]]
    )
    
    # LEO - circular orbit
    theta = np.linspace(0, 2*np.pi, 100)
    r_leo = 0.2  # ~2000 km altitude scaled
    fig.add_trace(go.Scatter3d(
        x=r_leo*np.cos(theta), y=r_leo*np.sin(theta), z=np.zeros_like(theta),
        mode='lines', line=dict(color='cyan', width=2),
        name='LEO'
    ), row=1, col=1)
    
    # GEO - circular orbit
    r_geo = 0.5
    fig.add_trace(go.Scatter3d(
        x=r_geo*np.cos(theta), y=r_geo*np.sin(theta), z=np.zeros_like(theta),
        mode='lines', line=dict(color='yellow', width=2),
        name='GEO'
    ), row=1, col=2)
    
    # Cislunar - L1 halo-like
    x_cis = 0.8 + 0.1*np.cos(theta)
    y_cis = 0.1*np.sin(theta)
    z_cis = 0.05*np.sin(2*theta)
    fig.add_trace(go.Scatter3d(
        x=x_cis, y=y_cis, z=z_cis,
        mode='lines', line=dict(color='magenta', width=2),
        name='Cislunar'
    ), row=1, col=3)
    
    fig.update_layout(
        title='Orbital Regime Comparison',
        height=500,
        showlegend=False
    )
    
    return fig
