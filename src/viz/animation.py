"""Animated trajectory visualization for cislunar space."""

import numpy as np
import plotly.graph_objects as go

def animate_trajectory(sol, L_points=None, title="Cislunar Trajectory Animation", 
                       fps=30, duration=10):
    """
    Create animated trajectory showing evolution over time.
    
    Parameters:
        sol: solve_ivp result
        L_points: Lagrange points dict
        fps: frames per second
        duration: animation duration in seconds
    Returns:
        plotly Figure with animation
    """
    n_frames = fps * duration
    frame_indices = np.linspace(0, len(sol.t)-1, n_frames, dtype=int)
    
    # Create frames
    frames = []
    for idx in frame_indices:
        frame_data = []
        
        # Earth
        frame_data.append(go.Scatter3d(
            x=[-0.012150585609624], y=[0], z=[0],
            mode='markers', marker=dict(size=15, color='blue'),
            name='Earth'
        ))
        
        # Moon
        frame_data.append(go.Scatter3d(
            x=[1-0.012150585609624], y=[0], z=[0],
            mode='markers', marker=dict(size=8, color='gray'),
            name='Moon'
        ))
        
        # Lagrange points
        if L_points:
            lx, ly, lz, labels = [], [], [], []
            for name, pos in L_points.items():
                lx.append(pos[0]); ly.append(pos[1]); lz.append(pos[2])
                labels.append(name)
            frame_data.append(go.Scatter3d(
                x=lx, y=ly, z=lz, mode='markers+text',
                marker=dict(size=6, color='red'), text=labels,
                name='Lagrange Points'
            ))
        
        # Trajectory up to current time
        frame_data.append(go.Scatter3d(
            x=sol.y[0][:idx+1], y=sol.y[1][:idx+1], z=sol.y[2][:idx+1],
            mode='lines', line=dict(width=3, color='green'),
            name='Path'
        ))
        
        # Current position
        frame_data.append(go.Scatter3d(
            x=[sol.y[0][idx]], y=[sol.y[1][idx]], z=[sol.y[2][idx]],
            mode='markers', marker=dict(size=10, color='yellow', symbol='diamond'),
            name='Current Position'
        ))
        
        # Velocity vector
        scale = 0.5
        frame_data.append(go.Scatter3d(
            x=[sol.y[0][idx], sol.y[0][idx] + scale*sol.y[3][idx]],
            y=[sol.y[1][idx], sol.y[1][idx] + scale*sol.y[4][idx]],
            z=[sol.y[2][idx], sol.y[2][idx] + scale*sol.y[5][idx]],
            mode='lines', line=dict(width=4, color='orange'),
            name='Velocity'
        ))
        
        frames.append(go.Frame(data=frame_data, name=f'frame{idx}'))
    
    # Initial figure
    fig = go.Figure(
        data=frames[0].data,
        frames=frames
    )
    
    # Animation controls
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='x', range=[-1.5, 1.5]),
            yaxis=dict(title='y', range=[-1.5, 1.5]),
            zaxis=dict(title='z', range=[-0.5, 0.5]),
            aspectmode='cube',
            bgcolor='rgb(10,10,10)'
        ),
        paper_bgcolor='rgb(5,5,5)',
        plot_bgcolor='rgb(5,5,5)',
        font=dict(color='white'),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': '▶ Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 1000/fps, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 0}
                    }]
                },
                {
                    'label': '⏸ Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ],
            'x': 0.1, 'y': 0.05
        }],
        sliders=[{
            'active': 0,
            'yanchor': 'top', 'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 16},
                'prefix': 'Time:',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 1000/fps, 'easing': 'cubic-in-out'},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1, 'y': 0,
            'steps': [
                {
                    'args': [[f'frame{frame_indices[i]}'], {
                        'frame': {'duration': 1000/fps, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 1000/fps}
                    }],
                    'label': f'{sol.t[frame_indices[i]]:.1f}',
                    'method': 'animate'
                }
                for i in range(len(frame_indices))
            ]
        }],
        width=900, height=700
    )
    
    return fig
