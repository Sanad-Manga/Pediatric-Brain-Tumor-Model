import plotly.graph_objects as go
import numpy as np

def create_slice_heatmap(volume_slice, mask_slice=None):
    """Generates an interactive Plotly heatmap for 2D axial/coronal/sagittal slices with mask overlays."""
    fig = go.Figure(data=go.Heatmap(
        z=volume_slice,
        colorscale='Gray',
        showscale=False
    ))
    
    if mask_slice is not None:
        fig.add_trace(go.Heatmap(
            z=mask_slice,
            colorscale='Reds',
            opacity=0.5,
            showscale=False
        ))
        
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=10),
        height=350
    )
    return fig