import plotly.graph_objects as px
import plotly.io as pio
from PIL import Image
from model_detection import ModelDetection

class IntersectionResult:
    def __init__(
        self,
        point_elbow,
        point_shoulder,
        plane,
        w_coord_human,
        inter_elbow,
        inter_shoulder,
        vector_e,
        vector_s,
    ):
        self.point_elbow = point_elbow
        self.point_shoulder = point_shoulder
        self.plane = plane
        self.w_coord_human = w_coord_human
        self.inter_elbow = inter_elbow
        self.inter_shoulder = inter_shoulder
        self.vector_e = vector_e
        self.vector_s = vector_s

    def create_fig(self):
        fig = px.Figure()

        world_landmarks_diagram(self.w_coord_human, fig)
        a = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        # for i, j in enumerate(self.plane):
        #     fig.add_annotation(
        #         x=j[0], y=j[1], z=j[2],
        #         text=a[i],
        #         showarrow=False
        #     )

        if self.inter_elbow is not None:
            fig.add_trace(px.Scatter3d(
                x=[self.inter_elbow[0]],
                y=[self.inter_elbow[1]],
                z=[self.inter_elbow[2]],
                mode='markers',
                marker=dict(color='purple', size=5)
            ))
            fig.add_trace(px.Cone(
                x=[self.point_elbow[0]], y=[self.point_elbow[1]], z=[self.point_elbow[2]],
                u=[self.vector_e[0]], v=[self.vector_e[1]], w=[self.vector_e[2]],
                sizemode='absolute', sizeref=0.05,
                anchor='tail',
                showscale=False,
                colorscale=[[0, 'red'], [1, 'red']]
            ))
        elif self.inter_shoulder is not None:
            fig.add_trace(px.Scatter3d(
                x=[self.inter_shoulder[0]],
                y=[self.inter_shoulder[1]],
                z=[self.inter_shoulder[2]],
                mode='markers',
                marker=dict(color='purple', size=5)
            ))
            fig.add_trace(px.Cone(
                x=[self.inter_shoulder[0]], y=[self.inter_shoulder[1]], z=[self.inter_shoulder[2]],
                u=[self.vector_s[0]], v=[self.vector_s[1]], w=[self.vector_s[2]],
                sizemode='absolute', sizeref=0.05,
                anchor='tail',
                showscale=False,
                colorscale=[[0, 'red'], [1, 'red']]
            ))

        fig.update_layout(scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data',
            xaxis=dict(range=[-1,1]),
            yaxis=dict(range=[-1,1]),
            zaxis=dict(range=[-1,1]),
        ))

        return fig

    def get_image(self):
        print("1")
        fig = self.create_fig()
        print("2")
        image_bytes = fig.to_image(format='png', width=10, height=10)
        print("3")
        image = Image.open(pio.BytesIO(image_bytes))
        print("4")
        image_rgb = image.convert('RGB')
        print("5")
        return image_rgb

def world_landmarks_diagram(w_coord_human, fig):
    w_coord_transposed = list(map(list, zip(*w_coord_human)))
    fig.add_trace(px.Scatter3d(
        x=w_coord_transposed[0],
        y=w_coord_transposed[1],
        z=w_coord_transposed[2],
        mode='markers',
        marker=dict(color='skyblue', size=5)
    ))

    for c in ModelDetection.POSE_CONNECTIONS:
        fig.add_trace(px.Scatter3d(
            x=[w_coord_human[c[0]][0], w_coord_human[c[1]][0]],
            y=[w_coord_human[c[0]][1], w_coord_human[c[1]][1]],
            z=[w_coord_human[c[0]][2], w_coord_human[c[1]][2]],
            mode='lines',
            line=dict(color='skyblue')
        ))
