import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.backends.backend_agg import FigureCanvasAgg
from model_detection import ModelDetection
import numpy as np
import cv2


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

    def create_fig(self, fig):
        plt.clf()
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(280, 270)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        w_coord_transposed = list(map(list, zip(*self.w_coord_human)))
        ax.scatter3D(
            w_coord_transposed[0],
            w_coord_transposed[1],
            w_coord_transposed[2],
            color="skyblue",
        )

        for c in ModelDetection.POSE_CONNECTIONS:
            ax.plot(
                [self.w_coord_human[c[0]][0], self.w_coord_human[c[1]][0]],
                [self.w_coord_human[c[0]][1], self.w_coord_human[c[1]][1]],
                [self.w_coord_human[c[0]][2], self.w_coord_human[c[1]][2]],
                color="skyblue",
            )
        ax.add_collection3d(Poly3DCollection([self.plane], facecolor="g", alpha=0.3))
        a = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        for i, j in enumerate(self.plane):
            ax.text(j[0], j[1], j[2], a[i])
        if self.inter_elbow is not None:
            ax.scatter3D(
                self.inter_elbow[0],
                self.inter_elbow[1],
                self.inter_elbow[2],
                color="purple",
            )
            ax.text(self.inter_elbow[0], self.inter_elbow[1], self.inter_elbow[2], "M")
            ax.quiver(
                self.point_elbow[0],
                self.point_elbow[1],
                self.point_elbow[2],
                self.vector_e[0],
                self.vector_e[1],
                self.vector_e[2],
                color="red",
            )
        elif self.inter_shoulder is not None:
            ax.scatter3D(
                self.inter_shoulder[0],
                self.inter_shoulder[1],
                self.inter_shoulder[2],
                color="purple",
            )
            ax.text(
                self.inter_shoulder[0],
                self.inter_shoulder[1],
                self.inter_shoulder[2],
                "M",
            )
            ax.quiver(
                self.inter_shoulder[0],
                self.inter_shoulder[1],
                self.inter_shoulder[2],
                self.vector_s[0],
                self.vector_s[1],
                self.vector_s[2],
                color="red",
            )
        return fig

    def plot_chart(self, figure):
        fig = self.create_fig(figure)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        plot_image = np.asarray(buf)
        plot_image = cv2.cvtColor(plot_image, cv2.COLOR_RGBA2RGB)
        plt.clf()
        return plot_image
