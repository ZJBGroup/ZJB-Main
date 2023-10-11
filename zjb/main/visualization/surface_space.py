"""
此脚本包含surface空间中应用到的可视化方法
"""
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal
from pyqtgraph.opengl.shaders import FragmentShader, ShaderProgram, VertexShader


class SurfaceViewWidget(gl.GLViewWidget):
    def __init__(self, surface):
        super().__init__()

        # color map
        self.surface = None
        self.md = None
        self.color_map = pg.colormap.get(
            "glasbey_bw_minc_20", source="colorcet"
        )  # tab20
        self.faces = None
        self.vertexes = None
        self._mouse_move = False
        self.setSurface(surface)

    def setSurface(self, surface):
        self.vertexes = surface.vertices  # 顶点
        self.faces = surface.faces  # 三角面

        self.md = gl.MeshData(vertexes=self.vertexes, faces=self.faces)

        self.surface = gl.GLMeshItem(meshdata=self.md, color=(1, 0, 0, 0.2))

        # ShaderProgram.names['ZJBedgeLow']
        self.surface.setShader(
            shader=ShaderProgram(
                "ZJBedgeLow",
                [
                    VertexShader(
                        """
                                    varying vec3 normal;
                                    void main() {
                                        // compute here for use in fragment shader
                                        normal = normalize(gl_NormalMatrix * gl_Normal);
                                        gl_FrontColor = gl_Color;
                                        gl_BackColor = gl_Color;
                                        gl_Position = ftransform();
                                    }
                                """
                    ),
                    FragmentShader(
                        """
                                    varying vec3 normal;
                                    void main() {
                                        vec4 color = gl_Color;
                                        float s = pow(normal.x*normal.x + normal.y*normal.y, 2.0);
                                        color.x = color.x - s * (1.0-color.x)/1.5;
                                        color.y = color.y - s * (1.0-color.y)/1.5;
                                        color.z = color.z - s * (1.0-color.z)/1.5;
                                        gl_FragColor = color;
                                    }
                                """
                    ),
                ],
            )
        )  # 'edgeHilight','pointSprite'
        self.surface.scale(0.06, 0.06, 0.06)  # 初始化大小设置

        self.addItem(self.surface)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self._mouse_move = True


class AtlasSurfaceViewWidget(SurfaceViewWidget):
    def __init__(self, atlas, surface, surface_region_mapping):
        super().__init__(surface)
        self.labels = None
        self.ampl = None
        self._split = False
        regioncolor_list = list(
            np.arange(atlas.labels.shape[0]) / atlas.labels.shape[0]
        )
        self.setRegionColor(surface_region_mapping, regioncolor_list)

    def setRegionColor(self, surface_region_mapping, regioncolor: list):
        # 标签; -1是因为regionmapping中有-1
        self.labels = surface_region_mapping.data - 1
        self.ampl = np.array(regioncolor)[self.labels]  # 颜色只和对应脑区有关
        colors = self.color_map.map(self.ampl, mode="float")
        self.md.setVertexColors(colors)
        self.surface.vertexes = None
        self.surface.update()
