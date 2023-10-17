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
    def __init__(self):
        super().__init__()

        # color map
        self.surface = None
        self.md = None
        self.color_map = pg.colormap.get(
            "glasbey_bw_minc_20", source="colorcet"
        )  # tab20
        self.faces = None
        self.vertexes = None

        # self.setSurface(surface)

    def setSurface(self, surface):
        self.vertexes = surface.vertices  # 顶点
        self.faces = surface.faces  # 三角面

        self.md = gl.MeshData(vertexes=self.vertexes, faces=self.faces)

        self.surface = gl.GLMeshItem(meshdata=self.md, color=(0.7, 0.7, 0.7, 1))

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
                                        color.x = color.x * (1 - s);
                                        color.y = color.y * (1 - s);
                                        color.z = color.z * (1 - s);
                                        gl_FragColor = color;
                                    }
                                """
                    ),
                ],
            )
        )  # 'edgeHilight','pointSprite'
        self.surface.scale(0.06, 0.06, 0.06)  # 初始化大小设置

        self.addItem(self.surface)

    def setColorMap(self, name: str, source=None):
        # 如果没有color map源，则读取路径
        if source is None:
            self.color_map = pg.colormap.get(name)
        # 如果有color源，则使用对应color map
        else:
            self.color_map = pg.colormap.get(name, source=source)
        colors = self.color_map.map(self.ampl, mode="float")
        self.md.setVertexColors(colors)
        self.surface.vertexes = None
        self.surface.update()

    def setShader(self, shader_program):
        self.surface.setShader(shader=shader_program)
        self.surface.vertexes = None
        self.surface.update()


class AtlasSurfaceViewWidget(SurfaceViewWidget):
    region_signal = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.labels = None
        self.ampl = None
        self._split = False
        self._mouse_move = False

    def setAtlas(self, atlas, surface, surface_region_mapping):
        self.setSurface(surface)
        regioncolor_list = list(
            np.arange(atlas.labels.shape[0]) / atlas.labels.shape[0]
        )
        self.setRegionColor(surface_region_mapping, regioncolor_list)

    def setRegions(self, atlas, surface_region_mapping):
        for region in range(atlas.labels.shape[0]):
            # region = 1
            index_vertexes = np.where(surface_region_mapping.data.squeeze() == region)
            vertexes = self.vertexes[[np.array(index_vertexes)]].squeeze()

            mask = np.isin(self.faces, index_vertexes)
            faces_result = np.all(mask, axis=1)
            index_faces = np.where(faces_result == True)

            faces = self.faces[[np.array(index_faces)]].squeeze()

            idx = np.argsort(index_vertexes)  # 返回一个一维数组，表示进行排序后的索引
            sorted_d = np.take(index_vertexes, idx)  # 返回一个一维数组，表示根据索引获取的元素
            result = np.where(
                sorted_d == faces[:, :, None]
            )  # 返回一个三维数组，表示在排序后的d中查找c中的每个元素的位置
            result = result[2]  # 取第三维度上的值，即在原始d中的位置
            faces = np.reshape(result, faces.shape)  # 将结果调整为和c相同的形状

            md = gl.MeshData(vertexes=vertexes, faces=faces)

            regions = gl.GLMeshItem(meshdata=md)
            regions.scale(0.06, 0.06, 0.06)

            regions.setObjectName(str(region))

            self.addItem(regions)
        self._split = True

    def setRegionColor(self, surface_region_mapping, regioncolor: list):
        # 标签; -1是因为regionmapping中有-1
        self.labels = surface_region_mapping.data - 1
        self.ampl = np.array(regioncolor)[self.labels]  # 颜色只和对应脑区有关
        colors = self.color_map.map(self.ampl, mode="float")
        self.md.setVertexColors(colors)
        self.surface.vertexes = None
        self.surface.update()

    def mouseReleaseEvent(self, event):
        if self._mouse_move == True:
            self._mouse_move = False
            return
        lpos = event.position() if hasattr(event, "position") else event.localPos()
        self.mousePos = lpos
        region = [lpos.x(), lpos.y(), 1, 1]
        # itemsAt seems to take in device pixels
        dpr = self.devicePixelRatioF()
        region = tuple([x * dpr for x in region])

        if self._split == True:
            _selected_items = self.itemsAt(region)

            if len(_selected_items) > 1:
                region_number = int(_selected_items[1].objectName()) - 1
                self.region_signal.emit(region_number)

        self._mouse_move = False

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self._mouse_move = True
