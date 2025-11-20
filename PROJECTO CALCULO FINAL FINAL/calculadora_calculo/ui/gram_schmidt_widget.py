from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QGridLayout, QSizePolicy, QComboBox, QTextEdit, QScrollArea
from PySide6.QtCore import Qt
import sympy as sp
from calculadora_calculo.ui.math_render import lines_to_html
import pyqtgraph as pg
import numpy as np
from calculadora_calculo.calculos.visualizacion import Visualizador3D

class GramSchmidtWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        root_layout = QVBoxLayout(self)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_container = QWidget()
        layout = QVBoxLayout(scroll_container)

        process_layout = QHBoxLayout()
        process_layout.addWidget(QLabel("Seleccionar proceso:"))
        self.proceso_combo = QComboBox()
        self.proceso_combo.addItems(["Ortogonal", "Ortonormal"])  # Default: Ortonormal
        self.proceso_combo.setCurrentIndex(1)
        process_layout.addWidget(self.proceso_combo)
        layout.addLayout(process_layout)

        count_layout = QHBoxLayout()
        self.vector_count_spin = QSpinBox()
        self.vector_count_spin.setRange(1, 10)
        self.vector_count_spin.setValue(3)
        self.dimension_count_spin = QSpinBox()
        self.dimension_count_spin.setRange(1, 10)
        self.dimension_count_spin.setValue(3)
        count_layout.addWidget(QLabel("Número de vectores:"))
        count_layout.addWidget(self.vector_count_spin)
        count_layout.addWidget(QLabel("Número de dimensiones:"))
        count_layout.addWidget(self.dimension_count_spin)
        layout.addLayout(count_layout)

        self.vector_input_layout = QGridLayout()
        layout.addLayout(self.vector_input_layout)

        self.vector_count_spin.valueChanged.connect(self.update_vector_inputs)
        self.dimension_count_spin.valueChanged.connect(self.update_vector_inputs)

        self.calc_button = QPushButton("Calcular")
        self.calc_button.clicked.connect(self.calculate)
        layout.addWidget(self.calc_button)

        # Áreas de texto al estilo de Integrales: Procedimiento y Resultado
        layout.addWidget(QLabel("Procedimiento:"))
        self.proceso_display = QTextEdit()
        self.proceso_display.setReadOnly(True)
        self.proceso_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.proceso_display.setMinimumHeight(100)
        self.proceso_display.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.proceso_display.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.proceso_display.setLineWrapMode(QTextEdit.WidgetWidth)
        self.proceso_display.setStyleSheet("color: #FFFFFF;")
        layout.addWidget(self.proceso_display)

        layout.addWidget(QLabel("Resultado:"))
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.result_display.setMinimumHeight(80)
        self.result_display.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.result_display.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.result_display.setLineWrapMode(QTextEdit.WidgetWidth)
        self.result_display.setStyleSheet("color: #FFFFFF;")
        layout.addWidget(self.result_display)

        layout.addWidget(QLabel("Visualización 2D:"))
        self.plot2d = pg.PlotWidget()
        self.plot2d.setBackground('w')
        self.plot2d.showGrid(x=True, y=True, alpha=0.3)
        self.plot2d.setAspectLocked(True)
        self.plot2d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.plot2d.setMinimumHeight(250)
        # Ejes y ticks en negro (fondo blanco)
        try:
            for which in ('left', 'bottom'):
                ax = self.plot2d.getPlotItem().getAxis(which)
                ax.setPen(pg.mkPen(color=(0, 0, 0), width=1))
                ax.setTextPen(pg.mkPen(color=(0, 0, 0)))
        except Exception:
            pass
        layout.addWidget(self.plot2d)
        self._lines2d = []
        self._axis2d_items = []

        # Leyenda 2D (vectores y ejes)
        legend2d = QWidget()
        h2 = QHBoxLayout(legend2d)
        h2.setContentsMargins(0, 0, 0, 0)
        h2.setSpacing(12)
        # Contenedor dinámico para vectores de entrada: Vector 1, Vector 2, ...
        self.legend2d_vecs = QWidget()
        self.legend2d_vecs_layout = QHBoxLayout(self.legend2d_vecs)
        self.legend2d_vecs_layout.setContentsMargins(0, 0, 0, 0)
        self.legend2d_vecs_layout.setSpacing(12)
        h2.addWidget(self.legend2d_vecs)
        # Ítems estáticos
        h2.addWidget(self._legend_item('#ffff00', 'Base Gram-Schmidt'))
        h2.addWidget(self._legend_item('#ff0000', 'Eje X'))
        h2.addWidget(self._legend_item('#00aa00', 'Eje Y'))
        h2.addStretch()
        layout.addWidget(legend2d)

        layout.addWidget(QLabel("Visualización 3D:"))
        self.visual3d = Visualizador3D()
        self.visual3d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.visual3d.setMinimumHeight(300)
        layout.addWidget(self.visual3d)

        # Leyenda 3D (ejes y vectores)
        legend3d = QWidget()
        h3 = QHBoxLayout(legend3d)
        h3.setContentsMargins(0, 0, 0, 0)
        h3.setSpacing(12)
        h3.addWidget(self._legend_item('#ff0000', 'Eje X'))
        h3.addWidget(self._legend_item('#00ff00', 'Eje Y'))
        h3.addWidget(self._legend_item('#0000ff', 'Eje Z'))
        # Contenedor dinámico para 'Vector i'
        self.legend3d_vecs = QWidget()
        self.legend3d_vecs_layout = QHBoxLayout(self.legend3d_vecs)
        self.legend3d_vecs_layout.setContentsMargins(0, 0, 0, 0)
        self.legend3d_vecs_layout.setSpacing(12)
        h3.addWidget(self.legend3d_vecs)
        h3.addWidget(self._legend_item('#ffff00', 'Base Gram-Schmidt'))
        h3.addStretch()
        layout.addWidget(legend3d)

        self.update_vector_inputs()
        scroll_area.setWidget(scroll_container)
        root_layout.addWidget(scroll_area)

    def _auto_resize_textedit(self, te: QTextEdit):
        try:
            doc = te.document()
            doc.adjustSize()
            h = int(doc.size().height()) + 2 * int(te.frameWidth()) + 4
            if h < 60:
                h = 60
            te.setFixedHeight(h)
        except Exception:
            pass

    def _legend_item(self, color_css: str, text: str) -> QWidget:
        box = QWidget()
        box.setFixedSize(12, 12)
        box.setStyleSheet(f"background-color: {color_css}; border: 1px solid #777;")
        item = QWidget()
        hl = QHBoxLayout(item)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(6)
        hl.addWidget(box)
        hl.addWidget(QLabel(text))
        return item

    def update_vector_inputs(self):
        for i in reversed(range(self.vector_input_layout.count())):
            item = self.vector_input_layout.itemAt(i)
            w = item.widget()
            if w is not None:
                w.setParent(None)

        vector_count = self.vector_count_spin.value()
        dimension_count = self.dimension_count_spin.value()

        for i in range(vector_count):
            self.vector_input_layout.addWidget(QLabel(f"Vector {i + 1}:"), i, 0)
            for j in range(dimension_count):
                input_field = QLineEdit()
                input_field.setPlaceholderText(f"x{j + 1}")
                self.vector_input_layout.addWidget(input_field, i, j + 1)

    def get_vectors(self):
        vectors = []
        vector_count = self.vector_count_spin.value()
        dimension_count = self.dimension_count_spin.value()
        for i in range(vector_count):
            components = []
            for j in range(dimension_count):
                item = self.vector_input_layout.itemAtPosition(i, j + 1)
                if item and isinstance(item.widget(), QLineEdit):
                    component = item.widget().text().strip()
                    if component:
                        components.append(sp.Rational(component))
                    else:
                        components.append(sp.Rational(0))
            if components:
                vectors.append(sp.Matrix(components))
        return vectors

    def gram_schmidt(self, vectors, orthonormal=True):
        basis = []
        for v in vectors:
            w = v - sum((v.dot(b) / b.dot(b) * b for b in basis), sp.zeros(v.shape[0], 1))
            if w.norm() == 0:
                raise ValueError("Input vectors are linearly dependent.")
            if orthonormal:
                w = w / w.norm()
            basis.append(w)
        return basis

    def vector_to_latex(self, vector):
        return "\\left[" + ", ".join(sp.latex(comp) for comp in vector) + "\\right]"

    def vector_to_str(self, vector):
        return "[" + ", ".join(str(comp) for comp in vector) + "]"

    def calculate(self):
        try:
            vectors = self.get_vectors()
            if not vectors:
                raise ValueError("No se ingresaron vectores válidos.")

            orthonormal = (self.proceso_combo.currentText() == "Ortonormal")
            basis = self.gram_schmidt(vectors, orthonormal)

            process_type = "Ortonormal" if orthonormal else "Ortogonal"
            # Construir procedimiento paso a paso (LaTeX)
            pasos_latex = []
            pasos_latex.append(f"\\text{{Proceso: {process_type}}}")
            pasos_latex.append("\\text{Vectores de entrada:}")
            for i, v in enumerate(vectors, 1):
                pasos_latex.append(f"v_{{{i}}} = {self.vector_to_latex(v)}")

            # Recalcular explícitamente para registrar pasos
            u_basis = []
            for k, v in enumerate(vectors, 1):
                w = v
                detalle_proy = []
                for j, b in enumerate(u_basis, 1):
                    coef = sp.simplify((v.dot(b) / b.dot(b)))
                    w = w - coef * b
                    detalle_proy.append(f"\\text{{Proy}}_{{v_{k} \to u_{j}}} = {sp.latex(coef)}\\, u_{{{j}}}")
                pasos_latex.append(f"\\text{{Paso {k}}}")
                if detalle_proy:
                    pasos_latex.extend(detalle_proy)
                else:
                    pasos_latex.append("\\text{No hay proyecciones previas}")
                pasos_latex.append(f"w_{{{k}}} = {self.vector_to_latex(sp.simplify(w))}")
                if w.norm() == 0:
                    raise ValueError("Los vectores de entrada son linealmente dependientes.")
                if orthonormal:
                    norm_w = sp.simplify(sp.sqrt((w.T*w)[0]))
                    u = sp.simplify(w / norm_w)
                    pasos_latex.append(f"u_{{{k}}} = \\frac{{w_{{{k}}}}}{{{sp.latex(norm_w)}}} = {self.vector_to_latex(u)}")
                else:
                    u = sp.simplify(w)
                    pasos_latex.append(f"u_{{{k}}} = w_{{{k}}} = {self.vector_to_latex(u)}")
                u_basis.append(u)

            # Mostrar procedimiento y resultado renderizados
            self.proceso_display.setHtml(lines_to_html(pasos_latex))
            self._auto_resize_textedit(self.proceso_display)

            resultado_latex = [f"u_{{{i+1}}} = {self.vector_to_latex(u)}" for i, u in enumerate(u_basis)]
            self.result_display.setHtml(lines_to_html(resultado_latex))
            self._auto_resize_textedit(self.result_display)
            # Actualizar visualizadores
            self.update_visuals(vectors, u_basis)
        except Exception as e:
            self.proceso_display.setPlainText("")
            self.result_display.setPlainText(f"Ocurrió un error: {str(e)}")
            self._auto_resize_textedit(self.proceso_display)
            self._auto_resize_textedit(self.result_display)

    def _clear_2d(self):
        for item in self._lines2d:
            try:
                self.plot2d.removeItem(item)
            except Exception:
                pass
        self._lines2d = []
        # limpiar elementos de ejes 2D
        if hasattr(self, '_axis2d_items'):
            for it in self._axis2d_items:
                try:
                    self.plot2d.removeItem(it)
                except Exception:
                    pass
            self._axis2d_items = []

    def _plot_vectors_2d(self, vectors, color, width: int = 2):
        if not vectors:
            return
        try:
            arr = np.array([np.array(v, dtype=object).astype(float).flatten() for v in vectors], dtype=float)
        except Exception:
            return
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        # Tomar las dos primeras componentes, completar con 0 si falta
        if arr.shape[1] < 2:
            pad = np.zeros((arr.shape[0], 2 - arr.shape[1]))
            arr = np.concatenate([arr, pad], axis=1)
        # Dibujar cada vector como una línea independiente
        max_abs_val = 1.0
        for v in arr:
            x_vals = [0.0, float(v[0])]
            y_vals = [0.0, float(v[1])]
            line = pg.PlotDataItem(x_vals, y_vals, pen=pg.mkPen(color=color, width=width))
            self.plot2d.addItem(line)
            self._lines2d.append(line)
            try:
                max_abs_val = max(max_abs_val, abs(float(v[0])), abs(float(v[1])))
            except Exception:
                pass
        # Ajuste de vista
        self.plot2d.setXRange(-max_abs_val, max_abs_val)
        self.plot2d.setYRange(-max_abs_val, max_abs_val)

    def _hex_to_rgb(self, hx: str):
        hx = hx.lstrip('#')
        return tuple(int(hx[i:i+2], 16) for i in (0, 2, 4))

    def _hex_to_rgba(self, hx: str, a: float = 1.0):
        r, g, b = self._hex_to_rgb(hx)
        return (r/255.0, g/255.0, b/255.0, a)

    def _vector_palette(self, n: int):
        base_hex = [
            '#00ffff',  # cyan
            '#ff00ff',  # magenta
            '#ffa500',  # orange
            '#32cd32',  # limegreen
            '#00bfff',  # deepskyblue
            '#ff69b4',  # hotpink
            '#00ced1',  # darkturquoise
            '#9370db',  # mediumpurple
            '#ff4500',  # orangered
            '#1e90ff',  # dodgerblue
        ]
        colors_hex = [base_hex[i % len(base_hex)] for i in range(n)]
        colors_2d = [self._hex_to_rgb(h) for h in colors_hex]
        colors_3d = [self._hex_to_rgba(h, 1.0) for h in colors_hex]
        return colors_hex, colors_2d, colors_3d

    def update_visuals(self, vectors, basis):
        self._clear_2d()
        self.plot2d.clear()
        self.plot2d.showGrid(x=True, y=True, alpha=0.3)
        self.plot2d.setBackground('w')
        self.visual3d.limpiar_escena()
        # Colores por vector (entrada) y línea más gruesa en 2D
        colors_hex, colors_2d, colors_3d = self._vector_palette(len(vectors))
        for i, v in enumerate(vectors):
            self._plot_vectors_2d([v], colors_2d[i], width=4)
        # Base en 2D (amarillo) con grosor medio
        self._plot_vectors_2d(basis, (255, 255, 0), width=3)
        # Leyenda dinámica: Vector 1, Vector 2, ...
        try:
            # Limpiar contenedores
            for lay in (self.legend2d_vecs_layout, self.legend3d_vecs_layout):
                while lay.count():
                    w = lay.takeAt(0).widget()
                    if w is not None:
                        w.deleteLater()
            # Reconstruir
            for i in range(len(vectors)):
                label = f"Vector {i+1}"
                self.legend2d_vecs_layout.addWidget(self._legend_item(colors_hex[i], label))
                self.legend3d_vecs_layout.addWidget(self._legend_item(colors_hex[i], label))
            self.legend2d_vecs_layout.addStretch()
            self.legend3d_vecs_layout.addStretch()
        except Exception:
            pass
        # Ejes 2D coloreados y etiquetas
        try:
            view = self.plot2d.getPlotItem().getViewBox()
            xr, yr = view.viewRange()[0], view.viewRange()[1]
            x_line = pg.InfiniteLine(pos=0, angle=0, movable=False, pen=pg.mkPen((255, 0, 0), width=1))
            y_line = pg.InfiniteLine(pos=0, angle=90, movable=False, pen=pg.mkPen((0, 170, 0), width=1))
            self.plot2d.addItem(x_line)
            self.plot2d.addItem(y_line)
            tx = pg.TextItem(text='X', color=(255, 0, 0))
            ty = pg.TextItem(text='Y', color=(0, 170, 0))
            self.plot2d.addItem(tx)
            self.plot2d.addItem(ty)
            tx.setPos(xr[1]*0.95, 0)
            ty.setPos(0, yr[1]*0.95)
            self._axis2d_items.extend([x_line, y_line, tx, ty])
        except Exception:
            pass
        # 3D: usar método del visualizador
        try:
            vecs3 = [np.array(v, dtype=object).astype(float).flatten()[:3] for v in vectors]
        except Exception:
            vecs3 = []
        try:
            basis3 = [np.array(v, dtype=object).astype(float).flatten()[:3] for v in basis]
        except Exception:
            basis3 = []
        # 3D: colores por vector a juego con la leyenda
        colores_vecs = [colors_3d[i] for i in range(len(vecs3))]
        colores_basis = [(1.0, 1.0, 0.0, 1.0) for _ in basis3] # amarillo
        self.visual3d.graficar_vectores(vecs3, colores=colores_vecs)
        self.visual3d.graficar_vectores(basis3, colores=colores_basis)
