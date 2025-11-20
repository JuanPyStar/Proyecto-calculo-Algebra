from PySide6.QtWidgets import (QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QTextEdit, QComboBox, QGroupBox, QFormLayout,
                             QMessageBox, QStackedWidget, QDialog, QScrollArea, QFrame, QSizePolicy)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
import pyqtgraph as pg
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from calculadora_calculo.calculos.visualizacion import Visualizador3D
from calculadora_calculo.ui.gram_schmidt_widget import GramSchmidtWidget
from calculadora_calculo.ui.math_render import lines_to_html

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calculadora Avanzada de Cálculo Vectorial")
        self.setMinimumSize(500, 500)
        
        # Configuración de la ventana principal
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # Crear pestañas
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # Agregar pestañas
        self.setup_integrales_tab()
        self.setup_teoremas_tab()
        self.setup_teoria_tab()
        self.setup_algebra_tab()
        # Carga perezosa de Gram-Schmidt para evitar conflictos de contexto
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        # Configurar gráficos
        pg.setConfigOptions(antialias=True)
        
    def _set_math_lines(self, text_edit: QTextEdit, lines: list[str]):
        """Establece contenido en formato matemático renderizado a partir de líneas de texto/LaTeX.
        Para texto plano, se envuelve con \\text{...}."""
        latex_lines = []
        for line in lines:
            if line is None:
                continue
            line = str(line)
            if line.strip() == "":
                continue
            # Si la línea parece contener LaTeX (\\, ^, _, {, }), la dejamos; si no, la envolvemos en \\text{}
            if any(ch in line for ch in ['\\', '^', '_', '{', '}', '∫', 'θ', 'φ', 'π']):
                # Asegurar que no haya saltos de línea crudos que rompan el parser
                safe_line = line.replace('\n', ' ').replace('\t', ' ')
                latex_lines.append(safe_line)
            else:
                safe = (
                    line.replace('\n', ' ').replace('\t', ' ')
                        .replace('{', '\\{').replace('}', '\\}')
                        .replace('#', '\\#').replace('%', '\\%')
                        .replace('_', '\\_').replace('^', '\\^{}')
                )
                latex_lines.append(f"\\text{{{safe}}}")
        text_edit.setHtml(lines_to_html(latex_lines))
        
    def setup_integrales_tab(self):
        """Configura la pestaña de integrales triples"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Grupo de selección de coordenadas
        coord_group = QGroupBox("Tipo de Coordenadas")
        coord_layout = QHBoxLayout()
        
        self.coord_type = QComboBox()
        self.coord_type.addItems(["Rectangulares", "Cilíndricas", "Esféricas"])
        coord_layout.addWidget(QLabel("Sistema de coordenadas:"))
        coord_layout.addWidget(self.coord_type)
        coord_group.setLayout(coord_layout)
        
        # Grupo de entrada de función y límites
        input_group = QGroupBox("Entrada de Función y Límites")
        form_layout = QFormLayout()
        
        self.func_input = QLineEdit()
        self.func_input.setPlaceholderText("Ingrese la función f(x,y,z)")
        
        # Campos para límites de integración
        self.x_lim = [QLineEdit(), QLineEdit()]
        self.y_lim = [QLineEdit(), QLineEdit()]
        self.z_lim = [QLineEdit(), QLineEdit()]
        
        form_layout.addRow("Función f(x,y,z):", self.func_input)
        # Teclado matemático compacto para facilitar la entrada de funciones
        keypad_row = QWidget()
        keypad_layout = QHBoxLayout(keypad_row)
        keypad_layout.setContentsMargins(0, 0, 0, 0)
        for label, insert in [
            ("x", "x"), ("y", "y"), ("z", "z"),
            ("(", "("), (")", ")"),
            ("+", "+"), ("-", "-"), ("×", "*"), ("÷", "/"),
            ("^", "**"), ("x²", "**2"),
            ("√", "sqrt()"), ("abs", "Abs()"),
            ("sin", "sin()"), ("cos", "cos()"), ("tan", "tan()"),
            ("ln", "log()"), ("exp", "exp()"),
            ("π", "pi"), ("e", "E"),
            ("θ", "θ"), ("φ", "φ")
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(24)
            btn.setStyleSheet("padding: 2px 6px;")
            btn.clicked.connect(lambda _, t=insert: self.insert_into_func(t))
            keypad_layout.addWidget(btn)
        form_layout.addRow("", keypad_row)
        # Filas dinámicas para límites (se ocultan inicialmente hasta detectar variables)
        self.x_label = QLabel("Límites en x:")
        self.x_group = self.create_horizontal_group(self.x_lim)
        form_layout.addRow(self.x_label, self.x_group)

        self.y_label = QLabel("Límites en y:")
        self.y_group = self.create_horizontal_group(self.y_lim)
        form_layout.addRow(self.y_label, self.y_group)

        self.z_label = QLabel("Límites en z:")
        self.z_group = self.create_horizontal_group(self.z_lim)
        form_layout.addRow(self.z_label, self.z_group)
        
        input_group.setLayout(form_layout)
        
        # Actualización dinámica de límites según variables usadas
        self.func_input.textChanged.connect(self.update_limit_fields)
        # Estado inicial (función vacía => ocultar)
        self.update_limit_fields()

        # Botón de cálculo
        self.calc_button = QPushButton("Calcular Integral")
        self.calc_button.clicked.connect(self.calcular_integral)
        
        # Apartado de procedimiento
        self.proceso_display = QTextEdit()
        self.proceso_display.setReadOnly(True)
        # Deshabilitar barras internas: usar solo la barra principal
        self.proceso_display.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.proceso_display.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.proceso_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.proceso_display.setMinimumHeight(60)
        
        # Área de resultados
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.result_display.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.result_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.result_display.setMinimumHeight(60)
        
        # Gráfico 3D
        self.visualizador3d = Visualizador3D()
        self.visualizador3d.setMinimumHeight(400)

        # Un solo contenedor desplazable para TODO el contenido de la pestaña
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_container = QWidget()
        scroll_layout = QVBoxLayout(scroll_container)

        # Sección superior (controles)
        scroll_layout.addWidget(coord_group)
        scroll_layout.addWidget(input_group)
        scroll_layout.addWidget(self.calc_button)

        # Sección inferior (resultado, procedimiento y 3D)
        scroll_layout.addWidget(QLabel("Resultado:"))
        scroll_layout.addWidget(self.result_display)
        scroll_layout.addWidget(QLabel("Procedimiento:"))
        scroll_layout.addWidget(self.proceso_display)
        scroll_layout.addWidget(QLabel("Visualización 3D:"))
        scroll_layout.addWidget(self.visualizador3d)
        # Leyenda de ejes por color (X rojo, Y verde, Z azul)
        legend = QWidget()
        legend_layout = QHBoxLayout(legend)
        legend_layout.setContentsMargins(0, 0, 0, 0)
        legend_layout.setSpacing(12)
        def add_legend_item(color_css: str, text: str):
            box = QFrame()
            box.setFixedSize(12, 12)
            box.setStyleSheet(f"background-color: {color_css}; border: 1px solid #777;")
            item = QWidget()
            hl = QHBoxLayout(item)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(6)
            hl.addWidget(box)
            hl.addWidget(QLabel(text))
            return item
        legend_layout.addWidget(add_legend_item('#ff0000', 'Eje X'))
        legend_layout.addWidget(add_legend_item('#00ff00', 'Eje Y'))
        legend_layout.addWidget(add_legend_item('#0000ff', 'Eje Z'))
        legend_layout.addStretch()
        scroll_layout.addWidget(legend)

        scroll_area.setWidget(scroll_container)
        layout.addWidget(scroll_area)
        
        self.tabs.addTab(tab, "Integrales Triples")
    
    def setup_teoremas_tab(self):
        """Configura la pestaña de teoremas vectoriales"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Selector de teorema
        teorema_group = QGroupBox("Seleccionar Teorema")
        teorema_layout = QVBoxLayout()
        
        self.teorema_combo = QComboBox()
        self.teorema_combo.addItems(["Teorema de Green", 
                                   "Teorema de Stokes", 
                                   "Teorema de la Divergencia"])
        self.teorema_combo.currentIndexChanged.connect(self.update_teorema_inputs)
        
        teorema_layout.addWidget(QLabel("Seleccione un teorema:"))
        teorema_layout.addWidget(self.teorema_combo)
        teorema_group.setLayout(teorema_layout)
        
        # Grupo de entrada específico para cada teorema
        self.teorema_inputs = QStackedWidget()
        
        # Páginas para cada teorema
        self.setup_green_inputs()
        self.setup_stokes_inputs()
        self.setup_divergencia_inputs()
        
        # Botón de cálculo
        self.calc_teorema_btn = QPushButton("Aplicar Teorema")
        self.calc_teorema_btn.clicked.connect(self.aplicar_teorema)
        
        # Área de procedimiento
        self.teorema_proceso = QTextEdit()
        self.teorema_proceso.setReadOnly(True)
        self.teorema_proceso.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.teorema_proceso.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.teorema_proceso.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.teorema_proceso.setMinimumHeight(60)
        
        # Área de resultados
        self.teorema_result = QTextEdit()
        self.teorema_result.setReadOnly(True)
        self.teorema_result.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.teorema_result.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.teorema_result.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.teorema_result.setMinimumHeight(60)
        
        # Agregar widgets al layout
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        scroll_layout.addWidget(teorema_group)
        scroll_layout.addWidget(self.teorema_inputs)
        scroll_layout.addWidget(self.calc_teorema_btn)
        scroll_layout.addWidget(QLabel("Procedimiento:"))
        scroll_layout.addWidget(self.teorema_proceso)
        scroll_layout.addWidget(QLabel("Resultado:"))
        scroll_layout.addWidget(self.teorema_result)
        scroll_layout.addStretch()
        
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        self.tabs.addTab(tab, "Teoremas Vectoriales")
    
    def setup_teoria_tab(self):
        """Configura la pestaña de teoría"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Título
        title = QLabel("Teoría de Cálculo Vectorial")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(16)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        
        # Contenido de teoría
        self.teoria_content = QTextEdit()
        self.teoria_content.setReadOnly(True)
        self.teoria_content.setHtml("""
        <h2>Integrales Triples</h2>
        <p>Las integrales triples extienden el concepto de integral doble a funciones de tres variables. 
        Se utilizan para calcular volúmenes, masas, centros de masa y otros conceptos físicos en el espacio tridimensional.</p>
        
        <h3>Coordenadas Rectangulares</h3>
        <p>La integral triple de una función f(x,y,z) sobre una región sólida E se expresa como:</p>
        <p>∭_E f(x,y,z) dV</p>
        
        <h3>Coordenadas Cilíndricas</h3>
        <p>Útiles para regiones con simetría cilíndrica. La transformación es:</p>
        <p>x = r·cos(θ), y = r·sin(θ), z = z</p>
        <p>El elemento de volumen: dV = r·dz·dr·dθ</p>
        
        <h3>Coordenadas Esféricas</h3>
        <p>Ideales para regiones con simetría esférica. La transformación es:</p>
        <p>x = ρ·sin(φ)·cos(θ), y = ρ·sin(φ)·sin(θ), z = ρ·cos(φ)</p>
        <p>El elemento de volumen: dV = ρ²·sin(φ)·dρ·dφ·dθ</p>
        
        <h2>Teorema de Green</h2>
        <p>Relaciona una integral de línea alrededor de una curva plana cerrada simple C con una integral doble sobre la región D encerrada por C:</p>
        <p>∮_C (P·dx + Q·dy) = ∬_D (∂Q/∂x - ∂P/∂y) dA</p>
        
        <h2>Teorema de Stokes</h2>
        <p>Generalización del teorema de Green a superficies en el espacio. Relaciona la integral de línea de un campo vectorial alrededor de la frontera de una superficie con la integral del rotacional del campo sobre la superficie:</p>
        <p>∮_C F·dr = ∬_S (∇ × F)·dS</p>
        
        <h2>Teorema de la Divergencia</h2>
        <p>Relaciona el flujo de un campo vectorial a través de una superficie cerrada con la integral de la divergencia del campo sobre el volumen encerrado:</p>
        <p>∯_S F·dS = ∭_V (∇·F) dV</p>
        """)
        
        # Agregar widgets al layout
        layout.addWidget(title)
        layout.addWidget(self.teoria_content)
        
        self.tabs.addTab(tab, "Teoría")
    
    def setup_algebra_tab(self):
        """Configura la pestaña de álgebra (Gram-Schmidt)"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Placeholder: se creará el widget bajo demanda
        self._algebra_container = QWidget()
        self._algebra_layout = QVBoxLayout(self._algebra_container)
        self._algebra_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._algebra_container)
        self.gram_schmidt_widget = None
        
        self.tabs.addTab(tab, "Álgebra")

    def on_tab_changed(self, index: int):
        """Crea/Destruye el widget de Gram-Schmidt al entrar/salir de la pestaña para evitar conflictos de OpenGL."""
        try:
            tab_text = self.tabs.tabText(index)
        except Exception:
            tab_text = ""
        # Si entramos a Álgebra, crear si no existe
        if tab_text == "Álgebra":
            if self.gram_schmidt_widget is None:
                self.gram_schmidt_widget = GramSchmidtWidget()
                self._algebra_layout.addWidget(self.gram_schmidt_widget)
        else:
            # Si salimos de Álgebra, destruir el WebEngine para liberar contexto
            if self.gram_schmidt_widget is not None:
                self.gram_schmidt_widget.setParent(None)
                self.gram_schmidt_widget.deleteLater()
                self.gram_schmidt_widget = None
    
    # Métodos auxiliares
    def create_horizontal_group(self, widgets):
        """Crea un grupo horizontal de widgets"""
        group = QWidget()
        layout = QHBoxLayout(group)
        for widget in widgets:
            layout.addWidget(widget)
        return group
    
    def update_limit_fields(self, *_):
        """Muestra/oculta filas de límites según variables detectadas en la función."""
        text = self.func_input.text() if hasattr(self, 'func_input') else ""
        used = set()
        if 'x' in text:
            used.add('x')
        if 'y' in text:
            used.add('y')
        if 'z' in text:
            used.add('z')
        # Si aún no existen las referencias (carrera en init), salir
        if not all(hasattr(self, attr) for attr in ['x_label','x_group','y_label','y_group','z_label','z_group']):
            return
        # Mostrar/ocultar
        self.x_label.setVisible('x' in used)
        self.x_group.setVisible('x' in used)
        self.y_label.setVisible('y' in used)
        self.y_group.setVisible('y' in used)
        self.z_label.setVisible('z' in used)
        self.z_group.setVisible('z' in used)
        # Prellenar por defecto cuando no se usan
        if 'x' not in used:
            for i, val in enumerate(['0', '1']):
                self.x_lim[i].setText(val)
        if 'y' not in used:
            for i, val in enumerate(['0', '1']):
                self.y_lim[i].setText(val)
        if 'z' not in used:
            for i, val in enumerate(['0', '1']):
                self.z_lim[i].setText(val)
    
    def setup_green_inputs(self):
        """Configura los inputs para el Teorema de Green"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Grupo para la función
        func_group = QGroupBox("Funciones del Campo Vectorial")
        func_layout = QFormLayout()
        
        # Campos para las funciones P y Q
        self.green_p = QLineEdit()
        self.green_q = QLineEdit()
        
        # Teclado para facilitar la entrada
        keypad_row = QWidget()
        keypad_layout = QHBoxLayout(keypad_row)
        keypad_layout.setContentsMargins(0, 0, 0, 0)
        for label, insert in [
            ("x", "x"), ("y", "y"), ("(", "("), (")", ")"),
            ("+", "+"), ("-", "-"), ("×", "*"), ("÷", "/"),
            ("^", "**"), ("x²", "**2"),
            ("√", "sqrt()"), ("abs", "Abs()"),
            ("sin", "sin()"), ("cos", "cos()"), ("tan", "tan()"),
            ("ln", "log()"), ("exp", "exp()"),
            ("π", "pi"), ("e", "E")
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(24)
            btn.setStyleSheet("padding: 2px 6px;")
            btn.clicked.connect(lambda _, t=insert: self.insert_into_func(t, self.green_p))
            keypad_layout.addWidget(btn)
        
        func_layout.addRow("P(x,y):", self.green_p)
        func_layout.addRow("", keypad_row)
        
        # Segundo teclado para Q
        keypad_row2 = QWidget()
        keypad_layout2 = QHBoxLayout(keypad_row2)
        keypad_layout2.setContentsMargins(0, 0, 0, 0)
        for label, insert in [
            ("x", "x"), ("y", "y"), ("(", "("), (")", ")"),
            ("+", "+"), ("-", "-"), ("×", "*"), ("÷", "/"),
            ("^", "**"), ("x²", "**2"),
            ("√", "sqrt()"), ("abs", "Abs()"),
            ("sin", "sin()"), ("cos", "cos()"), ("tan", "tan()"),
            ("ln", "log()"), ("exp", "exp()"),
            ("π", "pi"), ("e", "E")
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(24)
            btn.setStyleSheet("padding: 2px 6px;")
            btn.clicked.connect(lambda _, t=insert: self.insert_into_func(t, self.green_q))
            keypad_layout2.addWidget(btn)
        
        func_layout.addRow("Q(x,y):", self.green_q)
        func_layout.addRow("", keypad_row2)
        func_group.setLayout(func_layout)
        
        # Grupo para los límites de integración
        limits_group = QGroupBox("Límites de Integración")
        limits_layout = QHBoxLayout()
        
        # Límites para x
        x_group = QGroupBox("Límites en x")
        x_layout = QVBoxLayout()
        self.green_xmin = QLineEdit("-1")
        self.green_xmax = QLineEdit("1")
        x_layout.addWidget(QLabel("Mínimo:"))
        x_layout.addWidget(self.green_xmin)
        x_layout.addWidget(QLabel("Máximo:"))
        x_layout.addWidget(self.green_xmax)
        x_group.setLayout(x_layout)
        
        # Límites para y
        y_group = QGroupBox("Límites en y")
        y_layout = QVBoxLayout()
        self.green_ymin = QLineEdit("-1")
        self.green_ymax = QLineEdit("1")
        y_layout.addWidget(QLabel("Mínimo:"))
        y_layout.addWidget(self.green_ymin)
        y_layout.addWidget(QLabel("Máximo:"))
        y_layout.addWidget(self.green_ymax)
        y_group.setLayout(y_layout)
        
        limits_layout.addWidget(x_group)
        limits_layout.addWidget(y_group)
        limits_group.setLayout(limits_layout)
        
        # Botón de cálculo
        calc_btn = QPushButton("Aplicar Teorema de Green")
        calc_btn.clicked.connect(self.aplicar_teorema)
        
        # Agregar widgets al layout principal
        layout.addWidget(func_group)
        layout.addWidget(limits_group)
        layout.addWidget(calc_btn)
        layout.addStretch()
        
        # Agregar el widget al contenedor de teoremas
        self.teorema_inputs.addWidget(widget)
    
    def setup_stokes_inputs(self):
        """Configura los inputs para el Teorema de Stokes"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Grupo para el campo vectorial
        field_group = QGroupBox("Campo Vectorial F = (F₁, F₂, F₃)")
        field_layout = QFormLayout()
        
        # Campos para las componentes del campo vectorial
        self.stokes_f1 = QLineEdit()
        self.stokes_f2 = QLineEdit()
        self.stokes_f3 = QLineEdit()
        
        # Función para crear teclado de símbolos
        def create_symbol_keyboard(target_field):
            keypad = QWidget()
            layout = QHBoxLayout(keypad)
            layout.setContentsMargins(0, 0, 0, 0)
            
            symbols = [
                ("x", "x"), ("y", "y"), ("z", "z"), ("(", "("), (")", ")"),
                ("+", "+"), ("-", "-"), ("×", "*"), ("÷", "/"),
                ("^", "**"), ("x²", "**2"),
                ("√", "sqrt()"), ("abs", "Abs()"),
                ("sin", "sin()"), ("cos", "cos()"), ("tan", "tan()"),
                ("ln", "log()"), ("exp", "exp()"),
                ("π", "pi"), ("e", "E")
            ]
            
            for label, text in symbols:
                btn = QPushButton(label)
                btn.setFixedHeight(24)
                btn.setStyleSheet("padding: 2px 6px;")
                btn.clicked.connect(lambda _, t=text: self.insert_into_func(t, target_field))
                layout.addWidget(btn)
            
            return keypad
        
        # Agregar campos con sus teclados
        field_layout.addRow("F₁(x,y,z):", self.stokes_f1)
        field_layout.addRow("", create_symbol_keyboard(self.stokes_f1))
        
        field_layout.addRow("F₂(x,y,z):", self.stokes_f2)
        field_layout.addRow("", create_symbol_keyboard(self.stokes_f2))
        
        field_layout.addRow("F₃(x,y,z):", self.stokes_f3)
        field_layout.addRow("", create_symbol_keyboard(self.stokes_f3))
        
        field_group.setLayout(field_layout)
        
        # Grupo para la superficie
        surface_group = QGroupBox("Superficie S")
        surface_layout = QFormLayout()
        
        self.stokes_surface = QComboBox()
        self.stokes_surface.addItems(["Plano", "Esfera", "Cilindro", "Cono", "Personalizado"])
        
        # Parámetros de la superficie
        self.stokes_param1 = QLineEdit()
        self.stokes_param2 = QLineEdit()
        self.stokes_param3 = QLineEdit()
        
        # Función para actualizar los parámetros según la superficie seleccionada
        def update_surface_params():
            surface = self.stokes_surface.currentText()
            if surface == "Plano":
                self.stokes_param1.setPlaceholderText("a (coef. x)")
                self.stokes_param2.setPlaceholderText("b (coef. y)")
                self.stokes_param3.setPlaceholderText("c (coef. z)")
            elif surface == "Esfera":
                self.stokes_param1.setPlaceholderText("Radio")
                self.stokes_param2.setPlaceholderText("Centro X")
                self.stokes_param3.setPlaceholderText("Centro Y")
            elif surface == "Cilindro":
                self.stokes_param1.setPlaceholderText("Radio")
                self.stokes_param2.setPlaceholderText("Altura")
                self.stokes_param3.setPlaceholderText("Eje (x/y/z)")
            elif surface == "Cono":
                self.stokes_param1.setPlaceholderText("Radio base")
                self.stokes_param2.setPlaceholderText("Altura")
                self.stokes_param3.setPlaceholderText("Vértice (x,y,z)")
            else:  # Personalizado
                self.stokes_param1.setPlaceholderText("x(u,v)")
                self.stokes_param2.setPlaceholderText("y(u,v)")
                self.stokes_param3.setPlaceholderText("z(u,v)")
        
        self.stokes_surface.currentTextChanged.connect(update_surface_params)
        update_surface_params()  # Inicializar placeholders
        
        surface_layout.addRow("Tipo:", self.stokes_surface)
        surface_layout.addRow("Parámetro 1:", self.stokes_param1)
        surface_layout.addRow("Parámetro 2:", self.stokes_param2)
        surface_layout.addRow("Parámetro 3:", self.stokes_param3)
        
        surface_group.setLayout(surface_layout)
        
        # Botón de cálculo
        calc_btn = QPushButton("Aplicar Teorema de Stokes")
        calc_btn.clicked.connect(self.aplicar_teorema)
        
        # Agregar todo al layout principal
        layout.addWidget(field_group)
        layout.addWidget(surface_group)
        layout.addWidget(calc_btn)
        layout.addStretch()
        
        self.teorema_inputs.addWidget(widget)
    
    def setup_divergencia_inputs(self):
        """Configura los inputs para el Teorema de la Divergencia"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Grupo para el campo vectorial
        field_group = QGroupBox("Campo Vectorial F = (F₁, F₂, F₃)")
        field_layout = QFormLayout()
        
        # Campos para las componentes del campo vectorial
        self.div_f1 = QLineEdit()
        self.div_f2 = QLineEdit()
        self.div_f3 = QLineEdit()
        
        # Función para crear teclado de símbolos (reutilizada de setup_stokes_inputs)
        def create_symbol_keyboard(target_field):
            keypad = QWidget()
            layout = QHBoxLayout(keypad)
            layout.setContentsMargins(0, 0, 0, 0)
            
            symbols = [
                ("x", "x"), ("y", "y"), ("z", "z"), ("(", "("), (")", ")"),
                ("+", "+"), ("-", "-"), ("×", "*"), ("÷", "/"),
                ("^", "**"), ("x²", "**2"),
                ("√", "sqrt()"), ("abs", "Abs()"),
                ("sin", "sin()"), ("cos", "cos()"), ("tan", "tan()"),
                ("ln", "log()"), ("exp", "exp()"),
                ("π", "pi"), ("e", "E")
            ]
            
            for label, text in symbols:
                btn = QPushButton(label)
                btn.setFixedHeight(24)
                btn.setStyleSheet("padding: 2px 6px;")
                btn.clicked.connect(lambda _, t=text: self.insert_into_func(t, target_field))
                layout.addWidget(btn)
            
            return keypad
        
        # Agregar campos con sus teclados
        field_layout.addRow("F₁(x,y,z):", self.div_f1)
        field_layout.addRow("", create_symbol_keyboard(self.div_f1))
        
        field_layout.addRow("F₂(x,y,z):", self.div_f2)
        field_layout.addRow("", create_symbol_keyboard(self.div_f2))
        
        field_layout.addRow("F₃(x,y,z):", self.div_f3)
        field_layout.addRow("", create_symbol_keyboard(self.div_f3))
        
        field_group.setLayout(field_layout)
        
        # Grupo para la región de integración
        region_group = QGroupBox("Región de Integración E")
        region_layout = QFormLayout()
        
        self.div_region = QComboBox()
        self.div_region.addItems(["Esfera", "Cubo", "Cilindro", "Cono", "Personalizado"])
        
        # Parámetros de la región
        self.div_param1 = QLineEdit()
        self.div_param2 = QLineEdit()
        self.div_param3 = QLineEdit()
        
        # Función para actualizar los parámetros según la región seleccionada
        def update_region_params():
            region = self.div_region.currentText()
            if region == "Esfera":
                self.div_param1.setPlaceholderText("Radio")
                self.div_param2.setPlaceholderText("Centro X")
                self.div_param3.setPlaceholderText("Centro Y")
            elif region == "Cubo":
                self.div_param1.setPlaceholderText("Lado")
                self.div_param2.setPlaceholderText("Centro X")
                self.div_param3.setPlaceholderText("Centro Y")
            elif region == "Cilindro":
                self.div_param1.setPlaceholderText("Radio")
                self.div_param2.setPlaceholderText("Altura")
                self.div_param3.setPlaceholderText("Eje (x/y/z)")
            elif region == "Cono":
                self.div_param1.setPlaceholderText("Radio base")
                self.div_param2.setPlaceholderText("Altura")
                self.div_param3.setPlaceholderText("Vértice (x,y,z)")
            else:  # Personalizado
                self.div_param1.setPlaceholderText("Límite inferior (x,y,z)")
                self.div_param2.setPlaceholderText("Límite superior (x,y,z)")
                self.div_param3.setPlaceholderText("Frontera (opcional)")
        
        self.div_region.currentTextChanged.connect(update_region_params)
        update_region_params()  # Inicializar placeholders
        
        region_layout.addRow("Tipo de región:", self.div_region)
        region_layout.addRow("Parámetro 1:", self.div_param1)
        region_layout.addRow("Parámetro 2:", self.div_param2)
        region_layout.addRow("Parámetro 3:", self.div_param3)
        
        # Coordenadas a usar (Cartesianas, Cilíndricas, Esféricas)
        self.div_coords = QComboBox()
        self.div_coords.addItems(["Cartesianas", "Cilíndricas", "Esféricas"])
        region_layout.addRow("Coordenadas:", self.div_coords)
        
        region_group.setLayout(region_layout)
        
        # Botón de cálculo
        calc_btn = QPushButton("Aplicar Teorema de la Divergencia")
        calc_btn.clicked.connect(self.aplicar_teorema)
        
        # Agregar todo al layout principal
        layout.addWidget(field_group)
        layout.addWidget(region_group)
        layout.addWidget(calc_btn)
        layout.addStretch()
        
        self.teorema_inputs.addWidget(widget)
    
    def update_teorema_inputs(self):
        """Actualiza los inputs mostrados según el teorema seleccionado"""
        self.teorema_inputs.setCurrentIndex(self.teorema_combo.currentIndex())
    
    # Métodos de cálculo
    def calcular_integral(self):
        """Calcula la integral triple según las coordenadas seleccionadas"""
        try:
            # Obtener la función y los límites
            func_str = self.func_input.text().strip()
            if not func_str:
                raise ValueError("Por favor ingrese una función")
                
            # Convertir la función a una expresión sympy
            x, y, z = sp.symbols('x y z')
            try:
                func = parse_expr(
                    func_str,
                    local_dict={
                        "x": x, "y": y, "z": z,
                        "pi": sp.pi, "E": sp.E, "e": sp.E,
                        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                        "sqrt": sp.sqrt, "log": sp.log, "exp": sp.exp,
                        "Abs": sp.Abs,
                        # Alias de variables angulares y radiales comunes
                        "phi": sp.symbols('phi'), "varphi": sp.symbols('phi'), "ϕ": sp.symbols('phi'), "φ": sp.symbols('phi'),
                        "theta": sp.symbols('theta'), "vartheta": sp.symbols('theta'), "θ": sp.symbols('theta'),
                        "rho": sp.symbols('rho'), "ρ": sp.symbols('rho')
                    }
                )
            except Exception as e:
                raise ValueError(f"Error al analizar la función: {str(e)}")
            
            # Obtener los límites
            try:
                x_lim = [float(lim.text()) for lim in self.x_lim if lim.text()]
                y_lim = [float(lim.text()) for lim in self.y_lim if lim.text()]
                z_lim = [float(lim.text()) for lim in self.z_lim if lim.text()]
                
                if len(x_lim) != 2 or len(y_lim) != 2 or len(z_lim) != 2:
                    raise ValueError("Debe especificar ambos límites para x, y, z")
                
                x_min, x_max = sorted(x_lim)
                y_min, y_max = sorted(y_lim)
                z_min, z_max = sorted(z_lim)
                
            except ValueError as e:
                raise ValueError("Los límites deben ser números válidos")
            
            # Inicializar lista de pasos
            pasos = [
                f"Función a integrar: f(x,y,z) = {sp.pretty(func)}",
                f"Límites de integración:",
                f"  - x ∈ [{x_min}, {x_max}]",
                f"  - y ∈ [{y_min}, {y_max}]",
                f"  - z ∈ [{z_min}, {z_max}]"
            ]
            
            # Calcular la integral según el tipo de coordenadas
            coord_type = self.coord_type.currentText()
            pasos.append(f"\nSistema de coordenadas: {coord_type}")
            
            if coord_type == "Rectangulares":
                pasos.append("\nIntegrando en coordenadas rectangulares (x, y, z):")
                
                # Mostrar la integral original
                integral_str = f"∫∫∫ ({sp.pretty(func)}) dz dy dx"
                pasos.append(f"Expresión original: {integral_str}")
                
                # Integrar en z
                int_z = sp.integrate(func, (z, z_min, z_max))
                pasos.append(f"\n1. Integrando con respecto a z (de {z_min} a {z_max}):")
                pasos.append(f"   ∫({sp.pretty(func)}) dz = {sp.pretty(int_z)}")
                
                # Integrar en y
                int_y = sp.integrate(int_z, (y, y_min, y_max))
                pasos.append(f"\n2. Integrando el resultado con respecto a y (de {y_min} a {y_max}):")
                pasos.append(f"   ∫({sp.pretty(int_z)}) dy = {sp.pretty(int_y)}")
                
                # Integrar en x
                result = sp.integrate(int_y, (x, x_min, x_max))
                pasos.append(f"\n3. Integrando el resultado con respecto a x (de {x_min} a {x_max}):")
                pasos.append(f"   ∫({sp.pretty(int_y)}) dx = {sp.pretty(result)}")
                
            elif coord_type == "Cilíndricas":
                r, theta = sp.symbols('r theta')
                pasos.append("\nConversión a coordenadas cilíndricas (r, θ, z):")
                pasos.append("   x = r·cos(θ)")
                pasos.append("   y = r·sin(θ)")
                pasos.append("   z = z")
                pasos.append("   Jacobiano: |J| = r")
                
                # Aplicar la transformación
                func_cyl = func.subs({
                    x: r * sp.cos(theta),
                    y: r * sp.sin(theta),
                    z: z
                }) * r  # Jacobiano
                
                pasos.append(f"\nFunción en coordenadas cilíndricas: {sp.pretty(func_cyl)}")
                
                # Mostrar la integral
                integral_str = f"∫∫∫ ({sp.pretty(func_cyl)}) dz dr dθ"
                pasos.append(f"\nExpresión a integrar: {integral_str}")
                
                # Integrar en z
                int_z = sp.integrate(func_cyl, (z, z_min, z_max))
                pasos.append(f"\n1. Integrando con respecto a z (de {z_min} a {z_max}):")
                pasos.append(f"   ∫({sp.pretty(func_cyl)}) dz = {sp.pretty(int_z)}")
                
                # Integrar en r
                int_r = sp.integrate(int_z, (r, 0, x_max))
                pasos.append(f"\n2. Integrando el resultado con respecto a r (de 0 a {x_max}):")
                pasos.append(f"   ∫({sp.pretty(int_z)}) dr = {sp.pretty(int_r)}")
                
                # Integrar en theta
                result = sp.integrate(int_r, (theta, 0, 2*sp.pi))
                pasos.append(f"\n3. Integrando el resultado con respecto a θ (de 0 a 2π):")
                pasos.append(f"   ∫({sp.pretty(int_r)}) dθ = {sp.pretty(result)}")
                
            else:  # Esféricas
                rho, phi, theta = sp.symbols('rho phi theta')
                pasos.append("\nConversión a coordenadas esféricas (ρ, φ, θ):")
                pasos.append("   x = ρ·sin(φ)·cos(θ)")
                pasos.append("   y = ρ·sin(φ)·sin(θ)")
                pasos.append("   z = ρ·cos(φ)")
                pasos.append("   Jacobiano: |J| = ρ²·sin(φ)")
                
                # Aplicar la transformación
                func_sph = func.subs({
                    x: rho * sp.sin(phi) * sp.cos(theta),
                    y: rho * sp.sin(phi) * sp.sin(theta),
                    z: rho * sp.cos(phi)
                }) * rho**2 * sp.sin(phi)  # Jacobiano
                
                pasos.append(f"\nFunción en coordenadas esféricas: {sp.pretty(func_sph)}")
                
                # Mostrar la integral
                integral_str = f"∫∫∫ ({sp.pretty(func_sph)}) dρ dφ dθ"
                pasos.append(f"\nExpresión a integrar: {integral_str}")
                
                # Integrar en rho
                int_rho = sp.integrate(func_sph, (rho, 0, x_max))
                pasos.append(f"\n1. Integrando con respecto a ρ (de 0 a {x_max}):")
                pasos.append(f"   ∫({sp.pretty(func_sph)}) dρ = {sp.pretty(int_rho)}")
                
                # Integrar en phi
                int_phi = sp.integrate(int_rho, (phi, 0, sp.pi))
                pasos.append(f"\n2. Integrando el resultado con respecto a φ (de 0 a π):")
                pasos.append(f"   ∫({sp.pretty(int_rho)}) dφ = {sp.pretty(int_phi)}")
                
                # Integrar en theta
                result = sp.integrate(int_phi, (theta, 0, 2*sp.pi))
                pasos.append(f"\n3. Integrando el resultado con respecto a θ (de 0 a 2π):")
                pasos.append(f"   ∫({sp.pretty(int_phi)}) dθ = {sp.pretty(result)}")
            
            # Procedimiento y resumen en LaTeX (función y límites)
            encabezado_latex = [
                f"\\text{{Sistema de coordenadas:}}\\;\\text{{{coord_type}}}",
                f"f(x,y,z) = {sp.latex(func)}",
                f"x \\in [{sp.latex(sp.nsimplify(x_min))}, {sp.latex(sp.nsimplify(x_max))}]",
                f"y \\in [{sp.latex(sp.nsimplify(y_min))}, {sp.latex(sp.nsimplify(y_max))}]",
                f"z \\in [{sp.latex(sp.nsimplify(z_min))}, {sp.latex(sp.nsimplify(z_max))}]",
            ]
            # Combinar encabezado (LaTeX) con pasos (texto plano)
            self._set_math_lines(self.proceso_display, encabezado_latex + pasos)
            self._auto_resize_textedit(self.proceso_display)

            # Resultado final en LaTeX
            resultado_latex = [f"\\text{{Resultado final:}}\\; {sp.latex(sp.simplify(result))}"]
            self._set_math_lines(self.result_display, resultado_latex)
            self._auto_resize_textedit(self.result_display)
            
            # Actualizar visualización 3D: graficar z = f(x, y) con corte en z medio si aplica
            try:
                z_mid = (z_min + z_max) / 2
                func_xy = func.subs({z: z_mid})
                # Intentar graficar superficie en el rango de x,y provistos
                self.visualizador3d.graficar_superficie(func_xy, x_range=(x_min, x_max), y_range=(y_min, y_max))
            except Exception:
                # Si no es posible graficar, continuar sin interrumpir el flujo
                pass
            
            # Asegurarse de que estamos en la pestaña de Integrales
            self.tabs.setCurrentIndex(0)
            
        except Exception as e:
            error_msg = f"Error al calcular la integral: {str(e)}"
            self._set_math_lines(self.proceso_display, [f"\\text{{{error_msg}}}"])
            self.result_display.clear()
            self._auto_resize_textedit(self.proceso_display)
            self._auto_resize_textedit(self.result_display)
            # Asegurarse de que estamos en la pestaña de Integrales incluso si hay error
            self.tabs.setCurrentIndex(0)
    
    def insert_into_func(self, text: str, target_field=None):
        """Inserta texto en el QLineEdit de función, posicionando el cursor inteligentemente.
        
        Args:
            text: El texto a insertar
            target_field: El campo QLineEdit donde se insertará el texto. Si es None, se usa self.func_input
        """
        edit = target_field if target_field is not None else self.func_input
        pos = edit.cursorPosition()
        current_text = edit.text()
        new_text = current_text[:pos] + text + current_text[pos:]
        edit.setText(new_text)
        edit.setCursorPosition(pos + len(text))
        if text.endswith("()"):
            edit.setCursorPosition(pos + len(text) - 1)
        else:
            edit.setCursorPosition(pos + len(text))
        edit.setFocus()

    def _auto_resize_textedit(self, text_edit: QTextEdit, max_height: int = 1200):
        """Ajusta la altura del QTextEdit al contenido para evitar scroll interno."""
        doc = text_edit.document()
        doc.adjustSize()
        h = int(doc.size().height()) + 12  # padding
        h = max(60, min(h, max_height))
        text_edit.setFixedHeight(h)
    
    def mostrar_procedimiento_teorema(self, titulo, pasos):
        """Muestra un procedimiento paso a paso en el área de proceso de teoremas"""
        try:
            # Limpiar el área de proceso
            self.teorema_proceso.clear()
            
            # Construir el contenido HTML
            html_content = [f"<h2>{titulo}</h2><br>"]
            
            # Agregar cada paso
            for i, paso in enumerate(pasos, 1):
                # Si el paso es una cadena, mostrarlo como párrafo
                if isinstance(paso, str):
                    # Reemplazar ^ con formato de superíndice
                    paso = paso.replace('^', '<sup>').replace('</sup>', '') + ('</sup>' if '^' in paso else '')
                    # Si parece una ecuación (contiene operadores matemáticos)
                    if any(c in paso for c in ['=', '∫', '∂', '∑', '∬', '∭', '∇', '·', '×']):
                        html_content.append(f"<div style='margin: 10px 0; padding: 10px; background-color: #f8f9fa; border-left: 4px solid #3498db; border-radius: 4px;'>{paso}</div>")
                    else:
                        html_content.append(f"<p style='margin: 10px 0;'>{paso}</p>")
                # Si el paso es una lista, asumir que son subpasos
                elif isinstance(paso, list):
                    for subpaso in paso:
                        subpaso = subpaso.replace('^', '<sup>').replace('</sup>', '') + ('</sup>' if '^' in subpaso else '')
                        html_content.append(f"<p style='margin: 5px 0 5px 20px;'>{subpaso}</p>")
                
                # Agregar línea divisoria entre pasos (excepto después del último)
                if i < len(pasos):
                    html_content.append("<hr style='margin: 10px 0; border: 0; border-top: 1px solid #e0e0e0;'>")
            
            # Establecer el contenido HTML en el QTextEdit
            self.teorema_proceso.setHtml("".join(html_content))
            
            # Ajustar el tamaño del área de proceso
            self._auto_resize_textedit(self.teorema_proceso)
            
        except Exception as e:
            self.teorema_result.setPlainText(f"Error al mostrar el procedimiento: {str(e)}")
            self._auto_resize_textedit(self.teorema_result)
    
    def aplicar_teorema(self):
        """Aplica el teorema vectorial seleccionado"""
        teorema = self.teorema_combo.currentText()
        
        try:
            # Limpiar resultados anteriores
            self.teorema_proceso.clear()
            self.teorema_result.clear()
            
            # Asegurarse de que estamos en la pestaña de Teoremas
            self.tabs.setCurrentIndex(1)  # Índice 1 es la pestaña de Teoremas
            
            # Forzar la actualización de la interfaz
            self.repaint()
            
            # Aplicar el teorema correspondiente
            if teorema == "Teorema de Green":
                self.aplicar_green()
            elif teorema == "Teorema de Stokes":
                self.aplicar_stokes()
            else:  # Teorema de la Divergencia
                self.aplicar_divergencia()
            
            # Asegurarse de que estamos en la pestaña de Teoremas después de la ejecución
            self.tabs.setCurrentIndex(1)
            
        except Exception as e:
            error_msg = f"Error al aplicar el teorema: {str(e)}"
            self._set_math_lines(self.teorema_result, [f"\\text{{{error_msg}}}"])
            self._set_math_lines(self.teorema_proceso, [f"\\text{{{error_msg}}}"])
            self._auto_resize_textedit(self.teorema_result)
            self._auto_resize_textedit(self.teorema_proceso)
            # Asegurarse de que estamos en la pestaña de Teoremas incluso si hay error
            self.tabs.setCurrentIndex(1)
    
    def aplicar_green(self):
        """Aplica el Teorema de Green"""
        try:
            # Limpiar resultados anteriores
            self.teorema_proceso.clear()
            self.teorema_result.clear()
            
            # Asegurarse de que estamos en la pestaña de Teoremas
            self.tabs.setCurrentIndex(1)
            self.repaint()  # Forzar actualización de la interfaz
            
            # Obtener las funciones P y Q
            P_str = self.green_p.text().strip()
            Q_str = self.green_q.text().strip()
            
            if not P_str or not Q_str:
                raise ValueError("Por favor ingrese las funciones P y Q")
                
            # Obtener los límites de integración
            try:
                x_min = float(self.green_xmin.text().strip())
                x_max = float(self.green_xmax.text().strip())
                y_min = float(self.green_ymin.text().strip())
                y_max = float(self.green_ymax.text().strip())
            except ValueError:
                raise ValueError("Por favor ingrese valores numéricos válidos para los límites")
            
            # Crear símbolos
            x, y = sp.symbols('x y')
            
            # Parsear las expresiones
            P = sp.parsing.sympy_parser.parse_expr(P_str, local_dict={"x": x, "y": y})
            Q = sp.parsing.sympy_parser.parse_expr(Q_str, local_dict={"x": x, "y": y})
            
            # Calcular las derivadas parciales
            self._auto_resize_textedit(self.teorema_result)
            
            # Asegurarse de que estamos en la pestaña de Teoremas
            self.tabs.setCurrentIndex(1)
            
            return resultado
            
        except Exception as e:
            error_msg = f"Error al aplicar el Teorema de Green: {str(e)}"
            self.teorema_result.setPlainText(error_msg)
            self.teorema_proceso.setHtml(f"<div style='color: red;'>{error_msg}</div>")
            self._auto_resize_textedit(self.teorema_result)
            self._auto_resize_textedit(self.teorema_proceso)
            # Asegurarse de que estamos en la pestaña de Teoremas incluso si hay error
            self.tabs.setCurrentIndex(1)
            raise

    def aplicar_stokes(self):
        """Aplica el Teorema de Stokes"""
        try:
            # Limpiar resultados anteriores
            self.teorema_proceso.clear()
            self.teorema_result.clear()
            
            # Asegurarse de que estamos en la pestaña de Teoremas
            self.tabs.setCurrentIndex(1)
            self.repaint()  # Forzar actualización de la interfaz
            
            # Obtener los componentes del campo vectorial
            F1_str = self.stokes_f1.text().strip()
            F2_str = self.stokes_f2.text().strip()
            F3_str = self.stokes_f3.text().strip()
            
            if not F1_str or not F2_str or not F3_str:
                raise ValueError("Por favor ingrese todos los componentes del campo vectorial")
            
            # Obtener la superficie y parámetros
            superficie = self.stokes_superficie.currentText()
            param1 = self.stokes_param1.text().strip()
            param2 = self.stokes_param2.text().strip()
            param3 = self.stokes_param3.text().strip()
            
            # Convertir los parámetros a valores numéricos si es posible
            try:
                parametros = {}
                if param1: 
                    parametros['param1'] = float(param1) if param1.replace('.', '', 1).isdigit() else param1
                if param2: 
                    parametros['param2'] = float(param2) if param2.replace('.', '', 1).isdigit() else param2
                if param3: 
                    parametros['param3'] = float(param3) if param3.replace('.', '', 1).isdigit() else param3
            except ValueError:
                # Si no se pueden convertir a float, asumir que son expresiones simbólicas
                parametros = {
                    'param1': param1 if param1 else None,
                    'param2': param2 if param2 else None,
                    'param3': param3 if param3 else None
                }
            
            # Símbolos para las variables
            x, y, z = sp.symbols('x y z')
            
            # Parsear las expresiones
            F1 = sp.parsing.sympy_parser.parse_expr(F1_str, local_dict={"x": x, "y": y, "z": z})
            F2 = sp.parsing.sympy_parser.parse_expr(F2_str, local_dict={"x": x, "y": y, "z": z})
            F3 = sp.parsing.sympy_parser.parse_expr(F3_str, local_dict={"x": x, "y": y, "z": z})
            
            # Iniciar el procedimiento
            pasos = [
                "<b>Teorema de Stokes:</b>",
                "∮_C F·dr = ∬_S (∇ × F)·dS",
                "",
                "<b>Datos de entrada:</b>",
                f"F(x,y,z) = ({F1}, {F2}, {F3})",
                f"Superficie: {superficie}",
                f"Parámetros: {parametros}",
                "",
                "<b>1. Cálculo del rotacional de F (∇ × F):</b>",
                "   ∇ × F = (∂F₃/∂y - ∂F₂/∂z, ∂F₁/∂z - ∂F₃/∂x, ∂F₂/∂x - ∂F₁/∂y)"
            ]
            
            # Calcular el rotacional
            rot_F1 = sp.diff(F3, y) - sp.diff(F2, z)
            rot_F2 = sp.diff(F1, z) - sp.diff(F3, x)
            rot_F3 = sp.diff(F2, x) - sp.diff(F1, y)
            
            pasos.extend([
                f"   ∂F₃/∂y - ∂F₂/∂z = {sp.diff(F3, y)} - {sp.diff(F2, z)} = {rot_F1}",
                f"   ∂F₁/∂z - ∂F₃/∂x = {sp.diff(F1, z)} - {sp.diff(F3, x)} = {rot_F2}",
                f"   ∂F₂/∂x - ∂F₁/∂y = {sp.diff(F2, x)} - {sp.diff(F1, y)} = {rot_F3}",
                "",
                f"   ∇ × F = ({rot_F1}, {rot_F2}, {rot_F3})",
                "",
                "<b>2. Cálculo de la integral de superficie sobre S:</b>"
            ])
            
            # Llamar a la función del teorema de Stokes
            from calculos.teoremas import teorema_stokes
            try:
                resultado = teorema_stokes((F1, F2, F3), superficie, parametros)
                pasos.extend([
                    f"   ∬_S (∇ × F)·dS = {resultado}",
                    "",
                    "<b>3. Aplicación del teorema de Stokes:</b>"
                ])
                
                # Mostrar el procedimiento paso a paso en formato matemático
                self._set_math_lines(self.teorema_proceso, pasos)
                self._auto_resize_textedit(self.teorema_proceso)
                
                # Mostrar solo el resultado final en formato matemático
                self._set_math_lines(self.teorema_result, [sp.latex(sp.simplify(resultado))])
                self._auto_resize_textedit(self.teorema_result)
                
            except Exception as e:
                error_msg = f"Error al calcular la integral de superficie: {str(e)}"
                pasos.append(f"\nError: {error_msg}")
                self._set_math_lines(self.teorema_proceso, pasos)
                self._set_math_lines(self.teorema_result, [f"\\text{{Error: {error_msg}}}"])
                self._auto_resize_textedit(self.teorema_proceso)
                self._auto_resize_textedit(self.teorema_result)
                raise
                
        except Exception as e:
            error_msg = f"Error al aplicar el Teorema de Stokes: {str(e)}"
            self._set_math_lines(self.teorema_result, [f"\\text{{{error_msg}}}"])
            self._set_math_lines(self.teorema_proceso, [f"\\text{{{error_msg}}}"])
            self._auto_resize_textedit(self.teorema_result)
            self._auto_resize_textedit(self.teorema_proceso)
            # Asegurarse de que estamos en la pestaña de Teoremas incluso si hay error
            self.tabs.setCurrentIndex(1)
            raise
    
    def aplicar_divergencia(self):
        """Aplica el Teorema de la Divergencia"""
        try:
            F1_str = self.div_f1.text().strip()
            F2_str = self.div_f2.text().strip()
            F3_str = self.div_f3.text().strip()
            
            if not all([F1_str, F2_str, F3_str]):
                raise ValueError("Por favor ingrese todas las componentes del campo vectorial")
            
            # Obtener los parámetros de la región
            region = self.div_region.currentText()
            param1 = self.div_param1.text().strip()
            param2 = self.div_param2.text().strip()
            param3 = self.div_param3.text().strip()
            coord_type = self.div_coords.currentText()
            
            # Convertir los parámetros a valores numéricos si es posible
            try:
                parametros = {}
                if param1: parametros['param1'] = float(param1) if param1.replace('.', '', 1).isdigit() else param1
                if param2: parametros['param2'] = float(param2) if param2.replace('.', '', 1).isdigit() else param2
                if param3: parametros['param3'] = float(param3) if param3.replace('.', '', 1).isdigit() else param3
            except ValueError:
                # Si no se pueden convertir a float, asumir que son expresiones simbólicas
                parametros = {
                    'param1': param1 if param1 else None,
                    'param2': param2 if param2 else None,
                    'param3': param3 if param3 else None
                }
            
            # Símbolos para las variables
            x, y, z = sp.symbols('x y z')
            
            # Parsear las expresiones
            F1 = parse_expr(F1_str, local_dict={"x": x, "y": y, "z": z})
            F2 = parse_expr(F2_str, local_dict={"x": x, "y": y, "z": z})
            F3 = parse_expr(F3_str, local_dict={"x": x, "y": y, "z": z})
            
            # Iniciar el procedimiento
            pasos = [
                "Aplicando el Teorema de la Divergencia:",
                "∯_S F·dS = ∭_V (∇·F) dV",
                "",
                f"Datos de entrada:",
                f"F(x,y,z) = ({F1}, {F2}, {F3})",
                f"Región: {region}",
                f"Tipo de coordenadas: {coord_type}",
                f"Parámetros: {parametros}",
                "",
                "Paso 1: Calcular la divergencia de F (∇·F)",
                "   ∇·F = ∂F₁/∂x + ∂F₂/∂y + ∂F₃/∂z"
            ]
            
            # Calcular la divergencia
            div_F1 = sp.diff(F1, x)
            div_F2 = sp.diff(F2, y)
            div_F3 = sp.diff(F3, z)
            div_F = div_F1 + div_F2 + div_F3
            
            pasos.extend([
                f"   ∂F₁/∂x = {div_F1}",
                f"   ∂F₂/∂y = {div_F2}",
                f"   ∂F₃/∂z = {div_F3}",
                "",
                f"   ∇·F = {div_F1} + {div_F2} + {div_F3} = {div_F}",
                "",
                f"Paso 2: Calcular la integral de volumen en coordenadas {coord_type}"
            ])
            
            # Llamar a la función del teorema de la divergencia
            from calculos.teoremas import teorema_divergencia
            try:
                resultado = teorema_divergencia((F1, F2, F3), region, parametros, coord_type)
                pasos.extend([
                    f"   ∭_V (∇·F) dV = {resultado}",
                    "",
                    "Paso 3: Aplicar el teorema de la divergencia"
                ])
                
                # Mostrar el procedimiento paso a paso en formato matemático
                self._set_math_lines(self.teorema_proceso, pasos)
                self._auto_resize_textedit(self.teorema_proceso)
                
                # Mostrar solo el resultado final en formato matemático
                self._set_math_lines(self.teorema_result, [sp.latex(sp.simplify(resultado))])
                self._auto_resize_textedit(self.teorema_result)
                
            except Exception as e:
                error_msg = f"Error al calcular la integral de volumen: {str(e)}"
                pasos.append(f"Error: {error_msg}")
                self._set_math_lines(self.teorema_proceso, pasos)
                self._set_math_lines(self.teorema_result, [f"\\text{{Error: {error_msg}}}"])
                self._auto_resize_textedit(self.teorema_proceso)
                self._auto_resize_textedit(self.teorema_result)
                raise
                
        except Exception as e:
            error_msg = f"Error al aplicar el Teorema de la Divergencia: {str(e)}"
            self._set_math_lines(self.teorema_proceso, [f"\\text{{Error: {error_msg}}}"])
            self._set_math_lines(self.teorema_result, [f"\\text{{Error: {error_msg}}}"])
            self._auto_resize_textedit(self.teorema_proceso)
            self._auto_resize_textedit(self.teorema_result)
            # Asegurarse de que estamos en la pestaña de Teoremas incluso si hay error
            self.tabs.setCurrentIndex(1)
            raise
    
    def mostrar_procedimiento(self, titulo, pasos):
        """Muestra un procedimiento paso a paso en el área de proceso"""
        try:
            # Limpiar el área de proceso
            self.teorema_proceso.clear()
            
            # Construir el contenido como texto plano
            texto_contenido = f"{titulo}\n" + "="*len(titulo) + "\n\n"
            
            # Agregar cada paso
            for paso in pasos:
                if isinstance(paso, str):
                    # Limpiar etiquetas HTML y caracteres especiales
                    paso = paso.replace('<b>', '').replace('</b>', '')
                    paso = paso.replace('<br>', '\n').replace('</div>', '')
                    paso = paso.replace('<div style="color: red;">', '¡Error! ')
                    texto_contenido += paso + "\n\n"
                elif isinstance(paso, list):
                    for subpaso in paso:
                        subpaso = subpaso.replace('<b>', '').replace('</b>', '')
                        subpaso = '  • ' + subpaso if not subpaso.strip().startswith('•') else '  ' + subpaso
                        texto_contenido += subpaso + "\n"
                    texto_contenido += "\n"
            
            # Establecer el contenido en texto plano
            self.teorema_proceso.setPlainText(texto_contenido.strip())
            
            # Configurar la fuente para mejor legibilidad
            font = self.teorema_proceso.font()
            font.setFamily('Consolas')
            font.setPointSize(10)
            self.teorema_proceso.setFont(font)
            
            # Ajustar el tamaño del área de proceso
            self._auto_resize_textedit(self.teorema_proceso)
            
            # Asegurarse de que estamos en la pestaña de Teoremas
            self.tabs.setCurrentIndex(1)
            
        except Exception as e:
            self._set_math_lines(self.teorema_result, [f"\\text{{Error al mostrar el procedimiento: {str(e)} }}"]) 
            self._auto_resize_textedit(self.teorema_result)
            # Asegurarse de que estamos en la pestaña de Teoremas
            self.tabs.setCurrentIndex(1)
    
    # Esta función se mantiene por compatibilidad, pero redirige a mostrar_procedimiento
    def plot_3d_function(self, titulo, pasos):
        """Muestra un procedimiento paso a paso (compatibilidad con código existente)"""
        self.mostrar_procedimiento(titulo, pasos)
            
    def aplicar_green(self):
        """Aplica el Teorema de Green"""
        try:
            # Limpiar resultados anteriores
            self.teorema_proceso.clear()
            self.teorema_result.clear()
            
            # Asegurarse de que estamos en la pestaña de Teoremas
            self.tabs.setCurrentIndex(1)
            self.repaint()  # Forzar actualización de la interfaz
            
            # Obtener las funciones P y Q
            P_str = self.green_p.text().strip()
            Q_str = self.green_q.text().strip()
            
            if not P_str or not Q_str:
                raise ValueError("Por favor ingrese las funciones P y Q")
                
            # Obtener los límites de integración (usando valores por defecto si no están definidos)
            try:
                x_min = float(self.green_xmin.text().strip()) if hasattr(self, 'green_xmin') and self.green_xmin.text().strip() else -1
                x_max = float(self.green_xmax.text().strip()) if hasattr(self, 'green_xmax') and self.green_xmax.text().strip() else 1
                y_min = float(self.green_ymin.text().strip()) if hasattr(self, 'green_ymin') and self.green_ymin.text().strip() else -1
                y_max = float(self.green_ymax.text().strip()) if hasattr(self, 'green_ymax') and self.green_ymax.text().strip() else 1
            except ValueError:
                x_min, x_max, y_min, y_max = -1, 1, -1, 1
            
            # Crear símbolos
            x, y = sp.symbols('x y')
            
            # Parsear las expresiones
            P = sp.parsing.sympy_parser.parse_expr(P_str, local_dict={"x": x, "y": y})
            Q = sp.parsing.sympy_parser.parse_expr(Q_str, local_dict={"x": x, "y": y})
            
            # Calcular las derivadas parciales
            dQ_dx = sp.diff(Q, x)
            dP_dy = sp.diff(P, y)
            integrando = dQ_dx - dP_dy
            
            # Calcular la integral doble
            resultado = sp.integrate(
                sp.integrate(integrando, (y, y_min, y_max)),
                (x, x_min, x_max)
            )
            
            # Mostrar el procedimiento paso a paso (LaTeX)
            pasos_green = [
                "\\text{Teorema de Green}",
                "\\oint_C (P\,dx + Q\,dy) = \\iint_D (\\partial Q/\\partial x - \\partial P/\\partial y)\,dA",
                "\\text{Datos de entrada:}",
                f"P(x,y) = {sp.latex(P)}",
                f"Q(x,y) = {sp.latex(Q)}",
                f"x \\in [{x_min}, {x_max}]",
                f"y \\in [{y_min}, {y_max}]",
                "\\text{1. Cálculo de las derivadas parciales:}",
                f"\\partial Q/\\partial x = {sp.latex(dQ_dx)}",
                f"\\partial P/\\partial y = {sp.latex(dP_dy)}",
                "\\text{2. Aplicación del teorema:}",
                f"\\oint_C (P\,dx + Q\,dy) = \\iint_D ({sp.latex(dQ_dx)} - {sp.latex(dP_dy)})\,dA = \\iint_D ({sp.latex(integrando)})\,dA",
                "\\text{3. Cálculo de la integral doble:}",
                f"x \\in [{x_min}, {x_max}],\quad y \\in [{y_min}, {y_max}]",
            ]
            self._set_math_lines(self.teorema_proceso, pasos_green)
            self._auto_resize_textedit(self.teorema_proceso)
            
            # Mostrar solo el resultado final (LaTeX)
            self._set_math_lines(self.teorema_result, [sp.latex(sp.simplify(resultado))])
            self._auto_resize_textedit(self.teorema_result)
            
            # Asegurarse de que estamos en la pestaña de Teoremas
            self.tabs.setCurrentIndex(1)
            
            return resultado
            
        except Exception as e:
            error_msg = f"Error al aplicar el Teorema de Green: {str(e)}"
            self._set_math_lines(self.teorema_result, [f"\\text{{{error_msg}}}"])
            self._set_math_lines(self.teorema_proceso, [f"\\text{{{error_msg}}}"])
            self._auto_resize_textedit(self.teorema_result)
            self._auto_resize_textedit(self.teorema_proceso)
            # Asegurarse de que estamos en la pestaña de Teoremas incluso si hay error
            self.tabs.setCurrentIndex(1)
            raise

    def aplicar_stokes(self):
        """Aplica el Teorema de Stokes"""
        try:
            F1_str = self.stokes_f1.text().strip()
            F2_str = self.stokes_f2.text().strip()
            F3_str = self.stokes_f3.text().strip()
            
            if not all([F1_str, F2_str, F3_str]):
                raise ValueError("Por favor ingrese todas las componentes del campo vectorial")
            
            x, y, z = sp.symbols('x y z')
            F1 = parse_expr(F1_str, local_dict={"x": x, "y": y, "z": z})
            F2 = parse_expr(F2_str, local_dict={"x": x, "y": y, "z": z})
            F3 = parse_expr(F3_str, local_dict={"x": x, "y": y, "z": z})
            
            # Calcular el rotacional de F = (F1, F2, F3)
            # rot(F) = (∂F3/∂y - ∂F2/∂z, ∂F1/∂z - ∂F3/∂x, ∂F2/∂x - ∂F1/∂y)
            rot_F1 = sp.diff(F3, y) - sp.diff(F2, z)
            rot_F2 = sp.diff(F1, z) - sp.diff(F3, x)
            rot_F3 = sp.diff(F2, x) - sp.diff(F1, y)
            
            # Para simplificar, asumiremos una superficie plana z = 1 - x - y en el primer octante
            # En una aplicación real, esto debería ser configurable
            
            # Calcular la integral de superficie del rotacional
            # ∬_S (∇ × F) · dS = ∬_D (∇ × F) · (r_u × r_v) du dv
            # Para z = 1 - x - y, la normal es (1, 1, 1)
            
            # Proyectamos sobre el plano xy: D es el triángulo 0 ≤ x ≤ 1, 0 ≤ y ≤ 1-x
            # El producto punto (rot_F) · (1,1,1) = rot_F1 + rot_F2 + rot_F3
            integrando = rot_F1 + rot_F2 + rot_F3
            
            # Evaluamos z = 1 - x - y en el integrando
            integrando = integrando.subs(z, 1 - x - y)
            
            # Calculamos la integral doble
            result = sp.integrate(
                sp.integrate(integrando, (y, 0, 1 - x)),
                (x, 0, 1)
            )
            
            # Procedimiento en LaTeX
            pasos_stokes = [
                "\\text{Teorema de Stokes}",
                "\\oint_C F\\cdot dr = \\iint_S (\\nabla \\times F)\\cdot dS",
                "\\text{Datos de entrada:}",
                f"F(x,y,z) = ({sp.latex(F1)}, {sp.latex(F2)}, {sp.latex(F3)})",
                "\\text{1. Cálculo del rotacional de F (\\nabla \\times F):}",
                "\\nabla \\times F = (\\partial F_3/\\partial y - \\partial F_2/\\partial z,\\; \\partial F_1/\\partial z - \\partial F_3/\\partial x,\\; \\partial F_2/\\partial x - \\partial F_1/\\partial y)",
                f"\\partial F_3/\\partial y - \\partial F_2/\\partial z = {sp.latex(sp.diff(F3, y))} - {sp.latex(sp.diff(F2, z))} = {sp.latex(rot_F1)}",
                f"\\partial F_1/\\partial z - \\partial F_3/\\partial x = {sp.latex(sp.diff(F1, z))} - {sp.latex(sp.diff(F3, x))} = {sp.latex(rot_F2)}",
                f"\\partial F_2/\\partial x - \\partial F_1/\\partial y = {sp.latex(sp.diff(F2, x))} - {sp.latex(sp.diff(F1, y))} = {sp.latex(rot_F3)}",
                f"\\nabla \\times F = ({sp.latex(rot_F1)}, {sp.latex(rot_F2)}, {sp.latex(rot_F3)})",
                "\\text{2. Integral de superficie sobre S (z=1-x-y):}",
                f"\\iint_S (\\nabla \\times F)\\cdot dS = \\iint_D ({sp.latex(rot_F1)} + {sp.latex(rot_F2)} + {sp.latex(rot_F3)})\,dA",
                f"\\text{{Sustituyendo }} z = 1 - x - y: \\; {sp.latex(integrando)}",
                "0 \\leq x \\leq 1,\\; 0 \\leq y \\leq 1-x",
            ]
            self._set_math_lines(self.teorema_proceso, pasos_stokes)
            self._auto_resize_textedit(self.teorema_proceso)
            
            # Mostrar solo el resultado final en LaTeX
            self._set_math_lines(self.teorema_result, [sp.latex(sp.simplify(result))])
            self._auto_resize_textedit(self.teorema_result)
            
            # Asegurarse de que estamos en la pestaña de Teoremas
            self.tabs.setCurrentIndex(1)
            
        except Exception as e:
            raise Exception(f"Error en el Teorema de Stokes: {str(e)}")

    def aplicar_divergencia(self):
        """Aplica el Teorema de la Divergencia"""
        try:
            # Limpiar resultados anteriores
            self.teorema_proceso.clear()
            self.teorema_result.clear()
            
            # Asegurarse de que estamos en la pestaña de Teoremas
            self.tabs.setCurrentIndex(1)
            self.repaint()  # Forzar actualización de la interfaz
            
            # Obtener las componentes del campo vectorial
            F1_str = self.div_f1.text().strip()
            F2_str = self.div_f2.text().strip()
            F3_str = self.div_f3.text().strip()
            
            if not all([F1_str, F2_str, F3_str]):
                raise ValueError("Por favor ingrese todas las componentes del campo vectorial")
            
            # Obtener el tipo de región seleccionada
            region_type = self.div_region.currentText()
            
            # Obtener los parámetros según el tipo de región
            if region_type == "Esfera":
                try:
                    radio = float(self.div_param1.text().strip() or "1")
                    x0 = float(self.div_param2.text().strip() or "0")
                    y0 = float(self.div_param3.text().strip() or "0")
                    z0 = float(self.div_param3_2.text().strip() or "0") if hasattr(self, 'div_param3_2') else 0.0
                except ValueError:
                    raise ValueError("Los parámetros de la esfera deben ser números válidos")
                
                # Validar radio positivo
                if radio <= 0:
                    raise ValueError("El radio debe ser un número positivo")
                
                # Mostrar información de la región
                region_info = (
                    f"Región: Esfera\n"
                    f"  - Radio: {radio}\n"
                    f"  - Centro: ({x0}, {y0}, {z0})"
                )
                
                # Límites de integración para la esfera
                rho_lim = (0, radio)
                theta_lim = (0, 2*sp.pi)
                phi_lim = (0, sp.pi)
                
            elif region_type == "Cubo":
                try:
                    lado = float(self.div_param1.text().strip() or "2")
                    x0 = float(self.div_param2.text().strip() or "0")
                    y0 = float(self.div_param3.text().strip() or "0")
                    z0 = float(self.div_param3_2.text().strip() or "0") if hasattr(self, 'div_param3_2') else 0.0
                except ValueError:
                    raise ValueError("Los parámetros del cubo deben ser números válidos")
                
                if lado <= 0:
                    raise ValueError("El lado del cubo debe ser positivo")
                
                region_info = (
                    f"Región: Cubo\n"
                    f"  - Lado: {lado}\n"
                    f"  - Centro: ({x0}, {y0}, {z0})"
                )
                
                # Límites de integración para el cubo
                x_lim = (x0 - lado/2, x0 + lado/2)
                y_lim = (y0 - lado/2, y0 + lado/2)
                z_lim = (z0 - lado/2, z0 + lado/2)
                
            elif region_type == "Cilindro":
                try:
                    radio = float(self.div_param1.text().strip() or "1")
                    altura = float(self.div_param2.text().strip() or "2")
                    eje = self.div_param3.text().strip().lower() or "z"
                    if eje not in ['x', 'y', 'z']:
                        eje = 'z'  # Valor por defecto si no es válido
                except ValueError:
                    raise ValueError("Los parámetros del cilindro deben ser números válidos")
                
                if radio <= 0 or altura <= 0:
                    raise ValueError("El radio y la altura deben ser positivos")
                
                region_info = (
                    f"Región: Cilindro\n"
                    f"  - Radio: {radio}\n"
                    f"  - Altura: {altura}\n"
                    f"  - Eje: {eje.upper()}"
                )
                
                # Límites de integración para el cilindro (eje z por defecto)
                if eje == 'z':
                    r_lim = (0, radio)
                    theta_lim = (0, 2*sp.pi)
                    z_lim = (-altura/2, altura/2)
                elif eje == 'x':
                    r_lim = (0, radio)
                    theta_lim = (0, 2*sp.pi)
                    x_lim = (-altura/2, altura/2)
                else:  # eje 'y'
                    r_lim = (0, radio)
                    theta_lim = (0, 2*sp.pi)
                    y_lim = (-altura/2, altura/2)
                    
            else:
                raise ValueError("Tipo de región no soportado")
            
            # Obtener el sistema de coordenadas seleccionado
            coord_system = self.div_coords.currentText().lower()
            
            # Crear símbolos para las variables
            x, y, z = sp.symbols('x y z')
            rho, phi, theta = sp.symbols('rho phi theta')
            
            # Parsear las expresiones del campo vectorial
            F1 = sp.parsing.sympy_parser.parse_expr(F1_str, local_dict={"x": x, "y": y, "z": z})
            F2 = sp.parsing.sympy_parser.parse_expr(F2_str, local_dict={"x": x, "y": y, "z": z})
            F3 = sp.parsing.sympy_parser.parse_expr(F3_str, local_dict={"x": x, "y": y, "z": z})
            
            # Calcular la divergencia de F = (F1, F2, F3)
            div_F = sp.diff(F1, x) + sp.diff(F2, y) + sp.diff(F3, z)
            
            # Inicializar el procedimiento
            procedimiento = [
                "Teorema de la Divergencia",
                "∯_S F·dS = ∭_V (∇·F) dV\n",
                "Datos de entrada:",
                f"F(x,y,z) = ({F1}, {F2}, {F3})",
                region_info,
                f"Sistema de coordenadas: {coord_system.capitalize()}\n",
                "1. Cálculo de la divergencia (∇·F):",
                "   ∇·F = ∂F₁/∂x + ∂F₂/∂y + ∂F₃/∂z",
                f"   ∂F₁/∂x = {sp.diff(F1, x)}",
                f"   ∂F₂/∂y = {sp.diff(F2, y)}",
                f"   ∂F₃/∂z = {sp.diff(F3, z)}",
                f"   ∇·F = {sp.diff(F1, x)} + {sp.diff(F2, y)} + {sp.diff(F3, z)}",
                f"   ∇·F = {div_F}\n"
            ]
            
            # Calcular la integral según el sistema de coordenadas
            if coord_system == "esféricas" or region_type == "Esfera":
                # Usar coordenadas esféricas
                procedimiento.extend([
                    "2. Cambio a coordenadas esféricas:",
                    "   x = x0 + ρ·sin(φ)·cos(θ)",
                    "   y = y0 + ρ·sin(φ)·sin(θ)",
                    "   z = z0 + ρ·cos(φ)",
                    "   Jacobiano: ρ²·sin(φ)\n"
                ])
                
                # Expresar la divergencia en coordenadas esféricas
                div_F_sph = div_F.subs({
                    x: x0 + rho * sp.sin(phi) * sp.cos(theta),
                    y: y0 + rho * sp.sin(phi) * sp.sin(theta),
                    z: z0 + rho * sp.cos(phi)
                }) * rho**2 * sp.sin(phi)  # Jacobiano
                
                procedimiento.extend([
                    "3. Cálculo de la integral triple:",
                    f"   ∭_V (∇·F) dV = ∫₀^π ∫₀^2π ∫₀^{radio} ({sp.pretty(div_F_sph)}) dρ dθ dφ",
                    f"   Límites: 0 ≤ ρ ≤ {radio}, 0 ≤ θ ≤ 2π, 0 ≤ φ ≤ π"
                ])
                
                # Calcular la integral
                try:
                    resultado = sp.integrate(
                        sp.integrate(
                            sp.integrate(div_F_sph, (rho, *rho_lim)),
                            (theta, *theta_lim)
                        ),
                        (phi, *phi_lim)
                    )
                    resultado = sp.simplify(resultado)
                    
                except Exception as e:
                    raise Exception(f"Error al calcular la integral: {str(e)}")
                
            elif coord_system == "cilíndricas" or region_type == "Cilindro":
                # Usar coordenadas cilíndricas
                procedimiento.extend([
                    "2. Cambio a coordenadas cilíndricas:",
                    f"   {'z' if eje == 'z' else 'x' if eje == 'x' else 'y'} = {eje}",
                    f"   {'x' if eje != 'x' else 'y'} = r·cos(θ)",
                    f"   {'y' if eje != 'y' else 'x'} = r·sin(θ)",
                    "   Jacobiano: r\n"
                ])
                
                # Expresar la divergencia en coordenadas cilíndricas
                if eje == 'z':
                    div_F_cyl = div_F.subs({
                        x: 'r*cos(theta)',
                        y: 'r*sin(theta)',
                        z: 'z'
                    }) * r  # Jacobiano
                elif eje == 'x':
                    div_F_cyl = div_F.subs({
                        y: 'r*cos(theta)',
                        z: 'r*sin(theta)',
                        x: 'x'
                    }) * r  # Jacobiano
                else:  # eje 'y'
                    div_F_cyl = div_F.subs({
                        x: 'r*cos(theta)',
                        z: 'r*sin(theta)',
                        y: 'y'
                    }) * r  # Jacobiano
                
                procedimiento.extend([
                    "3. Cálculo de la integral triple:",
                    f"   ∭_V (∇·F) dV = "
                    f"∫_{z_lim[0] if eje == 'z' else x_lim[0] if eje == 'x' else y_lim[0]}^"
                    f"{z_lim[1] if eje == 'z' else x_lim[1] if eje == 'x' else y_lim[1]} "
                    f"∫_0^{2*sp.pi} ∫_0^{radio} ({sp.pretty(div_F_cyl)}) dr dθ "
                    f"{'dz' if eje == 'z' else 'dx' if eje == 'x' else 'dy'}",
                    f"   Límites: 0 ≤ r ≤ {radio}, 0 ≤ θ ≤ 2π, "
                    f"{'z' if eje == 'z' else 'x' if eje == 'x' else 'y'} ∈ "
                    f"[{z_lim[0] if eje == 'z' else x_lim[0] if eje == 'x' else y_lim[0]}, "
                    f"{z_lim[1] if eje == 'z' else x_lim[1] if eje == 'x' else y_lim[1]}]"
                ])
                
                # Calcular la integral
                try:
                    if eje == 'z':
                        resultado = sp.integrate(
                            sp.integrate(
                                sp.integrate(div_F_cyl, (r, *r_lim)),
                                (theta, *theta_lim)
                            ),
                            (z, *z_lim)
                        )
                    elif eje == 'x':
                        resultado = sp.integrate(
                            sp.integrate(
                                sp.integrate(div_F_cyl, (r, *r_lim)),
                                (theta, *theta_lim)
                            ),
                            (x, *x_lim)
                        )
                    else:  # eje 'y'
                        resultado = sp.integrate(
                            sp.integrate(
                                sp.integrate(div_F_cyl, (r, *r_lim)),
                                (theta, *theta_lim)
                            ),
                            (y, *y_lim)
                        )
                    
                    resultado = sp.simplify(resultado)
                    
                except Exception as e:
                    raise Exception(f"Error al calcular la integral: {str(e)}")
                
            else:  # Coordenadas cartesianas
                procedimiento.extend([
                    "2. Uso de coordenadas cartesianas:",
                    "   x, y, z",
                    "   Jacobiano: 1\n"
                ])
                
                procedimiento.extend([
                    "3. Cálculo de la integral triple:",
                    f"   ∭_V (∇·F) dV = "
                    f"∫_{x_lim[0]}^{x_lim[1]} "
                    f"∫_{y_lim[0]}^{y_lim[1]} "
                    f"∫_{z_lim[0]}^{z_lim[1]} "
                    f"({div_F}) dz dy dx"
                ])
                
                # Calcular la integral
                try:
                    resultado = sp.integrate(
                        sp.integrate(
                            sp.integrate(div_F, (z, *z_lim)),
                            (y, *y_lim)
                        ),
                        (x, *x_lim)
                    )
                    resultado = sp.simplify(resultado)
                    
                except Exception as e:
                    raise Exception(f"Error al calcular la integral: {str(e)}")
            
            # Mostrar el procedimiento paso a paso en LaTeX
            self._set_math_lines(self.teorema_proceso, procedimiento)
            self._auto_resize_textedit(self.teorema_proceso)
            
            # Mostrar el resultado final en LaTeX
            self._set_math_lines(self.teorema_result, [sp.latex(sp.simplify(resultado))])
            self._auto_resize_textedit(self.teorema_result)
            
            # Asegurarse de que estamos en la pestaña de Teoremas
            self.tabs.setCurrentIndex(1)
            
            return resultado
            
        except Exception as e:
            error_msg = f"Error en el Teorema de la Divergencia: {str(e)}"
            self._set_math_lines(self.teorema_proceso, [f"\\text{{{error_msg}}}"])
            self._set_math_lines(self.teorema_result, [f"\\text{{{error_msg}}}"])
            self._auto_resize_textedit(self.teorema_proceso)
            self._auto_resize_textedit(self.teorema_result)
            # Asegurarse de que estamos en la pestaña de Teoremas incluso si hay error
            self.tabs.setCurrentIndex(1)
            raise
