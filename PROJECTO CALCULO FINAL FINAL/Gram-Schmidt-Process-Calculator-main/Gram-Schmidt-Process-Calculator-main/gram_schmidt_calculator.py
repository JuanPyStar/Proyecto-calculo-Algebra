import os
import sys
import sympy as sp
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QRadioButton, QButtonGroup, QSpinBox, QGridLayout)
from PyQt5.QtWebEngineWidgets import QWebEngineView

# Set QT platform appropriately: use 'windows' on Windows; do not force Wayland
if sys.platform.startswith('win'):
    os.environ['QT_QPA_PLATFORM'] = 'windows'


class GramSchmidtCalculator(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Gram-Schmidt Process Calculator')
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()

        # seleccion de proceso
        process_layout = QHBoxLayout()
        self.orthogonal_btn = QRadioButton("Orthogonal")
        self.orthonormal_btn = QRadioButton("Orthonormal")
        self.orthonormal_btn.setChecked(True)
        process_group = QButtonGroup(self)
        process_group.addButton(self.orthogonal_btn)
        process_group.addButton(self.orthonormal_btn)
        process_layout.addWidget(QLabel("Select Process:"))
        process_layout.addWidget(self.orthogonal_btn)
        process_layout.addWidget(self.orthonormal_btn)
        layout.addLayout(process_layout)

        # numero de vectores y dimensiones
        count_layout = QHBoxLayout()
        self.vector_count_spin = QSpinBox()
        self.vector_count_spin.setRange(1, 10)
        self.vector_count_spin.setValue(3)
        self.dimension_count_spin = QSpinBox()
        self.dimension_count_spin.setRange(1, 10)
        self.dimension_count_spin.setValue(3)
        count_layout.addWidget(QLabel("Number of Vectors:"))
        count_layout.addWidget(self.vector_count_spin)
        count_layout.addWidget(QLabel("Number of Dimensions:"))
        count_layout.addWidget(self.dimension_count_spin)
        layout.addLayout(count_layout)

        # input de vectores
        self.vector_input_layout = QGridLayout()
        layout.addLayout(self.vector_input_layout)

        self.vector_count_spin.valueChanged.connect(self.update_vector_inputs)
        self.dimension_count_spin.valueChanged.connect(self.update_vector_inputs)

        # boton de calculo
        self.calc_button = QPushButton("Calculate")
        self.calc_button.clicked.connect(self.calculate)
        layout.addWidget(self.calc_button)

        # display de resultados
        self.results_display = QWebEngineView()
        layout.addWidget(self.results_display)

        self.setLayout(layout)

        # inicializar inputs de vectores
        self.update_vector_inputs()

    def update_vector_inputs(self):
        # limpiar inputs existentes
        for i in reversed(range(self.vector_input_layout.count())):
            self.vector_input_layout.itemAt(i).widget().setParent(None)

        vector_count = self.vector_count_spin.value()
        dimension_count = self.dimension_count_spin.value()

        # crear nuevos inputs
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

    def calculate(self):
        try:
            vectors = self.get_vectors()
            if not vectors:
                raise ValueError("No valid vectors entered.")

            orthonormal = self.orthonormal_btn.isChecked()
            basis = self.gram_schmidt(vectors, orthonormal)

            process_type = "Orthonormal" if orthonormal else "Orthogonal"
            latex_str = f"\\text{{{process_type} Basis Vectors:}}\\\\"
            for i, vec in enumerate(basis, start=1):
                latex_str += f"\\text{{Vector {i}: }} {self.vector_to_latex(vec)}\\\\"

            # Render LaTeX as HTML
            html_content = f"$$ {latex_str} $$"
            self.results_display.setHtml(f"""
                <html>
                <head>
                <script type='text/x-mathjax-config'>
                    MathJax.Hub.Config({{tex2jax: {{inlineMath: [['$','$'], ['\\(','\\)']]}}}});
                </script>
                <script type='text/javascript' src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-MML-AM_CHTML'></script>
                </head>
                <body>
                    <h2>{process_type} Basis Vectors Calculation</h2>
                    <p>The Gram-Schmidt process has been applied to the input vectors to generate the following basis vectors:</p>
                    <p>{html_content}</p>
                </body>
                </html>
            """)

        except Exception as e:
            self.results_display.setHtml(f"<html><body><p>An error occurred: {str(e)}</p></body></html>")


def main():
    app = QApplication(sys.argv)
    calc = GramSchmidtCalculator()
    calc.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
