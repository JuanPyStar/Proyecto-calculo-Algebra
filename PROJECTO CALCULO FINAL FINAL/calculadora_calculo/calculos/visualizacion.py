"""
Módulo para la visualización de funciones 3D, superficies y campos vectoriales.
"""
import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Qt
import sympy as sp

# Símbolos comunes
x, y, z = sp.symbols('x y z', real=True)

class Visualizador3D(QWidget):
    """Widget para visualizar gráficos 3D usando pyqtgraph."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Configurar el layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Crear el widget de visualización 3D
        self.view = gl.GLViewWidget()
        self.layout.addWidget(self.view)
        
        # Configurar la cámara
        self.view.setCameraPosition(distance=5)
        
        # Agregar ejes de referencia
        self.agregar_ejes()
        
        # Variables para almacenar los objetos gráficos
        self.surface_item = None
        self.vector_field = None
        self.curve = None
        self._vector_items = []
    
    def agregar_ejes(self):
        """Agrega ejes de referencia al visualizador 3D."""
        # Eje X (rojo)
        x_axis = gl.GLLinePlotItem(
            pos=np.array([
                [-2, 0, 0],
                [2, 0, 0]
            ]),
            color=(1, 0, 0, 1),
            width=2
        )
        
        # Eje Y (verde)
        y_axis = gl.GLLinePlotItem(
            pos=np.array([
                [0, -2, 0],
                [0, 2, 0]
            ]),
            color=(0, 1, 0, 1),
            width=2
        )
        
        # Eje Z (azul)
        z_axis = gl.GLLinePlotItem(
            pos=np.array([
                [0, 0, -2],
                [0, 0, 2]
            ]),
            color=(0, 0, 1, 1),
            width=2
        )
        
        # Agregar ejes a la escena
        self.view.addItem(x_axis)
        self.view.addItem(y_axis)
        self.view.addItem(z_axis)
        
        # Agregar etiquetas a los ejes
        self.agregar_etiqueta([2.1, 0, 0], "X", (1, 0, 0, 1))
        self.agregar_etiqueta([0, 2.1, 0], "Y", (0, 1, 0, 1))
        self.agregar_etiqueta([0, 0, 2.1], "Z", (0, 0, 1, 1))
    
    def agregar_etiqueta(self, pos, text, color):
        """Agrega una etiqueta de texto en la posición 3D especificada."""
        label = gl.GLTextItem(pos=pos, text=text, color=color)
        self.view.addItem(label)
    
    def graficar_superficie(self, func, x_range=(-2, 2), y_range=(-2, 2), num_points=50):
        """
        Grafica una superficie definida por z = f(x, y).
        
        Args:
            func: Función simbólica de x e y
            x_range: Tupla (x_min, x_max) para el rango de x
            y_range: Tupla (y_min, y_max) para el rango de y
            num_points: Número de puntos en cada dirección para la malla
        """
        # Convertir la función simbólica a una función numérica
        f_np = sp.lambdify((x, y), func, 'numpy')
        
        # Crear la malla de puntos
        x_vals = np.linspace(x_range[0], x_range[1], num_points)
        y_vals = np.linspace(y_range[0], y_range[1], num_points)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        # Evaluar la función en la malla
        try:
            Z = f_np(X, Y)
        except:
            # Si hay un error (por ejemplo, división por cero), usar ceros
            Z = np.zeros_like(X)
        
        # Crear los vértices de la malla
        vertices = np.zeros((num_points, num_points, 3))
        vertices[:, :, 0] = X
        vertices[:, :, 1] = Y
        vertices[:, :, 2] = Z
        
        # Crear las caras de la malla
        faces = []
        for i in range(num_points - 1):
            for j in range(num_points - 1):
                # Cada cuadrado se divide en dos triángulos
                v1 = i * num_points + j
                v2 = v1 + 1
                v3 = v1 + num_points
                v4 = v3 + 1
                
                # Primer triángulo (v1, v2, v3)
                faces.append([v1, v2, v3])
                # Segundo triángulo (v2, v4, v3)
                faces.append([v2, v4, v3])
        
        faces = np.array(faces)
        
        # Crear el item de la superficie
        if self.surface_item is not None:
            self.view.removeItem(self.surface_item)
        
        self.surface_item = gl.GLMeshItem(
            vertexes=vertices.reshape(-1, 3),
            faces=faces,
            smooth=True,
            color=(0.7, 0.85, 1.0, 0.9),
            shader='shaded'
        )
        
        self.view.addItem(self.surface_item)
        self.ajustar_vista()
    
    def graficar_campo_vectorial(self, F, x_range=(-2, 2), y_range=(-2, 2), z_range=(-2, 2), num_points=5):
        """
        Grafica un campo vectorial 3D.
        
        Args:
            F: Tupla con las componentes (F1, F2, F3) del campo vectorial
            x_range: Tupla (x_min, x_max) para el rango de x
            y_range: Tupla (y_min, y_max) para el rango de y
            z_range: Tupla (z_min, z_max) para el rango de z
            num_points: Número de puntos en cada dirección para la malla
        """
        F1, F2, F3 = F
        
        # Convertir las funciones simbólicas a funciones numéricas
        F1_np = sp.lambdify((x, y, z), F1, 'numpy')
        F2_np = sp.lambdify((x, y, z), F2, 'numpy')
        F3_np = sp.lambdify((x, y, z), F3, 'numpy')
        
        # Crear la malla de puntos
        x_vals = np.linspace(x_range[0], x_range[1], num_points)
        y_vals = np.linspace(y_range[0], y_range[1], num_points)
        z_vals = np.linspace(z_range[0], z_range[1], num_points)
        
        X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)
        
        # Evaluar las componentes del campo vectorial
        try:
            U = F1_np(X, Y, Z)
            V = F2_np(X, Y, Z)
            W = F3_np(X, Y, Z)
        except:
            # Si hay un error, usar un campo vectorial nulo
            U = np.zeros_like(X)
            V = np.zeros_like(Y)
            W = np.ones_like(Z)  # Pequeño valor en Z para evitar vectores nulos
        
        # Crear los vectores
        vectors = np.stack((X, Y, Z, U, V, W), axis=-1)
        
        # Eliminar el campo vectorial anterior si existe
        if self.vector_field is not None:
            self.view.removeItem(self.vector_field)
        
        # Crear el campo vectorial
        self.vector_field = gl.GLVectorPlotItem(
            vectors=vectors.reshape(-1, 6),
            color=(1, 1, 0, 0.8),
            arrowSize=0.5,
            glOptions='opaque'
        )
        
        self.view.addItem(self.vector_field)
        self.ajustar_vista()
    
    def graficar_curva(self, r, t_range=(0, 2*np.pi), num_points=100):
        """
        Grafica una curva paramétrica en 3D.
        
        Args:
            r: Tupla con las funciones paramétricas (x(t), y(t), z(t))
            t_range: Tupla (t_min, t_max) para el parámetro t
            num_points: Número de puntos para la curva
        """
        x_t, y_t, z_t = r
        
        # Convertir las funciones simbólicas a funciones numéricas
        x_np = sp.lambdify((), x_t, 'numpy')
        y_np = sp.lambdify((), y_t, 'numpy')
        z_np = sp.lambdify((), z_t, 'numpy')
        
        # Evaluar la curva en los puntos del parámetro
        t_vals = np.linspace(t_range[0], t_range[1], num_points)
        
        try:
            x_vals = np.array([x_np(t) for t in t_vals])
            y_vals = np.array([y_np(t) for t in t_vals])
            z_vals = np.array([z_np(t) for t in t_vals])
        except:
            # Si hay un error, usar una curva por defecto
            t_vals = np.linspace(0, 2*np.pi, num_points)
            x_vals = np.cos(t_vals)
            y_vals = np.sin(t_vals)
            z_vals = t_vals / (2*np.pi)
        
        # Crear los puntos de la curva
        points = np.column_stack((x_vals, y_vals, z_vals))
        
        # Eliminar la curva anterior si existe
        if self.curve is not None:
            self.view.removeItem(self.curve)
        
        # Crear la curva
        self.curve = gl.GLLinePlotItem(
            pos=points,
            color=(0, 1, 1, 1),
            width=2,
            antialias=True
        )
        
        self.view.addItem(self.curve)
        self.ajustar_vista()
    
    def graficar_vectores(self, vectores, colores=None, ancho=3):
        """
        Dibuja una colección de vectores como segmentos desde el origen.
        vectores: lista/array de forma (N, 3)
        colores: lista de tuplas RGBA en rango [0,1] o None para color por defecto
        ancho: grosor de línea
        """
        if vectores is None:
            return
        try:
            arr = np.array(vectores, dtype=float)
        except Exception:
            return
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < 3:
            # Extender a 3D con ceros si vienen 2D
            pad = np.zeros((arr.shape[0], 3 - arr.shape[1]))
            arr = np.concatenate([arr, pad], axis=1)

        for i, v in enumerate(arr):
            pos = np.array([[0.0, 0.0, 0.0], [v[0], v[1], v[2]]], dtype=float)
            color = (0.2, 0.6, 1.0, 1.0)
            if colores is not None and i < len(colores):
                color = colores[i]
            item = gl.GLLinePlotItem(pos=pos, color=color, width=ancho, antialias=True)
            self.view.addItem(item)
            self._vector_items.append(item)
        self.ajustar_vista()
    
    def ajustar_vista(self):
        """Ajusta la vista para que todos los objetos sean visibles."""
        self.view.show()
        self.view.pan(0, 0, 0)
        self.view.setCameraPosition(distance=10)
    
    def limpiar_escena(self):
        """Limpia todos los objetos de la escena."""
        self.view.clear()
        self.agregar_ejes()
        self.surface_item = None
        self.vector_field = None
        self.curve = None
        # Limpiar vectores dibujados
        self._vector_items = []
