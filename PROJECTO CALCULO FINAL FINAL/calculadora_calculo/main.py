import sys
import os
from pathlib import Path

# Añadir el directorio del proyecto al path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root.parent))

from PySide6.QtWidgets import QApplication # pyright: ignore[reportMissingImports]
from PySide6.QtCore import Qt, QCoreApplication
from calculadora_calculo.ui.main_window import MainWindow

def main():
    # Preferir OpenGL de escritorio y compartir contextos entre widgets GL
    os.environ["QT_OPENGL"] = "desktop"
    os.environ["QTWEBENGINE_DISABLE_GPU"] = "1"
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu --disable-software-rasterizer"
    QCoreApplication.setAttribute(Qt.AA_UseDesktopOpenGL, True)
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    app = QApplication(sys.argv)
    
    # Configuración del estilo de la aplicación
    app.setStyle('Fusion')
    # Tema oscuro con texto blanco
    app.setStyleSheet(
        """
        * { color: #FFFFFF; }
        QWidget { background-color: #121212; }
        QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QComboBox { background-color: #1e1e1e; }
        QPushButton { background-color: #2a2a2a; }
        QTabWidget::pane { border: 1px solid #333; }
        QTabBar::tab { background: #1e1e1e; padding: 6px 10px; }
        QTabBar::tab:selected { background: #2a2a2a; }
        QLabel { color: #FFFFFF; }
        """
    )
    
    # Crear y mostrar la ventana principal
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()


