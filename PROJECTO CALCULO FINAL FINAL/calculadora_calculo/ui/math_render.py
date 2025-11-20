import io
import base64
from matplotlib import pyplot as plt


def latex_to_html(latex: str, fontsize: int = 14, dpi: int = 200) -> str:
    """
    Renderiza una cadena LaTeX a una imagen PNG embebida en HTML (data URL).
    Usa el motor mathtext de matplotlib (no requiere instalación externa de LaTeX).
    """
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.patch.set_alpha(0)
    plt.axis('off')
    text = fig.text(0, 0, f"${latex}$", fontsize=fontsize, color='white')

    # Forzar render y ajustar tamaño exacto a la caja del texto
    fig.canvas.draw()
    bbox = text.get_window_extent()
    width, height = bbox.size / fig.dpi
    fig.set_size_inches(max(width, 0.01), max(height, 0.01))
    text.set_position((0, 0))

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'<img src="data:image/png;base64,{b64}"/>'


def lines_to_html(latex_lines: list[str], fontsize: int = 14, dpi: int = 200) -> str:
    """
    Convierte una lista de ecuaciones LaTeX en un bloque HTML apilando imágenes.
    """
    parts = [latex_to_html(line, fontsize=fontsize, dpi=dpi) for line in latex_lines]
    return "<div>" + "<br>".join(parts) + "</div>"
