"""
Módulo para el cálculo de integrales triples en diferentes sistemas de coordenadas.
"""
import sympy as sp
import numpy as np
from typing import Tuple, Union, Callable, Dict, Any

# Símbolos comunes
x, y, z = sp.symbols('x y z', real=True)
r, theta, rho, phi = sp.symbols('r theta rho phi', real=True, positive=True)


def calcular_integral_rectangular(
    func: sp.Expr, 
    x_lim: Tuple[float, float], 
    y_lim: Tuple[float, float], 
    z_lim: Tuple[float, float],
    **kwargs
) -> sp.Expr:
    """
    Calcula una integral triple en coordenadas rectangulares.
    
    Args:
        func: Expresión simbólica de la función a integrar
        x_lim: Límites de integración en x (min, max)
        y_lim: Límites de integración en y (min, max)
        z_lim: Límites de integración en z (min, max)
        
    Returns:
        Resultado simbólico de la integral
    """
    x_min, x_max = x_lim
    y_min, y_max = y_lim
    z_min, z_max = z_lim
    
    # Orden de integración: z, y, x (de adentro hacia afuera)
    integral = sp.integrate(
        sp.integrate(
            sp.integrate(func, (z, z_min, z_max)),
            (y, y_min, y_max)
        ),
        (x, x_min, x_max)
    )
    
    return integral


def calcular_integral_cilindrica(
    func: sp.Expr, 
    r_lim: Tuple[float, float], 
    theta_lim: Tuple[float, float], 
    z_lim: Tuple[float, float],
    **kwargs
) -> sp.Expr:
    """
    Calcula una integral triple en coordenadas cilíndricas.
    
    Args:
        func: Expresión simbólica de la función a integrar (ya en coordenadas cilíndricas)
        r_lim: Límites de integración en r (radio)
        theta_lim: Límites de integración en theta (ángulo azimutal en radianes)
        z_lim: Límites de integración en z
        
    Returns:
        Resultado simbólico de la integral
    """
    r_min, r_max = r_lim
    theta_min, theta_max = theta_lim
    z_min, z_max = z_lim
    
    # Jacobiano: r (para coordenadas cilíndricas)
    jacobian = r
    integrando = func * jacobian
    
    # Orden de integración: z, r, theta
    integral = sp.integrate(
        sp.integrate(
            sp.integrate(integrando, (z, z_min, z_max)),
            (r, r_min, r_max)
        ),
        (theta, theta_min, theta_max)
    )
    
    return integral


def calcular_integral_esferica(
    func: sp.Expr, 
    rho_lim: Tuple[float, float], 
    phi_lim: Tuple[float, float], 
    theta_lim: Tuple[float, float],
    **kwargs
) -> sp.Expr:
    """
    Calcula una integral triple en coordenadas esféricas.
    
    Args:
        func: Expresión simbólica de la función a integrar (ya en coordenadas esféricas)
        rho_lim: Límites de integración en rho (distancia al origen)
        phi_lim: Límites de integración en phi (ángulo polar en radianes, 0 a pi)
        theta_lim: Límites de integración en theta (ángulo azimutal en radianes, 0 a 2pi)
        
    Returns:
        Resultado simbólico de la integral
    """
    rho_min, rho_max = rho_lim
    phi_min, phi_max = phi_lim
    theta_min, theta_max = theta_lim
    
    # Jacobiano: rho² * sin(phi) (para coordenadas esféricas)
    jacobian = rho**2 * sp.sin(phi)
    integrando = func * jacobian
    
    # Orden de integración: rho, phi, theta
    integral = sp.integrate(
        sp.integrate(
            sp.integrate(integrando, (rho, rho_min, rho_max)),
            (phi, phi_min, phi_max)
        ),
        (theta, theta_min, theta_max)
    )
    
    return integral


def transformar_a_cilindricas(func: sp.Expr) -> sp.Expr:
    """
    Transforma una función de coordenadas rectangulares a cilíndricas.
    
    Args:
        func: Función en coordenadas rectangulares f(x,y,z)
        
    Returns:
        Función en coordenadas cilíndricas f(r,θ,z)
    """
    # Sustituir x = r*cos(θ), y = r*sin(θ), z = z
    func_cil = func.subs({
        x: r * sp.cos(theta),
        y: r * sp.sin(theta),
        z: z
    })
    
    return func_cil


def transformar_a_esfericas(func: sp.Expr) -> sp.Expr:
    """
    Transforma una función de coordenadas rectangulares a esféricas.
    
    Args:
        func: Función en coordenadas rectangulares f(x,y,z)
        
    Returns:
        Función en coordenadas esféricas f(ρ,φ,θ)
    """
    # Sustituir x = ρ*sin(φ)*cos(θ), y = ρ*sin(φ)*sin(θ), z = ρ*cos(φ)
    func_esf = func.subs({
        x: rho * sp.sin(phi) * sp.cos(theta),
        y: rho * sp.sin(phi) * sp.sin(theta),
        z: rho * sp.cos(phi)
    })
    
    return func_esf


def calcular_integral_triple(
    func_str: str, 
    coord_type: str,
    x_lim: Tuple[float, float],
    y_lim: Tuple[float, float],
    z_lim: Tuple[float, float],
    **kwargs
) -> sp.Expr:
    """
    Función principal para calcular integrales triples en diferentes sistemas de coordenadas.
    
    Args:
        func_str: Función como cadena (ej: 'x**2 + y**2 + z**2')
        coord_type: Tipo de coordenadas ('rectangular', 'cilindrica', 'esferica')
        x_lim: Límites de integración en x (o r/ρ según el sistema)
        y_lim: Límites de integración en y (o θ/φ según el sistema)
        z_lim: Límites de integración en z (o z/θ según el sistema)
        
    Returns:
        Resultado simbólico de la integral
    """
    # Convertir la cadena a una expresión simbólica
    func = sp.sympify(func_str)
    
    if coord_type.lower() == 'rectangular':
        return calcular_integral_rectangular(func, x_lim, y_lim, z_lim, **kwargs)
    
    elif coord_type.lower() == 'cilindrica':
        # Transformar la función a coordenadas cilíndricas
        func_cil = transformar_a_cilindricas(func)
        return calcular_integral_cilindrica(func_cil, x_lim, y_lim, z_lim, **kwargs)
    
    elif coord_type.lower() == 'esferica':
        # Transformar la función a coordenadas esféricas
        func_esf = transformar_a_esfericas(func)
        return calcular_integral_esferica(func_esf, x_lim, y_lim, z_lim, **kwargs)
    
    else:
        raise ValueError(f"Tipo de coordenadas no soportado: {coord_type}")
