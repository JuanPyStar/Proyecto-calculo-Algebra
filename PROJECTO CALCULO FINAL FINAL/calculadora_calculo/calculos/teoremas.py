"""
Módulo para implementar los teoremas fundamentales del cálculo vectorial:
- Teorema de Green
- Teorema de Stokes
- Teorema de la Divergencia
"""
import sympy as sp
from typing import Tuple, List, Dict, Any, Union

# Símbolos comunes para coordenadas rectangulares
x, y, z = sp.symbols('x y z', real=True)

# Símbolos para coordenadas cilíndricas
r, theta = sp.symbols('r theta', real=True, nonnegative=True)

# Símbolos para coordenadas esféricas
rho, phi = sp.symbols('rho phi', real=True, nonnegative=True)

# Vectores unitarios
i_hat, j_hat, k_hat = sp.symbols('i j k', real=True, commutative=False)
e_r, e_theta, e_z = sp.symbols('e_r e_theta e_z', real=True, commutative=False)
e_rho, e_phi, e_theta_sph = sp.symbols('e_rho e_phi e_theta', real=True, commutative=False)


def teorema_green(
    P: sp.Expr,
    Q: sp.Expr,
    region: str = 'rectangulo',
    parametros: Dict[str, Any] = None,
    sistema_coordenadas: str = 'cartesianas'
) -> sp.Expr:
    """
    Aplica el Teorema de Green para calcular la integral de línea alrededor de una curva cerrada C
    como una integral doble sobre la región D encerrada por C.
    
    En coordenadas cartesianas:
    ∮_C (P·dx + Q·dy) = ∬_D (∂Q/∂x - ∂P/∂y) dA
    
    En coordenadas polares (r, θ):
    ∮_C (P·dr + Q·r·dθ) = ∬_D (1/r)(∂(rQ)/∂r - ∂P/∂θ) r·dr·dθ
    
    Args:
        P: Función P(x,y) que multiplica a dx (o dr en polares)
        Q: Función Q(x,y) que multiplica a dy (o r·dθ en polares)
        region: Tipo de región ('rectangulo', 'circulo', 'elipse', 'personalizada')
        parametros: Parámetros específicos de la región
        sistema_coordenadas: 'cartesianas' o 'polares'
        
    Returns:
        Resultado simbólico de la integral doble
    """
    if parametros is None:
        parametros = {}
    
    if sistema_coordenadas == 'cartesianas':
        # Teorema de Green estándar en coordenadas cartesianas
        dQ_dx = sp.diff(Q, x)
        dP_dy = sp.diff(P, y)
        integrando = dQ_dx - dP_dy
        
    elif sistema_coordenadas == 'polares':
        # Versión en coordenadas polares del teorema de Green
        r_sym = sp.symbols('r', real=True, positive=True)
        theta_sym = sp.symbols('theta', real=True)
        
        # Convertir P y Q a coordenadas polares si es necesario
        # Asumimos que P y Q ya están en términos de r y theta
        # P = P(r, theta), Q = Q(r, theta)
        
        # En coordenadas polares: dA = r·dr·dθ
        # El teorema se convierte en:
        # ∮(P·dr + Q·r·dθ) = ∬(1/r)(∂(rQ)/∂r - ∂P/∂θ) r·dr·dθ
        rQ = r_sym * Q
        drQ_dr = sp.diff(rQ, r_sym)
        dP_dtheta = sp.diff(P, theta_sym)
        integrando = (drQ_dr - dP_dtheta)  # El r del jacobiano se cancela
    else:
        raise ValueError("Sistema de coordenadas no soportado. Use 'cartesianas' o 'polares'")
    
    # Definir los límites de integración según la región
    if region == 'rectangulo':
        if sistema_coordenadas == 'cartesianas':
            a = parametros.get('x_min', -1)
            b = parametros.get('x_max', 1)
            c = parametros.get('y_min', -1)
            d = parametros.get('y_max', 1)
            
            resultado = sp.integrate(
                sp.integrate(integrando, (y, c, d)),
                (x, a, b)
            )
        else:  # polares
            r_min = parametros.get('r_min', 0)
            r_max = parametros.get('r_max', 1)
            theta_min = parametros.get('theta_min', 0)
            theta_max = parametros.get('theta_max', 2*sp.pi)
            
            resultado = sp.integrate(
                sp.integrate(integrando * r_sym, (r_sym, r_min, r_max)),
                (theta_sym, theta_min, theta_max)
            )
    
    elif region == 'circulo':
        radio = parametros.get('radio', 1)
        x0 = parametros.get('x0', 0)
        y0 = parametros.get('y0', 0)
        
        if sistema_coordenadas == 'cartesianas':
            # Usar coordenadas polares para integrar sobre el círculo
            integrando_polar = integrando.subs({
                x: x0 + r_sym * sp.cos(theta_sym),
                y: y0 + r_sym * sp.sin(theta_sym)
            }) * r_sym  # Jacobiano
            
            resultado = sp.integrate(
                sp.integrate(integrando_polar, (r_sym, 0, radio)),
                (theta_sym, 0, 2*sp.pi)
            )
        else:  # polares
            # Asumimos que el círculo está centrado en el origen
            resultado = sp.integrate(
                sp.integrate(integrando * r_sym, (r_sym, 0, radio)),
                (theta_sym, 0, 2*sp.pi)
            )
    
    elif region == 'elipse':
        a = parametros.get('semi_eje_x', 2)
        b = parametros.get('semi_eje_y', 1)
        
        if sistema_coordenadas == 'cartesianas':
            # Usar coordenadas elípticas
            integrando_elip = integrando.subs({
                x: a * r_sym * sp.cos(theta_sym),
                y: b * r_sym * sp.sin(theta_sym)
            }) * a * b * r_sym  # Jacobiano
            
            resultado = sp.integrate(
                sp.integrate(integrando_elip, (r_sym, 0, 1)),
                (theta_sym, 0, 2*sp.pi)
            )
        else:  # polares
            # Para elipses en coordenadas polares, usamos coordenadas elípticas
            # con transformación a polares modificada
            integrando_elip = integrando.subs({
                r_sym: r_sym * sp.sqrt((a*sp.cos(theta_sym))**2 + (b*sp.sin(theta_sym))**2)
            }) * r_sym  # Jacobiano ya incluye la transformación
            
            resultado = sp.integrate(
                sp.integrate(integrando_elip, (r_sym, 0, 1)),
                (theta_sym, 0, 2*sp.pi)
            )
    
    elif region == 'personalizada' and 'limites' in parametros:
        # Región personalizada con límites dados
        limites = parametros['limites']
        # Asumimos que limites es una lista de tuplas (lim_inf, lim_sup, var_integracion)
        resultado = integrando
        for lim_inf, lim_sup, var in reversed(limites):
            resultado = sp.integrate(resultado, (var, lim_inf, lim_sup))
    
    else:
        raise ValueError(f"Tipo de región no soportado: {region}")
    
    return resultado


def teorema_stokes(
    F: Tuple[sp.Expr, sp.Expr, sp.Expr],
    superficie: str = 'plano',
    parametros: Dict[str, Any] = None,
    sistema_coordenadas: str = 'cartesianas'
) -> sp.Expr:
    """
    Aplica el Teorema de Stokes para calcular la integral de línea de un campo vectorial F
    alrededor de una curva cerrada C como la integral de superficie del rotacional de F
    sobre cualquier superficie S limitada por C.
    
    En coordenadas cartesianas:
    ∮_C F·dr = ∬_S (∇ × F)·dS
    
    En coordenadas cilíndricas (r, θ, z):
    ∮_C (F_r·dr + F_θ·r·dθ + F_z·dz) = ∬_S (∇ × F)·n dS
    
    En coordenadas esféricas (ρ, φ, θ):
    ∮_C (F_ρ·dρ + F_φ·ρ·dφ + F_θ·ρ·sin(φ)·dθ) = ∬_S (∇ × F)·n dS
    
    Args:
        F: Tupla con las componentes del campo vectorial (F1, F2, F3)
            - Cartesiano: (F_x, F_y, F_z)
            - Cilíndrico: (F_r, F_θ, F_z)
            - Esférico: (F_ρ, F_φ, F_θ)
        superficie: Tipo de superficie ('plano', 'esfera', 'cilindro', 'cono', 'personalizada')
        parametros: Parámetros específicos de la superficie
        sistema_coordenadas: 'cartesianas', 'cilindricas' o 'esfericas'
        
    Returns:
        Resultado simbólico de la integral de superficie
    """
    if parametros is None:
        parametros = {}
    
    F1, F2, F3 = F
    
    # Calcular el rotacional según el sistema de coordenadas
    if sistema_coordenadas == 'cartesianas':
        # ∇ × F en coordenadas cartesianas
        rot_F1 = sp.diff(F3, y) - sp.diff(F2, z)
        rot_F2 = sp.diff(F1, z) - sp.diff(F3, x)
        rot_F3 = sp.diff(F2, x) - sp.diff(F1, y)
        
    elif sistema_coordenadas == 'cilindricas':
        # ∇ × F en coordenadas cilíndricas
        # F = (F_r, F_θ, F_z)
        rot_r = (1/r) * sp.diff(F3, theta) - sp.diff(F2, z)
        rot_theta = sp.diff(F1, z) - sp.diff(F3, r)
        rot_z = (1/r) * (sp.diff(r*F2, r) - sp.diff(F1, theta))
        rot_F1, rot_F2, rot_F3 = rot_r, rot_theta, rot_z
        
    elif sistema_coordenadas == 'esfericas':
        # ∇ × F en coordenadas esféricas
        # F = (F_ρ, F_φ, F_θ)
        rot_rho = (1/(rho*sp.sin(phi))) * (sp.diff(sp.sin(phi)*F3, phi) - sp.diff(F2, theta))
        rot_phi = (1/rho) * (1/sp.sin(phi) * sp.diff(F1, theta) - sp.diff(rho*F3, rho))
        rot_theta = (1/rho) * (sp.diff(rho*F2, rho) - sp.diff(F1, phi))
        rot_F1, rot_F2, rot_F3 = rot_rho, rot_phi, rot_theta
        
    else:
        raise ValueError("Sistema de coordenadas no soportado. Use 'cartesianas', 'cilindricas' o 'esfericas'")
    
    # Definir la integral de superficie según el tipo de superficie
    if superficie == 'plano':
        # Plano z = ax + by + c
        a = parametros.get('a', 0)
        b = parametros.get('b', 0)
        c = parametros.get('c', 0)
        
        # Vector normal unitario (hacia arriba)
        n = (-a, -b, 1)
        norma = sp.sqrt(a**2 + b**2 + 1)
        n_unit = (n[0]/norma, n[1]/norma, n[2]/norma)
        
        # Producto punto (∇ × F)·n
        integrando = rot_F1*n_unit[0] + rot_F2*n_unit[1] + rot_F3*n_unit[2]
        
        # Área de integración en el plano xy
        x_lim = parametros.get('x_lim', (-1, 1))
        y_lim = parametros.get('y_lim', (-1, 1))
        
        # Multiplicar por el elemento de área dS = √(1 + (∂z/∂x)² + (∂z/∂y)²) dxdy
        dS = sp.sqrt(1 + a**2 + b**2)
        
        # Integrar sobre la región en el plano xy
        resultado = sp.integrate(
            sp.integrate(integrando * dS, (y, y_lim[0], y_lim[1])),
            (x, x_lim[0], x_lim[1])
        )
    
    elif superficie == 'esfera':
        # Esfera de radio r centrada en (x0, y0, z0)
        radio = parametros.get('radio', 1)
        x0 = parametros.get('x0', 0)
        y0 = parametros.get('y0', 0)
        z0 = parametros.get('z0', 0)
        
        if sistema_coordenadas == 'cartesianas':
            # Transformación a coordenadas esféricas
            x_sph = x0 + rho * sp.sin(phi) * sp.cos(theta)
            y_sph = y0 + rho * sp.sin(phi) * sp.sin(theta)
            z_sph = z0 + rho * sp.cos(phi)
            
            # Vector normal unitario (hacia afuera)
            n = (sp.sin(phi)*sp.cos(theta), sp.sin(phi)*sp.sin(theta), sp.cos(phi))
            
            # Evaluar el rotacional en las coordenadas esféricas
            rot_F_sph = (
                rot_F1.subs({x: x_sph, y: y_sph, z: z_sph}),
                rot_F2.subs({x: x_sph, y: y_sph, z: z_sph}),
                rot_F3.subs({x: x_sph, y: y_sph, z: z_sph})
            )
        else:
            # Ya estamos en coordenadas esféricas
            rot_F_sph = (rot_F1, rot_F2, rot_F3)
            n = (1, 0, 0)  # Vector radial unitario
        
        # Producto punto (∇ × F)·n
        integrando = (
            rot_F_sph[0] * n[0] +
            rot_F_sph[1] * n[1] * (1/rho if n[1] != 0 else 0) +  # Factores métricos
            rot_F_sph[2] * n[2] * (1/(rho*sp.sin(phi)) if n[2] != 0 else 0)
        ) * rho**2 * sp.sin(phi)  # Jacobiano para esféricas
        
        # Integrar sobre la esfera
        resultado = sp.integrate(
            sp.integrate(integrando, (theta, 0, 2*sp.pi)),
            (phi, 0, sp.pi)
        )
    
    # Si no se reconoce la superficie, devolver el rotacional para que se calcule la integral de línea
    return rot_F1, rot_F2, rot_F3


def teorema_divergencia(
    F: Tuple[sp.Expr, sp.Expr, sp.Expr],
    region: str = 'cubo',
    parametros: Dict[str, Any] = None,
    sistema_coordenadas: str = 'cartesianas'
) -> sp.Expr:
    """
    Aplica el Teorema de la Divergencia para calcular el flujo de un campo vectorial F
    a través de una superficie cerrada S como la integral de volumen de la divergencia
    de F sobre la región E encerrada por S.
    
    En coordenadas cartesianas:
    ∯_S F·dS = ∭_E (∇·F) dV
    
    En coordenadas cilíndricas (r, θ, z):
    ∯_S F·dS = ∭_E (1/r)(∂(rF_r)/∂r + ∂F_θ/∂θ + r·∂F_z/∂z) r·dr·dθ·dz
    
    En coordenadas esféricas (ρ, φ, θ):
    ∯_S F·dS = ∭_E (1/ρ²)(∂(ρ²F_ρ)/∂ρ + (1/sin(φ))∂(sin(φ)F_φ)/∂φ + (1/sin(φ))∂F_θ/∂θ) ρ²·sin(φ)·dρ·dφ·dθ
    
    Args:
        F: Tupla con las componentes del campo vectorial (F1, F2, F3)
            - Cartesiano: (F_x, F_y, F_z)
            - Cilíndrico: (F_r, F_θ, F_z)
            - Esférico: (F_ρ, F_φ, F_θ)
        region: Tipo de región ('cubo', 'esfera', 'cilindro', 'elipsoide', 'personalizada')
        parametros: Parámetros específicos de la región
        sistema_coordenadas: 'cartesianas', 'cilindricas' o 'esfericas'
        
    Returns:
        Resultado simbólico de la integral de volumen
    """
    if parametros is None:
        parametros = {}
    
    F1, F2, F3 = F
    
    # Calcular la divergencia según el sistema de coordenadas
    if sistema_coordenadas == 'cartesianas':
        # ∇·F en coordenadas cartesianas
        div_F = sp.diff(F1, x) + sp.diff(F2, y) + sp.diff(F3, z)
        
    elif sistema_coordenadas == 'cilindricas':
        # ∇·F en coordenadas cilíndricas
        # F = (F_r, F_θ, F_z)
        div_F = (1/r) * sp.diff(r*F1, r) + (1/r) * sp.diff(F2, theta) + sp.diff(F3, z)
        
    elif sistema_coordenadas == 'esfericas':
        # ∇·F en coordenadas esféricas
        # F = (F_ρ, F_φ, F_θ)
        div_F = (1/rho**2) * sp.diff(rho**2 * F1, rho) + \
                (1/(rho*sp.sin(phi))) * sp.diff(sp.sin(phi)*F2, phi) + \
                (1/(rho*sp.sin(phi))) * sp.diff(F3, theta)
    else:
        raise ValueError("Sistema de coordenadas no soportado. Use 'cartesianas', 'cilindricas' o 'esfericas'")
    
    # Definir la integral de volumen según el tipo de región
    if region == 'cubo':
        if sistema_coordenadas == 'cartesianas':
            # Cubo [a,b]x[c,d]x[e,f]
            a = parametros.get('x_min', -1)
            b = parametros.get('x_max', 1)
            c = parametros.get('y_min', -1)
            d = parametros.get('y_max', 1)
            e = parametros.get('z_min', -1)
            f = parametros.get('z_max', 1)
            
            # Integrar sobre el cubo
            resultado = sp.integrate(
                sp.integrate(
                    sp.integrate(div_F, (z, e, f)),
                    (y, c, d)
                ),
                (x, a, b)
            )
        else:
            # Para otros sistemas de coordenadas, usar la región apropiada
            raise NotImplementedError("Cubo solo soportado en coordenadas cartesianas")
    
    elif region == 'esfera':
        # Esfera de radio r centrada en (x0, y0, z0)
        radio = parametros.get('radio', 1)
        
        if sistema_coordenadas == 'cartesianas':
            x0 = parametros.get('x0', 0)
            y0 = parametros.get('y0', 0)
            z0 = parametros.get('z0', 0)
            
            # Transformación a coordenadas esféricas
            x_sph = x0 + rho * sp.sin(phi) * sp.cos(theta)
            y_sph = y0 + rho * sp.sin(phi) * sp.sin(theta)
            z_sph = z0 + rho * sp.cos(phi)
            
            # Evaluar la divergencia en coordenadas esféricas
            div_F_sph = div_F.subs({
                x: x_sph,
                y: y_sph,
                z: z_sph
            })
        else:
            # Ya estamos en coordenadas esféricas o cilíndricas
            div_F_sph = div_F
        
        # Jacobiano para coordenadas esféricas: ρ²·sin(φ)
        jacobiano = rho**2 * sp.sin(phi)
        
        # Límites para la esfera
        rho_lim = (0, radio)
        phi_lim = (0, sp.pi)
        theta_lim = (0, 2*sp.pi)
        
        # Integrar sobre la esfera
        resultado = sp.integrate(
            sp.integrate(
                sp.integrate(div_F_sph * jacobiano, (rho, *rho_lim)),
                (phi, *phi_lim)
            ),
            (theta, *theta_lim)
        )
    
    elif region == 'cilindro':
        # Cilindro de radio r y altura h, con eje en z
        radio = parametros.get('radio', 1)
        altura = parametros.get('altura', 2)
        
        if sistema_coordenadas in ['cartesianas', 'cilindricas']:
            if sistema_coordenadas == 'cartesianas':
                x0 = parametros.get('x0', 0)
                y0 = parametros.get('y0', 0)
                
                # Transformación a coordenadas cilíndricas
                x_cyl = x0 + r * sp.cos(theta)
                y_cyl = y0 + r * sp.sin(theta)
                
                # Evaluar la divergencia en coordenadas cilíndricas
                div_F_cyl = div_F.subs({
                    x: x_cyl,
                    y: y_cyl
                })
            else:
                # Ya estamos en coordenadas cilíndricas
                div_F_cyl = div_F
            
            # Jacobiano para coordenadas cilíndricas: r
            jacobiano = r
            
            # Límites para el cilindro
            r_lim = (0, radio)
            theta_lim = (0, 2*sp.pi)
            z_lim = (-altura/2, altura/2)
            
            # Integrar sobre el cilindro
            resultado = sp.integrate(
                sp.integrate(
                    sp.integrate(div_F_cyl * jacobiano, (r, *r_lim)),
                    (theta, *theta_lim)
                ),
                (z, *z_lim)
            )
        else:
            raise NotImplementedError("Cilindro no soportado en coordenadas esféricas")
    
    elif region == 'elipsoide':
        # Elipsoide (x/a)² + (y/b)² + (z/c)² = 1
        a = parametros.get('semi_eje_x', 2)
        b = parametros.get('semi_eje_y', 1.5)
        c = parametros.get('semi_eje_z', 1)
        
        if sistema_coordenadas == 'cartesianas':
            # Usar coordenadas esféricas modificadas para el elipsoide
            r_sym = sp.symbols('r', real=True, nonnegative=True)
            
            # Transformación a coordenadas elipsoidales
            x_elip = a * r_sym * sp.sin(phi) * sp.cos(theta)
            y_elip = b * r_sym * sp.sin(phi) * sp.sin(theta)
            z_elip = c * r_sym * sp.cos(phi)
            
            # Evaluar la divergencia en coordenadas elipsoidales
            div_F_elip = div_F.subs({
                x: x_elip,
                y: y_elip,
                z: z_elip
            })
            
            # Jacobiano para coordenadas elipsoidales: a·b·c·r²·sin(φ)
            jacobiano = a * b * c * r_sym**2 * sp.sin(phi)
            
            # Límites para el elipsoide
            r_lim = (0, 1)
            phi_lim = (0, sp.pi)
            theta_lim = (0, 2*sp.pi)
            
            # Integrar sobre el elipsoide
            resultado = sp.integrate(
                sp.integrate(
                    sp.integrate(div_F_elip * jacobiano, (r_sym, *r_lim)),
                    (phi, *phi_lim)
                ),
                (theta, *theta_lim)
            )
        else:
            raise NotImplementedError("Elipsoide solo soportado en coordenadas cartesianas")
    
    elif region == 'personalizada' and 'limites' in parametros:
        # Región personalizada con límites dados
        limites = parametros['limites']
        # Asumimos que limites es una lista de tuplas (lim_inf, lim_sup, var_integracion)
        resultado = div_F
        for lim_inf, lim_sup, var in reversed(limites):
            # Aplicar el jacobiano apropiado según la variable de integración
            if var == r or var == rho:
                resultado = resultado * var  # r·dr o ρ·dρ
            elif var == phi:
                resultado = resultado * sp.sin(phi)  # sin(φ)·dφ
            
            resultado = sp.integrate(resultado, (var, lim_inf, lim_sup))
    
    else:
        raise ValueError(f"Tipo de región no soportado: {region}")
    
    return resultado
