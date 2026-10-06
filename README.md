# PINNs — Tesis de maestría

Redes neuronales informadas por la física (PINNs) aplicadas a problemas de elasticidad,
con el **problema de Boussinesq** —carga puntual sobre un semiespacio elástico— como
caso central.

## El cuaderno principal

> **[`Boussinesq_PINN_corregido.ipynb`](Boussinesq_PINN_corregido.ipynb)**

Es la versión que resuelve el problema con la singularidad real, sin regularizarla ni
evitarla. Está ejecutado de principio a fin, con sus figuras y resultados.

La idea central es no pedirle a la red que aprenda la singularidad, sino escribirla a
mano en la parametrización:

```
u_r = r·q      q = k₀·A(s,c)/ρ²      w = k₀·B(s,c)/ρ      k₀ = 1/2π
ρ = √(r²+z²)   c = z/ρ = cos θ       s = ln ρ
```

La red solo produce `A` y `B`, que son campos suaves y acotados. Todo el comportamiento
singular (u ~ 1/ρ, σ ~ 1/ρ²) vive en los factores explícitos. La carga puntual se impone
por equilibrio de hemisferios, no con una gaussiana.

**Precisión alcanzada**, error L2 relativo contra la solución analítica de Boussinesq:

| σ_zz | σ_rr | σ_θθ | τ_rz | u_r | w |
|---|---|---|---|---|---|
| 6.9·10⁻⁶ | 1.8·10⁻⁴ | 5.7·10⁻⁴ | 1.8·10⁻⁵ | 9.1·10⁻⁵ | 2.7·10⁻⁵ |

Los campos son correctos hasta ρ = 10⁻⁴, donde las tensiones son del orden de 10⁸. La
fuerza resultante sobre hemisferios da −99.99945 frente a −100 exacto, para todo radio
entre 10⁻⁴ y 10.

La explicación detallada de la parametrización está en
[`docs/`](docs/Parametrizacion_PINN_Boussinesq.docx).

## Estructura

```
Boussinesq_PINN_corregido.ipynb   cuaderno principal
docs/                             explicación de la parametrización
boussinesq/
  checkpoints/                    iteraciones sucesivas del enfoque anterior
  variantes/                      polinómica (SiLU, GELU), restricciones duras,
                                  Fourier, curriculum learning
  love/                           formulación con la función de tensión de Love
  modelos/                        pesos entrenados (.pth)
vigas/                            Euler-Bernoulli, Love-Kirchhoff, viga de Airy
fundamentos/                      Laplace y ecuaciones diferenciales de práctica
```

## Sobre el enfoque anterior

Todo lo que está bajo `boussinesq/checkpoints`, `variantes` y `love` corresponde al
enfoque previo, que **evita** la singularidad: reemplaza la carga puntual por una carga
distribuida sobre un radio pequeño y genera puntos de colocación solo lejos del origen
(ρ ≥ 0.04). Los modelos de `modelos/` se entrenaron así, con `P = 10000 N`, `a = 0.1`,
dominio 20×20 y ν = 0.3.

Se conserva como registro del trabajo y como contraste: el cuaderno principal no es una
corrección de estos, sino una reformulación del problema.

## Requisitos

PyTorch, NumPy, SciPy y Matplotlib. El cuaderno principal usa precisión doble
(`float64`), necesaria para L-BFGS y las derivadas de segundo orden. Su entrenamiento
completo tarda unos 9 minutos en CPU de un núcleo y bastante menos con GPU.
