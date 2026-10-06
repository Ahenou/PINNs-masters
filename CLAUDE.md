# CLAUDE.md

Contexto para Claude Code al trabajar en este repositorio.

## Qué es esto

Material de una **tesis de maestría sobre PINNs** (redes neuronales informadas por la
física) aplicadas a problemas de elasticidad, con el **problema de Boussinesq** —carga
puntual sobre un semiespacio elástico— como caso central.

Es un archivo histórico de experimentos, no una librería: notebooks sueltos en la raíz,
sin paquete ni tests. Varios exploran el mismo problema con enfoques distintos.

## Mapa de archivos

```
Boussinesq_PINN_corregido.ipynb   cuaderno principal (enfoque actual)
docs/                             explicación de la parametrización (.docx)
boussinesq/
  checkpoints/                    Checkpoint1..6, NEWCheckpoint5
  variantes/                      PINN Boussinesq*: polinómica SiLU/GELU,
                                  restricciones duras, Fourier, curriculum learning
  love/                           Bousinessq Love.py, Resume.ipynb
  modelos/                        boussinesq_model_distributed_load_v2/v3 (.pth)
vigas/                            Euler-Bernoulli, Love-Kirchhoff, viga de Airy
fundamentos/                      Laplace, ED4_1V, motionDE
```

Todo lo que está bajo `boussinesq/` salvo el cuaderno principal corresponde al **enfoque
anterior**, que evita la singularidad: carga distribuida sobre un radio pequeño y puntos
de colocación solo donde ρ ≥ 0.04 (`P = 10000 N`, `a = 0.1`, dominio 20×20, ν = 0.3). Se
conserva como registro, no como base sobre la que construir.

`boussinesq/checkpoints/Checkpoint1.ipynb` **no** coincide con la copia que circuló fuera
del repositorio. Confirmar cuál es la versión buena antes de modificarlo.

## Convenciones del enfoque corregido

- **Precisión doble** (`torch.set_default_dtype(torch.float64)`). Imprescindible para
  L-BFGS y derivadas de segundo orden.
- **Signos**: z hacia abajo, compresión negativa.
- **La singularidad va en la parametrización**, no en la red:
  ```
  u_r = r·q      q = k₀·A(s,c)/ρ²      w = k₀·B(s,c)/ρ      k₀ = 1/2π
  ρ = √(r²+z²)   c = z/ρ = cos θ       s = ln ρ
  ```
  La red solo produce `A` y `B`, suaves y acotados. Nunca predecir tensiones o
  desplazamientos directamente: divergen como 1/ρ² y 1/ρ.
- **Nada se divide por r.** `ε_θθ = q` y `(σ_rr − σ_θθ)/r = 2G·∂q/∂r` son identidades
  exactas que se siguen de `u_r = r·q`.
- **La carga puntual se impone por equilibrio de hemisferios** (∫t_z dA = −P con
  cuadratura de Gauss-Legendre), nunca regularizada con una gaussiana.
- **Residuos adimensionalizados** (`ρ³·eq`, `ρ²·σ`) y **muestreo uniforme en ln ρ**.
- `USE_S = False` impone la autosemejanza y da ~10⁻⁵ de error; `USE_S = True` deja que
  la red la descubra y da ~10⁻⁴.

### Resultados validados

`USE_S=False`, 1500 épocas de Adam + 4 rondas de L-BFGS, 3298 parámetros, ~9 min en CPU
de un núcleo. Error L2 relativo contra la solución analítica:

| σ_zz | σ_rr | σ_θθ | τ_rz | u_r | w |
|---|---|---|---|---|---|
| 6.9·10⁻⁶ | 1.8·10⁻⁴ | 5.7·10⁻⁴ | 1.8·10⁻⁵ | 9.1·10⁻⁵ | 2.7·10⁻⁵ |

Fuerza resultante sobre hemisferios: −99.99945 frente a −100 exacto, para todo
ρ₀ ∈ [10⁻⁴, 10].

## Entorno

- PyTorch **no viene instalado** en sesiones nuevas. `pip install --break-system-packages
  torch` descarga ~2 GB de ruedas CUDA y puede llenar el disco; si falla con
  `No space left on device`, comprobar `python -c "import torch"` antes de concluir que
  falló, porque suele funcionar igual.
- Sin GPU y un solo núcleo. Lanzar entrenamientos largos con `setsid nohup ... &`; un
  `nohup` simple se corta al terminar la llamada.

## Cómo trabajar aquí

- **En español.** Todo el material está en español.
- **Verificar contra la solución analítica** antes de afirmar precisión. La función
  `analytic()` del notebook corregido ya está comprobada: cumple equilibrio a precisión
  de máquina. Ojo con σ_rr, cuya fórmula es fácil de escribir mal.
- **Ejecutar los notebooks antes de darlos por buenos**
  (`jupyter nbconvert --to notebook --execute --inplace`) y mirar las figuras, no solo
  que no haya excepciones.
- **Revisar los `.docx` renderizándolos**: convertir a PDF con `soffice`, pasar a JPG con
  `pdftoppm` y mirar las páginas. Los subíndices Unicode (uᵣ) y algunos símbolos
  matemáticos no se dibujan bien en negrita.
- Los notebooks se versionan **con sus salidas**, así que cada reejecución produce un
  diff enorme. Tenerlo en cuenta antes de reejecutar y commitear por costumbre.
- Al informar resultados, distinguir lo medido de lo supuesto, y decir qué **no** se
  comprobó.
