# surco

Banco de pruebas para la segmentación automática de la ROI del surco de microirradiación.
El código vive en `src/surco/`; los notebooks orquestan y muestran, no contienen lógica. Las
decisiones de diseño están en la memoria del trabajo.

**Empieza por `notebooks/GUIA.ipynb`**: es el índice ejecutable, con una sección por fase
que dice qué hace, qué scripts ejecuta, en qué entorno, qué ficheros produce y qué notebook
los muestra. Lleva la secuencia entera desde cero, y su última celda comprueba si esta copia
está completa.

## Instalación

    conda env create -f environment.yml
    conda run -n surco pip install -e .

Comprobación:

    conda run -n surco pytest

Y si `salidas/` cuadra con lo que dice la memoria:

    conda run -n surco python -c "from surco import verificacion; t = verificacion.comprobar(); print(verificacion.resumen(t))"

## El trabajo está partido en dos mitades

**Cada modelo necesita su propio entorno conda**, con versiones de torch y de sus repositorios
que no conviven. El entorno del banco, `surco`, no lleva torch ni ningún modelo, y **las
únicas fases que necesitan otro son F5-F6, F7 y F8**; el resto corre entero aquí, incluido el
factor selector de F10, porque reordenar candidatas ya calculadas es aritmética.

Las recetas están en `entornos/*.yml`, los pesos van fuera del repositorio en `SURCO_PESOS` y
los repositorios clonados de terceros, en `SURCO_REPOS`.

## Los notebooks se editan a mano

Son la fuente, no una salida. Se editan como cualquier otro fichero y después se ejecutan:

    conda run -n surco jupyter nbconvert --to notebook --execute --inplace notebooks/<nombre>.ipynb

Lo que **no** se edita a mano es la lógica: los notebooks importan de `surco` y leen los CSV
de `salidas/`, y cada celda que lee uno dice al lado qué script lo generó.

## Las dos tablas finales

`salidas/resumen.csv`, una fila por *checkpoint* medido y con su `LEEME_resumen.md` al lado, y
`salidas/contrastes.csv`, una fila por comparación pareada. Qué lleva cada una, en la guía.

**De `salidas/` solo viajan esas tres.** El resto, unos 60 CSV y las figuras, se regenera
siguiendo la guía. Lo único irrecuperable es `anotaciones/`, que sí está versionada: los cinco
clics del autor y las ROIs de las dos anotadoras.

## Las carpetas de `scripts/`

- **`scripts/`**, raíz: lo que se ejecuta para reproducir el trabajo, en el orden de la guía.
- **`scripts/entornos/`**: parches a paquetes de terceros y verificaciones de exclusión. Se
  ejecutan al construir un entorno, no al reproducir resultados.
- **`scripts/unicos/`**: de un solo uso, con su resultado ya congelado en un fichero
  versionado o citado en la memoria. Se conservan porque documentan por qué un dato está como
  está.

`entornos/LEEME.md` explica las recetas, incluidas las dos de modelos excluidos y los tres
entornos que llevan parche.
