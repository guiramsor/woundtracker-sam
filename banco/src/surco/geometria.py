"""Medidas sobre la mascara de una lesion: entra una mascara booleana, salen
numeros.

Las componentes de una lesion partida se miden **como una sola region**: el eje
del surco es uno aunque el trazo este roto, y es ese eje el que necesita la
descomposicion en PCA del apartado 8. El numero de componentes se reporta
aparte, como propiedad del par (unidad, frame).
"""

import numpy as np
from skimage.measure import label as etiquetar
from skimage.measure import regionprops

# La conectividad no esta fijada en la base. Se reportan las dos y que decida
# quien lea la tabla: 1 = 4 vecinos, 2 = 8 vecinos (skimage).
CONECTIVIDAD_4 = 1
CONECTIVIDAD_8 = 2


def n_componentes(mascara: np.ndarray, conectividad: int) -> int:
    _, n = etiquetar(mascara, connectivity=conectividad, return_num=True)
    return int(n)


def medir(mascara: np.ndarray) -> dict:
    """Area, ejes de la elipse equivalente, centroide y numero de componentes.

    `elongacion` es eje mayor / eje menor, la misma razon con la que el
    apartado 3 estima la geometria del surco. `semiancho_px` es el semieje
    menor: es lo que F2 compara contra la deriva perpendicular, y es por lesion,
    no un valor global.
    """
    if not mascara.any():
        raise ValueError("mascara vacia: no hay nada que medir")

    props = regionprops(mascara.astype(np.uint8))[0]
    eje_mayor = float(props.axis_major_length)
    eje_menor = float(props.axis_minor_length)
    return {
        "area_px2": int(props.area),
        "eje_mayor_px": eje_mayor,
        "eje_menor_px": eje_menor,
        "semiancho_px": eje_menor / 2,
        "elongacion": eje_mayor / eje_menor if eje_menor > 0 else float("nan"),
        "orientacion_rad": float(props.orientation),
        "centroide_fila": float(props.centroid[0]),
        "centroide_col": float(props.centroid[1]),
        "n_comp_c4": n_componentes(mascara, CONECTIVIDAD_4),
        "n_comp_c8": n_componentes(mascara, CONECTIVIDAD_8),
    }
