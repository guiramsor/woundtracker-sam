"""Medida previa del apartado 8: deriva del centroide del GT.

El clic de M1 se coloca en `start` y no se vuelve a mover. La pregunta es si la
lesion se desplaza lo suficiente como para que ese clic acabe fuera de ella. El
desplazamiento del centroide se descompone en los ejes del PCA de la lesion: la
componente paralela al surco es inofensiva, porque el clic sigue dentro; la
perpendicular es la que lo saca.

La deriva perpendicular se reporta **normalizada por el semiancho de la
unidad**. En pixeles no es comparable entre unidades: el semiancho medido en F1
va de 5.50 a 21.28 px, asi que los mismos 8 px son media anchura en una lesion y
dos anchuras en otra. El valor en pixeles se conserva como diagnostico.

El criterio es el **test directo**: se toma el pixel del centroide del GT en
`start` y se mira si sigue cayendo dentro de la mascara de la misma unidad en
cada frame posterior. Eso es literalmente lo que le pasa al clic de M1. El
cociente contra el semiancho acompana como diagnostico continuo, normalizado
por el semiancho **del frame t**.

El origen, los ejes y el pixel del clic quedan fijados en `start`, que es donde
se pone el clic. `giro_eje_grados` mide cuanto se aparta el eje de cada frame
del de `start`, para que se vea si fijar ahi la base es inocuo.
"""

import numpy as np
import pandas as pd

from surco import config, datos, geometria, inventario


def centroide(mascara: np.ndarray) -> np.ndarray:
    """Centro de masas de la mascara, en coordenadas (fila, columna)."""
    filas, columnas = np.nonzero(mascara)
    return np.array([filas.mean(), columnas.mean()])


def ejes_pca(mascara: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ejes mayor y menor de la lesion, vectores unitarios en (fila, columna).

    Salen de los autovectores de la covarianza de las coordenadas y no del
    `orientation` de skimage, para no depender de su convencion de signo. El
    signo de un autovector es arbitrario, asi que solo son fiables los valores
    absolutos de las proyecciones entre unidades distintas; dentro de una
    unidad la base es la misma en todos los frames y el signo si informa.
    """
    filas, columnas = np.nonzero(mascara)
    if len(filas) < 2:
        raise ValueError("hacen falta al menos dos pixeles para orientar la lesion")
    coords = np.column_stack([filas, columnas]).astype(float)
    covarianza = np.cov(coords, rowvar=False)
    _, vectores = np.linalg.eigh(covarianza)
    return vectores[:, -1], vectores[:, 0]


def _giro_del_eje(eje_referencia: np.ndarray, mascara: np.ndarray) -> float:
    """Angulo en grados entre el eje mayor de este frame y el de `start`."""
    try:
        eje, _ = ejes_pca(mascara)
    except ValueError:
        return float("nan")
    coseno = min(1.0, abs(float(eje @ eje_referencia)))
    return float(np.degrees(np.arccos(coseno)))


def pixel_del_centroide(mascara: np.ndarray) -> tuple[int, int]:
    """El centroide redondeado al pixel mas cercano.

    Es el pixel que ocuparia el clic central del protocolo si se pusiera en el
    centro de masas del GT, y el que se sigue frame a frame en el test directo.
    """
    fila, columna = centroide(mascara)
    return int(round(fila)), int(round(columna))


def _base_de_referencia(mascara_start: np.ndarray) -> dict:
    """Todo lo que queda fijado en `start`: origen, ejes, pixel del clic."""
    eje_mayor, eje_menor = ejes_pca(mascara_start)
    pixel = pixel_del_centroide(mascara_start)
    return {
        "origen": centroide(mascara_start),
        "eje_mayor": eje_mayor,
        "eje_menor": eje_menor,
        "pixel_clic": pixel,
        "clic_dentro_en_start": bool(mascara_start[pixel]),
        "semiancho_start_px": geometria.medir(mascara_start)["semiancho_px"],
    }


def _fila_de_frame(mascara: np.ndarray, frame: int, base: dict) -> dict:
    """El estado de la unidad en un frame respecto de la base de `start`."""
    desplazamiento = centroide(mascara) - base["origen"]
    perpendicular = float(desplazamiento @ base["eje_menor"])
    semiancho_t = geometria.medir(mascara)["semiancho_px"]
    return {
        "frame": frame,
        "clic_dentro": bool(mascara[base["pixel_clic"]]),
        "clic_dentro_en_start": base["clic_dentro_en_start"],
        "paralela_px": float(desplazamiento @ base["eje_mayor"]),
        "perpendicular_px": perpendicular,
        "semiancho_start_px": base["semiancho_start_px"],
        "semiancho_frame_px": semiancho_t,
        "perpendicular_norm": abs(perpendicular) / semiancho_t,
        "giro_eje_grados": _giro_del_eje(base["eje_mayor"], mascara),
    }


def deriva_de_unidad(gt_serie: np.ndarray, serie: int, etiqueta: int) -> pd.DataFrame:
    """Una fila por frame posterior a `start`.

    El cociente se normaliza por el semiancho **del frame t**, no por el de
    `start`: la lesion se ensancha con el tiempo y normalizar por el de `start`
    infla el cociente en los frames tardios.
    """
    frames = inventario.frames_de_unidad(gt_serie, etiqueta)
    start = frames[0]
    base = _base_de_referencia(gt_serie[start - 1] == etiqueta)
    identificacion = {"serie": serie, "etiqueta": etiqueta, "start": start}
    filas = [
        identificacion | _fila_de_frame(gt_serie[frame - 1] == etiqueta, frame, base)
        for frame in frames[1:]
    ]
    return pd.DataFrame(filas)


def construir_deriva() -> pd.DataFrame:
    """La deriva de las 17 unidades, leyendo el GT de disco."""
    trozos = []
    for serie in config.SERIES_CON_GT:
        gt_serie = datos.cargar_gt_serie(serie)
        for etiqueta in inventario.etiquetas_de_serie(gt_serie):
            trozos.append(deriva_de_unidad(gt_serie, serie, etiqueta))
    return pd.concat(trozos, ignore_index=True)


def resumen_por_unidad(deriva: pd.DataFrame) -> pd.DataFrame:
    """Una fila por unidad con el peor caso de su ventana.

    `sale_alguna_vez` es el resultado del test directo, el que decide. El
    cociente `perp_max_norm` acompana como diagnostico continuo.
    """
    filas = []
    for (serie, etiqueta), grupo in deriva.groupby(["serie", "etiqueta"]):
        peor = grupo.loc[grupo["perpendicular_norm"].idxmax()]
        fuera = ~grupo["clic_dentro"]
        filas.append(
            {
                "serie": int(serie),
                "etiqueta": int(etiqueta),
                "start": int(grupo["start"].iloc[0]),
                "n_frames": len(grupo),
                "clic_dentro_en_start": bool(grupo["clic_dentro_en_start"].iloc[0]),
                "pares_fuera": int(fuera.sum()),
                "sale_alguna_vez": bool(fuera.any()),
                "primer_frame_fuera": (
                    int(grupo.loc[fuera, "frame"].min()) if fuera.any() else None
                ),
                "semiancho_start_px": float(grupo["semiancho_start_px"].iloc[0]),
                "perp_max_px": float(grupo["perpendicular_px"].abs().max()),
                "perp_max_norm": float(grupo["perpendicular_norm"].max()),
                "frame_del_maximo": int(peor["frame"]),
                "paralela_max_px": float(grupo["paralela_px"].abs().max()),
                "giro_eje_max_grados": float(grupo["giro_eje_grados"].max()),
                "supera_semiancho": bool(grupo["perpendicular_norm"].max() > 1),
            }
        )
    return pd.DataFrame(filas)


def fracciones_del_clic_fijo(deriva: pd.DataFrame) -> pd.Series:
    """El test directo del apartado 8, que es el numero que va a la memoria.

    Se toma el pixel del centroide del GT en `start` y se mira si sigue cayendo
    dentro de la mascara de la misma unidad en cada frame posterior. Si nunca se
    sale, M1 y M2 dan lo mismo y M2 sobra.
    """
    fuera = ~deriva["clic_dentro"]
    por_unidad = deriva.groupby(["serie", "etiqueta"])["clic_dentro"].apply(
        lambda dentro: bool((~dentro).any())
    )
    return pd.Series(
        {
            "pares_fuera": int(fuera.sum()),
            "pares_totales": int(len(deriva)),
            "fraccion_pares_fuera": float(fuera.mean()),
            "unidades_que_se_salen": int(por_unidad.sum()),
            "unidades_totales": int(len(por_unidad)),
            "fraccion_unidades_que_se_salen": float(por_unidad.mean()),
        }
    )


def cruzar_con_inventario(
    deriva_df: pd.DataFrame, inventario_df: pd.DataFrame
) -> pd.DataFrame:
    """Une la deriva con la fragmentacion y el area de cada par.

    El centroide de una mascara partida se mueve aunque la celula este quieta:
    basta con que desaparezca un fragmento de un extremo. Sin este cruce no hay
    forma de separar deriva real de efecto de la fragmentacion.
    """
    columnas = ["serie", "etiqueta", "frame", config.COLUMNA_FRAGMENTACION, "area_px2"]
    return deriva_df.merge(
        inventario_df[columnas], on=["serie", "etiqueta", "frame"]
    )


def norma_con_semiancho_de_start(deriva: pd.DataFrame) -> pd.Series:
    """El cociente como se calculaba antes, contra el semiancho de `start`.

    Se conserva solo para poder ensenar cuanto infla: la lesion se ensancha con
    el tiempo, asi que dividir por el semiancho inicial exagera los frames
    tardios. El cociente bueno es el de `perpendicular_norm`.
    """
    return deriva["perpendicular_px"].abs() / deriva["semiancho_start_px"]
