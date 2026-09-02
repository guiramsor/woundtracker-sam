"""F10: el factor selector, S0 contra S1, sobre las candidatas ya guardadas.

**No corre ningun modelo.** Las candidatas de F8 estan en
`salidas/proceso/candidatas/`, y reordenar
candidatas ya calculadas es aritmetica: no consume GPU. Por eso este script vive
en el entorno del banco y no en ninguno de los cuatro de modelo.

Solo M0 y M1. **M3 x S1 es una celda degenerada** (apartado 5.5): el predictor de
video devuelve una unica mascara y sin puntuacion, asi que no hay nada que
reordenar y S0 y S1 dan literalmente lo mismo. Enseñarla como dos columnas
iguales seria fingir una comparacion.

En M0 el selector actua **una sola vez, en el frame semilla**, y la mascara
elegida se copia al resto de la ventana: es lo que define el modo.

    conda run -n surco python scripts/selector_s0_s1.py
"""

import numpy as np
import pandas as pd

from surco import config, datos, inventario, metricas, modelos, prompts, selector
from surco.selector import desempaquetar

import profundidad


def _fila(nombre, modo, protocolo, unidad, frame, dices, extra) -> dict:
    return {
        "modelo": nombre,
        "modo": modo,
        "protocolo": protocolo,
        "serie": unidad["serie"],
        "etiqueta": unidad["etiqueta"],
        "frame": frame,
        **dices,
        **extra,
    }


def decidir(guardado, clave, gris, unidad, protocolo):
    """Las cuatro elecciones sobre el mismo conjunto de candidatas.

    Ademas de S0 y S1 se guardan **las dos mitades de S1 por separado**: solo la
    puerta, con el orden del modelo detras, y solo el ranking por z, sin puerta.
    Con el respaldo actuando en buena parte de los prompts de P4, sin esta
    descomposicion no se sabe cual de las dos partes sostiene el resultado.
    """
    candidatas = desempaquetar(guardado, clave)
    puntos, etiquetas = profundidad.prompt_de(unidad, protocolo)
    indice_s1, respaldo = selector.elegir(candidatas, gris, puntos, etiquetas)
    indice_s0 = selector.elegir_con("S0", candidatas, gris, puntos, etiquetas)
    indices = {
        "s0": indice_s0,
        "s1": indice_s1,
        "puerta": selector.elegir_solo_puerta(candidatas, gris, puntos, etiquetas),
        "ranking": selector.elegir_solo_ranking(candidatas, gris, puntos, etiquetas),
    }
    extra = {
        "n_candidatas": len(candidatas),
        "respaldo": respaldo,
        "coincide": indice_s0 == indice_s1,
    }
    return {nombre: candidatas.mascaras[i] for nombre, i in indices.items()}, extra


def _dices(elegidas: dict, referencia) -> dict:
    return {f"dice_{nombre}": metricas.dice(mascara, referencia)
            for nombre, mascara in elegidas.items()}


def de_m0(guardados, unidad, gt_serie, ventana) -> list[dict]:
    """El selector decide en la semilla y la mascara elegida se copia."""
    gris = modelos.preparar_frame(unidad["serie"], unidad["start"])[:, :, 0]
    filas = []
    for nombre, guardado in guardados.items():
        for protocolo in config.PROTOCOLOS:
            clave = f"M0_{protocolo}_t{unidad['start']:02d}"
            if clave not in guardado:
                continue
            elegidas, extra = decidir(guardado, clave, gris, unidad, protocolo)
            for frame in ventana:
                referencia = gt_serie[frame - 1] == unidad["etiqueta"]
                filas.append(
                    _fila(nombre, "M0", protocolo, unidad, frame,
                          _dices(elegidas, referencia), extra)
                )
    return filas


def de_m1(guardados, unidad, gt_serie, ventana) -> list[dict]:
    """El selector decide en cada frame, sobre las candidatas de ese frame."""
    filas = []
    for frame in ventana:
        gris = modelos.preparar_frame(unidad["serie"], frame)[:, :, 0]
        referencia = gt_serie[frame - 1] == unidad["etiqueta"]
        for nombre, guardado in guardados.items():
            for protocolo in config.PROTOCOLOS:
                clave = f"M1_{protocolo}_t{frame:02d}"
                if clave not in guardado:
                    continue
                elegidas, extra = decidir(guardado, clave, gris, unidad, protocolo)
                filas.append(
                    _fila(nombre, "M1", protocolo, unidad, frame,
                          _dices(elegidas, referencia), extra)
                )
    return filas


def cargar_candidatas(unidad: dict) -> dict:
    """Los cuatro ficheros de esa unidad, uno por modelo de la rejilla."""
    guardados = {}
    for nombre, checkpoint in profundidad.REJILLA.items():
        ruta = selector.ruta_de_candidatas(
            nombre, checkpoint, unidad["serie"], unidad["etiqueta"]
        )
        if not ruta.exists():
            raise FileNotFoundError(f"faltan las candidatas de F8: {ruta}")
        with np.load(ruta) as fichero:
            guardados[nombre] = dict(fichero)
    return guardados


def main() -> None:
    filas = []
    for unidad in prompts.cargar()["unidades"]:
        gt_serie = datos.cargar_gt_serie(unidad["serie"])
        frames = inventario.frames_de_unidad(gt_serie, unidad["etiqueta"])
        ventana = metricas.ventana_de_evaluacion(frames, unidad["start"])
        guardados = cargar_candidatas(unidad)
        filas.extend(de_m0(guardados, unidad, gt_serie, ventana))
        filas.extend(de_m1(guardados, unidad, gt_serie, ventana))
        print(f"  {unidad['serie']}-L{unidad['etiqueta']}", flush=True)

    tabla = pd.DataFrame(filas)
    tabla.to_csv(config.DIR_PROCESO / "selector.csv", index=False)
    print(f"\n{len(tabla)} filas en {config.DIR_PROCESO / 'selector.csv'}")


if __name__ == "__main__":
    main()
