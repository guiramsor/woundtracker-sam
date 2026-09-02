"""F8: la rejilla de profundidad sobre los k = 4 del apartado 5.3.

**Modos x protocolos x S0** sobre las cuatro filas elegidas: M0 y M1 en los
cuatro, y **M3 solo donde existe**, que en esta seleccion es SAM 2.1. La celda
de **M2 se declara vacia por dependencia no disponible** (apartado 5.1): no es
que no se haya medido, es que con estos datos no se puede medir, y por eso este
script no la intenta ni la deja como fallo.

**Se guardan las candidatas, todas, empaquetadas a bits.** Sin ellas F10 no
puede rederivar los umbrales de S1: su guarda anti-nucleo mira la forma, asi que
el Dice y la puntuacion no bastan. Van a `salidas/proceso/candidatas/`, un fichero por modelo y
unidad, con `np.packbits` sobre el ultimo eje, de modo que la forma se conserva
salvo el ancho, que se recupera de `config.LADO_PX`. Ocupan la octava parte.

En M1 el bucle lleva el frame por fuera y el protocolo por dentro, para que los
cuatro protocolos de un mismo frame compartan el embedding: es lo que hace la
cache de `Modelo.hay_que_codificar`.

    conda run -n <entorno> python scripts/profundidad.py
"""

import os

import numpy as np
import pandas as pd

from surco import adaptadores, config, datos, inventario, metricas, modelos, prompts

# Los cuatro del apartado 5.3, cada uno con su representante del cribado. La
# lista vive en el config: la comparten este script, M4, el selector, las
# figuras y el resumen, y tenerla repetida es la unica forma que tiene el
# trabajo de medir modelos distintos sin enterarse.
REJILLA = config.REJILLA_PROFUNDIDAD
SELECTOR = "S0"


# Los clics del protocolo y sus etiquetas, sobre la anotacion unica. Se
# reexporta con este nombre porque media docena de scripts lo importan de aqui.
prompt_de = prompts.puntos_y_etiquetas


def _fila(modelo, unidad, modo, protocolo, frame, dice, n_candidatas) -> dict:
    return {
        "modelo": modelo.nombre,
        "checkpoint": modelo.checkpoint,
        "familia": modelo.familia,
        "modo": modo,
        "protocolo": protocolo,
        "selector": SELECTOR,
        "serie": unidad["serie"],
        "etiqueta": unidad["etiqueta"],
        "frame": frame,
        "lado_entrada": modelo.lado_entrada,
        "n_candidatas": n_candidatas,
        "dice": dice,
    }


def _guardar(almacen: dict, modo: str, protocolo: str, frame: int, candidatas) -> None:
    """Empaqueta las candidatas de un prompt. El ancho se recupera del config."""
    clave = f"{modo}_{protocolo}_t{frame:02d}"
    almacen[clave] = np.packbits(candidatas.mascaras, axis=-1)
    almacen[f"{clave}_score"] = candidatas.puntuaciones


def secuencia_de_unidad(gt_serie: np.ndarray, unidad: dict) -> tuple[int, ...]:
    """Los frames en que existe la lesion, desde su `start`.

    Se exige que sean contiguos porque M3 propaga sobre la secuencia: un hueco
    cambiaria el espaciado temporal que ve el banco de memoria sin que se note.
    """
    frames = inventario.frames_de_unidad(gt_serie, unidad["etiqueta"])
    esperados = tuple(range(frames[0], frames[-1] + 1))
    if frames != esperados or frames[0] != unidad["start"]:
        raise ValueError(
            f"{unidad['serie']}-L{unidad['etiqueta']}: frames {frames}, "
            f"start {unidad['start']}"
        )
    return frames


def correr_m0(modelo, unidad, gt_serie, ventana, protocolos, almacen) -> list[dict]:
    """La mascara del frame semilla, copiada al resto. Es el suelo."""
    semilla = modelos.preparar_frame(unidad["serie"], unidad["start"])
    filas = []
    for protocolo in protocolos:
        puntos, etiquetas = prompt_de(unidad, protocolo)
        candidatas = modelo.segmentar(semilla, puntos, etiquetas)
        _guardar(almacen, "M0", protocolo, unidad["start"], candidatas)
        copiada = candidatas.mejor()
        for frame in ventana:
            referencia = gt_serie[frame - 1] == unidad["etiqueta"]
            filas.append(
                _fila(modelo, unidad, "M0", protocolo, frame,
                      metricas.dice(copiada, referencia), len(candidatas))
            )
    return filas


def correr_m1(modelo, unidad, gt_serie, ventana, protocolos, almacen) -> list[dict]:
    """Se re-promptea cada frame con los mismos clics, sin memoria."""
    filas = []
    for frame in ventana:
        imagen = modelos.preparar_frame(unidad["serie"], frame)
        referencia = gt_serie[frame - 1] == unidad["etiqueta"]
        for protocolo in protocolos:
            puntos, etiquetas = prompt_de(unidad, protocolo)
            candidatas = modelo.segmentar(imagen, puntos, etiquetas)
            _guardar(almacen, "M1", protocolo, frame, candidatas)
            filas.append(
                _fila(modelo, unidad, "M1", protocolo, frame,
                      metricas.dice(candidatas.mejor(), referencia), len(candidatas))
            )
    return filas


def correr_m3(modelo, unidad, gt_serie, frames, ventana, protocolos, almacen) -> list[dict]:
    """Propagacion con banco de memoria desde el frame semilla.

    La secuencia arranca en `start`, que es donde estan los clics, asi que la
    propagacion es solo hacia delante. El frame semilla sale de la propagacion
    pero no de la evaluacion: lo deja fuera el apartado 6.2.
    """
    secuencia = [modelos.preparar_frame(unidad["serie"], f) for f in frames]
    filas = []
    for protocolo in protocolos:
        puntos, etiquetas = prompt_de(unidad, protocolo)
        salida = modelo.propagar(secuencia, puntos, etiquetas, 0)
        for frame, candidatas in zip(frames, salida):
            if frame not in ventana:
                continue
            _guardar(almacen, "M3", protocolo, frame, candidatas)
            referencia = gt_serie[frame - 1] == unidad["etiqueta"]
            filas.append(
                _fila(modelo, unidad, "M3", protocolo, frame,
                      metricas.dice(candidatas.mejor(), referencia), len(candidatas))
            )
    return filas


def correr_unidad(modelo, unidad: dict, gt_serie: np.ndarray) -> list[dict]:
    frames = secuencia_de_unidad(gt_serie, unidad)
    ventana = metricas.ventana_de_evaluacion(frames, unidad["start"])
    protocolos = modelo.protocolos_admitidos()

    almacen: dict[str, np.ndarray] = {}
    filas = correr_m0(modelo, unidad, gt_serie, ventana, protocolos, almacen)
    filas += correr_m1(modelo, unidad, gt_serie, ventana, protocolos, almacen)
    if modelo.soporta_memoria:
        filas += correr_m3(
            modelo, unidad, gt_serie, frames, ventana, protocolos, almacen
        )

    config.DIR_CANDIDATAS.mkdir(parents=True, exist_ok=True)
    destino = config.DIR_CANDIDATAS / config.PATRON_CANDIDATAS.format(
        prefijo="", nombre=modelo.nombre, checkpoint=modelo.checkpoint,
        serie=unidad["serie"], etiqueta=unidad["etiqueta"],
    )
    np.savez_compressed(destino, **almacen)
    return filas


def main() -> None:
    adaptadores.cargar_disponibles()
    entorno = os.environ.get("CONDA_DEFAULT_ENV", "desconocido")
    registro = modelos.registrados()
    mios = [
        (nombre, punto)
        for nombre, punto in REJILLA.items()
        if nombre in registro and registro[nombre].entorno == entorno
    ]
    if not mios:
        print(f"ningun modelo de la rejilla en el entorno {entorno!r}")
        return

    unidades = prompts.cargar()["unidades"]
    gt = {u["serie"]: datos.cargar_gt_serie(u["serie"]) for u in unidades}

    filas: list[dict] = []
    caidos: list[dict] = []
    for nombre, checkpoint in mios:
        modelo = modelos.crear(nombre, checkpoint)
        try:
            modelo.cargar()
            modos = "M0 M1" + (" M3" if modelo.soporta_memoria else "")
            protocolos = " ".join(modelo.protocolos_admitidos())
            print(f"{modelo.identificador}: {modos} x {protocolos}", flush=True)
            for unidad in unidades:
                filas.extend(correr_unidad(modelo, unidad, gt[unidad["serie"]]))
                print(f"  {unidad['serie']}-L{unidad['etiqueta']}", flush=True)
        except Exception as error:  # noqa: BLE001
            # Lo que ya haya medido se queda: las filas se acumulan por unidad y
            # las candidatas se escriben por unidad, asi que un fallo a mitad no
            # tira la corrida entera.
            caidos.append(
                {"fila": modelo.identificador, "fallo": f"{type(error).__name__}: {error}"}
            )
            print(f"{modelo.identificador}: NO -> {type(error).__name__}: {error}", flush=True)

    if caidos:
        pd.DataFrame(caidos).to_csv(
            config.DIR_PROCESO / f"profundidad_caidos_{entorno}.csv", index=False
        )
    if not filas:
        return

    tabla = pd.DataFrame(filas)
    tabla.to_csv(config.DIR_PROCESO / f"profundidad_{entorno}.csv", index=False)
    print(f"\n=== {entorno}: {len(tabla)} filas ===")
    for clave, grupo in tabla.groupby(["modelo", "modo", "protocolo"]):
        media = metricas.media_entre_unidades(metricas.por_unidad(grupo))
        print(f"  {clave[0]:14s} {clave[1]}  {clave[2]}  {media:7.4f}")


if __name__ == "__main__":
    main()
