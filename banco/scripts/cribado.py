"""F7, primera pasada: el cribado del apartado 5.3.

Configuracion fija e identica para todos: **M1 x P1 x S0**, sobre las 17
unidades y en la ventana del apartado 6.2, desde `start + 1`.

Corre **solo los modelos de su propio entorno**, asi que hay que lanzarlo una
vez por entorno y la tabla final es la union de los CSV. Por defecto solo el
checkpoint por defecto de cada modelo, que es el orden de medicion del apartado
5.2.

**El coste va en dos columnas** (apartado 6.6): coste por imagen, que es
codificar el frame y resolver el primer prompt, y coste por prompt adicional
sobre la misma imagen. Se obtienen llamando dos veces a `segmentar` sobre el
mismo frame y quedandose con cada tiempo, sin anadir al contrato una operacion
de codificacion separada: asi se mide lo que el pipeline ejecuta de verdad. Se
descarta una llamada de calentamiento antes de cronometrar, porque la primera
sobre una imagen nueva arrastra reservas de memoria y compilacion de kernels que
no se repiten.

    conda run -n <entorno> python scripts/cribado.py
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from surco import adaptadores, config, datos, inventario, metricas, modelos, prompts

PROTOCOLO = "P1"
MODO = "M1"
SELECTOR = "S0"


def _clics(unidad: dict) -> tuple[list, list]:
    return prompts.puntos_y_etiquetas(unidad, PROTOCOLO)


def _cronometrar(modelo, imagen, otra, puntos, etiquetas) -> tuple[float, float]:
    """Coste por imagen y coste por prompt adicional, en segundos.

    El calentamiento va **sobre otra imagen**, no sobre la que se cronometra.
    Si fuera la misma dejaria su embedding en la cache y la primera medida ya no
    incluiria la codificacion: las dos columnas volverian a medir lo mismo, esta
    vez las dos por lo bajo.

    Asi, la primera llamada encuentra la cache fria y paga codificar; la segunda
    la encuentra caliente y paga solo el decodificador. En los modelos sin
    `set_image` las dos pagan lo mismo, que es su comportamiento real.
    """
    modelo.segmentar(otra, puntos, etiquetas)  # calentamiento, se tira

    inicio = time.perf_counter()
    modelo.segmentar(imagen, puntos, etiquetas)
    por_imagen = time.perf_counter() - inicio

    inicio = time.perf_counter()
    modelo.segmentar(imagen, puntos, etiquetas)
    por_prompt = time.perf_counter() - inicio
    return por_imagen, por_prompt


def medir_unidad(modelo, unidad: dict, gt_serie: np.ndarray) -> list[dict]:
    serie, etiqueta = unidad["serie"], unidad["etiqueta"]
    puntos, etiquetas = _clics(unidad)
    frames = inventario.frames_de_unidad(gt_serie, etiqueta)
    ventana = metricas.ventana_de_evaluacion(frames, unidad["start"])

    filas = []
    for frame in ventana:
        imagen = modelos.preparar_frame(serie, frame)
        candidatas = modelo.segmentar(imagen, puntos, etiquetas)
        filas.append(
            {
                "modelo": modelo.nombre,
                "checkpoint": modelo.checkpoint,
                "familia": modelo.familia,
                "modo": MODO,
                "protocolo": PROTOCOLO,
                "selector": SELECTOR,
                "serie": serie,
                "etiqueta": etiqueta,
                "frame": frame,
                "lado_entrada": modelo.lado_entrada,
                "n_candidatas": len(candidatas),
                # S0 es quedarse con la de mayor puntuacion del modelo.
                "dice": metricas.dice(candidatas.mejor(), gt_serie[frame - 1] == etiqueta),
            }
        )
    return filas


def coste_de(modelo, unidad: dict) -> dict:
    puntos, etiquetas = _clics(unidad)
    imagen = modelos.preparar_frame(unidad["serie"], unidad["start"] + 1)
    otra = modelos.preparar_frame(unidad["serie"], unidad["start"])
    por_imagen, por_prompt = _cronometrar(modelo, imagen, otra, puntos, etiquetas)
    return {
        "modelo": modelo.nombre,
        "checkpoint": modelo.checkpoint,
        "dispositivo": getattr(modelo, "dispositivo", None),
        "coste_por_imagen_s": por_imagen,
        "coste_prompt_adicional_s": por_prompt,
    }


def main(argv: list[str] | None = None) -> None:
    analizador = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    analizador.add_argument(
        "--todos-los-checkpoints",
        action="store_true",
        help="no solo el por defecto de cada modelo",
    )
    analizador.add_argument(
        "--solo-coste",
        action="store_true",
        help="rehace unicamente las dos columnas de coste, sin volver a medir "
        "el Dice, que no depende de la cache del embedding",
    )
    argumentos = analizador.parse_args(argv)

    adaptadores.cargar_disponibles()
    entorno = os.environ.get("CONDA_DEFAULT_ENV", "desconocido")
    pares = modelos.filas(
        solo_por_defecto=not argumentos.todos_los_checkpoints, entorno=entorno
    )
    if not pares:
        print(f"ningun modelo del entorno {entorno!r}")
        return

    unidades = prompts.cargar()["unidades"]
    gt = {u["serie"]: datos.cargar_gt_serie(u["serie"]) for u in unidades}

    filas, costes, caidos = [], [], []
    for nombre, checkpoint in pares:
        modelo = modelos.crear(nombre, checkpoint)
        try:
            modelo.cargar()
        except Exception as error:  # noqa: BLE001
            # Un checkpoint que no arranca no puede tumbar a los demas: sale
            # como fila con su motivo, que es lo que hace la tabla honesta.
            caidos.append({"fila": modelo.identificador, "fallo": f"{type(error).__name__}: {error}"})
            print(f"{modelo.identificador}: NO carga -> {type(error).__name__}", flush=True)
            continue
        print(f"{modelo.identificador}: cargado", flush=True)
        try:
            costes.append(coste_de(modelo, unidades[0]))
            if argumentos.solo_coste:
                continue
            propias = []
            for unidad in unidades:
                propias.extend(medir_unidad(modelo, unidad, gt[unidad["serie"]]))
                print(f"  {unidad['serie']}-L{unidad['etiqueta']}", flush=True)
        except Exception as error:  # noqa: BLE001
            # Cargar y segmentar fallan por motivos distintos: TinySAM w8a8
            # carga bien y revienta al inferir. Sin este segundo cerco, un
            # checkpoint asi se lleva por delante los de su mismo entorno.
            caidos.append(
                {
                    "fila": modelo.identificador,
                    "fallo": f"{type(error).__name__}: {error}",
                }
            )
            print(f"{modelo.identificador}: NO mide -> {type(error).__name__}", flush=True)
            continue
        filas.extend(propias)

    pd.DataFrame(costes).to_csv(
        config.DIR_PROCESO / f"coste_{entorno}.csv", index=False
    )
    print(f"\n=== {entorno} ===")
    print(pd.DataFrame(costes).to_string(index=False))
    if caidos:
        pd.DataFrame(caidos).to_csv(
            config.DIR_PROCESO / f"cribado_caidos_{entorno}.csv", index=False
        )
        for caido in caidos:
            print(f"  no medido -> {caido['fila']}: {caido['fallo'][:90]}")

    if argumentos.solo_coste:
        return

    tabla = pd.DataFrame(filas)
    tabla.to_csv(config.DIR_PROCESO / f"cribado_{entorno}.csv", index=False)
    por_unidad = metricas.por_unidad(tabla)
    print(f"{len(tabla)} pares medidos")
    print(f"Dice medio entre unidades: {metricas.media_entre_unidades(por_unidad):.4f}")


if __name__ == "__main__":
    main()
