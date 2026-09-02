"""Comprobacion de carga: que modelos estan disponibles de verdad.

Un modelo esta disponible si **se instancia, traga un punto sobre una imagen
sintetica y devuelve candidatas de la forma correcta**. Nada mas. Aqui no se
mide calidad ni tiempo: eso es F7, y mezclarlo llevaria a instalar y medir a la
vez, que es justo lo que el apartado 11 dice que no hay que hacer.

El apartado 11 tambien dice por que esto existe: el cuello de botella no es la
GPU, son los entornos, y un modelo que se resiste en la semana cuatro es un
hueco en la tabla. Esto convierte "se resiste" en una fila con un motivo.

    conda run -n surco python scripts/comprobar_carga.py
"""

import argparse
import os
import traceback

import pandas as pd

from surco import adaptadores, config, modelos


def comprobar_gpu() -> str:
    """Si la GPU no solo se detecta, sino que ejecuta un kernel de verdad.

    `torch.cuda.is_available()` puede devolver True y fallar despues todo lo que
    se le pida: pasa cuando el build de torch no trae la arquitectura de la
    tarjeta. Como el apartado 6.6 reporta el coste en GPU, conviene que eso
    salte aqui y no al medir.
    """
    try:
        import torch
    except ImportError:
        return "sin torch"
    if not torch.cuda.is_available():
        return "sin CUDA"
    try:
        tensor = torch.randn(64, 64, device="cuda")
        float((tensor @ tensor).sum())
        torch.cuda.synchronize()
    except Exception as error:  # noqa: BLE001
        return f"CUDA se detecta pero no ejecuta: {type(error).__name__}"
    return f"CUDA ok en {torch.cuda.get_device_name(0)}"


def comprobar(nombre: str, checkpoint: str) -> dict:
    """Instancia, carga y pide una mascara. Devuelve la fila del informe."""
    fila = {
        "modelo": nombre,
        "checkpoint": checkpoint,
        "por_defecto": None,
        "familia": None,
        "acepta_negativos": None,
        "protocolos": None,
        "modos": None,
        "propaga": None,
        "instancia": False,
        "carga": False,
        "segmenta": False,
        "n_candidatas": None,
        "forma_correcta": False,
        "disponible": False,
        "fallo": "",
    }
    try:
        modelo = modelos.crear(nombre, checkpoint)
        fila["instancia"] = True
        fila["por_defecto"] = modelo.es_por_defecto
        fila["familia"] = modelo.familia
        fila["acepta_negativos"] = modelo.acepta_negativos
        fila["protocolos"] = " ".join(modelo.protocolos_admitidos())
        fila["modos"] = " ".join(modelo.modos_admitidos())

        modelo.cargar()
        fila["carga"] = True

        imagen = modelos.imagen_de_prueba_rgb()
        candidatas = modelo.segmentar(imagen, [modelos.punto_de_prueba()], [1])
        fila["segmenta"] = True
        fila["n_candidatas"] = len(candidatas)
        fila["forma_correcta"] = candidatas.mascaras.shape[1:] == imagen.shape[:2]
        fila["disponible"] = fila["forma_correcta"]

        # Quien dice tener banco de memoria tiene que demostrarlo aqui: sin
        # esto, un M3 roto no aparece hasta la rejilla de F8.
        if modelo.soporta_memoria:
            secuencia = modelos.secuencia_de_prueba()
            salida = modelo.propagar(secuencia, [modelos.punto_de_prueba()], [1], 0)
            fila["propaga"] = len(salida) == len(secuencia) and all(
                c.mascaras.shape[1:] == imagen.shape[:2] for c in salida
            )
            fila["disponible"] = fila["forma_correcta"] and bool(fila["propaga"])
    except Exception as error:  # noqa: BLE001 - se quiere el motivo, no la traza
        fila["fallo"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    return fila


def main(argv: list[str] | None = None) -> None:
    analizador = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    analizador.add_argument(
        "--solo-por-defecto",
        action="store_true",
        help="solo el checkpoint por defecto de cada modelo, que es el orden "
        "de medicion que fija el apartado 5.2",
    )
    analizador.add_argument(
        "--todos-los-entornos",
        action="store_true",
        help="no filtrar por entorno. Solo para depurar: el informe saldra con "
        "modelos que en este entorno no estan instalados",
    )
    argumentos = analizador.parse_args(argv)

    entorno = os.environ.get("CONDA_DEFAULT_ENV", "desconocido")
    importados, fallidos = adaptadores.cargar_disponibles()
    print(f"entorno: {entorno}")
    print(f"adaptadores importados: {', '.join(importados) or 'ninguno'}")
    for modulo, motivo in fallidos.items():
        print(f"  {modulo}: no se pudo importar -> {motivo}")
    print(f"gpu: {comprobar_gpu()}\n")

    # Cada informe cubre solo el modelo de su entorno. Sin este filtro salen
    # filas de modelos que aqui no estan instalados, y como la tabla final es la
    # union de los informes, esas filas ensucian con fallos que no significan
    # nada. TinySAM es el caso claro: no se instala, entra por sys.path, y por
    # tanto es importable desde todos los entornos.
    pares = modelos.filas(
        solo_por_defecto=argumentos.solo_por_defecto,
        entorno=None if argumentos.todos_los_entornos else entorno,
    )
    if not pares:
        print(f"ningun adaptador declara pertenecer al entorno {entorno!r}")
        return

    informe = pd.DataFrame([comprobar(nombre, punto) for nombre, punto in pares])
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print(informe.to_string(index=False))

    disponibles = int(informe["disponible"].sum())
    print(f"\ndisponibles: {disponibles} de {len(informe)}")
    caidos = informe[~informe["disponible"]]
    for _, fila in caidos.iterrows():
        etiqueta = f"{fila['modelo']}:{fila['checkpoint']}"
        print(f"  {etiqueta}: {fila['fallo'] or 'forma de mascara incorrecta'}")

    # Un informe por entorno, con solo sus modelos. La tabla final de
    # disponibilidad es la union de estos ficheros.
    informe.insert(0, "entorno", entorno)
    destino = config.DIR_PROCESO / f"comprobacion_carga_{entorno}.csv"
    informe.to_csv(destino, index=False)
    print(f"\ninforme en {destino}")


if __name__ == "__main__":
    main()
