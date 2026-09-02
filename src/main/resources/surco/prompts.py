"""El fichero de prompts: los cinco clics de cada unidad, congelados.

Es la unica entrada manual de todo el trabajo y por eso se versiona. Este
modulo no dibuja nada: define el formato, lo lee, lo escribe y calcula los dos
diagnosticos del apartado 4.3. La ventana de anotacion esta en `picker.py`.

Guarda tambien el `start` de cada unidad. El apartado 6.2 evalua desde
`start + 1`, asi que sin ese dato ninguna fase posterior sabe donde empieza su
ventana: es lo que engancha F3 con todo lo demas.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from surco import config

VERSION_FORMATO = 1

Unidad = tuple[int, int]
Clic = tuple[int, int]


def esquema_vacio() -> dict:
    return {"version": VERSION_FORMATO, "n_clics": config.N_CLICS, "unidades": []}


def _completar(datos: dict) -> dict:
    """Rellena los campos que una anotacion antigua pueda no tener.

    `consulta_temporal` se anadio despues de anotar las primeras unidades. No
    hace falta migrar el fichero ni subir la version: lo que se lee queda
    completo en memoria, y la proxima escritura deja el fichero uniforme.
    """
    for unidad in datos["unidades"]:
        unidad.setdefault("consulta_temporal", False)
    return datos


def cargar(ruta: Path | None = None) -> dict:
    """Lee el fichero. Si todavia no existe, devuelve el esquema vacio."""
    ruta = ruta or config.RUTA_CLICS
    if not ruta.exists():
        return esquema_vacio()
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    if datos.get("version") != VERSION_FORMATO:
        raise ValueError(
            f"{ruta.name}: version {datos.get('version')}, "
            f"se esperaba {VERSION_FORMATO}"
        )
    return _completar(datos)


def guardar(datos: dict, ruta: Path | None = None) -> None:
    """Escribe atomicamente: primero un temporal, despues el reemplazo.

    Anotar es una sesion larga y por tandas. Una interrupcion a mitad de
    escritura no puede dejar a medias el unico fichero irrecuperable del
    trabajo.
    """
    ruta = ruta or config.RUTA_CLICS
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(ruta.name + ".tmp")
    temporal.write_text(
        json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporal.replace(ruta)


def anotadas(datos: dict) -> set[Unidad]:
    return {(u["serie"], u["etiqueta"]) for u in datos["unidades"]}


def pendientes(datos: dict, todas: list[Unidad]) -> list[Unidad]:
    """Las unidades que faltan por anotar, en el orden en que se dieron."""
    hechas = anotadas(datos)
    return [unidad for unidad in todas if unidad not in hechas]


def validar_clics(clics: list[Clic]) -> None:
    if len(clics) != config.N_CLICS:
        raise ValueError(
            f"hacen falta {config.N_CLICS} clics, llegaron {len(clics)}"
        )
    for fila, columna in clics:
        dentro = 0 <= fila < config.LADO_PX and 0 <= columna < config.LADO_PX
        if not dentro:
            raise ValueError(f"clic fuera de la imagen: ({fila}, {columna})")


def diagnosticos(clics: list[Clic], mascara_gt: np.ndarray) -> dict:
    """Los dos campos de diagnostico del apartado 4.3, sobre el clic central.

    No son metrica. Cuando una unidad se hunda en todos los modelos, diran si
    fallo el modelo o fallo el clic.
    """
    fila, columna = clics[0]
    filas, columnas = np.nonzero(mascara_gt)
    centroide = np.array([filas.mean(), columnas.mean()])
    distancia = float(np.hypot(*(np.array([fila, columna]) - centroide)))
    return {
        "dist_clic_centroide": distancia,
        "clic_dentro_gt": bool(mascara_gt[int(fila), int(columna)]),
    }


def registrar(
    datos: dict,
    serie: int,
    etiqueta: int,
    start: int,
    clics: list[Clic],
    excepcion: bool,
    mascara_gt: np.ndarray,
    consulta_temporal: bool = False,
) -> dict:
    """Anade o reemplaza la anotacion de una unidad y devuelve el dict nuevo.

    `excepcion` dice si de verdad hubo que deslizar el clic **1 o el 2** (`un
    cuarto` y `tres cuartos`, numerados desde 0 como en todo el trabajo) para
    buscar senal. Es campo distinto de `partida_en_start`, que sale medido del
    GT en F2: una unidad puede arrancar partida y que el hueco no caiga donde
    van los clics.

    `consulta_temporal` dice si para localizar la lesion hubo que mirar otros
    frames de la secuencia antes de clicar. No es contaminacion: la imagen es
    dato y el GT es respuesta, y es lo que haria un usuario del plugin con la
    secuencia delante. Pero es una desviacion del procedimiento del resto de
    unidades, y se registra para poder declararla.
    """
    validar_clics(clics)
    entrada = {
        "serie": int(serie),
        "etiqueta": int(etiqueta),
        "start": int(start),
        "clics": [[int(fila), int(columna)] for fila, columna in clics],
        "excepcion": bool(excepcion),
        "consulta_temporal": bool(consulta_temporal),
        "anotado_en": datetime.now().isoformat(timespec="seconds"),
    }
    entrada.update(diagnosticos(clics, mascara_gt))

    otras = [
        u
        for u in datos["unidades"]
        if (u["serie"], u["etiqueta"]) != (int(serie), int(etiqueta))
    ]
    nuevo = dict(datos)
    nuevo["unidades"] = sorted(
        otras + [entrada], key=lambda u: (u["serie"], u["etiqueta"])
    )
    return nuevo


def subconjunto(clics: list[Clic], protocolo: str) -> list[Clic]:
    """Los clics de un protocolo P1..P4, que son subconjuntos de indices."""
    return [clics[i] for i in config.PROTOCOLOS[protocolo]]


def puntos_y_etiquetas(unidad: dict, protocolo: str) -> tuple[list, list]:
    """Lo que se le pasa a un modelo para un protocolo: puntos y sus etiquetas.

    Vive aqui, y no en cada script, porque la traduccion de un protocolo a
    puntos es lo que define el factor prompt: si un script la escribiera
    distinto, estaria midiendo otro nivel del factor con el mismo nombre.

    Y aqui en vez de en un modulo mas nuevo porque **este no arrastra nada**:
    `profundidad.py` y `m4_encadenado.py` corren dentro de los entornos de
    modelo, y cualquier cosa que meta skimage en el mismo proceso que torch
    reproduce el conflicto de OpenMP que ya tumbo tres entornos en M4.
    """
    indices = config.PROTOCOLOS[protocolo]
    puntos = [tuple(unidad["clics"][i]) for i in indices]
    etiquetas = [0 if i in config.CLICS_NEGATIVOS else 1 for i in indices]
    return puntos, etiquetas
