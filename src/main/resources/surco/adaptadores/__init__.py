"""Adaptadores de modelos, uno por fichero.

**No se importan desde `surco`.** Cada uno vive en el entorno de su modelo, y el
entorno del banco no tiene torch ni los repositorios de nadie. Importar un
adaptador desde el banco tiene que ser inofensivo, y por eso los adaptadores no
importan sus dependencias pesadas en el cuerpo del modulo sino dentro de
`cargar()`.

`cargar_disponibles()` recorre los adaptadores conocidos y devuelve los que se
pueden importar, con el motivo de los que no. Un modelo que se resiste tiene que
salir como una fila con su motivo, no como un error que corta la comprobacion.
"""

import importlib
import pkgutil


def cargar_disponibles() -> tuple[list[str], dict[str, str]]:
    """Importa todos los adaptadores. Devuelve los importados y los fallidos."""
    importados: list[str] = []
    fallidos: dict[str, str] = {}
    for modulo in pkgutil.iter_modules(__path__):
        if modulo.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{__name__}.{modulo.name}")
            importados.append(modulo.name)
        except Exception as error:  # noqa: BLE001 - interesa el motivo
            fallidos[modulo.name] = f"{type(error).__name__}: {error}"
    return importados, fallidos
