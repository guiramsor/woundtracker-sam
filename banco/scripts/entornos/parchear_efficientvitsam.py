"""Parche reproducible del paquete EfficientViT instalado.

**No modifica el modelo.** Modifica dos importaciones de cabecera que arrastran
Triton, que en Windows no tiene distribucion oficial:

    efficientvit/models/nn/__init__.py:  from .triton_rms_norm import *
    efficientvit/models/nn/norm.py:      from ...triton_rms_norm import TritonRMSNorm2dFunc

Triton solo hace falta para la normalizacion `trms2d`, que en todo el paquete
usan unicamente los modelos de difusion de `dc_ae.py`. Los cinco EfficientViT-SAM
usan `bn2d`, comprobado en `models/efficientvit/sam.py`. Es decir: sin este
parche el paquete no se importa en Windows, y con el, el camino de SAM queda
igual porque nunca pasa por ahi.

Mismo procedimiento que con EdgeSAM: script versionado e idempotente, y
`scripts/entornos/verificar_efficientvitsam.py` demuestra despues que el camino de prompt
de punto no toca nada de lo parcheado.

    conda run -n surco-efficientvitsam python scripts/entornos/parchear_efficientvitsam.py
"""

import importlib.util
from pathlib import Path

MARCA = "# --- parche surco: Triton solo si esta"

ORIGINAL_INIT = "from .triton_rms_norm import *"
PARCHE_INIT = f"""{MARCA} disponible ---
# Triton no tiene distribucion oficial en Windows y aqui solo sirve para la
# normalizacion trms2d, que ningun EfficientViT-SAM usa.
try:
    from .triton_rms_norm import *
except ImportError:
    pass
# --- fin del parche surco ---"""

ORIGINAL_NORM = (
    "from efficientvit.models.nn.triton_rms_norm import TritonRMSNorm2dFunc"
)
PARCHE_NORM = f'''{MARCA} disponible ---
try:
    from efficientvit.models.nn.triton_rms_norm import TritonRMSNorm2dFunc
except ImportError as _error_surco:
    # Python borra el nombre del `as` al salir del bloque, asi que la causa se
    # rebota a otro nombre para que siga viva cuando estalle el sustituto.
    _causa_surco = _error_surco

    class _SinTriton:
        """Estalla al usarse, para que nada pase de largo en silencio."""

        def _fallar(self, *_args, **_kwargs):
            raise ImportError(
                "la normalizacion trms2d de EfficientViT necesita Triton, que "
                "no esta instalado. Los modelos EfficientViT-SAM usan bn2d y no "
                "pasan por aqui."
            ) from _causa_surco

        apply = _fallar
        __call__ = _fallar

        def __getattr__(self, _nombre):
            self._fallar()

    TritonRMSNorm2dFunc = _SinTriton()
# --- fin del parche surco ---'''


def raiz_del_paquete() -> Path:
    especificacion = importlib.util.find_spec("efficientvit")
    if especificacion is None or not especificacion.submodule_search_locations:
        raise ModuleNotFoundError("efficientvit no esta instalado en este entorno")
    return Path(list(especificacion.submodule_search_locations)[0])


def parchear(fichero: Path, original: str, parche: str) -> bool:
    texto = fichero.read_text(encoding="utf-8")
    if MARCA in texto:
        return False
    if original not in texto:
        raise RuntimeError(
            f"{fichero} no contiene la importacion esperada. El repositorio ha "
            "cambiado y el parche hay que revisarlo."
        )
    fichero.write_text(texto.replace(original, parche), encoding="utf-8")
    return True


def main() -> None:
    raiz = raiz_del_paquete() / "models" / "nn"
    hechos = [
        parchear(raiz / "__init__.py", ORIGINAL_INIT, PARCHE_INIT),
        parchear(raiz / "norm.py", ORIGINAL_NORM, PARCHE_NORM),
    ]
    if not any(hechos):
        print(f"ya estaba parcheado: {raiz}")
        return
    print(f"parcheados {sum(hechos)} ficheros en {raiz}")
    print("dos importaciones de Triton envueltas; el modelo no se toca")


if __name__ == "__main__":
    main()
