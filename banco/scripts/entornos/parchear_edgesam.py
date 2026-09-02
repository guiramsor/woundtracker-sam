"""Parche reproducible del paquete EdgeSAM instalado.

**No modifica el modelo.** Modifica cuatro lineas de importacion que el
repositorio pone en la cabecera de `edge_sam/modeling/sam.py` para una cabeza de
propuestas opcional (`rpn_head`) que este trabajo no activa:

    from mmdet.models.dense_heads import RPNHead, CenterNetUpdateHead
    from mmdet.models.necks import FPN
    from projects.EfficientDet import efficientdet
    from mmengine import ConfigDict

Las cuatro se usan **solo** dentro de las ramas `if rpn_head == 'centernet' |
'rpn' | 'efficient_det'` del constructor y en `forward_rpn`. El valor por
defecto es `rpn_head=None`, que es el del prompt de punto, y con el ninguna se
ejecuta. Ademas `projects.EfficientDet` **no se instala con pip**: es una
carpeta del repositorio de MMDetection, asi que el paquete publicado no es
importable ni instalando mmdet.

Es un defecto de empaquetado ajeno, no una modificacion del modelo. El parche
envuelve esas importaciones para que fallen solo si alguien pide de verdad esa
cabeza, y `scripts/entornos/verificar_edgesam.py` demuestra despues que el camino de
punto no pasa por ninguna de ellas.

Es idempotente: si ya esta parcheado, no toca nada.

    conda run -n surco-edgesam python scripts/entornos/parchear_edgesam.py
"""

import importlib.util
from pathlib import Path

MARCA = "# --- parche surco: importaciones perezosas de MMDetection"

ORIGINAL = """from mmdet.models.dense_heads import RPNHead, CenterNetUpdateHead
from mmdet.models.necks import FPN
from projects.EfficientDet import efficientdet
from mmengine import ConfigDict"""

PARCHE = f'''{MARCA} ---------------
# Las cuatro importaciones de abajo solo sirven para la cabeza de propuestas
# opcional (`rpn_head`). Con `rpn_head=None`, que es el valor por defecto y el
# del prompt de punto, ninguna se usa. `projects.EfficientDet` ni siquiera se
# instala con pip. Se envuelven para que fallen solo si alguien las usa.
try:
    from mmdet.models.dense_heads import RPNHead, CenterNetUpdateHead
    from mmdet.models.necks import FPN
    from projects.EfficientDet import efficientdet
    from mmengine import ConfigDict
except ImportError as _error_surco:
    # Python borra el nombre del `as` al salir del bloque, asi que la causa se
    # rebota a otro nombre para que siga viva cuando estalle el sustituto.
    _causa_surco = _error_surco

    class _SinMMDet:
        """Estalla al usarse, para que nada pase de largo en silencio."""

        def _fallar(self, *_args, **_kwargs):
            raise ImportError(
                "esta parte de EdgeSAM necesita MMDetection, que no esta "
                "instalado. El camino de prompt de punto no pasa por aqui."
            ) from _causa_surco

        __call__ = _fallar

        def __getattr__(self, _nombre):
            self._fallar()

    RPNHead = CenterNetUpdateHead = FPN = efficientdet = ConfigDict = _SinMMDet()
# --- fin del parche surco ------------------------------------------------'''


def ruta_de_sam() -> Path:
    """Localiza el fichero sin importar el paquete, que hoy no importa."""
    especificacion = importlib.util.find_spec("edge_sam")
    if especificacion is None or not especificacion.submodule_search_locations:
        raise ModuleNotFoundError("edge_sam no esta instalado en este entorno")
    raiz = Path(list(especificacion.submodule_search_locations)[0])
    return raiz / "modeling" / "sam.py"


def main() -> None:
    fichero = ruta_de_sam()
    texto = fichero.read_text(encoding="utf-8")

    if MARCA in texto:
        print(f"ya estaba parcheado: {fichero}")
        return
    if ORIGINAL not in texto:
        raise RuntimeError(
            f"{fichero} no contiene el bloque de importaciones esperado. "
            "El repositorio ha cambiado y el parche hay que revisarlo."
        )

    fichero.write_text(texto.replace(ORIGINAL, PARCHE), encoding="utf-8")
    print(f"parcheado: {fichero}")
    print("cuatro importaciones de cabecera envueltas; el modelo no se toca")


if __name__ == "__main__":
    main()
