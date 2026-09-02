"""Parche reproducible del paquete SAM 3 instalado.

**No modifica el modelo.** Envuelve una importacion de cabecera que arrastra
Triton, que no tiene distribucion oficial en Windows:

    sam3/model/sam3_tracker_utils.py:  from sam3.model.edt import edt_triton

Se parchea **el sitio que importa y no el modulo importado**: `edt.py` define
sus nucleos con `@triton.jit` en el cuerpo, asi que no hay forma de importarlo
sin Triton, pero si de no importarlo.

`edt_triton` es una transformada de distancia euclidea y solo la usa
`sample_one_point_from_error_center`, alcanzable por `get_next_point`, que a su
vez solo se llama desde `video_tracking_multiplex.py` en dos sitios, y **los dos
reciben `gt_masks`**: son el camino de entrenamiento y de simulacion de
correcciones interactivas, donde el ground truth esta disponible. Ni el prompt
de punto sobre imagen ni la propagacion en video a partir de clics del usuario
pasan por ahi, porque en inferencia no hay ground truth que muestrear.

    conda run -n surco-sam3 python scripts/entornos/parchear_sam3.py
"""

import importlib.util
from pathlib import Path

MARCA = "# --- parche surco: Triton solo si esta disponible"

ORIGINAL = "from sam3.model.edt import edt_triton"

PARCHE = f'''{MARCA} ---
# edt_triton solo se usa al muestrear puntos de correccion contra el ground
# truth, que es entrenamiento y simulacion, no inferencia. Triton no tiene
# distribucion oficial en Windows.
try:
    from sam3.model.edt import edt_triton
except ImportError as _error_surco:
    # Python borra el nombre del `as` al salir del bloque, asi que la causa se
    # rebota a otro nombre para que siga viva cuando estalle el sustituto.
    _causa_surco = _error_surco

    def edt_triton(*_args, **_kwargs):
        """Estalla al usarse, para que nada pase de largo en silencio."""
        raise ImportError(
            "la transformada de distancia de SAM 3 necesita Triton, que no "
            "esta instalado. Solo se usa al muestrear puntos de correccion "
            "contra el ground truth; la inferencia no pasa por aqui."
        ) from _causa_surco
# --- fin del parche surco ---'''


def ruta_del_fichero() -> Path:
    especificacion = importlib.util.find_spec("sam3")
    if especificacion is None or not especificacion.submodule_search_locations:
        raise ModuleNotFoundError("sam3 no esta instalado en este entorno")
    raiz = Path(list(especificacion.submodule_search_locations)[0])
    return raiz / "model" / "sam3_tracker_utils.py"


def main() -> None:
    fichero = ruta_del_fichero()
    texto = fichero.read_text(encoding="utf-8")

    if MARCA in texto:
        print(f"ya estaba parcheado: {fichero}")
        return
    if ORIGINAL not in texto:
        raise RuntimeError(
            f"{fichero} no contiene la importacion esperada. El repositorio ha "
            "cambiado y el parche hay que revisarlo."
        )

    fichero.write_text(texto.replace(ORIGINAL, PARCHE), encoding="utf-8")
    print(f"parcheado: {fichero}")
    print("una importacion de Triton envuelta; el modelo no se toca")


if __name__ == "__main__":
    main()
