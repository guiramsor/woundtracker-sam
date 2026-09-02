"""Prueba de que el camino de prompt de punto de EfficientViT-SAM es el publicado.

Mismo argumento que en `verificar_edgesam.py`, y se sostiene solo si las tres
comprobaciones pasan:

1. Triton **no esta instalado** en este entorno. Se comprueba.
2. Por tanto el parche entra siempre por su rama de excepcion y
   `TritonRMSNorm2dFunc` queda sustituido por un objeto que **estalla en cuanto
   se le llama o se le pide un atributo**. Se comprueba que estalla.
3. Aun asi, cargar el modelo y segmentar con un punto **termina sin error**.

Si el camino de punto tocara la normalizacion `trms2d`, el paso 3 habria
reventado. Como no revienta, no la toca: los EfficientViT-SAM usan `bn2d`.

    conda run -n surco-efficientvitsam python scripts/entornos/verificar_efficientvitsam.py
"""

import importlib.util

from surco import adaptadores, modelos


def comprobar_sin_triton() -> None:
    if importlib.util.find_spec("triton") is not None:
        raise AssertionError(
            "triton SI esta instalado: la prueba no vale, porque el parche no "
            "entraria por la rama de excepcion"
        )
    print("1. triton no esta instalado: correcto")


def comprobar_que_estalla() -> None:
    from efficientvit.models.nn import norm

    try:
        norm.TritonRMSNorm2dFunc.apply(None, None, None, None)
    except ImportError:
        print("2. TritonRMSNorm2dFunc estalla al usarse: correcto")
        return
    raise AssertionError("TritonRMSNorm2dFunc no estallo; el parche no protege")


def comprobar_que_el_punto_funciona() -> None:
    adaptadores.cargar_disponibles()
    modelo = modelos.crear("efficientvitsam")
    modelo.cargar()
    imagen = modelos.imagen_de_prueba_rgb()
    candidatas = modelo.segmentar(imagen, [modelos.punto_de_prueba()], [1])
    if candidatas.mascaras.shape[1:] != imagen.shape[:2]:
        raise AssertionError(
            f"la mascara sale {candidatas.mascaras.shape[1:]} y la imagen es "
            f"{imagen.shape[:2]}"
        )
    print(
        f"3. cargar y segmentar con un punto termina sin error: correcto "
        f"({len(candidatas)} candidatas de {candidatas.mascaras.shape[1:]})"
    )


def main() -> None:
    comprobar_sin_triton()
    comprobar_que_estalla()
    comprobar_que_el_punto_funciona()
    print(
        "\nEl camino de prompt de punto no pasa por la normalizacion trms2d.\n"
        "El modelo evaluado es el publicado."
    )


if __name__ == "__main__":
    main()
