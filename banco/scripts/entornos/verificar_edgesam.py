"""Prueba de que el camino de prompt de punto de EdgeSAM es el publicado.

El argumento es este, y se sostiene solo si las tres comprobaciones pasan:

1. MMDetection **no esta instalado** en este entorno. Se comprueba.
2. Por tanto el parche de `scripts/entornos/parchear_edgesam.py` entra siempre por su
   rama de excepcion, y las cinco referencias que envuelve (`RPNHead`,
   `CenterNetUpdateHead`, `FPN`, `efficientdet` y `ConfigDict`) quedan
   sustituidas por un objeto que **estalla en cuanto se le llama o se le pide
   un atributo**. Se comprueba que efectivamente estalla.
3. Aun asi, cargar el modelo y segmentar con un punto **termina sin error**. Se
   comprueba ejecutandolo.

Si el camino de punto tocara cualquiera de esas cinco cosas, el paso 3 habria
reventado. Como no revienta, no las toca. De ahi que lo que se evalua sea el
modelo publicado: el parche solo afecta a codigo que no se ejecuta.

    conda run -n surco-edgesam python scripts/entornos/verificar_edgesam.py
"""

import importlib.util


from surco import adaptadores, modelos


def comprobar_sin_mmdet() -> None:
    for paquete in ("mmdet", "mmengine", "projects"):
        if importlib.util.find_spec(paquete) is not None:
            raise AssertionError(
                f"{paquete} SI esta instalado: la prueba no vale, porque el "
                "parche no entraria por la rama de excepcion"
            )
    print("1. mmdet, mmengine y projects no estan instalados: correcto")


def comprobar_que_estallan() -> None:
    from edge_sam.modeling import sam as modulo

    nombres = ["RPNHead", "CenterNetUpdateHead", "FPN", "efficientdet", "ConfigDict"]
    for nombre in nombres:
        objeto = getattr(modulo, nombre)
        try:
            objeto()
        except ImportError:
            continue
        raise AssertionError(f"{nombre} no estallo al usarse; el parche no protege")
    print(f"2. las {len(nombres)} referencias parcheadas estallan al usarse: correcto")


def comprobar_que_el_punto_funciona() -> None:
    adaptadores.cargar_disponibles()
    modelo = modelos.crear("edgesam")
    modelo.cargar()
    imagen = modelos.imagen_de_prueba_rgb()
    candidatas = modelo.segmentar(imagen, [modelos.punto_de_prueba()], [1])
    if candidatas.mascaras.shape[1:] != imagen.shape[:2]:
        raise AssertionError("la mascara no tiene la forma de la imagen")
    print(
        f"3. cargar y segmentar con un punto termina sin error: correcto "
        f"({len(candidatas)} candidatas de {candidatas.mascaras.shape[1:]})"
    )


def main() -> None:
    comprobar_sin_mmdet()
    comprobar_que_estallan()
    comprobar_que_el_punto_funciona()
    print(
        "\nEl camino de prompt de punto no pasa por nada de lo parcheado.\n"
        "El modelo evaluado es el publicado."
    )


if __name__ == "__main__":
    main()
