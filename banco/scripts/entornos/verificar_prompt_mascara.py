"""Que modelos de la rejilla admiten prompt de mascara: la condicion de M4.

M4 encadena la mascara de `t-1` como `maskPrompt` en `t` (apartado 5.1). Antes de
implementarlo hay que saber **quien puede recibirla**, igual que se hizo con M3 y
el banco de memoria: un nivel del factor modo no puede darse por supuesto.

**No basta con mirar la firma.** Con SAM 3 ya se aprendio que un camino puede
existir en el codigo, aceptar la llamada y devolver formas correctas sin que
signifique nada. Asi que aqui se comprueban tres cosas, en este orden:

1. que `predict` declara el parametro `mask_input`;
2. que `predict` devuelve **logits de baja resolucion**, que es lo que ese
   parametro espera recibir de vuelta, y con que forma;
3. que una segunda llamada pasandole esos logits **corre y cambia el resultado**.
   Si devolviera exactamente lo mismo, el prompt de mascara se estaria ignorando
   en silencio y M4 seria M1 con otro nombre.

    conda run -n <entorno> python scripts/entornos/verificar_prompt_mascara.py
"""

import inspect
import os

import numpy as np
import pandas as pd

from surco import adaptadores, config, modelos

# Los cuatro de la rejilla de F8 (apartado 5.3), con su representante.
REJILLA = config.REJILLA_PROFUNDIDAD


def _prueba(modelo):
    """Un punto sobre la barra clara de la imagen sintetica."""
    imagen = modelos.imagen_de_prueba_rgb(modelo.lado_entrada)
    fila, columna = modelos.punto_de_prueba(modelo.lado_entrada)
    coordenadas = np.array([[columna, fila]], dtype=float)
    return imagen, coordenadas, np.array([1], dtype=int)


def comprobar(modelo) -> dict:
    modelo.cargar()
    predictor = modelo._predictor  # noqa: SLF001 - es una comprobacion del paquete
    firma = inspect.signature(predictor.predict)
    fila = {
        "modelo": modelo.identificador,
        "acepta_mask_input": "mask_input" in firma.parameters,
    }
    if not fila["acepta_mask_input"]:
        fila["motivo"] = "su predict no declara mask_input"
        return fila

    # `multimask_output` solo si su firma lo tiene: el `predict` de TinySAM no
    # lo declara, y pasarselo tumbaria la comprobacion por un motivo que no es
    # el que se esta comprobando.
    extra = {"multimask_output": True} if "multimask_output" in firma.parameters else {}

    imagen, coordenadas, etiquetas = _prueba(modelo)
    predictor.set_image(imagen)
    mascaras, puntuaciones, logits = predictor.predict(
        point_coords=coordenadas, point_labels=etiquetas, **extra
    )
    mascaras = np.atleast_3d(np.asarray(mascaras))
    puntuaciones = np.atleast_1d(np.asarray(puntuaciones))
    logits = np.asarray(logits)
    fila["forma_logits"] = str(logits.shape)
    mejor = int(np.argmax(puntuaciones))

    segundas, _, _ = predictor.predict(
        point_coords=coordenadas,
        point_labels=etiquetas,
        mask_input=logits[mejor][None],
        **extra,
    )
    segundas = np.atleast_3d(np.asarray(segundas))
    fila["corre_con_mascara"] = True
    fila["cambia_el_resultado"] = not np.array_equal(
        np.asarray(segundas, dtype=bool), np.asarray(mascaras, dtype=bool)
    )
    fila["px_sin_mascara"] = int(np.asarray(mascaras, dtype=bool)[mejor].sum())
    fila["px_con_mascara"] = int(np.asarray(segundas, dtype=bool)[mejor].sum())
    return fila


def main() -> None:
    adaptadores.cargar_disponibles()
    entorno = os.environ.get("CONDA_DEFAULT_ENV", "desconocido")
    registro = modelos.registrados()
    mios = [
        (n, c) for n, c in REJILLA.items()
        if n in registro and registro[n].entorno == entorno
    ]
    if not mios:
        print(f"ningun modelo de la rejilla en el entorno {entorno!r}")
        return
    filas = []
    for nombre, checkpoint in mios:
        modelo = modelos.crear(nombre, checkpoint)
        try:
            fila = comprobar(modelo)
        except Exception as error:  # noqa: BLE001
            fila = {
                "modelo": modelo.identificador,
                "acepta_mask_input": False,
                "motivo": f"{type(error).__name__}: {error}",
            }
        for clave, valor in fila.items():
            print(f"  {clave}: {valor}")
        filas.append(fila)

    # La forma de los logits y el encogimiento al reencadenar sostienen el
    # apartado 5.1 (SAM-Med2D pasa de 287 a 151 px con logits de 64 x 64 frente
    # a los 256 x 256 de los otros tres) y hasta ahora solo salian por consola.
    destino = config.DIR_PROCESO / f"prompt_mascara_{entorno}.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(destino, index=False)
    print(f"\nescrito en {destino}")


if __name__ == "__main__":
    main()
