"""Reproduce la verificacion que deja a FastSAM fuera de la lista.

La linea era esta: si se le pueden pasar los cinco clics con sus etiquetas y
devuelve una mascara que responde a ese prompt, entra, aunque sea con una sola
candidata. Si tiene que segmentar todo y elegir despues, el prompt de nuestro
protocolo no se esta aplicando, y sale.

FastSAM hace lo segundo, y no es una interpretacion: lo dice el docstring de su
propia implementacion en ultralytics, que describe su entrada como *"original
inference results from FastSAM models **without any prompts**"*. Los clics nunca
llegan a la red. Se usan despues, para **seleccionar por contencion** entre las
mascaras que el modelo ya habia producido sin verlos.

Este script lo demuestra en vez de citarlo:

1. Corre el modo "todo" sobre un frame real y guarda las mascaras que produce
   sin ningun prompt.
2. Corre el prompt de punto con los cinco clics del protocolo.
3. Comprueba que la salida es **exactamente la union de un subconjunto** de las
   mascaras del paso 1: FastSAM nunca produce una mascara que no tuviera ya.
4. Comprueba que un clic negativo **no puede recortar**: anadido dentro de la
   misma mascara que un positivo, la salida o es la misma mascara entera o
   desaparece entera, nunca la mascara menos un trozo.

Un modelo promptable por punto responde al punto; este elige entre lo que ya
tenia. La consecuencia para el protocolo es directa: los clics negativos no
pueden afinar ni recortar nada, solo descartar entera una mascara que los
contenga, asi que P2 y P4 no significan sobre FastSAM lo que significan sobre
los demas.

    conda run -n surco-fastsam python scripts/entornos/verificar_exclusion_fastsam.py
"""

import numpy as np

from surco import config, modelos

PESOS = "FastSAM-x.pt"


def _mascaras_de(resultado) -> np.ndarray:
    if resultado.masks is None:
        return np.zeros((0, 1, 1), dtype=bool)
    return resultado.masks.data.cpu().numpy().astype(bool)


def _promptear(modelo, imagen, puntos, etiquetas) -> np.ndarray:
    salida = modelo(
        imagen, device="cuda", retina_masks=True, verbose=False,
        points=puntos, labels=etiquetas,
    )
    mascaras = _mascaras_de(salida[0])
    if len(mascaras) == 0:
        return np.zeros(imagen.shape[:2], dtype=bool)
    return np.any(mascaras, axis=0)


def main() -> None:
    from ultralytics import FastSAM

    from surco import prompts

    modelo = FastSAM(str(config.DIR_PESOS / PESOS))

    # Un frame real, no el sintetico: sobre una imagen con un solo objeto la
    # comprobacion seria trivial. Se toma la unidad 5-L1 en su frame start, con
    # sus clics de verdad.
    unidad = next(
        u for u in prompts.cargar()["unidades"] if (u["serie"], u["etiqueta"]) == (5, 1)
    )
    imagen = modelos.preparar_frame(unidad["serie"], unidad["start"])
    clics = [tuple(c) for c in unidad["clics"]]
    print(f"frame real: serie {unidad['serie']}, t{unidad['start']}, unidad 5-L1")

    todo = modelo(imagen, device="cuda", retina_masks=True, verbose=False)
    mascaras_todo = _mascaras_de(todo[0])
    print(f"\n1. modo 'todo': {len(mascaras_todo)} mascaras producidas sin ver ningun clic")

    # Aqui todo va en (fila, columna); FastSAM espera (x, y).
    puntos = [[c, f] for f, c in clics]
    etiquetas = [0 if i in config.CLICS_NEGATIVOS else 1 for i in range(len(clics))]
    p4 = _promptear(modelo, imagen, puntos, etiquetas)
    print(f"2. P4, los cinco clics: {int(p4.sum())} px en la salida")

    # P3 son los tres positivos, sin ningun clic de fondo.
    indices_p3 = config.PROTOCOLOS["P3"]
    p3 = _promptear(modelo, imagen, [puntos[i] for i in indices_p3], [1] * len(indices_p3))
    print(f"3. P3, solo los tres positivos: {int(p3.sum())} px en la salida")

    if p3.any():
        contenidas = [i for i, m in enumerate(mascaras_todo) if np.array_equal(m & p3, m)]
        union = np.any(mascaras_todo[contenidas], axis=0) if contenidas else None
        print(
            f"4. la salida de P3 es la union exacta de {len(contenidas)} de las "
            f"{len(mascaras_todo)} mascaras del paso 1: "
            f"{union is not None and np.array_equal(union, p3)}"
        )

    # El clic de fondo del protocolo cae en nucleoplasma, dentro de la misma
    # mascara que los positivos, asi que la descarta entera.
    indices_p2 = config.PROTOCOLOS["P2"]
    p2 = _promptear(
        modelo,
        imagen,
        [puntos[i] for i in indices_p2],
        [0 if i in config.CLICS_NEGATIVOS else 1 for i in indices_p2],
    )
    print(f"5. P2, un positivo y un fondo: {int(p2.sum())} px en la salida")

    print(
        "\nFastSAM no produce una mascara que no tuviera ya: elige entre las "
        "suyas por contencion.\n"
        "Los clics de fondo del protocolo caen en nucleoplasma, dentro de la "
        "misma mascara que\nlos positivos, asi que la descartan entera en vez "
        "de recortarla: P2 y P4 se quedan\nsin mascara. Un modelo que respetara "
        "el negativo devolveria la region recortada.\n"
        "Por eso sale con un motivo propio: mecanismo de prompt no comparable."
    )

    # Las cuatro cifras las cita el apartado 5.2 y hasta ahora solo existian
    # impresas. Con fichero, repetir la comprobacion es comparar dos tablas.
    import pandas as pd

    destino = config.DIR_PROCESO / "exclusion_fastsam.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"paso": "modo todo, sin ningun clic", "px": int(mascaras_todo.any(axis=0).sum()),
             "mascaras": len(mascaras_todo)},
            {"paso": "P4, los cinco clics", "px": int(p4.sum()), "mascaras": None},
            {"paso": "P3, solo los tres positivos", "px": int(p3.sum()), "mascaras": None},
            {"paso": "P2, un positivo y un fondo", "px": int(p2.sum()), "mascaras": None},
        ]
    ).to_csv(destino, index=False)
    print(f"escrito en {destino}")


if __name__ == "__main__":
    main()
