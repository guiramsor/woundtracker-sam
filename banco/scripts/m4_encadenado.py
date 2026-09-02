"""M4: encadenado de mascara, sobre las cuatro filas de F8.

**Que es M4.** En el frame semilla se resuelve **como M1**, con los clics, porque
todavia no hay mascara previa. En cada frame posterior entra la mascara ganadora
del frame anterior como prompt de mascara, **junto con los clics del protocolo,
que se mantienen**. Los clics no se sustituyen: si desaparecieran, M4 dejaria de
ser comparable con M0 y M1 en el eje de prompt y la rejilla dejaria de cruzarse.

**Se encadenan los logits, no la mascara binaria.** El `mask_input` de la familia
de SAM espera lo mismo que devuelve su `predict` como tercer valor; convertir una
mascara booleana en logits obligaria a inventar una escala.

**M4 x S0 y M4 x S1 son dos cadenas distintas y hay que correr las dos.** Lo que
se reencadena es lo que elige el selector activo, asi que las dos cadenas se
separan en el primer frame y ya no vuelven a coincidir. No es como M0 o M1, donde
S1 se puede aplicar despues sobre las candidatas guardadas.

El bucle va unidad, frame, y despues las ocho cadenas (cuatro protocolos por dos
selectores), para que las ocho compartan el embedding de ese frame.

    conda run -n <entorno> python scripts/m4_encadenado.py
"""

import os

# M4 es el primer modo que necesita **el selector dentro del proceso del
# modelo**: la mascara que se reencadena es la que elige el selector activo, asi
# que S1 tiene que evaluarse en cada frame y no despues. Eso mete skimage, que
# trae su propio runtime de OpenMP, en el mismo proceso que torch, que trae otro,
# y tres de los cuatro entornos abortan con `OMP: Error #15`.
#
# Se declara el conflicto en vez de esconderlo. La variable no cambia el
# calculo, solo permite que convivan las dos copias del runtime, y **eso esta
# comprobado**: micro-sam corrio entero antes de ponerla, y con ella puesta su
# CSV sale identico fila a fila.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from surco import (  # noqa: E402
    adaptadores, config, datos, inventario, metricas, modelos, prompts, selector,
)

from profundidad import REJILLA, prompt_de  # noqa: E402

SELECTORES = ("S0", "S1")


def correr_unidad(modelo, unidad: dict, gt_serie: np.ndarray) -> tuple[list[dict], dict]:
    serie, etiqueta = unidad["serie"], unidad["etiqueta"]
    frames = inventario.frames_de_unidad(gt_serie, etiqueta)
    ventana = metricas.ventana_de_evaluacion(frames, unidad["start"])
    protocolos = modelo.protocolos_admitidos()

    # Estado de cada cadena: los logits de la candidata que gano el frame
    # anterior. En el semilla todavia no hay ninguna, y por eso M4 arranca
    # exactamente como M1.
    cadenas = {(p, s): None for p in protocolos for s in SELECTORES}
    filas, almacen = [], {}

    for frame in [unidad["start"], *ventana]:
        imagen = modelos.preparar_frame(serie, frame)
        gris = imagen[:, :, 0]
        referencia = gt_serie[frame - 1] == etiqueta
        for (protocolo, nombre_selector), previa in list(cadenas.items()):
            puntos, etiquetas = prompt_de(unidad, protocolo)
            candidatas, logits = modelo.segmentar_con_mascara(
                imagen, puntos, etiquetas, previa
            )
            indice = selector.elegir_con(
                nombre_selector, candidatas, gris, puntos, etiquetas
            )
            cadenas[(protocolo, nombre_selector)] = logits[indice]

            modo = f"M4{nombre_selector}"
            clave = f"{modo}_{protocolo}_t{frame:02d}"
            almacen[clave] = np.packbits(candidatas.mascaras, axis=-1)
            almacen[f"{clave}_score"] = candidatas.puntuaciones
            if frame == unidad["start"]:
                continue  # el semilla se calcula pero no se evalua (apartado 6.2)
            filas.append(
                {
                    "modelo": modelo.nombre,
                    "checkpoint": modelo.checkpoint,
                    "familia": modelo.familia,
                    "modo": "M4",
                    "protocolo": protocolo,
                    "selector": nombre_selector,
                    "serie": serie,
                    "etiqueta": etiqueta,
                    "frame": frame,
                    "n_candidatas": len(candidatas),
                    "dice": metricas.dice(candidatas.mascaras[indice], referencia),
                }
            )
    return filas, almacen


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

    unidades = prompts.cargar()["unidades"]
    gt = {u["serie"]: datos.cargar_gt_serie(u["serie"]) for u in unidades}
    config.DIR_CANDIDATAS.mkdir(parents=True, exist_ok=True)

    filas = []
    for nombre, checkpoint in mios:
        modelo = modelos.crear(nombre, checkpoint)
        if not modelo.soporta_mascara:
            print(f"{nombre}: no acepta prompt de mascara, fuera de M4")
            continue
        modelo.cargar()
        print(f"{modelo.identificador}: M4 x {' '.join(modelo.protocolos_admitidos())}"
              f" x {' '.join(SELECTORES)}", flush=True)
        for unidad in unidades:
            propias, almacen = correr_unidad(modelo, unidad, gt[unidad["serie"]])
            filas.extend(propias)
            np.savez_compressed(
                config.DIR_CANDIDATAS / config.PATRON_CANDIDATAS.format(
                    prefijo="m4_", nombre=nombre, checkpoint=checkpoint,
                    serie=unidad["serie"], etiqueta=unidad["etiqueta"],
                ),
                **almacen,
            )
            print(f"  {unidad['serie']}-L{unidad['etiqueta']}", flush=True)

    tabla = pd.DataFrame(filas)
    tabla.to_csv(config.DIR_PROCESO / f"m4_{entorno}.csv", index=False)
    print(f"\n=== {entorno}: {len(tabla)} filas ===")
    for clave, grupo in tabla.groupby(["modelo", "selector", "protocolo"]):
        media = metricas.media_entre_unidades(metricas.por_unidad(grupo))
        print(f"  {clave[0]:14s} {clave[1]}  {clave[2]}  {media:7.4f}")


if __name__ == "__main__":
    main()
