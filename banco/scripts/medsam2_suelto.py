"""MedSAM2 suelto: M3 x P1-P4 y M1 x P2-P4, sobre las 17 unidades.

**No entra en la rejilla de profundidad ni en la agregacion por familias.**
MedSAM2 quedo fuera de los k = 4 del apartado 5.3 porque su familia ya la
representa SAM-Med2D, y esa decision no se toca. Esto es otra cosa: entra solo
en la comparacion pareada con SAM 2.1 sobre el eje temporal.

El motivo es que **son el mismo modelo y su fork**: misma mecanica de memoria,
distinto dominio de ajuste. Medir M1 contra M3 en los dos separa si lo que
aporta la memoria temporal depende del dominio o de la memoria en si, y es la
unica comparacion del conjunto que puede contestar eso.

**M1 solo en P2, P3 y P4.** El de P1 ya esta medido en el cribado, que fijo
M1 x P1 para todos, y no se rehace: es el mismo camino de codigo y daria el
mismo resultado. Sin estos tres, la comparacion de memoria existiria en un
protocolo de los cuatro.

Escribe sus propios CSV, `m3_suelto_*` y `m1_suelto_*`, y no `profundidad_*`,
para que la tabla de la rejilla no los recoja por accidente al unir los entornos.

    conda run -n surco-medsam2 python scripts/medsam2_suelto.py
"""

import numpy as np
import pandas as pd

from surco import adaptadores, config, datos, metricas, modelos, prompts

import profundidad

# El representante de MedSAM2 en el cribado, el mismo con el que se le comparo
# contra los elegidos: no se cambia de checkpoint a mitad de comparacion.
CHECKPOINT = "2411"

# Los que le faltan al cribado. P1 sale de alli y no se vuelve a medir.
PROTOCOLOS_M1 = ("P2", "P3", "P4")


def resumir(tabla: pd.DataFrame, titulo: str) -> None:
    print(f"\n=== {titulo}: {len(tabla)} filas ===")
    for (modo, protocolo), grupo in tabla.groupby(["modo", "protocolo"]):
        media = metricas.media_entre_unidades(metricas.por_unidad(grupo))
        print(f"  {modo} {protocolo}  {media:7.4f}")


def main() -> None:
    adaptadores.cargar_disponibles()
    modelo = modelos.crear("medsam2", CHECKPOINT)
    if not modelo.soporta_memoria:
        raise RuntimeError("MedSAM2 tendria que soportar banco de memoria")
    modelo.cargar()
    admitidos = modelo.protocolos_admitidos()
    protocolos_m1 = tuple(p for p in PROTOCOLOS_M1 if p in admitidos)
    print(
        f"{modelo.identificador}: M3 x {' '.join(admitidos)} + "
        f"M1 x {' '.join(protocolos_m1)}"
    )

    de_m3, de_m1 = [], []
    for unidad in prompts.cargar()["unidades"]:
        gt_serie = datos.cargar_gt_serie(unidad["serie"])
        frames = profundidad.secuencia_de_unidad(gt_serie, unidad)
        ventana = metricas.ventana_de_evaluacion(frames, unidad["start"])

        almacen: dict[str, np.ndarray] = {}
        de_m3.extend(
            profundidad.correr_m3(
                modelo, unidad, gt_serie, frames, ventana, admitidos, almacen
            )
        )
        de_m1.extend(
            profundidad.correr_m1(
                modelo, unidad, gt_serie, ventana, protocolos_m1, almacen
            )
        )
        config.DIR_CANDIDATAS.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            config.DIR_CANDIDATAS / config.PATRON_CANDIDATAS.format(
                prefijo="suelto_", nombre="medsam2", checkpoint=CHECKPOINT,
                serie=unidad["serie"], etiqueta=unidad["etiqueta"],
            ),
            **almacen,
        )
        print(f"  {unidad['serie']}-L{unidad['etiqueta']}", flush=True)

    for filas, nombre, titulo in (
        (de_m3, "m3_suelto_medsam2.csv", "M3"),
        (de_m1, "m1_suelto_medsam2.csv", "M1"),
    ):
        tabla = pd.DataFrame(filas)
        tabla.to_csv(config.DIR_PROCESO / nombre, index=False)
        resumir(tabla, titulo)


if __name__ == "__main__":
    main()
