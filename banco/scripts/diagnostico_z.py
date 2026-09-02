"""Por que S1 pierde en la serie 7: la condicion de fallo del z.

La hipotesis a comprobar es que en las unidades mas brillantes la ventana del
clic **satura**, el fondo local se queda sin dispersion y el MAD se acerca a
cero. Si eso pasa, el z deja de ordenar: o se va a infinito, o empata entre
candidatas, y en los dos casos manda el desempate por elongacion en vez del
contraste.

Se mide en dos planos, porque son dos cosas distintas:

- **la ventana**, que no depende de ningun modelo: fraccion de pixeles saturados
  a 255 despues de la preparacion, y MAD de la ventana entera;
- **las candidatas**, donde se cuenta cuantas veces el z sale infinito y cuantas
  veces las tres candidatas empatan, que es cuando el ranking no puede decidir.

    conda run -n surco python scripts/diagnostico_z.py
"""

import numpy as np
import pandas as pd

from surco import clasicos, config, datos, inventario, metricas, modelos, prompts, selector

import profundidad
from selector_s0_s1 import cargar_candidatas, desempaquetar

SATURADO = 255


def de_la_ventana(gris: np.ndarray, punto) -> dict:
    ventana, _ = clasicos.recortar(gris, punto)
    return {
        "saturacion": float((ventana == SATURADO).mean()),
        "mad_ventana": float(np.median(np.abs(ventana - np.median(ventana)))),
        "mediana_ventana": float(np.median(ventana)),
    }


def de_las_candidatas(guardados, unidad, frame, gris) -> dict:
    """Cuantas veces el z no puede ordenar, sobre los cuatro modelos y P1-P4."""
    infinitos = empates = total = 0
    for nombre, guardado in guardados.items():
        for protocolo in config.PROTOCOLOS:
            clave = f"M1_{protocolo}_t{frame:02d}"
            if clave not in guardado:
                continue
            candidatas = desempaquetar(guardado, clave)
            if len(candidatas) < 2:
                continue
            puntos, etiquetas = profundidad.prompt_de(unidad, protocolo)
            zetas = [
                selector.z_robusto(gris, mascara, puntos[0])
                for mascara in candidatas.mascaras
            ]
            total += 1
            infinitos += any(np.isinf(z) for z in zetas)
            empates += len(set(zetas)) == 1
    return {"prompts": total, "z_infinito": infinitos, "z_empatado": empates}


def main() -> None:
    pd.set_option("display.width", 200)
    filas = []
    for unidad in prompts.cargar()["unidades"]:
        gt_serie = datos.cargar_gt_serie(unidad["serie"])
        frames = inventario.frames_de_unidad(gt_serie, unidad["etiqueta"])
        guardados = cargar_candidatas(unidad)
        for frame in metricas.ventana_de_evaluacion(frames, unidad["start"]):
            gris = modelos.preparar_frame(unidad["serie"], frame)[:, :, 0]
            fila = {
                "serie": unidad["serie"],
                "etiqueta": unidad["etiqueta"],
                "frame": frame,
            }
            fila.update(de_la_ventana(gris, tuple(unidad["clics"][0])))
            fila.update(de_las_candidatas(guardados, unidad, frame, gris))
            filas.append(fila)
        print(f"  {unidad['serie']}-L{unidad['etiqueta']}", flush=True)

    tabla = pd.DataFrame(filas)
    tabla.to_csv(config.DIR_PROCESO / "diagnostico_z.csv", index=False)

    por_serie = tabla.groupby("serie").agg(
        saturacion=("saturacion", "mean"),
        mad_ventana=("mad_ventana", "mean"),
        mediana_ventana=("mediana_ventana", "mean"),
        frac_z_infinito=("z_infinito", "sum"),
        frac_z_empatado=("z_empatado", "sum"),
        prompts=("prompts", "sum"),
    )
    por_serie["frac_z_infinito"] /= por_serie["prompts"]
    por_serie["frac_z_empatado"] /= por_serie["prompts"]
    print("\n=== LA VENTANA DEL CLIC, POR SERIE ===")
    print(por_serie.to_string(float_format=lambda v: f"{v:8.4f}"))

    print("\n=== LAS DOS UNIDADES DE LA SERIE 7 ===")
    print(
        tabla[tabla["serie"] == 7]
        .groupby("etiqueta")
        .agg(
            saturacion=("saturacion", "mean"),
            mad_ventana=("mad_ventana", "mean"),
            mad_minimo=("mad_ventana", "min"),
            frames=("frame", "count"),
        )
        .to_string(float_format=lambda v: f"{v:8.4f}")
    )


if __name__ == "__main__":
    main()
