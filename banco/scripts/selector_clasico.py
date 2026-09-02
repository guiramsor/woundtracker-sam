"""El mismo S1, aplicado al filtro de cresta: es el selector o es SAM?

F10 midio que S1 sube a los modelos hasta +0.1480. Pero SAM propone tres
mascaras y el clasico solo una, asi que compararlos con selector puesto medía
dos cosas a la vez. Aqui el clasico recibe **la misma oportunidad**: tres
candidatas anidadas de la respuesta de Sato y el mismo selector encima.

Si el clasico tambien sube, lo que aporta es el selector y no el modelo.

Tres columnas por prompt, sobre las **mismas** tres candidatas:

- `dice_s0`, quedarse con la de mayor puntuacion, que aqui es la respuesta media
  de cresta dentro de la candidata;
- `dice_s1`, el criterio del apartado 5.5;
- `dice_una`, la referencia de F9: una sola mascara con Otsu simple.

    conda run -n surco python scripts/selector_clasico.py
"""

import numpy as np
import pandas as pd

from surco import clasicos, config, datos, inventario, metricas, modelos, prompts, selector

import profundidad

METODO = "sato"


def medir_frame(unidad, frame, imagen, referencia) -> list[dict]:
    gris = imagen[:, :, 0]
    filas = []
    for protocolo in config.PROTOCOLOS:
        puntos, etiquetas = profundidad.prompt_de(unidad, protocolo)
        mascaras, puntuaciones = clasicos.candidatas_de_cresta(
            imagen, puntos, etiquetas
        )
        candidatas = modelos.Candidatas(mascaras, puntuaciones)
        indice_s1, respaldo = selector.elegir(candidatas, gris, puntos, etiquetas)
        indice_s0 = int(np.argmax(puntuaciones))
        filas.append(
            {
                "modelo": METODO,
                "modo": "M1",
                "protocolo": protocolo,
                "serie": unidad["serie"],
                "etiqueta": unidad["etiqueta"],
                "frame": frame,
                "dice_s0": metricas.dice(mascaras[indice_s0], referencia),
                "dice_s1": metricas.dice(mascaras[indice_s1], referencia),
                "dice_una": metricas.dice(
                    clasicos.cresta_sato(imagen, puntos, etiquetas), referencia
                ),
                "n_candidatas": len(candidatas),
                "respaldo": respaldo,
                "coincide": indice_s0 == indice_s1,
            }
        )
    return filas


def main() -> None:
    filas = []
    for unidad in prompts.cargar()["unidades"]:
        gt_serie = datos.cargar_gt_serie(unidad["serie"])
        frames = inventario.frames_de_unidad(gt_serie, unidad["etiqueta"])
        for frame in metricas.ventana_de_evaluacion(frames, unidad["start"]):
            imagen = modelos.preparar_frame(unidad["serie"], frame)
            referencia = gt_serie[frame - 1] == unidad["etiqueta"]
            filas.extend(medir_frame(unidad, frame, imagen, referencia))
        print(f"  {unidad['serie']}-L{unidad['etiqueta']}", flush=True)

    tabla = pd.DataFrame(filas)
    tabla.to_csv(config.DIR_PROCESO / "selector_clasico.csv", index=False)

    print(f"\n=== EL MISMO S1 SOBRE EL FILTRO DE CRESTA, n = 17 ===")
    for protocolo, grupo in tabla.groupby("protocolo"):
        medias = {
            columna: metricas.media_entre_unidades(metricas.por_unidad(grupo, columna))
            for columna in ("dice_una", "dice_s0", "dice_s1")
        }
        diferencias = metricas.diferencia_pareada(
            metricas.por_unidad(grupo, "dice_s1"),
            metricas.por_unidad(grupo, "dice_s0"),
        )
        bajo, alto = metricas.intervalo_bootstrap(
            diferencias.to_numpy(), config.N_REMUESTREOS,
            config.NIVEL_CONFIANZA, config.SEMILLA_BOOTSTRAP,
        )
        print(
            f"  {protocolo}  una {medias['dice_una']:7.4f}  "
            f"S0 {medias['dice_s0']:7.4f}  S1 {medias['dice_s1']:7.4f}  "
            f"S1-S0 {diferencias.mean():+.4f} [{bajo:+.4f}, {alto:+.4f}]  "
            f"gana {int((diferencias > 0).sum())}/{len(diferencias)}  "
            f"respaldo {grupo['respaldo'].mean():.4f}"
        )


if __name__ == "__main__":
    main()
