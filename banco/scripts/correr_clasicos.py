"""F9: los tres metodos clasicos del apartado 5.4 sobre las 17 unidades.

La hipotesis nula del trabajo. Si uno de estos llega lo bastante cerca, embeber
SAM no esta justificado.

Los tres x P1-P4, sobre la misma ventana de evaluacion y con el mismo frame
preparado que reciben los modelos. **En el sentido de M1**: se vuelven a aplicar
en cada frame con los mismos clics y sin memoria, que es literalmente lo que
hacen, asi que la comparacion que les toca es contra la fila M1 de la rejilla.

**No entra el factor selector.** Los clasicos devuelven una mascara y no
proponen candidatas ni las puntuan, asi que S0 y S1 son la misma cosa en su fila
y la columna sale como `-` en vez de fingir un nivel.

No depende de ningun entorno de modelo: corre en el del banco.

    conda run -n surco python scripts/correr_clasicos.py
"""

import pandas as pd

from surco import clasicos, config, datos, inventario, metricas, modelos, prompts


prompt_de = prompts.puntos_y_etiquetas


def medir_frame(unidad: dict, frame: int, imagen, referencia) -> list[dict]:
    """Los tres metodos por los cuatro protocolos sobre un mismo frame.

    El frame va por fuera y el metodo por dentro para que la respuesta de Sato,
    que es la parte cara y no depende del prompt, se calcule una vez por imagen.
    """
    filas = []
    for metodo in config.METODOS_CLASICOS:
        for protocolo in config.PROTOCOLOS:
            puntos, etiquetas = prompt_de(unidad, protocolo)
            mascara = clasicos.segmentar(metodo, imagen, puntos, etiquetas)
            filas.append(
                {
                    "modelo": metodo,
                    "checkpoint": "-",
                    "familia": "clasico",
                    "modo": "M1",
                    "protocolo": protocolo,
                    "selector": "-",
                    "serie": unidad["serie"],
                    "etiqueta": unidad["etiqueta"],
                    "frame": frame,
                    "dice": metricas.dice(mascara, referencia),
                }
            )
    return filas


def correr_unidad(unidad: dict, gt_serie) -> list[dict]:
    frames = inventario.frames_de_unidad(gt_serie, unidad["etiqueta"])
    ventana = metricas.ventana_de_evaluacion(frames, unidad["start"])
    filas = []
    for frame in ventana:
        imagen = modelos.preparar_frame(unidad["serie"], frame)
        referencia = gt_serie[frame - 1] == unidad["etiqueta"]
        filas.extend(medir_frame(unidad, frame, imagen, referencia))
    return filas


def main() -> None:
    unidades = prompts.cargar()["unidades"]
    filas = []
    for unidad in unidades:
        gt_serie = datos.cargar_gt_serie(unidad["serie"])
        filas.extend(correr_unidad(unidad, gt_serie))
        print(f"  {unidad['serie']}-L{unidad['etiqueta']}", flush=True)

    tabla = pd.DataFrame(filas)
    tabla.to_csv(config.DIR_PROCESO / "clasicos.csv", index=False)

    print(f"\n=== LOS TRES CLASICOS: Dice medio entre unidades, n = 17 ===")
    resumen = []
    for (metodo, protocolo), grupo in tabla.groupby(["modelo", "protocolo"]):
        por_unidad = metricas.por_unidad(grupo)
        valores = por_unidad.to_numpy()
        bajo, alto = metricas.intervalo_bootstrap(
            valores, config.N_REMUESTREOS, config.NIVEL_CONFIANZA,
            config.SEMILLA_BOOTSTRAP,
        )
        resumen.append(
            {
                "metodo": metodo,
                "protocolo": protocolo,
                "dice": metricas.media_entre_unidades(por_unidad),
                "ic_bajo": bajo,
                "ic_alto": alto,
                "catastroficas": metricas.fraccion_catastrofica(
                    valores, config.UMBRAL_CATASTROFICO
                ),
            }
        )
    resumen = pd.DataFrame(resumen)
    print(resumen.to_string(index=False, float_format=lambda v: f"{v:7.4f}"))
    resumen.to_csv(config.DIR_PROCESO / "tabla_clasicos.csv", index=False)


if __name__ == "__main__":
    main()
