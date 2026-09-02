"""Si el filtro de cresta gana por fuerza bruta o por la forma del surco.

Diez escalas de 2 a 11 en paso 1 cubren el rango entero, asi que el resultado
podria venir de barrerlo de punta a punta y no de que el a priori de cresta sea
el adecuado. Se repite **Sato x P3**, que es su mejor celda, con cada
paso del 1 al 6, y se compara pareado unidad a unidad contra el paso 1.

Si el resultado aguanta, el que se queda es **el paso mas grosero que lo
mantenga**: menos escalas es menos computo dentro de Fiji.

**El paso 1 se mide aqui**, aunque cueste, y no se lee de `clasicos.csv`. Esa
tabla se rehace cada vez que cambia el config, asi que leerla como si fuera el
paso 1 daria una referencia distinta segun cuando se mirara, y la comparacion
diria cosas distintas sin que nada avisara.

    conda run -n surco python scripts/robustez_sigma.py
"""

import pandas as pd

from surco import clasicos, config, datos, inventario, metricas, modelos, prompts

PROTOCOLO = "P3"
PASOS = (1, 2, 3, 4, 5, 6)
REFERENCIA = 1


def escalas(paso: int) -> tuple[int, ...]:
    """El rango declarado del config, barrido con otro paso.

    Sale de `RANGO_SIGMAS_SATO` y **no de los extremos de `SIGMAS_SATO`**, que
    es lo que hacia antes: la tupla ya construida acaba donde la deja el paso
    elegido, asi que al adoptar el paso 4 este barrido paso a ir de 2 a 10 en
    vez de 2 a 11 sin que nada avisara. Un script que decide un paso no puede
    sacar su rango de una constante que ya depende de ese paso.
    """
    primera, ultima = config.RANGO_SIGMAS_SATO
    return tuple(range(primera, ultima + 1, paso))


def medir(sigmas) -> pd.DataFrame:
    indices = config.PROTOCOLOS[PROTOCOLO]
    filas = []
    for unidad in prompts.cargar()["unidades"]:
        gt_serie = datos.cargar_gt_serie(unidad["serie"])
        frames = inventario.frames_de_unidad(gt_serie, unidad["etiqueta"])
        puntos = [tuple(unidad["clics"][i]) for i in indices]
        etiquetas = [0 if i in config.CLICS_NEGATIVOS else 1 for i in indices]
        for frame in metricas.ventana_de_evaluacion(frames, unidad["start"]):
            imagen = modelos.preparar_frame(unidad["serie"], frame)
            mascara = clasicos.cresta_sato(imagen, puntos, etiquetas, sigmas)
            filas.append(
                {
                    "serie": unidad["serie"],
                    "etiqueta": unidad["etiqueta"],
                    "frame": frame,
                    "dice": metricas.dice(
                        mascara, gt_serie[frame - 1] == unidad["etiqueta"]
                    ),
                }
            )
        print(f"  {unidad['serie']}-L{unidad['etiqueta']}", flush=True)
    return pd.DataFrame(filas)


def main() -> None:
    medidas = {}
    for paso in PASOS:
        sigmas = escalas(paso)
        print(f"\npaso {paso}  {sigmas}")
        medidas[paso] = metricas.por_unidad(medir(sigmas))

    referencia = medidas[REFERENCIA]
    filas = []
    for paso, por_unidad in medidas.items():
        sigmas = escalas(paso)
        diferencias = metricas.diferencia_pareada(por_unidad, referencia)
        bajo, alto = metricas.intervalo_bootstrap(
            diferencias.to_numpy(), config.N_REMUESTREOS,
            config.NIVEL_CONFIANZA, config.SEMILLA_BOOTSTRAP,
        )
        filas.append(
            {
                "paso": paso,
                "n_escalas": len(sigmas),
                "sigmas": " ".join(str(s) for s in sigmas),
                "dice": metricas.media_entre_unidades(por_unidad),
                "menos_paso_1": float(diferencias.mean()),
                "ic_bajo": bajo,
                "ic_alto": alto,
                "unidades_ganadas": int((diferencias > 0).sum()),
                "catastroficas": metricas.fraccion_catastrofica(
                    por_unidad.to_numpy(), config.UMBRAL_CATASTROFICO
                ),
            }
        )

    tabla = pd.DataFrame(filas)
    print(f"\n=== SATO x {PROTOCOLO}: EL PASO DEL BARRIDO, n = 17 ===")
    print(tabla.to_string(index=False, float_format=lambda v: f"{v:7.4f}"))
    tabla.to_csv(config.DIR_PROCESO / "robustez_sigma.csv", index=False)


if __name__ == "__main__":
    main()
