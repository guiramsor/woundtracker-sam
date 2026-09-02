"""Cuanto cuesta pasar la entrada por JPEG, medido sobre las 17 unidades.

El predictor de video de SAM 2 solo admite carpeta de JPEG o MP4, mientras que
el de imagen recibe el array preparado tal cual. Si la perdida del JPEG importa,
M1 y M3 verian imagenes distintas del mismo frame y la comparacion entre ambos
dentro del mismo modelo, que es lo que pide el apartado 5.1, dejaria de medir la
memoria temporal para medir tambien el codec.

Esto no es un frame suelto: se corre M1 entero por las dos vias y se compara con
la maquinaria del apartado 6.4, diferencia pareada unidad a unidad. La pregunta
es si la perdida es sistematica o ruido.

Configuracion fija: **M1 x P1 x S0**, y el checkpoint **que corre en la rejilla**,
que sale de `config.REJILLA_PROFUNDIDAD` y hoy es `tiny`. No el por defecto del
adaptador: la primera version llamaba a `modelos.crear("sam21")` sin checkpoint y
media sobre `large`, que no es el que hace la comparacion M1 contra M3, asi que la
medida no cubria la configuracion que autorizaba.

    conda run -n surco-sam21 python scripts/unicos/medir_perdida_jpeg.py
"""

import io

import numpy as np
import pandas as pd
from PIL import Image

from surco import adaptadores, config, datos, inventario, metricas, modelos, prompts

PROTOCOLO = "P1"
CALIDAD = 100


def por_jpeg(imagen_rgb: np.ndarray) -> np.ndarray:
    """Ida y vuelta por JPEG a calidad maxima y sin submuestreo de croma."""
    memoria = io.BytesIO()
    Image.fromarray(imagen_rgb).save(
        memoria, format="JPEG", quality=CALIDAD, subsampling=0
    )
    memoria.seek(0)
    return np.asarray(Image.open(memoria).convert("RGB"))


def _clics_del_protocolo(unidad: dict) -> tuple[list, list]:
    indices = config.PROTOCOLOS[PROTOCOLO]
    puntos = [tuple(unidad["clics"][i]) for i in indices]
    etiquetas = [0 if i in config.CLICS_NEGATIVOS else 1 for i in indices]
    return puntos, etiquetas


def medir_unidad(modelo, unidad: dict, gt_serie: np.ndarray) -> list[dict]:
    serie, etiqueta = unidad["serie"], unidad["etiqueta"]
    puntos, etiquetas = _clics_del_protocolo(unidad)
    frames = inventario.frames_de_unidad(gt_serie, etiqueta)
    ventana = metricas.ventana_de_evaluacion(frames, unidad["start"])

    filas = []
    for frame in ventana:
        referencia = gt_serie[frame - 1] == etiqueta
        array = modelos.preparar_frame(serie, frame)
        jpeg = por_jpeg(array)
        filas.append(
            {
                "serie": serie,
                "etiqueta": etiqueta,
                "frame": frame,
                "dice_array": metricas.dice(
                    modelo.segmentar(array, puntos, etiquetas).mejor(), referencia
                ),
                "dice_jpeg": metricas.dice(
                    modelo.segmentar(jpeg, puntos, etiquetas).mejor(), referencia
                ),
                "dif_pixel_media": float(
                    np.abs(array.astype(np.int16) - jpeg.astype(np.int16)).mean()
                ),
            }
        )
    return filas


def informar(tabla: pd.DataFrame) -> None:
    por_array = metricas.por_unidad(tabla, "dice_array")
    por_jpeg_ = metricas.por_unidad(tabla, "dice_jpeg")
    diferencias = metricas.diferencia_pareada(por_array, por_jpeg_)

    comparacion = pd.DataFrame(
        {"array": por_array, "jpeg": por_jpeg_, "diferencia": diferencias}
    )
    print("\n=== DICE POR UNIDAD ===")
    print(comparacion.to_string(float_format=lambda v: f"{v:8.4f}"))

    bajo, alto = metricas.intervalo_bootstrap(
        diferencias.to_numpy(),
        config.N_REMUESTREOS,
        config.NIVEL_CONFIANZA,
        config.SEMILLA_BOOTSTRAP,
    )
    print("\n=== DIFERENCIA PAREADA (array menos jpeg) ===")
    print(f"  media            : {diferencias.mean():+.5f}")
    print(f"  IC {config.NIVEL_CONFIANZA:.0%} bootstrap : [{bajo:+.5f}, {alto:+.5f}]")
    print(f"  mayor a favor del array : {diferencias.max():+.5f}")
    print(f"  mayor a favor del jpeg  : {diferencias.min():+.5f}")
    print(
        f"  unidades en que gana el array: "
        f"{metricas.fraccion_de_unidades_ganadas(diferencias):.1%} "
        f"({int((diferencias > 0).sum())} de {len(diferencias)})"
    )
    print(f"  unidades identicas: {int((diferencias == 0).sum())}")

    print("\n=== MEDIA GLOBAL ===")
    print(f"  array: {metricas.media_entre_unidades(por_array):.4f}")
    print(f"  jpeg : {metricas.media_entre_unidades(por_jpeg_):.4f}")
    print(f"\n  diferencia media de pixel que mete el JPEG: "
          f"{tabla['dif_pixel_media'].mean():.3f} niveles de 255")


def main() -> None:
    adaptadores.cargar_disponibles()
    checkpoint = config.REJILLA_PROFUNDIDAD["sam21"]
    modelo = modelos.crear("sam21", checkpoint)
    modelo.cargar()

    datos_clics = prompts.cargar()
    filas = []
    for unidad in datos_clics["unidades"]:
        gt_serie = datos.cargar_gt_serie(unidad["serie"])
        filas.extend(medir_unidad(modelo, unidad, gt_serie))
        print(f"  {unidad['serie']}-L{unidad['etiqueta']} listo", flush=True)

    tabla = pd.DataFrame(filas)
    tabla.insert(0, "checkpoint", checkpoint)
    tabla.to_csv(config.DIR_PROCESO / "perdida_jpeg.csv", index=False)
    informar(tabla)
    print(f"\npares medidos: {len(tabla)}   checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
