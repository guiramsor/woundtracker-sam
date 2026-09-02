"""Muestra para el estudio de concordancia entre anotadoras (apartado 7).

Tarea al margen de las fases. **No toca el pipeline**: solo lee, y por eso vive
en `scripts/` y no en `src/surco/`. Tampoco declara nada en `config.py`, porque
lo de aqui no es un parametro del banco de pruebas sino de una exportacion.

Selecciona 34 pares (unidad, frame), dos por unidad, uno de cada estrato, y
exporta los frames correspondientes en 8 bits, sin marcas ni overlay del GT.
Las anotadoras devuelven ROIs en .zip del ROI Manager de Fiji, asi que aqui no
se generan mascaras.

**La concordancia se reporta por estrato, temprano y tardio por separado, nunca
agregada en un solo numero.** En los frames tardios la fragmentacion es maxima,
asi que un desacuerdo alto ahi puede venir de ambiguedad del protocolo y no de
falta de precision; promediarlo con el estrato temprano deflactaria el techo
humano entero.

Se ejecuta con:

    conda run -n surco python scripts/concordancia_seleccion.py
"""

import numpy as np
import pandas as pd
import tifffile

from surco import config, datos, inventario

DESTINO = config.DIR_PROCESO / "concordancia"
DIR_IMAGENES = DESTINO / "imagenes"

# `contraste.csv` es el unico resultado que no vive en `salidas/`, y va a la
# raiz **versionado**. El motivo no es que sea irregenerable, lo produce este
# mismo script, sino que `salidas/` no viaja con el repositorio y su rango, de
# 13 a 165 niveles, se cita en la memoria.
DIR_CONTRASTE = config.RAIZ

# Punto blanco de la normalizacion, en percentil sobre los pixeles reunidos de
# todos los frames exportados. El surco ocupa unas milesimas de la imagen, asi
# que estirar hasta el maximo global deja el 99.9% de los pixeles en los doce
# primeros niveles de gris y las imagenes salen negras. El punto negro es el
# minimo global, sin recorte.
PERCENTIL_BLANCO = 99.9


def pares_excluidos() -> set[tuple[int, int, int]]:
    """Los 10 pares cuya mascara es copia del frame anterior (medido en F1).

    Se excluye `frame_b`, que es la copia; `frame_a` si es anotacion propia.
    """
    duplicados = inventario.duplicados_de_todas_las_series()
    return set(
        zip(duplicados["serie"], duplicados["etiqueta"], duplicados["frame_b"])
    )


def seleccionar(inventario_df: pd.DataFrame) -> pd.DataFrame:
    """Dos pares por unidad, uno por estrato.

    El temprano arranca en `start + 1`, no en `start`: el frame semilla queda
    fuera de la ventana de evaluacion del apartado 6.2, asi que medir ahi el
    techo humano seria medirlo donde ningun modelo compite, y ademas es el
    frame mas debil. El tardio es el ultimo frame elegible.
    """
    excluidos = pares_excluidos()
    filas = []
    for (serie, etiqueta), grupo in inventario_df.groupby(["serie", "etiqueta"]):
        start = int(min(grupo["frame"]))
        elegibles = sorted(
            f
            for f in grupo["frame"]
            if f > start and (serie, etiqueta, f) not in excluidos
        )
        for estrato, frame in (("temprano", elegibles[0]), ("tardio", elegibles[-1])):
            filas.append(
                {
                    "serie": int(serie),
                    "etiqueta": int(etiqueta),
                    "estrato": estrato,
                    "frame": int(frame),
                    "start": start,
                }
            )
    return pd.DataFrame(filas)


def nombre_de_fichero(serie: int, frame: int) -> str:
    return f"conc_s{serie:02d}_t{frame:02d}.tif"


def limites_de_intensidad(frames: list[tuple[int, int]]) -> tuple[float, float]:
    """Punto negro y punto blanco, comunes a toda la exportacion."""
    reunidos = np.concatenate([datos.cargar_frame(s, f).ravel() for s, f in frames])
    return float(reunidos.min()), float(np.percentile(reunidos, PERCENTIL_BLANCO))


def a_ocho_bits(imagen: np.ndarray, negro: float, blanco: float) -> np.ndarray:
    escalada = np.clip((imagen.astype(np.float32) - negro) / (blanco - negro), 0, 1)
    return (escalada * 255).round().astype(np.uint8)


def exportar(frames: list[tuple[int, int]], negro: float, blanco: float) -> None:
    DIR_IMAGENES.mkdir(parents=True, exist_ok=True)
    for serie, frame in frames:
        ocho = a_ocho_bits(datos.cargar_frame(serie, frame), negro, blanco)
        tifffile.imwrite(DIR_IMAGENES / nombre_de_fichero(serie, frame), ocho)


def tabla_de_correspondencia(seleccion: pd.DataFrame) -> pd.DataFrame:
    """Que pares (unidad, frame) le tocan a cada fichero."""
    tabla = seleccion.copy()
    tabla["fichero"] = [
        nombre_de_fichero(s, f) for s, f in zip(tabla["serie"], tabla["frame"])
    ]
    tabla["unidad"] = [
        f"{s}-L{e}" for s, e in zip(tabla["serie"], tabla["etiqueta"])
    ]
    unidades_por_fichero = tabla.groupby("fichero")["unidad"].transform("size")
    tabla["unidades_en_el_fichero"] = unidades_por_fichero
    tabla["compartido"] = unidades_por_fichero > 1
    columnas = [
        "fichero",
        "unidad",
        "serie",
        "etiqueta",
        "estrato",
        "frame",
        "start",
        "unidades_en_el_fichero",
        "compartido",
    ]
    return tabla[columnas].sort_values(["serie", "frame", "etiqueta"])


def contraste_por_fichero(
    tabla: pd.DataFrame, negro: float, blanco: float
) -> pd.DataFrame:
    """Cuanto destaca el surco sobre el fondo en cada fichero, ya en 8 bits.

    Sirve para responder si la imagen mas tenue sigue siendo anotable con el
    mapeo comun. Usa el GT, que las anotadoras no van a ver: es un control
    nuestro, no informacion que viaje con las imagenes.
    """
    filas = []
    for fichero, grupo in tabla.groupby("fichero"):
        serie = int(grupo["serie"].iloc[0])
        frame = int(grupo["frame"].iloc[0])
        ocho = a_ocho_bits(datos.cargar_frame(serie, frame), negro, blanco)
        gt = datos.cargar_gt(serie, frame)
        for etiqueta in grupo["etiqueta"]:
            surco = ocho[gt == etiqueta]
            filas.append(
                {
                    "fichero": fichero,
                    "unidad": f"{serie}-L{etiqueta}",
                    "estrato": grupo.loc[grupo["etiqueta"] == etiqueta, "estrato"].iloc[0],
                    "fondo_mediana": float(np.median(ocho)),
                    "surco_mediana": float(np.median(surco)),
                    "surco_p90": float(np.percentile(surco, 90)),
                    "contraste": float(np.median(surco) - np.median(ocho)),
                }
            )
    return pd.DataFrame(filas).sort_values("contraste")


def main() -> None:
    inventario_df = inventario.construir_inventario()
    seleccion = seleccionar(inventario_df)
    tabla = tabla_de_correspondencia(seleccion)
    frames = sorted(set(zip(seleccion["serie"], seleccion["frame"])))

    negro, blanco = limites_de_intensidad(frames)
    exportar(frames, negro, blanco)

    DESTINO.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(DESTINO / "correspondencia.csv", index=False)

    contraste = contraste_por_fichero(tabla, negro, blanco)
    DIR_CONTRASTE.mkdir(parents=True, exist_ok=True)
    contraste.to_csv(DIR_CONTRASTE / "contraste.csv", index=False)

    print(f"pares seleccionados : {len(tabla)}")
    print(f"ficheros exportados : {len(frames)}")
    print(f"normalizacion comun : {negro:.0f} -> {blanco:.0f} (percentil {PERCENTIL_BLANCO})")
    print(f"destino             : {DIR_IMAGENES}")
    print()
    print(tabla.to_string(index=False))
    print("\n=== contraste del surco sobre el fondo, en niveles de 8 bits ===")
    print(contraste.to_string(index=False))


if __name__ == "__main__":
    main()
