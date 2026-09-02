"""Metricas de calidad (apartados 6.1 a 6.5).

Una sola metrica con su curva y su intervalo, en lugar de las nueve columnas
del trabajo anterior. Aqui no hay modelos: entran mascaras o numeros y salen
numeros, para poder probarlo todo sobre casos sinteticos.

El orden de las operaciones importa y es el del apartado 6.3: primero se
promedia **dentro** de cada lesion y despues **entre** las 17. El promedio
plano sobre los 312 pares daria el triple de peso a unas lesiones que a otras,
porque los pares por serie van de 13 a 56, y estrecharia los intervalos en un
factor de unos 4.3 tratando como independientes frames casi identicos.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from surco import config


def dice(prediccion: np.ndarray, referencia: np.ndarray) -> float:
    """Dice de la mascara predicha contra el GT de esa lesion (apartado 6.1).

    Si la prediccion esta vacia sale 0, que es lo correcto. Si el GT esta
    vacio, el Dice no esta definido y se avisa en vez de inventar un convenio:
    dentro de la ventana de evaluacion eso no puede pasar, porque la ventana se
    construye con los frames en que la lesion existe.
    """
    if not referencia.any():
        raise ValueError("el GT de referencia esta vacio: el Dice no esta definido")
    interseccion = int(np.logical_and(prediccion, referencia).sum())
    total = int(prediccion.sum()) + int(referencia.sum())
    return float(2 * interseccion / total)


def iou_desde_dice(valor: float) -> float:
    """IoU = Dice / (2 - Dice).

    Se reporta porque la literatura lo usa, pero es monotona en el Dice, asi
    que el orden de los modelos es identico en las dos columnas y no aporta
    informacion nueva (apartado 6.1).
    """
    return float(valor / (2 - valor))


def ventana_de_evaluacion(frames_anotados, start: int) -> tuple[int, ...]:
    """Desde `start + 1` en adelante, intersecado con los frames anotados.

    Deja fuera dos cosas (apartado 6.2): los frames anteriores a `start`, donde
    no hay prediccion sino ceros que el codigo nunca rellenó, y el propio frame
    semilla, donde la mascara es la entrada y no la salida. Asi nadie recibe un
    frame regalado ni es castigado por uno que no se le pidio.
    """
    return tuple(frame for frame in sorted(frames_anotados) if frame > start)


def por_unidad(tabla: pd.DataFrame, columna: str = "dice") -> pd.Series:
    """Media dentro de cada lesion: el primer paso de la agregacion."""
    return tabla.groupby(["serie", "etiqueta"])[columna].mean()


def media_entre_unidades(valores_por_unidad: pd.Series) -> float:
    """Media entre las unidades. n = 17, no 312."""
    return float(valores_por_unidad.mean())


def diferencia_pareada(uno: pd.Series, otro: pd.Series) -> pd.Series:
    """Diferencia unidad a unidad (apartado 6.4).

    Se compara asi porque todos los modelos ven exactamente las mismas
    lesiones. Exige que las dos series cubran las mismas unidades: comparar
    sobre conjuntos distintos no seria una comparacion pareada.
    """
    if set(uno.index) != set(otro.index):
        raise ValueError("las dos series no cubren las mismas unidades")
    return uno - otro.reindex(uno.index)


def fraccion_de_unidades_ganadas(diferencias: pd.Series) -> float:
    """En que fraccion de unidades gana el primero.

    Es lo que impide leer los intervalos como un semaforo: dos modelos pueden
    tener intervalos solapados y aun asi diferir de forma consistente, si uno
    gana al otro en 15 de las 17.
    """
    return float((diferencias > 0).mean())


def intervalo_bootstrap(
    valores, n_remuestreos: int, nivel: float, semilla: int
) -> tuple[float, float]:
    """Intervalo de percentiles remuestreando **unidades**, no frames."""
    muestra = np.asarray(valores, dtype=float)
    generador = np.random.default_rng(semilla)
    replicas = generador.choice(
        muestra, size=(n_remuestreos, muestra.size), replace=True
    )
    medias = replicas.mean(axis=1)
    cola = (1 - nivel) / 2
    return float(np.quantile(medias, cola)), float(np.quantile(medias, 1 - cola))


@dataclass(frozen=True)
class Contraste:
    """El resultado de comparar dos celdas, con todo lo que hay que reportar.

    Las cuatro cifras van juntas a proposito. La diferencia media sola no dice
    si es solida; el intervalo solo no dice en cuantas unidades pasa; y sin la
    n no se sabe sobre que se ha calculado. Separarlas fue lo que hizo que cada
    fase del trabajo acabara imprimiendo su comparacion en un formato distinto.
    """

    diferencia: float
    ic_bajo: float
    ic_alto: float
    ganadas: int
    n: int

    @property
    def separado(self) -> bool:
        """Si el intervalo no cruza el cero."""
        return self.ic_bajo > 0 or self.ic_alto < 0

    def como_fila(self) -> dict:
        """Para meterlo en un DataFrame sin desmontarlo a mano."""
        return {
            "diferencia": self.diferencia,
            "ic_bajo": self.ic_bajo,
            "ic_alto": self.ic_alto,
            "ganadas": self.ganadas,
            "n": self.n,
        }

    def __str__(self) -> str:
        return (
            f"{self.diferencia:+.4f} [{self.ic_bajo:+.4f}, {self.ic_alto:+.4f}], "
            f"gana en {self.ganadas} de {self.n}"
        )


def contraste(uno: pd.Series, otro: pd.Series) -> Contraste:
    """Compara dos celdas pareando por unidad, con el bootstrap del 6.4.

    Es la operacion que responde casi todas las preguntas del trabajo, y por eso
    vive aqui y no repetida en cada script de tabla: cambiar como se compara
    tiene que cambiarlo en todas las fases a la vez.

    Los tres parametros del bootstrap salen del config y no se pasan: son los
    mismos en todo el trabajo, y tenerlos en la firma solo servia para que siete
    llamadas los repitieran.
    """
    diferencias = diferencia_pareada(uno, otro)
    bajo, alto = intervalo_bootstrap(
        diferencias.to_numpy(),
        config.N_REMUESTREOS,
        config.NIVEL_CONFIANZA,
        config.SEMILLA_BOOTSTRAP,
    )
    return Contraste(
        diferencia=float(diferencias.mean()),
        ic_bajo=bajo,
        ic_alto=alto,
        ganadas=int((diferencias > 0).sum()),
        n=len(diferencias),
    )


def filas(tabla: pd.DataFrame, **filtros) -> pd.DataFrame:
    """Las filas por frame de una celda, filtrando por las columnas que se den.

    Va aparte de `celda` porque las figuras necesitan los frames sueltos y las
    comparaciones necesitan la media por unidad, y las dos empezaban por el
    mismo encadenado de filtros.
    """
    filtro = pd.Series(True, index=tabla.index)
    for nombre, valor in filtros.items():
        filtro &= tabla[nombre] == valor
    return tabla[filtro]


def celda(tabla: pd.DataFrame, columna: str = "dice", **filtros) -> pd.Series:
    """El Dice por unidad de una celda, filtrando por las columnas que se den.

    `celda(rejilla, modelo="sam21", modo="M3", protocolo="P4")` es lo que todos
    los scripts escribian a mano, cada uno con su propia forma de encadenar los
    filtros.
    """
    return por_unidad(filas(tabla, **filtros), columna)


def curva_de_umbrales(valores) -> pd.DataFrame:
    """Fraccion de casos que supera cada umbral, barriendo de 0 a 1.

    Se construye sobre la distribucion empirica y no sobre una rejilla, asi que
    no hay ningun paso que elegir y el area bajo la curva sale exactamente
    igual a la media, no aproximada.
    """
    muestra = np.asarray(valores, dtype=float)
    if muestra.min() < 0 or muestra.max() > 1:
        raise ValueError("los valores de Dice tienen que caer entre 0 y 1")
    umbrales = np.unique(np.concatenate(([0.0], muestra, [1.0])))
    fracciones = [float((muestra > umbral).mean()) for umbral in umbrales]
    return pd.DataFrame({"umbral": umbrales, "fraccion": fracciones})


def curvas(tabla: pd.DataFrame, columna: str = "dice") -> dict[str, pd.DataFrame]:
    """Las dos curvas del apartado 6.5, que responden preguntas distintas.

    `"unidad"` se construye sobre las medias de lesion y es la que acompana a la
    tabla principal y a los intervalos: su area **es** la media reportada, y su
    punto t = 0.1 son unidades perdidas enteras.

    `"frame"` se construye sobre los pares de la ventana y es la curva de
    utilidad: que fraccion de ROIs se aceptaria sin retocar. Su punto t = 0.1
    son frames catastroficos, y su area es la media por frame, que **no**
    coincide con la de la tabla.

    Que las dos discrepen es informacion: dice si los fallos estan repartidos
    por la muestra o concentrados en unas pocas lesiones.
    """
    return {
        "unidad": curva_de_umbrales(por_unidad(tabla, columna).to_numpy()),
        "frame": curva_de_umbrales(tabla[columna].to_numpy()),
    }


def area_bajo_curva(curva: pd.DataFrame) -> float:
    """Integra la curva de umbrales. Da la media, asi que no se pierde nada."""
    umbral = curva["umbral"].to_numpy()
    fraccion = curva["fraccion"].to_numpy()
    return float((fraccion[:-1] * np.diff(umbral)).sum())


def fraccion_que_supera(valores, umbral: float) -> float:
    """El valor de la curva en un umbral: que fraccion queda por encima."""
    return float((np.asarray(valores, dtype=float) > umbral).mean())


def fraccion_catastrofica(valores, umbral: float) -> float:
    """Complemento de la curva en el umbral bajo.

    Los catastroficos no son una metrica aparte: son el punto t = 0.1 de la
    misma curva, leido por el otro lado.
    """
    return 1.0 - fraccion_que_supera(valores, umbral)
