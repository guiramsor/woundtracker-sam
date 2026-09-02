"""Figuras de la memoria: overlays de mascara predicha sobre la imagen.

Vive aqui y no en los notebooks porque los notebooks orquestan y muestran, no
contienen logica. Lo que sale de aqui es lo que se pega en la memoria.

**Las mismas unidades y los mismos frames en todas las figuras.** El conjunto no
se elige a ojo: es la muestra estratificada del apartado 7, que ya estaba
definida por una regla (el frame temprano es `start + 1` y el tardio es el
ultimo elegible de cada unidad) y cubre las 17 unidades sin dejar ninguna fuera.
Elegir a mano cuatro ejemplos bonitos seria escoger el resultado.
"""

from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from surco import config, datos, inventario, metricas, modelos, prompts, resultados

# Colores de los overlays. El GT va en contorno y la prediccion en relleno
# translucido, para que se vea donde se solapan y donde no.
COLOR_PREDICCION = "#ff3b30"
COLOR_GT = "#00e5ff"
ALFA_PREDICCION = 0.45


def pares_de_figura() -> pd.DataFrame:
    """Las 34 parejas (unidad, frame) sobre las que se dibuja todo.

    Se reconstruyen del GT y del fichero de clics, no se leen de un CSV de
    salidas: asi la figura no depende de que se haya corrido antes el montaje de
    la muestra de concordancia.
    """
    filas = []
    for unidad in prompts.cargar()["unidades"]:
        gt_serie = datos.cargar_gt_serie(unidad["serie"])
        frames = inventario.frames_de_unidad(gt_serie, unidad["etiqueta"])
        elegibles = [f for f in frames if f > unidad["start"]]
        if not elegibles:
            continue
        for estrato, frame in (
            ("temprano", elegibles[0]),
            ("tardio", elegibles[-1]),
        ):
            filas.append(
                {
                    "serie": unidad["serie"],
                    "etiqueta": unidad["etiqueta"],
                    "unidad": f"{unidad['serie']}-L{unidad['etiqueta']}",
                    "estrato": estrato,
                    "frame": frame,
                }
            )
    return pd.DataFrame(filas)


# El mejor modelo del trabajo (apartado 5.5) sale del config, que es donde vive
# para que no se repita por los scripts.
MEJOR_MODELO = config.MEJOR_CELDA


@lru_cache(maxsize=1)
def mejor_clasico() -> dict:
    """La mejor celda de los metodos clasicos, preguntando a `resultados`.

    **No es una constante escrita a mano**, y ese es el arreglo: antes ponia
    `sato x P3` a pelo, que era la mejor celda cuando se escribio pero dejaba
    fuera el filtro con selector. Se cachea porque la rejilla de overlays llama
    a esto una vez por panel y los CSV no cambian a mitad de figura.
    """
    return resultados.mejor_celda_de_clasico()


def _clics_por_unidad() -> dict:
    return {(u["serie"], u["etiqueta"]): u for u in prompts.cargar()["unidades"]}


def mascara_del_mejor_modelo(fila) -> np.ndarray:
    """La mascara que elige S1 sobre las candidatas que guardo F8."""
    from surco import selector

    return selector.mascara_guardada(
        MEJOR_MODELO, fila["serie"], fila["etiqueta"], fila["frame"]
    )


def mascara_del_mejor_clasico(fila) -> np.ndarray:
    """La mascara del mejor metodo clasico, en la variante en que lo es.

    Las tres variantes se dibujan distinto y hay que respetarlo: con una sola
    mascara sale de `clasicos.segmentar`, y con selector hay que construir las
    tres candidatas y dejar que elija S0 o S1.
    """
    from surco import clasicos, selector

    celda = mejor_clasico()
    unidad = _clics_por_unidad()[(fila["serie"], fila["etiqueta"])]
    puntos, etiquetas = prompts.puntos_y_etiquetas(unidad, celda["protocolo"])
    imagen = modelos.preparar_frame(fila["serie"], fila["frame"])
    if celda["variante"] == "una mascara":
        return clasicos.segmentar(celda["metodo"], imagen, puntos, etiquetas)

    mascaras, puntuaciones = clasicos.candidatas_de_cresta(imagen, puntos, etiquetas)
    candidatas = modelos.Candidatas(mascaras, puntuaciones)
    indice = selector.elegir_con(
        celda["variante"], candidatas, imagen[:, :, 0], puntos, etiquetas
    )
    return candidatas.mascaras[indice]


def recuadro(gt: np.ndarray, margen: int = 60) -> tuple[slice, slice]:
    """Recorte alrededor de la lesion, para que el surco se vea.

    Sin esto cada panel es un 1024 x 1024 con un trazo de 17 px de ancho dentro,
    que a tamano de figura no se distingue de nada.
    """
    filas, columnas = np.nonzero(gt)
    lado = config.LADO_PX
    return (
        slice(max(0, filas.min() - margen), min(lado, filas.max() + margen + 1)),
        slice(max(0, columnas.min() - margen), min(lado, columnas.max() + margen + 1)),
    )


def dibujar_overlay(ejes, imagen_rgb, prediccion, gt, titulo: str) -> None:
    """Un panel: imagen de fondo, prediccion en relleno, GT en contorno."""
    filas, columnas = recuadro(gt)
    ejes.imshow(imagen_rgb[filas, columnas, 0], cmap="gray", vmin=0, vmax=255)
    capa = np.zeros((*imagen_rgb[filas, columnas, 0].shape, 4))
    capa[prediccion[filas, columnas]] = (*_rgb(COLOR_PREDICCION), ALFA_PREDICCION)
    ejes.imshow(capa)
    ejes.contour(gt[filas, columnas], levels=[0.5], colors=COLOR_GT, linewidths=1.0)
    ejes.set_title(titulo, fontsize=7)
    ejes.set_xticks([])
    ejes.set_yticks([])


def _rgb(color: str) -> tuple[float, float, float]:
    return tuple(int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))


def rejilla_de_overlays(pares, mascara_de, titulo: str, destino=None):
    """Una figura con un panel por pareja, en el orden de `pares_de_figura`.

    `mascara_de(fila)` devuelve la mascara predicha de esa pareja. Se pasa como
    funcion para que la misma figura sirva para un modelo y para un metodo
    clasico sin duplicar el dibujo.
    """
    columnas = 6
    filas_rejilla = int(np.ceil(len(pares) / columnas))
    figura, ejes = plt.subplots(
        filas_rejilla, columnas, figsize=(2.2 * columnas, 2.2 * filas_rejilla)
    )
    planos = ejes.ravel()
    for plano in planos[len(pares) :]:
        plano.axis("off")
    for plano, (_, fila) in zip(planos, pares.iterrows()):
        gt = datos.cargar_gt(fila["serie"], fila["frame"]) == fila["etiqueta"]
        imagen = modelos.preparar_frame(fila["serie"], fila["frame"])
        dibujar_overlay(
            plano,
            imagen,
            mascara_de(fila),
            gt,
            f"{fila['unidad']} t{fila['frame']} ({fila['estrato'][:4]})",
        )
    figura.suptitle(titulo, fontsize=11)
    figura.tight_layout()
    if destino is not None:
        guardar(figura, destino)
    return figura


# Cinco lineas: los cuatro de la rejilla y el mejor clasico. El clasico va en
# negro y de trazo discontinuo porque es la hipotesis nula, no un modelo mas.
COLORES_CURVAS = ("#c44e52", "#4c72b0", "#55a868", "#8172b2")
COLOR_CLASICO = "#111111"


def _filas_de_celda_de_modelo(celda: tuple) -> pd.DataFrame:
    """Las filas por frame de una celda de modelo, de la fuente que le toque.

    Cual le toca lo decide `resultados`, que es el unico que sabe cuantas
    fuentes hay.
    """
    patron, columna, filtros = resultados.fuente_de_celda(celda)
    return _con_dice(resultados.unir(patron), columna, filtros)


def _filas_de_celda_clasica(celda: tuple) -> pd.DataFrame:
    patron, columna, filtros = resultados.fuente_de_celda_clasica(celda)
    return _con_dice(resultados.unir(patron), columna, filtros)


def _con_dice(tabla: pd.DataFrame, columna: str, filtros: dict) -> pd.DataFrame:
    """Las filas de la celda, con su columna de Dice renombrada a `dice`."""
    return metricas.filas(tabla, **filtros).rename(columns={columna: "dice"})[
        ["serie", "etiqueta", "frame", "dice"]
    ]


def lineas_de_las_curvas() -> list[tuple[str, pd.DataFrame]]:
    """Que va en la figura del 6.5: cada modelo de la rejilla en su mejor celda.

    La celda **no se elige a mano**: se le pregunta a `resultados`, que es el
    unico que sabe que las celdas viven en tres ficheros. Y con ellos el mejor
    metodo clasico, que es contra quien se comparan en el 5.4; sin esa linea la
    figura pierde justo la comparacion que sostiene el apartado.
    """
    celdas = resultados.celdas_de_modelos()
    lineas = []
    for modelo in sorted(config.REJILLA_PROFUNDIDAD):
        celda, _ = resultados.mejor({k: v for k, v in celdas.items() if k[0] == modelo})
        etiqueta = f"{celda[0]} {celda[1]} x {celda[2]} x {celda[3]}"
        lineas.append((etiqueta, _filas_de_celda_de_modelo(celda)))
    mejor_clasico = resultados.mejor_celda_de_clasico()
    clave = (mejor_clasico["metodo"], mejor_clasico["protocolo"],
             mejor_clasico["variante"])
    lineas.append((f"{clave[0]} x {clave[1]} x {clave[2]}",
                   _filas_de_celda_clasica(clave)))
    return lineas


# Los umbrales que se leen en el 5.4. El 0.1 es el de los catastroficos y el 0.4
# es donde el clasico se descuelga; los demas van para ver la forma.
UMBRALES_DE_LECTURA = (0.1, 0.2, 0.3, 0.4, 0.5)


def fracciones_en_umbrales(umbrales=UMBRALES_DE_LECTURA) -> pd.DataFrame:
    """La figura en numeros: que fraccion supera cada umbral en cada curva.

    **Existe porque describir un cruce mirando el dibujo ya salio mal una vez.**
    La lectura del 5.4 se apoya en estas filas, no en lo que parece la figura, y
    asi vuelve a salir la misma tabla cada vez que alguien la recalcule.

    La fraccion se cuenta con la misma comparacion estricta que
    `metricas.curva_de_umbrales`, no interpolando entre escalones: la curva es
    escalonada y entre dos escalones no hay nada que interpolar.
    """
    filas = []
    for etiqueta, tabla in lineas_de_las_curvas():
        muestras = {
            "unidad": metricas.por_unidad(tabla).to_numpy(),
            "frame": tabla["dice"].to_numpy(),
        }
        for cual, muestra in muestras.items():
            fila = {"linea": etiqueta, "curva": cual, "n": len(muestra)}
            fila.update({f"t={t}": float((muestra > t).mean()) for t in umbrales})
            filas.append(fila)
    return pd.DataFrame(filas)


def curvas_de_umbral(destino: str | None = None):
    """Las dos curvas del apartado 6.5, que responden preguntas distintas.

    **Por unidad** acompana a la tabla principal: su area es la media reportada
    y su punto t = 0.1 son unidades perdidas enteras. **Por frame** es la curva
    de utilidad: que fraccion de ROIs se aceptaria sin retocar, que es el
    objetivo declarado del plugin, y su area **no coincide** con el numero de la
    tabla.

    No lleva ninguna linea del techo humano, y no por descuido: esa comparacion
    la retiro el apartado 7. Estas curvas son Dice **contra el GT** y el techo es
    acuerdo entre dos personas que no lo vieron, asi que superponerlos seria la
    lectura que ese apartado corrigio con el emparejamiento por solape.
    """
    lineas = lineas_de_las_curvas()
    figura, ejes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    titulos = {"unidad": "por unidad (n = 17)", "frame": "por frame (n = 295)"}
    for plano, cual in zip(ejes, ("unidad", "frame")):
        for i, (etiqueta, tabla) in enumerate(lineas):
            curva = metricas.curvas(tabla)[cual]
            es_clasico = i == len(lineas) - 1
            plano.step(
                curva["umbral"], curva["fraccion"], where="post", label=etiqueta,
                color=COLOR_CLASICO if es_clasico else COLORES_CURVAS[i],
                linestyle="--" if es_clasico else "-",
                linewidth=2 if es_clasico else 1.6,
            )
        plano.axvline(config.UMBRAL_CATASTROFICO, color="#999999", linewidth=1,
                      linestyle=":")
        plano.set_title(titulos[cual])
        plano.set_xlabel("umbral de Dice")
        plano.grid(alpha=0.3)
        plano.set_xlim(0, 1)
    ejes[0].set_ylabel("fraccion que lo supera")
    ejes[0].legend(fontsize=7, loc="upper right")
    figura.suptitle(
        "Fraccion que supera cada umbral. La vertical punteada es t = 0.1, "
        "el punto de los catastroficos"
    )
    figura.tight_layout()
    if destino:
        guardar(figura, destino)
    return figura


def guardar(figura, nombre) -> None:
    """Deja la figura en `salidas/figuras/`, creando la carpeta si hace falta.

    Los notebooks pasan solo el nombre del fichero: donde van las figuras lo
    decide el config, no cada celda que dibuja.
    """
    config.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    figura.savefig(config.DIR_FIGURAS / nombre, dpi=150, bbox_inches="tight")
