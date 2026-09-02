"""Ventana de anotacion de los cinco clics. Fase 1, y solo la fase 1.

No hay ningun modelo de por medio y **el GT no se dibuja nunca sobre la
imagen**: se anota mirando solo la microscopia. Esa separacion es el motivo de
partir el picker en dos fases. En el anterior los clics se afinaban contra lo
que proponia el modelo semilla y luego se pasaban identicos a los demas, con lo
que el semilla jugaba en casa; partido en dos, la realimentacion no es que este
prohibida, es que no puede ocurrir.

Se lanza con:

    conda run -n surco python -m surco.picker

Teclas: clic izquierdo pone el punto siguiente, retroceso deshace el ultimo,
`e` marca la unidad como excepcion, `r` la reinicia, `intro` la guarda y avanza,
`s` la salta sin guardar, `q` sale, flechas arriba y abajo ajustan el brillo. La
barra de matplotlib da zoom y desplazamiento; mientras esta activa, los clics no
cuentan como anotacion.

Se guarda tras cada unidad, asi que se puede parar y retomar: al abrir de nuevo
solo se ofrecen las que faltan. Para rehacer una que ya estaba:

    conda run -n surco python -m surco.picker --rehacer 9-L2

Las flechas izquierda y derecha recorren la secuencia de la serie. Sirve para
localizar una lesion por su desarrollo temporal cuando en `start` apenas se ve.
No es contaminacion: la imagen es dato y el GT es respuesta, y es lo que haria
un usuario del plugin con la secuencia delante. Pero **solo se clica sobre
`start`**, y que se haya consultado la secuencia queda registrado en el fichero
como `consulta_temporal`, para poder declarar la desviacion.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from surco import config, datos, inventario, prompts

# Punto blanco de pantalla, solo para ver. No entra en ningun calculo ni se
# guarda. Las flechas lo mueven en vivo.
PERCENTIL_BLANCO = 99.9
PASO_BRILLO = 1.1

COLOR_POSITIVO = "#37d67a"
COLOR_NEGATIVO = "#ff5c5c"

GUION = [
    "1  centro del surco",
    "2  sobre el trazo, a un cuarto",
    "3  sobre el trazo, a tres cuartos",
    "4  fondo en nucleoplasma, lado de arriba",
    "5  fondo, lado opuesto",
]

AVISO_CURVA = (
    "Los clics 2 y 3 van SOBRE EL TRAZO,\n"
    "siguiendo la curva si el surco esta\n"
    "arqueado. No sobre una recta."
)

AVISO_EXCEPCION = (
    "!! ESTA UNIDAD ARRANCA PARTIDA.\n"
    "Si el 2 o el 3 cae en hueco oscuro,\n"
    "deslizalo sobre el trazo hacia el\n"
    "centro hasta encontrar senal, y marca\n"
    "la unidad como excepcion con  e"
)

AVISO_FUERA_DE_START = (
    ">> ESTAS MIRANDO OTRO FRAME.\n"
    "Aqui no se clica: los cinco clics van\n"
    "sobre start. Vuelve con  inicio.\n"
    "Queda registrado que se consulto la\n"
    "secuencia para localizar la lesion."
)

AYUDA = [
    "clic   poner el punto siguiente",
    "retro  deshacer el ultimo",
    "e      marcar excepcion",
    "r      reiniciar la unidad",
    "intro  guardar y avanzar",
    "s      saltar sin guardar",
    "izq/der  moverse por la secuencia",
    "inicio   volver al frame start",
    "arriba/abajo   brillo",
    "q      salir (lo guardado se queda)",
]


def unidades_a_anotar() -> list[dict]:
    """Las 17 unidades con su `start` y si arrancan partidas (medido en F2)."""
    fragmentacion = inventario.fragmentacion_en_start(
        inventario.construir_inventario()
    )
    return [
        {
            "serie": int(fila["serie"]),
            "etiqueta": int(fila["etiqueta"]),
            "start": int(fila["start"]),
            "partida_en_start": bool(fila["partida_en_start"]),
        }
        for _, fila in fragmentacion.iterrows()
    ]


class Picker:
    """Estado de la sesion. La logica de fichero vive en `prompts`."""

    def __init__(
        self,
        unidades: list[dict],
        datos_previos: dict,
        rehacer: list[tuple[int, int]] | None = None,
    ):
        self.unidades = unidades
        self.datos = datos_previos
        claves = [(u["serie"], u["etiqueta"]) for u in unidades]
        if rehacer:
            desconocidas = [u for u in rehacer if u not in claves]
            if desconocidas:
                raise KeyError(f"unidades que no existen: {desconocidas}")
            self.cola = list(rehacer)
        else:
            self.cola = prompts.pendientes(self.datos, claves)
        self.indice = 0
        self.clics: list[tuple[int, int]] = []
        self.excepcion = False
        self.consulta_temporal = False
        self.factor_brillo = 1.0
        self.frame_visible = self.unidad["start"] if self.cola else None
        self._cache: tuple[tuple[int, int], np.ndarray] | None = None

    @property
    def terminado(self) -> bool:
        return self.indice >= len(self.cola)

    @property
    def unidad(self) -> dict:
        clave = self.cola[self.indice]
        return next(u for u in self.unidades if (u["serie"], u["etiqueta"]) == clave)

    def nombre(self) -> str:
        return f"{self.unidad['serie']}-L{self.unidad['etiqueta']}"

    @property
    def en_start(self) -> bool:
        """Solo se clica sobre `start`. Los demas frames son para mirar."""
        return self.frame_visible == self.unidad["start"]

    def mover_frame(self, paso: int) -> None:
        """Navega por la secuencia de la serie. Nunca toca el GT.

        Sirve para localizar una lesion por su desarrollo temporal cuando en
        `start` apenas se ve, que es lo que haria un usuario del plugin con la
        secuencia delante. Que se haya usado queda registrado en el fichero.
        """
        destino = self.frame_visible + paso
        if not 1 <= destino <= config.N_FRAMES:
            return
        self.frame_visible = destino
        if not self.en_start:
            self.consulta_temporal = True

    def volver_a_start(self) -> None:
        self.frame_visible = self.unidad["start"]

    def frame_actual(self) -> np.ndarray:
        """La imagen visible, leida una sola vez por frame."""
        clave = (self.unidad["serie"], self.frame_visible)
        if self._cache is None or self._cache[0] != clave:
            self._cache = (clave, datos.cargar_frame(*clave))
        return self._cache[1]

    def limites(self) -> tuple[float, float]:
        imagen = self.frame_actual()
        blanco = float(np.percentile(imagen, PERCENTIL_BLANCO)) * self.factor_brillo
        negro = float(imagen.min())
        return negro, max(blanco, negro + 1.0)

    def poner_clic(self, fila: float, columna: float) -> None:
        if self.en_start and len(self.clics) < config.N_CLICS:
            self.clics.append((int(round(fila)), int(round(columna))))

    def confirmar(self) -> None:
        """Guarda la unidad en curso en disco y pasa a la siguiente."""
        unidad = self.unidad
        mascara = (
            datos.cargar_gt(unidad["serie"], unidad["start"]) == unidad["etiqueta"]
        )
        self.datos = prompts.registrar(
            self.datos,
            unidad["serie"],
            unidad["etiqueta"],
            unidad["start"],
            self.clics,
            self.excepcion,
            mascara,
            self.consulta_temporal,
        )
        prompts.guardar(self.datos)
        self.avanzar()

    def avanzar(self) -> None:
        self.indice += 1
        self.clics = []
        self.excepcion = False
        self.consulta_temporal = False
        self.frame_visible = None if self.terminado else self.unidad["start"]

    def texto_lateral(self) -> str:
        hechas = len(prompts.anotadas(self.datos))
        lineas = [
            f"unidad {self.nombre()}",
            f"frame de anotacion (start): {self.unidad['start']}",
            f"frame en pantalla:          {self.frame_visible}",
            f"anotadas {hechas} de {len(self.unidades)}",
            "",
        ]
        if not self.en_start:
            lineas += [AVISO_FUERA_DE_START, ""]
        lineas += [
            "ORDEN DE LOS CINCO CLICS",
            *GUION,
            "",
            AVISO_CURVA,
            "",
        ]
        if self.unidad["partida_en_start"]:
            lineas += [AVISO_EXCEPCION, ""]
        lineas += [
            f"puestos: {len(self.clics)} de {config.N_CLICS}",
            f"excepcion: {'SI' if self.excepcion else 'no'}",
            f"consulta temporal: {'SI' if self.consulta_temporal else 'no'}",
            "",
            *AYUDA,
        ]
        return "\n".join(lineas)


class Ventana:
    """La parte de matplotlib: dibuja, escucha teclas y clics, y nada mas."""

    def __init__(self, picker: Picker):
        self.picker = picker
        self.fig = plt.figure(figsize=(14, 9))
        self.ax_imagen = self.fig.add_axes([0.02, 0.03, 0.64, 0.94])
        self.ax_texto = self.fig.add_axes([0.68, 0.03, 0.31, 0.94])
        self.ax_imagen.set_xticks([])
        self.ax_imagen.set_yticks([])
        self.ax_texto.axis("off")
        self.artista = None
        self.marcas: list = []
        self.fig.canvas.mpl_connect("button_press_event", self.al_clicar)
        self.fig.canvas.mpl_connect("key_press_event", self.al_teclear)

    def barra_activa(self) -> bool:
        """Con el zoom o el desplazamiento puestos, un clic no anota."""
        return bool(getattr(getattr(self.fig.canvas, "toolbar", None), "mode", ""))

    def al_clicar(self, evento) -> None:
        fuera = evento.inaxes is not self.ax_imagen or evento.xdata is None
        if fuera or self.barra_activa():
            return
        self.picker.poner_clic(evento.ydata, evento.xdata)
        self.refrescar()

    def al_teclear(self, evento) -> None:
        picker = self.picker
        tecla = evento.key
        if tecla == "backspace" and picker.clics:
            picker.clics.pop()
        elif tecla == "e":
            picker.excepcion = not picker.excepcion
        elif tecla == "r":
            picker.clics, picker.excepcion = [], False
        elif tecla == "enter" and len(picker.clics) == config.N_CLICS:
            picker.confirmar()
        elif tecla == "s":
            picker.avanzar()
        elif tecla in ("up", "down"):
            factor = PASO_BRILLO if tecla == "down" else 1 / PASO_BRILLO
            picker.factor_brillo *= factor
        elif tecla in ("left", "right"):
            picker.mover_frame(-1 if tecla == "left" else 1)
        elif tecla == "home":
            picker.volver_a_start()
        elif tecla == "q":
            plt.close(self.fig)
            return
        self.refrescar()

    def dibujar_clics(self) -> None:
        for marca in self.marcas:
            marca.remove()
        self.marcas = []
        for indice, (fila, columna) in enumerate(self.picker.clics):
            positivo = indice in config.CLICS_POSITIVOS
            color = COLOR_POSITIVO if positivo else COLOR_NEGATIVO
            punto = self.ax_imagen.plot(
                columna, fila, "+", color=color, markersize=16, markeredgewidth=2
            )[0]
            numero = self.ax_imagen.annotate(
                str(indice + 1),
                (columna, fila),
                color=color,
                xytext=(9, 7),
                textcoords="offset points",
                fontsize=12,
            )
            self.marcas += [punto, numero]

    def refrescar(self) -> None:
        if self.picker.terminado:
            plt.close(self.fig)
            return
        imagen = self.picker.frame_actual()
        negro, blanco = self.picker.limites()
        if self.artista is None:
            self.artista = self.ax_imagen.imshow(
                imagen, cmap="gray", vmin=negro, vmax=blanco
            )
        else:
            self.artista.set_data(imagen)
            self.artista.set_clim(negro, blanco)
        self.dibujar_clics()
        self.ax_texto.clear()
        self.ax_texto.axis("off")
        self.ax_texto.text(
            0, 1, self.picker.texto_lateral(), va="top", ha="left",
            family="monospace", fontsize=9,
        )
        self.fig.canvas.draw_idle()

    def abrir(self) -> None:
        self.refrescar()
        plt.show()


def parsear_unidad(texto: str) -> tuple[int, int]:
    """Convierte "9-L2" en (9, 2)."""
    serie, _, etiqueta = texto.strip().upper().partition("-L")
    if not etiqueta:
        raise ValueError(f"unidad mal escrita: {texto!r}. Se espera algo como 9-L2")
    return int(serie), int(etiqueta)


def main(argv: list[str] | None = None) -> None:
    analizador = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    analizador.add_argument(
        "--rehacer",
        nargs="+",
        metavar="UNIDAD",
        help="reabre estas unidades aunque ya esten anotadas, p. ej. 9-L2",
    )
    argumentos = analizador.parse_args(argv)
    rehacer = [parsear_unidad(t) for t in argumentos.rehacer or []]

    unidades = unidades_a_anotar()
    picker = Picker(unidades, prompts.cargar(), rehacer)
    hechas = len(prompts.anotadas(picker.datos))
    print(f"unidades totales : {len(unidades)}")
    print(f"ya anotadas      : {hechas}")
    if rehacer:
        nombres = ", ".join(f"{s}-L{e}" for s, e in rehacer)
        print(f"se reanotan      : {nombres}")
        print("lo anterior de esas unidades se reemplaza al guardar")
    else:
        print(f"pendientes       : {len(picker.cola)}")
    if picker.terminado:
        print(f"\nNo falta ninguna. El fichero esta en {config.RUTA_CLICS}")
        return
    if not rehacer:
        faltan = ", ".join(f"{s}-L{e}" for s, e in picker.cola)
        print(f"por anotar       : {faltan}")
    print()
    Ventana(picker).abrir()
    guardadas = len(prompts.anotadas(prompts.cargar()))
    print(f"guardadas {guardadas} unidades en {config.RUTA_CLICS}")


if __name__ == "__main__":
    main()
