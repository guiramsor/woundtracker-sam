"""Tests del contrato de modelos, contra modelos falsos.

Ninguno carga pesos ni toca la GPU: lo que se prueba es el contrato, que es lo
que tiene que estar bien antes de que llegue el primer modelo de verdad.
"""

import numpy as np
import pytest

from surco import config, modelos


def _mascaras(k: int = 3, lado: int = 32) -> np.ndarray:
    mascaras = np.zeros((k, lado, lado), dtype=bool)
    for indice in range(k):
        mascaras[indice, : indice + 2, : indice + 2] = True
    return mascaras


class ModeloFalso(modelos.Modelo):
    nombre = "falso"
    familia = "linaje"
    acepta_negativos = True
    checkpoints = ("pequeno", "mediano", "grande")
    checkpoint_por_defecto = "mediano"
    entorno = "surco-falso"

    def __init__(self, checkpoint: str | None = None, k: int = 3):
        super().__init__(checkpoint)
        self.k = k
        self.cargado = False

    def cargar(self) -> None:
        self.cargado = True

    def segmentar(self, imagen_rgb, puntos, etiquetas):
        alto, ancho = imagen_rgb.shape[:2]
        mascaras = np.zeros((self.k, alto, ancho), dtype=bool)
        for indice in range(self.k):
            fila, columna = puntos[0]
            mascaras[
                indice, fila - indice - 1 : fila + indice + 1, columna - 5 : columna + 5
            ] = True
        return modelos.Candidatas(mascaras, np.linspace(0.5, 0.9, self.k))


class ModeloSinNegativos(ModeloFalso):
    nombre = "falso_sin_negativos"
    acepta_negativos = False
    checkpoints = ("unico",)
    checkpoint_por_defecto = "unico"


# -- candidatas -------------------------------------------------------------


def test_las_candidatas_llevan_una_puntuacion_por_mascara():
    candidatas = modelos.Candidatas(_mascaras(3), np.array([0.1, 0.9, 0.5]))
    assert len(candidatas) == 3


def test_no_se_admiten_menos_puntuaciones_que_mascaras():
    with pytest.raises(ValueError, match="puntuaciones"):
        modelos.Candidatas(_mascaras(3), np.array([0.1, 0.9]))


def test_las_mascaras_tienen_que_venir_apiladas():
    with pytest.raises(ValueError, match="k, alto, ancho"):
        modelos.Candidatas(np.zeros((32, 32), dtype=bool), np.array([1.0]))


def test_las_mascaras_tienen_que_ser_booleanas():
    with pytest.raises(ValueError, match="booleanas"):
        modelos.Candidatas(np.zeros((2, 8, 8), dtype=np.uint8), np.array([1.0, 0.5]))


def test_la_mejor_candidata_es_el_selector_s0():
    candidatas = modelos.Candidatas(_mascaras(3), np.array([0.1, 0.9, 0.5]))
    np.testing.assert_array_equal(candidatas.mejor(), candidatas.mascaras[1])


def test_un_modelo_de_una_sola_candidata_es_valido():
    candidatas = modelos.Candidatas(_mascaras(1), np.array([0.7]))
    assert len(candidatas) == 1
    np.testing.assert_array_equal(candidatas.mejor(), candidatas.mascaras[0])


# -- protocolos admitidos ---------------------------------------------------


def test_quien_admite_negativos_compite_en_los_cuatro_protocolos():
    assert ModeloFalso().protocolos_admitidos() == tuple(config.PROTOCOLOS)


def test_quien_no_admite_negativos_compite_solo_en_p1_y_p3():
    assert ModeloSinNegativos().protocolos_admitidos() == ("P1", "P3")


def test_los_protocolos_admitidos_salen_de_los_indices_y_no_de_una_lista():
    admitidos = ModeloSinNegativos().protocolos_admitidos()
    negativos = set(config.CLICS_NEGATIVOS)
    for nombre in admitidos:
        assert not negativos & set(config.PROTOCOLOS[nombre])
    for nombre in set(config.PROTOCOLOS) - set(admitidos):
        assert negativos & set(config.PROTOCOLOS[nombre])


# -- modos y banco de memoria -----------------------------------------------


class ModeloConBanco(modelos.ModeloConMemoria):
    nombre = "falso_con_memoria"
    familia = "linaje"
    acepta_negativos = True
    checkpoints = ("unico",)
    checkpoint_por_defecto = "unico"
    entorno = "surco-falso"

    def cargar(self) -> None:
        pass

    def segmentar(self, imagen_rgb, puntos, etiquetas):
        alto, ancho = imagen_rgb.shape[:2]
        return modelos.Candidatas(np.ones((1, alto, ancho), dtype=bool), np.array([0.8]))

    def propagar(self, imagenes_rgb, puntos, etiquetas, indice_semilla):
        return [self.segmentar(imagen, puntos, etiquetas) for imagen in imagenes_rgb]


class ModeloConPromptDeMascara(modelos.ModeloConMascara):
    nombre = "falso_con_mascara"
    familia = "linaje"
    acepta_negativos = True
    checkpoints = ("unico",)
    checkpoint_por_defecto = "unico"
    entorno = "surco-falso"

    def cargar(self) -> None:
        pass

    def segmentar(self, imagen_rgb, puntos, etiquetas):
        alto, ancho = imagen_rgb.shape[:2]
        return modelos.Candidatas(np.ones((1, alto, ancho), dtype=bool), np.array([0.8]))

    def segmentar_con_mascara(self, imagen_rgb, puntos, etiquetas, mascara_previa=None):
        candidatas = self.segmentar(imagen_rgb, puntos, etiquetas)
        return candidatas, np.zeros((1, 256, 256), dtype=np.float32)


def test_sin_banco_de_memoria_no_hay_m3():
    modelo = ModeloFalso()
    assert not modelo.soporta_memoria
    assert "M3" not in modelo.modos_admitidos()
    assert modelo.modos_admitidos() == ("M0", "M1")


def test_con_banco_de_memoria_si_hay_m3():
    modelo = ModeloConBanco()
    assert modelo.soporta_memoria
    assert modelo.modos_admitidos() == ("M0", "M1", "M3")


def test_sin_prompt_de_mascara_no_hay_m4():
    modelo = ModeloFalso()
    assert not modelo.soporta_mascara
    assert "M4" not in modelo.modos_admitidos()


def test_con_prompt_de_mascara_si_hay_m4():
    modelo = ModeloConPromptDeMascara()
    assert modelo.soporta_mascara
    assert modelo.modos_admitidos() == ("M0", "M1", "M4")


def test_los_dos_modos_temporales_son_independientes():
    """Un modelo puede tener uno, el otro, los dos o ninguno.

    Es lo que pasa de verdad en la rejilla: SAM 2.1 tiene los dos, los otros
    tres solo M4, y ninguno de los once tiene M3 sin M4.
    """
    assert set(config.MODOS) == {"M0", "M1", "M3", "M4"}
    assert ModeloConBanco().modos_admitidos() != ModeloConPromptDeMascara().modos_admitidos()


def test_la_capacidad_de_m3_se_sabe_por_herencia_y_no_por_un_atributo():
    # Nadie puede declararse capaz de M3 sin implementar propagar: o hereda de
    # ModeloConMemoria, y entonces es abstracto hasta implementarlo, o no esta.
    assert not hasattr(ModeloFalso, "propagar")
    with pytest.raises(TypeError):

        class SinImplementar(modelos.ModeloConMemoria):
            nombre = "sin_implementar"
            familia = "linaje"
            acepta_negativos = True
            checkpoints = ("unico",)
            checkpoint_por_defecto = "unico"

            def cargar(self):
                pass

            def segmentar(self, imagen_rgb, puntos, etiquetas):
                pass

        SinImplementar()


def test_propagar_devuelve_una_entrada_por_frame_incluida_la_semilla():
    secuencia = modelos.secuencia_de_prueba(4, lado=32)
    salida = ModeloConBanco().propagar(secuencia, [(16, 16)], [1], 0)
    assert len(salida) == len(secuencia)
    for candidatas in salida:
        assert candidatas.mascaras.shape[1:] == (32, 32)


def test_segmentar_no_carga_con_parametros_de_secuencia():
    # La propagacion es una segunda operacion, no un parametro de segmentar:
    # los once modelos sin memoria no tienen que saber nada de secuencias.
    import inspect

    firma = inspect.signature(modelos.Modelo.segmentar)
    assert list(firma.parameters) == ["self", "imagen_rgb", "puntos", "etiquetas"]


# -- cache del embedding ----------------------------------------------------


class ModeloQueCuenta(ModeloFalso):
    """Cuenta cuantas veces habria codificado la imagen."""

    nombre = "falso_que_cuenta"

    def __init__(self, checkpoint=None, k=3):
        super().__init__(checkpoint, k)
        self.codificaciones = 0

    def segmentar(self, imagen_rgb, puntos, etiquetas):
        if self.hay_que_codificar(imagen_rgb):
            self.codificaciones += 1
        return super().segmentar(imagen_rgb, puntos, etiquetas)


def test_la_misma_imagen_se_codifica_una_vez():
    modelo = ModeloQueCuenta()
    imagen = modelos.imagen_de_prueba_rgb(64)
    for _ in range(4):
        modelo.segmentar(imagen, [(32, 32)], [1])
    assert modelo.codificaciones == 1, "los cuatro protocolos comparten embedding"


def test_una_imagen_igual_pero_otro_objeto_tampoco_se_recodifica():
    modelo = ModeloQueCuenta()
    modelo.segmentar(modelos.imagen_de_prueba_rgb(64), [(32, 32)], [1])
    modelo.segmentar(modelos.imagen_de_prueba_rgb(64), [(32, 32)], [1])
    assert modelo.codificaciones == 1


def test_una_imagen_distinta_si_se_recodifica():
    modelo = ModeloQueCuenta()
    modelo.segmentar(modelos.imagen_de_prueba_rgb(64), [(32, 32)], [1])
    otra = modelos.imagen_de_prueba_rgb(64).copy()
    otra[0, 0] = [255, 255, 255]
    modelo.segmentar(otra, [(32, 32)], [1])
    assert modelo.codificaciones == 2


def test_otra_forma_se_recodifica():
    modelo = ModeloQueCuenta()
    modelo.segmentar(modelos.imagen_de_prueba_rgb(64), [(32, 32)], [1])
    modelo.segmentar(modelos.imagen_de_prueba_rgb(48), [(24, 24)], [1])
    assert modelo.codificaciones == 2


def test_olvidar_imagen_obliga_a_recodificar():
    modelo = ModeloQueCuenta()
    imagen = modelos.imagen_de_prueba_rgb(64)
    modelo.segmentar(imagen, [(32, 32)], [1])
    modelo.olvidar_imagen()
    modelo.segmentar(imagen, [(32, 32)], [1])
    assert modelo.codificaciones == 2


# -- registro ---------------------------------------------------------------


def test_el_registro_rechaza_una_familia_que_no_existe():
    with pytest.raises(ValueError, match="familia desconocida"):

        @modelos.registrar
        class Raro(ModeloFalso):
            nombre = "raro"
            familia = "inventada"


def test_el_registro_rechaza_un_defecto_que_no_esta_entre_los_checkpoints():
    with pytest.raises(ValueError, match="por defecto"):

        @modelos.registrar
        class Incoherente(ModeloFalso):
            nombre = "incoherente"
            checkpoint_por_defecto = "enorme"


def test_el_registro_exige_declarar_el_entorno():
    with pytest.raises(ValueError, match="falta declarar su entorno"):

        @modelos.registrar
        class SinEntorno(ModeloFalso):
            nombre = "sin_entorno"
            entorno = ""


def test_no_se_puede_pedir_un_adaptador_que_no_existe():
    with pytest.raises(KeyError, match="no hay ningun adaptador"):
        modelos.crear("no_existe")


# -- filtro por entorno -----------------------------------------------------


def test_las_filas_se_pueden_filtrar_por_entorno():
    # Con filtro solo salen las de ese entorno. Es lo que impide que TinySAM,
    # que entra por sys.path y por tanto es importable desde cualquier entorno,
    # ensucie informes ajenos.
    ninguna = modelos.filas(entorno="surco-que-no-existe")
    assert ninguna == []


def test_el_filtro_por_entorno_respeta_el_checkpoint_por_defecto():
    filas = modelos.filas(solo_por_defecto=True, entorno="surco-que-no-existe")
    assert filas == []


# -- checkpoints ------------------------------------------------------------


def test_sin_decir_nada_se_instancia_el_checkpoint_por_defecto():
    modelo = ModeloFalso()
    assert modelo.checkpoint == "mediano"
    assert modelo.es_por_defecto


def test_se_puede_pedir_otro_checkpoint():
    modelo = ModeloFalso("grande")
    assert modelo.checkpoint == "grande"
    assert not modelo.es_por_defecto


def test_no_se_admite_un_checkpoint_que_el_modelo_no_tiene():
    with pytest.raises(ValueError, match="no tiene el checkpoint"):
        ModeloFalso("gigantesco")


def test_el_identificador_lleva_modelo_y_checkpoint():
    assert ModeloFalso("pequeno").identificador == "falso:pequeno"


# -- imagen de prueba -------------------------------------------------------


def test_la_preparacion_da_tres_bandas_de_ocho_bits():
    rgb = modelos.imagen_de_prueba_rgb(64)
    assert rgb.shape == (64, 64, 3)
    assert rgb.dtype == np.uint8


def test_las_tres_bandas_son_la_misma():
    rgb = modelos.imagen_de_prueba_rgb(64)
    np.testing.assert_array_equal(rgb[:, :, 0], rgb[:, :, 1])
    np.testing.assert_array_equal(rgb[:, :, 0], rgb[:, :, 2])


def test_la_preparacion_usa_todo_el_rango_disponible():
    rgb = modelos.imagen_de_prueba_rgb(64)
    assert rgb.min() == 0
    assert rgb.max() == 255


def test_la_preparacion_no_revienta_con_una_imagen_plana():
    plana = np.full((16, 16), 400, dtype=np.uint16)
    rgb = modelos.preparar(plana, (400.0, 400.0))
    assert rgb.shape == (16, 16, 3)
    assert (rgb == 0).all()


def test_el_mismo_frame_con_limites_distintos_da_imagenes_distintas():
    # El motivo de normalizar por serie: un frame debil dentro de una serie
    # brillante tiene que salir oscuro, no estirado hasta el rango entero.
    imagen = modelos.imagen_de_prueba(64)
    propios = modelos.preparar(imagen, (float(imagen.min()), 3000.0))
    de_serie = modelos.preparar(imagen, (float(imagen.min()), 12000.0))
    assert propios.max() == 255
    assert de_serie.max() < 100


def test_dos_frames_de_la_misma_serie_comparten_limites():
    limites = (400.0, 3000.0)
    uno = modelos.preparar(modelos.imagen_de_prueba(64), limites)
    otro = modelos.preparar(np.full((64, 64), 700, dtype=np.uint16), limites)
    # El segundo es mas debil y tiene que verse mas oscuro, no igualado.
    assert otro.max() < uno.max()


def test_la_imagen_de_prueba_tiene_la_forma_del_conjunto():
    imagen = modelos.imagen_de_prueba()
    assert imagen.shape == (config.LADO_PX, config.LADO_PX)
    assert imagen.dtype == np.uint16


def test_el_punto_de_prueba_cae_sobre_la_barra_clara():
    lado = 200
    imagen = modelos.imagen_de_prueba(lado)
    fila, columna = modelos.punto_de_prueba(lado)
    assert imagen[fila, columna] > imagen[0, 0]


# -- el contrato de extremo a extremo ---------------------------------------


def test_un_modelo_que_cumple_el_contrato_devuelve_la_forma_correcta():
    modelo = ModeloFalso()
    modelo.cargar()
    assert modelo.cargado

    imagen = modelos.imagen_de_prueba_rgb(64)
    candidatas = modelo.segmentar(imagen, [modelos.punto_de_prueba(64)], [1])
    assert candidatas.mascaras.shape[1:] == imagen.shape[:2]
    assert candidatas.mascaras.any()


def test_cargar_esta_separado_de_instanciar():
    # Instanciar tiene que ser barato: se listan modelos sin pagar la carga.
    assert ModeloFalso().cargado is False
