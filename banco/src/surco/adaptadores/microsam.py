"""Adaptadores de micro-sam: dos modelos, LM y EM organelles.

Son dos entradas distintas de la lista congelada, no dos checkpoints del mismo
modelo: estan ajustados a dominios distintos y el apartado 5.2 los cuenta por
separado dentro de la familia de microscopia.

Los nombres estan leidos de `micro_sam.util.models()` en la version instalada,
no escritos de memoria. Cada uno trae tres tamanos, `vit_t`, `vit_b` y `vit_l`,
y el por defecto es `vit_b`, que es tambien el que `get_sam_model` toma cuando
no se le dice nada.

Las entradas terminadas en `_decoder` que aparecen en el registro **no** son
checkpoints aparte: son el decodificador de segmentacion automatica de
instancias, que no se prompteapor punto y por tanto queda fuera del criterio
del apartado 5.2.

`get_sam_model` devuelve un `segment_anything.predictor.SamPredictor`, asi que
la mecanica es la misma que la de SAM 1 y MobileSAM.
"""

import os

import numpy as np

from surco import config, modelos

TAMANOS = ("vit_t", "vit_b", "vit_l")
POR_DEFECTO = "vit_b"


class _MicroSAM(modelos.ModeloConMascara):
    """Lo comun a las dos variantes. El sufijo distingue el dominio."""

    familia = "microscopia"
    acepta_negativos = True
    checkpoints = TAMANOS
    checkpoint_por_defecto = POR_DEFECTO
    entorno = "surco-microsam"
    lado_entrada = 1024
    sufijo: str

    def __init__(self, checkpoint: str | None = None) -> None:
        super().__init__(checkpoint)
        self._predictor = None
        self.dispositivo: str | None = None

    @property
    def tipo_de_modelo(self) -> str:
        """El nombre tal como lo conoce micro-sam, p. ej. `vit_b_lm`."""
        return f"{self.checkpoint}{self.sufijo}"

    def cargar(self) -> None:
        self.olvidar_imagen()
        import torch

        # micro-sam se baja los pesos a su propia cache, que por defecto cae en
        # AppData. Se redirige a SURCO_PESOS para que todo el peso del trabajo
        # viva en el mismo sitio y no repartido por el disco del sistema.
        os.environ.setdefault(
            "MICROSAM_CACHEDIR", str(config.DIR_PESOS / "micro_sam")
        )
        from micro_sam import util

        self.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        self._predictor = util.get_sam_model(
            model_type=self.tipo_de_modelo, device=self.dispositivo
        )

    def segmentar(
        self,
        imagen_rgb: np.ndarray,
        puntos: list[tuple[int, int]],
        etiquetas: list[int],
    ) -> modelos.Candidatas:
        if self._predictor is None:
            raise RuntimeError("hay que llamar a cargar() antes de segmentar")

        if self.hay_que_codificar(imagen_rgb):
            self._predictor.set_image(imagen_rgb)
        # Aqui todo va en (fila, columna); SAM espera (x, y) = (columna, fila).
        coordenadas = np.array([[col, fila] for fila, col in puntos], dtype=float)
        mascaras, puntuaciones, _ = self._predictor.predict(
            point_coords=coordenadas,
            point_labels=np.array(etiquetas, dtype=int),
            multimask_output=True,
        )
        return modelos.Candidatas(
            np.asarray(mascaras, dtype=bool), np.asarray(puntuaciones, dtype=float)
        )

    def segmentar_con_mascara(
        self,
        imagen_rgb: np.ndarray,
        puntos: list[tuple[int, int]],
        etiquetas: list[int],
        mascara_previa: np.ndarray | None = None,
    ) -> tuple[modelos.Candidatas, np.ndarray]:
        return modelos.encadenar_por_predict(
            self, self._predictor, imagen_rgb, puntos, etiquetas, mascara_previa
        )


@modelos.registrar
class MicroSAMLM(_MicroSAM):
    """Ajustado a microscopia optica: celulas, nucleos y organoides."""

    nombre = "microsam_lm"
    sufijo = "_lm"


@modelos.registrar
class MicroSAMEM(_MicroSAM):
    """Ajustado a microscopia electronica de organulos."""

    nombre = "microsam_em"
    sufijo = "_em_organelles"
