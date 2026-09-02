"""Adaptador de SAM 1.

El molde del que salen los demas: `sam_model_registry` y `SamPredictor`, la
mecanica que ya se ha visto en MobileSAM, EdgeSAM, TinySAM, micro-sam y
SAM-Med2D. Sin sorpresas, que es exactamente por lo que se dejo casi para el
final del orden de F6.

Es el unico de los once cuyo registro **declara un checkpoint por defecto de
forma explicita**: `sam_model_registry["default"]` es `build_sam_vit_h`. En los
demas, o hay uno solo, o el defecto sale del README, o lo elegimos nosotros y
queda declarado como eleccion nuestra en el apartado 5.2.
"""

import numpy as np

from surco import config, modelos

PESOS = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}


@modelos.registrar
class SAM1(modelos.Modelo):
    nombre = "sam1"
    familia = "linaje"
    acepta_negativos = True
    checkpoints = tuple(PESOS)
    checkpoint_por_defecto = "vit_h"
    entorno = "surco-sam1"
    lado_entrada = 1024

    def __init__(self, checkpoint: str | None = None) -> None:
        super().__init__(checkpoint)
        self._predictor = None
        self.dispositivo: str | None = None

    @property
    def ruta_pesos(self):
        return config.DIR_PESOS / PESOS[self.checkpoint]

    def cargar(self) -> None:
        self.olvidar_imagen()
        import torch
        from segment_anything import SamPredictor, sam_model_registry

        if not self.ruta_pesos.exists():
            raise FileNotFoundError(
                f"faltan los pesos en {self.ruta_pesos}. Se bajan de "
                "https://dl.fbaipublicfiles.com/segment_anything/"
            )
        self.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        red = sam_model_registry[self.checkpoint](checkpoint=str(self.ruta_pesos))
        red.to(self.dispositivo)
        red.eval()
        self._predictor = SamPredictor(red)

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
