"""Adaptador de MobileSAM.

Primer modelo del banco, elegido por ser el mas facil de instalar: 40 MB de
pesos en un solo fichero y la misma API que SAM 1, asi que este adaptador es
tambien el molde de los que vienen detras.

`torch` y `mobile_sam` se importan **dentro de `cargar()`** y no en la cabecera.
Asi el modulo se puede importar desde el entorno del banco, que no tiene ninguno
de los dos, y la comprobacion de carga reporta "no instalado" como una fila mas
en lugar de reventar. El apartado 11 quiere que un modelo que se resiste sea un
hueco documentado en la tabla.
"""

import numpy as np

from surco import config, modelos

# Los pesos no se versionan: 40 MB que se bajan del repositorio de MobileSAM.
PESOS = {"vit_t": "mobile_sam.pt"}


@modelos.registrar
class MobileSAM(modelos.Modelo):
    nombre = "mobilesam"
    familia = "eficiencia"
    acepta_negativos = True
    checkpoints = ("vit_t",)
    checkpoint_por_defecto = "vit_t"
    entorno = "surco-mobilesam"
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
        from mobile_sam import SamPredictor, sam_model_registry

        if not self.ruta_pesos.exists():
            raise FileNotFoundError(
                f"faltan los pesos en {self.ruta_pesos}. Se bajan de "
                "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
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
