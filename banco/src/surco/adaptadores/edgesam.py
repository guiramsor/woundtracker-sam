"""Adaptador de EdgeSAM.

Es un fork de SAM 1, asi que trae `sam_model_registry` y `SamPredictor` y la
mecanica es la de MobileSAM. Dos diferencias, leidas del paquete instalado:

1. `predict` recibe **`num_multimask_outputs: int`** en lugar del
   `multimask_output: bool` de SAM 1. Se piden tres, que es su valor por
   defecto y lo mismo que devuelven los demas.
2. El registro tiene una sola entrada propia, `edge_sam`, y los dos checkpoints
   publicados se distinguen por el fichero de pesos, no por la arquitectura:
   `edge_sam_3x` es el mismo modelo entrenado mas tiempo.

`edge_sam_3x` es el por defecto porque lo declara el repositorio: su README dice
que la demo espera `edge_sam_3x.pth` salvo que se le diga otra cosa. Aqui no hay
eleccion nuestra, al contrario que en SAM 2.1 o EfficientSAM.

**Este paquete necesita el parche de `scripts/entornos/parchear_edgesam.py`** antes de
poder importarse: el repositorio importa MMDetection en la cabecera de su modelo
para una cabeza de propuestas que no se usa, y una de esas importaciones ni
siquiera se instala con pip. `scripts/entornos/verificar_edgesam.py` demuestra despues que
el camino de prompt de punto no pasa por nada de lo parcheado.
"""

import numpy as np

from surco import config, modelos

TIPO = "edge_sam"
PESOS = {"edge_sam": "edge_sam.pth", "edge_sam_3x": "edge_sam_3x.pth"}


@modelos.registrar
class EdgeSAM(modelos.Modelo):
    nombre = "edgesam"
    familia = "eficiencia"
    acepta_negativos = True
    checkpoints = tuple(PESOS)
    checkpoint_por_defecto = "edge_sam_3x"
    entorno = "surco-edgesam"
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
        from edge_sam import SamPredictor, sam_model_registry

        if not self.ruta_pesos.exists():
            raise FileNotFoundError(
                f"faltan los pesos en {self.ruta_pesos}. Se bajan de "
                "https://huggingface.co/chongzhou/EdgeSAM/"
            )
        self.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        red = sam_model_registry[TIPO](checkpoint=str(self.ruta_pesos))
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
            num_multimask_outputs=3,
        )
        return modelos.Candidatas(
            np.asarray(mascaras, dtype=bool), np.asarray(puntuaciones, dtype=float)
        )
