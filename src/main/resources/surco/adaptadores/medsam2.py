"""Adaptador de MedSAM2.

Es un fork de SAM 2, y se instala **ocupando el nombre `sam2`**. Eso trae dos
consecuencias que hay que tener presentes:

1. No puede convivir en el mismo entorno con SAM 2.1, porque los dos serian
   `sam2`. Con un entorno por modelo (apartado 11) no hay choque, pero deja de
   ser una preferencia y pasa a ser una necesidad.
2. **Su wheel no incluye los ficheros de configuracion** que `build_sam2`
   necesita: la carpeta `configs` llega vacia. Por eso se clona el repositorio a
   `SURCO_REPOS` y se antepone a `sys.path`, de modo que el `sam2` que se
   importa es el del clon, que si los trae.

Su lado de entrada es **512**, declarado en `sam2.1_hiera_t512.yaml`.

Los defectos no los elegimos: los declaran sus propios scripts de inferencia,
que traen `checkpoints/MedSAM2_latest.pt` y `configs/sam2.1_hiera_t512.yaml`
como valores por defecto.

Publica cinco checkpoints. Entran **los dos generales**, `latest` y `2411`. Los
otros tres estan afinados a un organo concreto (ecografia de corazon, resonancia
de higado y lesion en TC) y quedan fuera por especificidad de organo, que es un
motivo propio en la tabla de exclusiones del apartado 5.2.

**Tiene banco de memoria**, comprobado propagando sobre una secuencia
sintetica. Es un fork de SAM 2, asi que comparte su predictor de video y la
propagacion sale del mismo sitio: `modelos.propagar_por_jpeg`.
"""

import sys

import numpy as np

from surco import config, modelos

CONFIGURACION = "configs/sam2.1_hiera_t512.yaml"

# Solo los dos generales. Los afinados a organo concreto quedan excluidos por
# especificidad de organo, no por incumplir el criterio.
PESOS = {
    "latest": "MedSAM2_latest.pt",
    "2411": "MedSAM2_2411.pt",
}


@modelos.registrar
class MedSAM2(modelos.ModeloConMemoria):
    nombre = "medsam2"
    familia = "radiologia"
    acepta_negativos = True
    checkpoints = tuple(PESOS)
    checkpoint_por_defecto = "latest"
    entorno = "surco-medsam2"
    lado_entrada = 512

    def __init__(self, checkpoint: str | None = None) -> None:
        super().__init__(checkpoint)
        self._predictor = None
        self._video = None
        self.dispositivo: str | None = None

    @property
    def ruta_repositorio(self):
        return config.DIR_REPOS / "MedSAM2"

    @property
    def ruta_pesos(self):
        return config.DIR_PESOS / PESOS[self.checkpoint]

    def _preparar_ruta(self) -> None:
        """Antepone el clon para que su `sam2`, con configs, gane al instalado."""
        if not self.ruta_repositorio.is_dir():
            raise FileNotFoundError(
                f"falta el clon en {self.ruta_repositorio}. Se obtiene con "
                "git clone --depth 1 https://github.com/bowang-lab/MedSAM2.git"
            )
        ruta = str(self.ruta_repositorio)
        if sys.path[:1] != [ruta]:
            sys.path.insert(0, ruta)

    def cargar(self) -> None:
        self.olvidar_imagen()
        import torch

        self._preparar_ruta()
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if not self.ruta_pesos.exists():
            raise FileNotFoundError(
                f"faltan los pesos en {self.ruta_pesos}. Se bajan de "
                "https://huggingface.co/wanglab/MedSAM2/"
            )
        self.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        red = build_sam2(
            CONFIGURACION, str(self.ruta_pesos), device=self.dispositivo
        )
        self._predictor = SAM2ImagePredictor(red)
        self._video = build_sam2_video_predictor(
            CONFIGURACION, str(self.ruta_pesos), device=self.dispositivo
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

    def propagar(
        self,
        imagenes_rgb: list[np.ndarray],
        puntos: list[tuple[int, int]],
        etiquetas: list[int],
        indice_semilla: int,
    ) -> list[modelos.Candidatas]:
        if self._video is None:
            raise RuntimeError("hay que llamar a cargar() antes de propagar")
        return modelos.propagar_por_jpeg(
            self._video, imagenes_rgb, puntos, etiquetas, indice_semilla
        )
