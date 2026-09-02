"""Adaptador de SAM 2.1.

Los nombres de configuracion y de pesos estan leidos del paquete instalado
(`sam2.build_sam.HF_MODEL_ID_TO_FILENAMES`), no escritos de memoria.

**M3 pasa por JPEG, y esa perdida esta medida.** El predictor de video de SAM 2
solo acepta como entrada una carpeta de JPEG o un MP4, mientras que el de imagen
recibe el array preparado tal cual, asi que M1 y M3 no ven exactamente lo mismo.
Antes de aceptarlo se corrio M1 entero por las dos vias sobre las 17 unidades, con
el checkpoint que corre en la rejilla (`scripts/unicos/medir_perdida_jpeg.py`): la
diferencia pareada es de **-0.00003 de Dice** de media, con intervalo
[-0.00013, +0.00006], que contiene el cero. La perdida del codec es
indistinguible de cero, asi que se acepta y se declara como limitacion.

El JPEG se escribe a calidad 100 y sin submuestreo de croma; con eso la
diferencia media por pixel es de 0.093 niveles sobre 255.

SAM 2.1 no tiene un checkpoint "por defecto" en su codigo, al contrario que
SAM 1, cuyo registro llama `default` a `vit_h`. Se toma `large`, que es el que
encabeza el README y el que usa la demo.
"""

import numpy as np

from surco import config, modelos

# Pares (configuracion, fichero de pesos) tal como los declara el paquete.
CHECKPOINTS = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
}


@modelos.registrar
class SAM21(modelos.ModeloConMemoria, modelos.ModeloConMascara):
    nombre = "sam21"
    familia = "linaje"
    acepta_negativos = True
    checkpoints = tuple(CHECKPOINTS)
    checkpoint_por_defecto = "large"
    entorno = "surco-sam21"
    lado_entrada = 1024

    def __init__(self, checkpoint: str | None = None) -> None:
        super().__init__(checkpoint)
        self._predictor = None
        self._video = None
        self.dispositivo: str | None = None

    @property
    def configuracion(self) -> str:
        return CHECKPOINTS[self.checkpoint][0]

    @property
    def ruta_pesos(self):
        return config.DIR_PESOS / CHECKPOINTS[self.checkpoint][1]

    def cargar(self) -> None:
        self.olvidar_imagen()
        import torch
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if not self.ruta_pesos.exists():
            raise FileNotFoundError(
                f"faltan los pesos en {self.ruta_pesos}. Se bajan de "
                "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
            )
        self.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        red = build_sam2(
            self.configuracion, str(self.ruta_pesos), device=self.dispositivo
        )
        self._predictor = SAM2ImagePredictor(red)
        self._video = build_sam2_video_predictor(
            self.configuracion, str(self.ruta_pesos), device=self.dispositivo
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
