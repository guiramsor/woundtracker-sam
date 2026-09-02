"""Adaptador de TinySAM.

Es el unico de los trece que **no se instala**: su repositorio no trae setup.py
ni pyproject.toml, asi que se clona a `SURCO_REPOS` y este adaptador mete esa
ruta en `sys.path` dentro de `cargar()`. Por eso no hay entrada suya en pip y el
entorno documenta el `git clone` como un paso mas.

Por dentro es un fork de SAM 1, con `sam_model_registry` y `SamPredictor`, asi
que la mecanica es la de MobileSAM. La diferencia esta en el segundo checkpoint:

- `vit_t` es TinySAM y se carga por el registro, como cualquier fork;
- `w8a8` es Q-TinySAM, la version cuantizada a 8 bits, y **es el modelo entero
  encurtido**, no un `state_dict`. Se carga con `torch.load` y se le pasa
  directamente a `SamPredictor`, que es lo que hace su propio `demo_quant.py`.

Ese segundo camino necesita `weights_only=False`, porque desencurtir un modelo
completo no es leer tensores. Es la misma confianza que ya se le da al fichero
de pesos al bajarlo de su repositorio oficial, pero conviene que se vea escrito.
"""

import sys

import numpy as np

from surco import config, modelos

PESOS = {"vit_t": "tinysam_42.3.pth", "w8a8": "tinysam_w8a8.pth"}
CUANTIZADO = "w8a8"


@modelos.registrar
class TinySAM(modelos.ModeloConMascara):
    nombre = "tinysam"
    familia = "eficiencia"
    acepta_negativos = True
    checkpoints = tuple(PESOS)
    checkpoint_por_defecto = "vit_t"
    entorno = "surco-tinysam"
    lado_entrada = 1024

    def __init__(self, checkpoint: str | None = None) -> None:
        super().__init__(checkpoint)
        self._predictor = None
        self.dispositivo: str | None = None

    @property
    def ruta_repositorio(self):
        return config.DIR_REPOS / "TinySAM"

    @property
    def ruta_pesos(self):
        return config.DIR_PESOS / PESOS[self.checkpoint]

    def _preparar_ruta(self) -> None:
        """Mete el clon en sys.path. No se instala porque no es instalable.

        Van **dos** rutas, no una. La del repositorio hace importable `tinysam`.
        La de dentro hace importable `quantization_layer` como modulo de primer
        nivel, que es como lo referencia el modelo encurtido de Q-TinySAM: en el
        arbol cuelga de `tinysam/`, pero al desencurtir se busca arriba del
        todo. Sin la segunda, `w8a8` no carga.
        """
        if not self.ruta_repositorio.is_dir():
            raise FileNotFoundError(
                f"falta el clon en {self.ruta_repositorio}. Se obtiene con "
                "git clone --depth 1 https://github.com/xinghaochen/TinySAM.git"
            )
        for ruta in (self.ruta_repositorio, self.ruta_repositorio / "tinysam"):
            if str(ruta) not in sys.path:
                sys.path.insert(0, str(ruta))

    def cargar(self) -> None:
        self.olvidar_imagen()
        import torch

        self._preparar_ruta()
        from tinysam import SamPredictor, sam_model_registry

        if not self.ruta_pesos.exists():
            raise FileNotFoundError(
                f"faltan los pesos en {self.ruta_pesos}. Estan en las releases "
                "de https://github.com/xinghaochen/TinySAM"
            )
        self.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"

        if self.checkpoint == CUANTIZADO:
            # El fichero es el modelo entero encurtido, no un state_dict.
            red = torch.load(self.ruta_pesos, weights_only=False)
        else:
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
        # Su `predict` **no tiene** `multimask_output`: llama al decodificador
        # sin ese argumento, asi que devuelve lo que el decodificador decida.
        # Cuantas candidatas salen es propiedad del modelo, no algo que se le
        # pida, y por eso aqui no se fuerza nada.
        mascaras, puntuaciones, _ = self._predictor.predict(
            point_coords=coordenadas,
            point_labels=np.array(etiquetas, dtype=int),
        )
        if mascaras.ndim == 2:  # una sola candidata, sin eje de mascara
            mascaras = mascaras[None]
            puntuaciones = np.atleast_1d(puntuaciones)
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
        # `multimask=False` porque su `predict` no declara ese argumento, igual
        # que en `segmentar`: cuantas candidatas salen lo decide el modelo.
        return modelos.encadenar_por_predict(
            self, self._predictor, imagen_rgb, puntos, etiquetas, mascara_previa,
            multimask=False,
        )
