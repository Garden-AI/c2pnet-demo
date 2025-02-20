import modal
import yaml
from typing import TYPE_CHECKING, Any
import subprocess

if TYPE_CHECKING:
    import torch
    import numpy as np
    from torch import Tensor

app = modal.App("C2PNets")


def download_models():
    # Initialize git-lfs
    _ = subprocess.run(["git", "lfs", "install"], check=True)

    # Clone repository
    _ = subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/semihkacmaz/C2PNets.git",
            "/root/C2PNets",
        ],
        check=True,
    )

    # Change to repository directory and pull LFS files
    _ = subprocess.run(["git", "lfs", "pull"], cwd="/root/C2PNets", check=True)


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "git-lfs")
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "pyyaml",
    )
    .run_function(download_models)
)


class C2PNetBase:
    def __init__(self, config_path: str) -> None:
        import torch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_path = None  # Will be set by subclasses
        self.config = self._load_config(config_path)

        # Use hardcoded normalization constants from the C++ inference code
        self.input_mean = torch.tensor(
            [0.0, 0.0, 0.0], dtype=torch.float32, device=self.device
        )  # Zero mean for [D, Sx, tau]
        self.input_std = torch.tensor(
            [1.0, 1.0, 1.0], dtype=torch.float32, device=self.device
        )  # Unit std for [D, Sx, tau]
        self.output_mean = torch.tensor(
            0.00083617732161656022, dtype=torch.float32, device=self.device
        )
        self.output_std = torch.tensor(
            0.00070963409962132573, dtype=torch.float32, device=self.device
        )

    @modal.enter()
    def _initialize_model(self) -> None:
        """Load model on container startup"""
        if self.model_path is None:
            raise ValueError("model_path must be set by subclass")
        self.load_model(self.model_path)

    def _load_config(self, config_path: str) -> dict[str, Any]:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def load_model(self, model_path: str) -> None:
        """Load TorchScript model"""
        import torch

        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

    def _to_float_list(self, data: Any) -> list[list[float]]:
        """Convert input data to list of lists of floats for serialization"""
        # If already a list of lists, verify and convert elements to float
        if isinstance(data, list) and all(isinstance(x, list) for x in data):
            return [[float(val) for val in row] for row in data]
        # If numpy array or torch tensor, convert to list of lists
        if hasattr(data, "tolist"):  # Handles numpy arrays and torch tensors
            return [[float(val) for val in row] for row in data.tolist()]
        raise ValueError("Input must be a list of lists or numpy-like array")

    def preprocess(self, input_data: list[list[float]]) -> "torch.Tensor":
        """Convert input to normalized torch tensor"""
        import torch

        # Convert to tensor and normalize
        x = torch.tensor(input_data, dtype=torch.float32, device=self.device)
        x = (x - self.input_mean) / self.input_std
        return x

    def postprocess(self, output: "torch.Tensor") -> list[list[float]]:
        """Denormalize model output and convert to list"""
        # Denormalize while still in tensor form
        output = output * self.output_std + self.output_mean
        # Convert to list and ensure float type
        return [[float(val) for val in row] for row in output.cpu().tolist()]

    def _predict(self, input_data: list[list[float]]) -> list[list[float]]:
        """Base prediction method"""
        import torch

        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Ensure input is in the correct format
        input_data = self._to_float_list(input_data)

        with torch.no_grad():
            x = self.preprocess(input_data)
            y = self.model(x)
            return self.postprocess(y)


@app.cls(gpu="L4", image=image)
class NNC2PS(C2PNetBase):
    def __init__(self) -> None:
        super().__init__("/root/C2PNets/configs/nnc2ps_config.yaml")
        self.model_path = "/root/C2PNets/models/NNC2PS/NNC2PS.pth"

    @modal.method()
    def predict(self, input_data: list[list[float]]) -> list[list[float]]:
        """Predict primitive pressure values from conservative variables using the shallow NNC2PS model.

        This model is optimized for piecewise polytropic equations of state (EoS) and uses a shallow
        neural network architecture for fast inference.

        Arguments:
        * `input_data`: A list of lists where each inner list contains 3 float values `[D, Sx, tau]`:
          * `D`: Conserved rest-mass density (D = ρW where ρ is rest-mass density and W is Lorentz factor)
          * `Sx`: Conserved momentum in x-direction
          * `tau`: Conserved energy density minus D

        Returns:
        * A list of lists where each inner list contains a single float value `[p]`:
          * `p`: Primitive pressure value

        Notes:
        * Input values are automatically normalized using pre-computed statistics from training
        * Output values are automatically denormalized before being returned
        """
        return self._predict(input_data)


@app.cls(gpu="L4", image=image)
class NNC2PL(C2PNetBase):
    def __init__(self) -> None:
        super().__init__("/root/C2PNets/configs/nnc2pl_config.yaml")
        self.model_path = "/root/C2PNets/models/NNC2PL/NNC2PL.pth"

    @modal.method()
    def predict(self, input_data: list[list[float]]) -> list[list[float]]:
        """Predict primitive pressure values from conservative variables using the deep NNC2PL model.

        This model extends NNC2PS with a deeper architecture for higher accuracy on piecewise
        polytropic equations of state (EoS).

        Arguments:
        * `input_data`: A list of lists where each inner list contains 3 float values `[D, Sx, tau]`:
          * `D`: Conserved rest-mass density (D = ρW where ρ is rest-mass density and W is Lorentz factor)
          * `Sx`: Conserved momentum in x-direction
          * `tau`: Conserved energy density minus D

        Returns:
        * A list of lists where each inner list contains a single float value `[p]`:
          * `p`: Primitive pressure value

        Notes:
        * Input values are automatically normalized using pre-computed statistics from training
        * Output values are automatically denormalized before being returned
        * This model may be slower but more accurate than NNC2PS
        """
        return self._predict(input_data)


@app.cls(gpu="L4", image=image)
class NNC2PTabulated(C2PNetBase):
    def __init__(self) -> None:
        import torch

        super().__init__("/root/C2PNets/configs/nnc2p_tabulated_config.yaml")
        self.model_path = "/root/C2PNets/models/NNC2P_Tabulated/NNC2P_Tabulated.pth"
        # Override normalization for tabulated model
        self.input_mean = torch.tensor(
            [0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device
        )  # Zero mean for [log10(D), log10(Sx), log10(tau), ye]
        self.input_std = torch.tensor(
            [1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=self.device
        )  # Unit std for [log10(D), log10(Sx), log10(tau), ye]

    @modal.method()
    def predict(self, input_data: list[list[float]]) -> list[list[float]]:
        """Predict primitive pressure values from conservative variables using the NNC2PTabulated model.

        This model is specifically designed for tabulated equations of state (EoS) and includes
        electron fraction (ye) as an additional input parameter. The model operates in log space
        for both inputs and outputs.

        Arguments:
        * `input_data`: A list of lists where each inner list contains 4 float values `[log10(D), log10(Sx), log10(tau), ye]`:
          * `log10(D)`: Log10 of conserved rest-mass density
          * `log10(Sx)`: Log10 of conserved momentum in x-direction
          * `log10(tau)`: Log10 of conserved energy density minus D
          * `ye`: Electron fraction (linear scale)

        Returns:
        * A list of lists where each inner list contains a single float value `[p]`:
          * `p`: Primitive pressure value (converted back to linear scale)

        Notes:
        * Input values for D, Sx, and tau must be provided in log10 scale
        * The model predicts log10(pressure) which is then converted back to linear scale
        * Input values are normalized using pre-computed statistics from training
        * Output values are denormalized and converted from log space before being returned
        * The model was trained on the LS220 equation of state table
        """
        return self._predict(input_data)


@app.local_entrypoint()
def main() -> None:
    # Generate test data based on the physics model
    n_samples = 1000

    # Constants from the data generation notebook
    rho_min = 2e-5
    rho_max = 2e-3
    ye_value = 0.5  # Electron fraction for tabulated model

    # Use pure Python for data generation to avoid numpy serialization issues
    import random

    random.seed(42)  # For reproducibility

    rho = [rho_min + (rho_max - rho_min) * random.random() for _ in range(n_samples)]
    vx = [0.721 * random.random() for _ in range(n_samples)]
    W = [1 / (1 - v**2) ** 0.5 for v in vx]  # Lorentz factor

    # Calculate conserved variables
    D = [r * w for r, w in zip(rho, W)]
    Sx = [r * w * w * v for r, w, v in zip(rho, W, vx)]  # Simplified without h
    tau = [r * w * w - r * w for r, w in zip(rho, W)]  # Simplified without h and p

    # Prepare input data for standard models [D, Sx, tau]
    input_data = [[d, sx, t] for d, sx, t in zip(D, Sx, tau)]

    # Prepare input data for tabulated model [log10(D), log10(Sx), log10(tau), ye]
    # Use math.log10 instead of np.log10 to avoid serialization issues
    from math import log10

    input_data_tabulated = [
        [log10(abs(d)), log10(abs(sx)), log10(abs(t)), ye_value]
        for d, sx, t in zip(D, Sx, tau)
    ]

    # Initialize models
    c2ps = NNC2PS()
    c2pl = NNC2PL()
    c2pt = NNC2PTabulated()

    # Run predictions
    r1 = c2ps.predict.remote(input_data)
    r2 = c2pl.predict.remote(input_data)
    r3 = c2pt.predict.remote(input_data_tabulated)

    # Print results
    print("C2PS Results:", r1)
    print("C2PL Results:", r2)
    print("C2PTabulated Results:", r3)
