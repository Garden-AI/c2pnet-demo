import modal
from typing import TYPE_CHECKING, Any, List
import subprocess

TYPE_CHECKING = False
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
    modal.Image.debian_slim()
    .apt_install(
        "git",
        "git-lfs",
        "build-essential",
        "python3-dev",
    )
    .pip_install(
        "torch>=2.0.0",
        "torchvision",
        "numpy",
        "tensorrt>=8.6.1",
        "torch-tensorrt>=1.4.0",  # Specify minimum version
        "pyyaml",
    )
    .run_function(download_models)
)

class C2PNetBase:
    def __init__(self, config_path: str, is_scaled: bool = True,
                 input_mean: List[float] = [0.0, 0.0, 0.0],
                 input_std: List[float] = [1.0, 1.0, 1.0],
                 output_mean: float = 0.00083617732161656022,
                 output_std: float = 0.00070963409962132573) -> None:
        import torch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_path = None  # Will be set by subclasses
        self.config = self._load_config(config_path)
        self.is_scaled = is_scaled

        # Use hardcoded normalization constants from the C++ inference code
        self.input_mean = torch.tensor(
            input_mean, dtype=torch.float32, device=self.device
        )  # Zero mean for [D, Sx, tau]
        self.input_std = torch.tensor(
            input_std, dtype=torch.float32, device=self.device
        )  # Unit std for [D, Sx, tau]
        self.output_mean = torch.tensor(
            output_mean, dtype=torch.float32, device=self.device
        )
        self.output_std = torch.tensor(
            output_std, dtype=torch.float32, device=self.device
        )

    @modal.enter()
    def _initialize_model(self) -> None:
        """Load model on container startup"""
        if self.model_path is None:
            raise ValueError("model_path must be set by subclass")
        self.load_model(self.model_path)

    def _load_config(self, config_path: str) -> dict[str, Any]:
        import yaml
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def load_model(self, model_path: str) -> None:
        """Load model or TensorRT engine"""
        if model_path.endswith(".engine"):
            import tensorrt as trt

            logger = trt.Logger(trt.Logger.WARNING)
            with open(model_path, "rb") as f, trt.Runtime(logger) as runtime:
                self.model = runtime.deserialize_cuda_engine(f.read())
        else:
            import torch

            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()

    def preprocess(self, input_data: "torch.Tensor") -> "torch.Tensor":
        """Convert input to normalized torch tensor"""
        import torch

        # Convert to tensor if not already
        if not isinstance(input_data, torch.Tensor):
            x = torch.tensor(input_data, dtype=torch.float32, device=self.device)
        else:
            # If already a tensor, ensure it's on the correct device
            x = input_data.to(self.device)

        # Normalize
        x = (x - self.input_mean) / self.input_std
        return x

    def postprocess(self, output: "torch.Tensor") -> "torch.Tensor":
        """Denormalize model output and convert to list"""
        # Denormalize while still in tensor form
        output = output * self.output_std + self.output_mean
        # Convert to list and ensure float type
        # return [[float(val) for val in row] for row in output.cpu().tolist()]
        return output.cpu()

    def _predict(self, input_data: "torch.Tensor") -> "torch.Tensor":
        """Base prediction method"""
        import torch

        if self.model_path.endswith(".engine"):
            import torch_tensorrt

            # Convert input to tensor if it's not already
            if not isinstance(input_data, torch.Tensor):
                input_data = torch.tensor(input_data, dtype=torch.float32, device=self.device)

            # Load and compile model with TensorRT if not already done
            if not hasattr(self, 'engine'):
                # Load the PyTorch model first
                base_model = torch.jit.load(self.model_path.replace('.engine', '.pth'))

                input_dim = input_data.shape[1]  # Assuming input_data is a 2D tensor
                # Compile with TensorRT
                compiled_model = torch_tensorrt.compile(
                    base_model,
                    inputs=[torch_tensorrt.Input(
                        min_shape=[1, input_dim],
                        opt_shape=[32, input_dim],
                        max_shape=[128, input_dim],
                    )],
                    enabled_precisions={torch.float32},  # Run with FP32
                    workspace_size=1 << 28  # 256MiB workspace size
                )
                self.engine = compiled_model

            # Run inference
            with torch.no_grad():
                if not self.is_scaled:
                    input_data = self.preprocess(input_data)
                predictions = self.engine(input_data)
            return self.postprocess(predictions)
        else:
            # Regular PyTorch inference path
            if not isinstance(input_data, torch.Tensor):
                input_data = torch.tensor(input_data, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                if not self.is_scaled:
                    input_data = self.preprocess(input_data)
                predictions = self.model(input_data)
            return self.postprocess(predictions)


@app.cls(gpu="L4", image=image)
class NNC2PS(C2PNetBase):
    def __init__(self) -> None:
        super().__init__("/root/C2PNets/configs/nnc2ps_config.yaml")
        self.model_path = "/root/C2PNets/models/NNC2PS/NNC2PS.pth"

    @modal.method()
    def predict(self, input_data: "torch.Tensor") -> "torch.Tensor":
        """Predict primitive pressure values from conservative variables using the shallow NNC2PS model.

        This model is optimized for piecewise polytropic equations of state (EoS) and uses a shallow
        neural network architecture for fast inference.

        Arguments:
        * `input_data`: (N, c=3) tensor where channels, c, are `[D, Sx, tau]`:
          * `D`: Conserved rest-mass density (D = ρW where ρ is rest-mass density and W is Lorentz factor)
          * `Sx`: Conserved momentum in x-direction
          * `tau`: Conserved energy density minus D

        Returns:
        * (N, 1) tensor with `[p]`:
          * `p`: Primitive pressure value

        Notes:
        * Input values are automatically normalized using pre-computed statistics from training if `is_scaled` is False
        * Output values are automatically denormalized before being returned
        """
        import torch
        # Convert to tensor if not already
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(input_data, dtype=torch.float32, device=self.device)
        else:
            # If already a tensor, ensure it's on the correct device
            input_data = input_data.to(dtype=torch.float32, device=self.device)
        return self._predict(input_data)

@app.cls(gpu="L4", image=image)
class NNC2PS_Engine(C2PNetBase):
    def __init__(self) -> None:
        super().__init__("/root/C2PNets/configs/nnc2ps_config.yaml")
        self.model_path = "/root/C2PNets/models/NNC2PS/NNC2PS.engine"

    @modal.method()
    def predict(self, input_data: "torch.Tensor") -> "torch.Tensor":
        """Predict primitive pressure values from conservative variables using the shallow NNC2PS model.

        This model is optimized for piecewise polytropic equations of state (EoS) and uses a shallow
        neural network architecture for fast inference.

        Arguments:
        * `input_data`: (N, c=3) tensor where channels, c, are `[D, Sx, tau]`:
          * `D`: Conserved rest-mass density (D = ρW where ρ is rest-mass density and W is Lorentz factor)
          * `Sx`: Conserved momentum in x-direction
          * `tau`: Conserved energy density minus D

        Returns:
        * (N, 1) tensor with `[p]`:
          * `p`: Primitive pressure value

        Notes:
        * Input values are automatically normalized using pre-computed statistics from training if `is_scaled` is False
        * Output values are automatically denormalized before being returned
        * This model will be counterintuitively slower due to remote execution overhead-- TRT Engines are normally much faster than PyTorch models
        """
        import torch
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(input_data, dtype=torch.float32, device=self.device)
        else:
            # If already a tensor, ensure it's on the correct device
            input_data = input_data.to(dtype=torch.float32, device=self.device)
        return self._predict(input_data)


@app.cls(gpu="L4", image=image)
class NNC2PL(C2PNetBase):
    def __init__(self) -> None:
        super().__init__("/root/C2PNets/configs/nnc2pl_config.yaml")
        self.model_path = "/root/C2PNets/models/NNC2PL/NNC2PL.pth"

    @modal.method()
    def predict(self, input_data: "torch.Tensor") -> "torch.Tensor":
        """Predict primitive pressure values from conservative variables using the deep NNC2PL model.

        This model extends NNC2PS with a deeper architecture for higher accuracy on piecewise
        polytropic equations of state (EoS).

        Arguments:
        * `input_data`: (N, c=3) tensor where channels, c, are `[D, Sx, tau]`:
          * `D`: Conserved rest-mass density (D = ρW where ρ is rest-mass density and W is Lorentz factor)
          * `Sx`: Conserved momentum in x-direction
          * `tau`: Conserved energy density minus D

        Returns:
        * (N, 1) tensor with `[p]`:
          * `p`: Primitive pressure value

        Notes:
        * Input values are automatically normalized using pre-computed statistics from training if `is_scaled` is False
        * Output values are automatically denormalized before being returned
        * This model may be slower but more accurate than NNC2PS
        """
        import torch
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(input_data, dtype=torch.float32, device=self.device)
        else:
            # If already a tensor, ensure it's on the correct device
            input_data = input_data.to(dtype=torch.float32, device=self.device)
        return self._predict(input_data)

@app.cls(gpu="L4", image=image)
class NNC2PL_Engine(C2PNetBase):
    def __init__(self) -> None:
        super().__init__("/root/C2PNets/configs/nnc2ps_config.yaml")
        self.model_path = "/root/C2PNets/models/NNC2PL/NNC2PL.engine"

    @modal.method()
    def predict(self, input_data: "torch.Tensor") -> "torch.Tensor":
        """Predict primitive pressure values from conservative variables using the shallow NNC2PS model.

        This model is optimized for piecewise polytropic equations of state (EoS) and uses a shallow
        neural network architecture for fast inference.

        Arguments:
        * `input_data`: (N, c=3) tensor where channels, c, are `[D, Sx, tau]`:
          * `D`: Conserved rest-mass density (D = ρW where ρ is rest-mass density and W is Lorentz factor)
          * `Sx`: Conserved momentum in x-direction
          * `tau`: Conserved energy density minus D

        Returns:
        * (N, 1) tensor with `[p]`:
          * `p`: Primitive pressure value

        Notes:
        * Input values are automatically normalized using pre-computed statistics from training if `is_scaled` is False
        * Output values are automatically denormalized before being returned
        * This model will be counterintuitively slower due to remote execution overhead-- TRT Engines are normally much faster than PyTorch models
        """
        import torch
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(input_data, dtype=torch.float32, device=self.device)
        else:
            # If already a tensor, ensure it's on the correct device
            input_data = input_data.to(dtype=torch.float32, device=self.device)
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
        self.output_mean = 30.08158874511718750
        self.output_std = 4.34816455841064453
    @modal.method()
    def predict(self, input_data: "torch.Tensor") -> "torch.Tensor":
        """Predict primitive pressure values from conservative variables using the NNC2PTabulated model.

        This model is specifically designed for tabulated equations of state (EoS) and includes
        electron fraction (ye) as an additional input parameter. The model operates in log space
        for both inputs and outputs.

        Arguments:
        * `input_data`: (N, 4) tensor where channels, c, are `[log10(D), log10(Sx), log10(tau), ye]`:
          * `log10(D)`: Log10 of conserved rest-mass density
          * `log10(Sx)`: Log10 of conserved momentum in x-direction
          * `log10(tau)`: Log10 of conserved energy density minus D
          * `ye`: Electron fraction (linear scale)

        Returns:
        * (N, 1) tensor with `[p]`:
          * `p`: Primitive pressure value (converted back to linear scale)

        Notes:
        * Input values for D, Sx, and tau must be provided in log10 scale
        * The model predicts log10(pressure) which is then converted back to linear scale
        * Input values are normalized using pre-computed statistics from training
        * Output values are denormalized and converted from log space before being returned
        * The model was trained on the LS220 equation of state table
        """
        import torch
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(input_data, dtype=torch.float32, device=self.device)
        else:
            # If already a tensor, ensure it's on the correct device
            input_data = input_data.to(dtype=torch.float32, device=self.device)
        return self._predict(input_data)

@app.cls(gpu="L4", image=image)
class NNC2PTabulated_Engine(C2PNetBase):
    def __init__(self) -> None:
        import torch

        super().__init__("/root/C2PNets/configs/nnc2ps_config.yaml")
        self.model_path = "/root/C2PNets/models/NNC2P_Tabulated/NNC2P_Tabulated.engine"

         # Override normalization for tabulated model
        self.input_mean = torch.tensor(
            [0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device
        )  # Zero mean for [log10(D), log10(Sx), log10(tau), ye]
        self.input_std = torch.tensor(
            [1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=self.device
        )  # Unit std for [log10(D), log10(Sx), log10(tau), ye]
        self.output_mean = 30.08158874511718750
        self.output_std = 4.34816455841064453
    @modal.method()
    def predict(self, input_data: "torch.Tensor") -> "torch.Tensor":
        """Predict primitive pressure values from conservative variables using the shallow NNC2PS model.

        This model is optimized for piecewise polytropic equations of state (EoS) and uses a shallow
        neural network architecture for fast inference.

        Arguments:
        * `input_data`: (N, 4) tensor where channels, c, are `[log10(D), log10(Sx), log10(tau), ye]`:
          * `log10(D)`: Log10 of conserved rest-mass density
          * `log10(Sx)`: Log10 of conserved momentum in x-direction
          * `log10(tau)`: Log10 of conserved energy density minus D
          * `ye`: Electron fraction (linear scale)

        Returns:
        * (N, 1) tensor with `[p]`:
          * `p`: Primitive pressure value (converted back to linear scale)

        Notes:
        * Input values for D, Sx, and tau must be provided in log10 scale
        * The model predicts log10(pressure) which is then converted back to linear scale
        * Input values are normalized using pre-computed statistics from training
        * Output values are denormalized and converted from log space before being returned
        * The model was trained on the LS220 equation of state table
        * This model will be counterintuitively slower due to remote execution overhead-- TRT Engines are normally much faster than PyTorch models
        """
        import torch
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(input_data, dtype=torch.float32, device=self.device)
        else:
            # If already a tensor, ensure it's on the correct device
            input_data = input_data.to(dtype=torch.float32, device=self.device)
        return self._predict(input_data)


@app.local_entrypoint()
def main() -> None:
    import random
    import torch
    import numpy as np

    n_samples = 5
    # Load data
    input_data = torch.from_numpy(
        np.loadtxt("../C2PNets/src/speed_test/gpu/inputs_test_scaled_m_pen350.txt")[:n_samples]
    )
    input_data_tabulated = torch.from_numpy(
        np.loadtxt("../C2PNets/src/speed_test/gpu/inputs_test_scaled_tabulated.txt")[:n_samples]
    )

    # Initialize models
    c2ps = NNC2PS()
    c2ps_engine = NNC2PS_Engine()
    c2pl = NNC2PL()
    c2pl_engine = NNC2PL_Engine()
    c2pt = NNC2PTabulated()
    c2pt_engine = NNC2PTabulated_Engine()

    # Run predictions
    r1 = c2ps.predict.remote(input_data)
    r1e = c2ps_engine.predict.remote(input_data)
    r2 = c2pl.predict.remote(input_data)
    r2e = c2pl_engine.predict.remote(input_data)
    r3 = c2pt.predict.remote(input_data_tabulated)
    r3e = c2pt_engine.predict.remote(input_data_tabulated)

    # Print results
    print("C2PS Results:", r1)
    print("C2PSEngine Results:", r1e)
    print("C2PL Results:", r2)
    print("C2PLEngine Results:", r2e)
    print("C2PTabulated Results:", r3)
    print("C2PTabulatedEngine Results:", r3e)
