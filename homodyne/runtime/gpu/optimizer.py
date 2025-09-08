#!/usr/bin/env python3
"""
GPU Optimizer for Homodyne
===========================

Intelligent GPU detection, benchmarking, and optimization system.
Automatically configures JAX for optimal performance based on hardware.
"""

import json
import logging
import os
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class GPUOptimizer:
    """Intelligent GPU optimization for homodyne analysis."""

    def __init__(self):
        self.gpu_info = {}
        self.optimal_settings = {}
        self.cache_file = Path.home() / ".cache" / "homodyne" / "gpu_optimization.json"
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

    def detect_gpu_hardware(self) -> dict:
        """Detect GPU hardware and capabilities."""
        info: dict = {
            "available": False,
            "cuda_available": False,
            "devices": [],
            "cuda_version": None,
            "driver_version": None,
            "compute_capability": [],
        }

        # Check for NVIDIA GPU
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,compute_cap",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                info["available"] = True
                for line in result.stdout.strip().split("\n"):
                    parts = line.split(", ")
                    if len(parts) >= 3:
                        info["devices"].append(
                            {
                                "name": parts[0],
                                "memory_mb": int(parts[1].replace(" MiB", "")),
                                "compute_capability": parts[2],
                            }
                        )
                        info["compute_capability"].append(parts[2])
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        # Check CUDA installation
        cuda_paths = ["/usr/local/cuda", "/opt/cuda", os.environ.get("CUDA_HOME", "")]
        for cuda_path in cuda_paths:
            if cuda_path and Path(cuda_path).exists():
                info["cuda_available"] = True
                # Get CUDA version
                nvcc_path = Path(cuda_path) / "bin" / "nvcc"
                if nvcc_path.exists():
                    try:
                        result = subprocess.run(
                            [str(nvcc_path), "--version"],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        if "release" in result.stdout:
                            for line in result.stdout.split("\n"):
                                if "release" in line:
                                    version = (
                                        line.split("release")[1].split(",")[0].strip()
                                    )
                                    info["cuda_version"] = version
                                    break
                    except subprocess.SubprocessError:
                        pass
                break

        # Check JAX GPU support
        try:
            import jax

            devices = jax.devices()
            info["jax_devices"] = [str(d) for d in devices]
            info["jax_gpu_available"] = any(
                "gpu" in str(d).lower() or "cuda" in str(d).lower() for d in devices
            )
        except ImportError:
            info["jax_devices"] = []
            info["jax_gpu_available"] = False

        self.gpu_info = info
        return info

    def benchmark_gpu(self, matrix_sizes: list[int] | None = None) -> dict:
        """Benchmark GPU performance for typical homodyne operations."""
        if matrix_sizes is None:
            matrix_sizes = [100, 500, 1000, 2000]

        benchmarks: dict[str, dict] = {
            "matrix_multiply": {},
            "fft": {},
            "eigenvalue": {},
        }

        try:
            import jax
            import jax.numpy as jnp
            from jax import jit

            # Force GPU if available
            if self.gpu_info.get("jax_gpu_available"):
                devices = jax.devices()
                gpu_device = next(
                    (
                        d
                        for d in devices
                        if "gpu" in str(d).lower() or "cuda" in str(d).lower()
                    ),
                    None,
                )

                if gpu_device:
                    for size in matrix_sizes:
                        # Matrix multiplication benchmark
                        @jit
                        def matmul_bench(x):
                            return jnp.dot(x, x.T)

                        x = jax.device_put(jnp.ones((size, size)), gpu_device)

                        # Warmup
                        _ = matmul_bench(x).block_until_ready()

                        # Benchmark
                        start = time.perf_counter()
                        for _ in range(10):
                            _ = matmul_bench(x).block_until_ready()
                        elapsed = (time.perf_counter() - start) / 10

                        benchmarks["matrix_multiply"][size] = {
                            "time_ms": elapsed * 1000,
                            "gflops": (2 * size**3) / (elapsed * 1e9),
                        }

                        # FFT benchmark
                        @jit
                        def fft_bench(x):
                            return jnp.fft.fft2(x)

                        # Warmup
                        _ = fft_bench(x).block_until_ready()

                        # Benchmark
                        start = time.perf_counter()
                        for _ in range(10):
                            _ = fft_bench(x).block_until_ready()
                        elapsed = (time.perf_counter() - start) / 10

                        benchmarks["fft"][size] = {
                            "time_ms": elapsed * 1000,
                        }

                    # Memory bandwidth test
                    size = 10000
                    x = jax.device_put(
                        jnp.ones((size, size), dtype=jnp.float32), gpu_device
                    )

                    @jit
                    def memory_bench(x):
                        return x + 1.0

                    # Warmup
                    _ = memory_bench(x).block_until_ready()

                    start = time.perf_counter()
                    for _ in range(10):
                        _ = memory_bench(x).block_until_ready()
                    elapsed = (time.perf_counter() - start) / 10

                    # Memory bandwidth in GB/s (read + write)
                    bytes_transferred = 2 * x.nbytes
                    benchmarks["memory_bandwidth_gb_s"] = bytes_transferred / (
                        elapsed * 1e9
                    )

        except Exception as e:
            logger.warning(f"GPU benchmarking failed: {e}")

        return benchmarks

    def determine_optimal_settings(self) -> dict:
        """Determine optimal settings based on hardware and benchmarks."""
        settings: dict = {
            "use_gpu": False,
            "xla_flags": [],
            "jax_settings": {},
            "recommended_batch_size": 1000,
            "memory_fraction": 0.9,
        }

        if not self.gpu_info.get("jax_gpu_available"):
            return settings

        settings["use_gpu"] = True

        # Set XLA flags based on GPU
        if self.gpu_info.get("devices"):
            device = self.gpu_info["devices"][0]
            memory_mb = device.get("memory_mb", 8192)

            # Adjust memory fraction based on available memory
            if memory_mb < 4096:
                settings["memory_fraction"] = 0.7
            elif memory_mb < 8192:
                settings["memory_fraction"] = 0.8
            else:
                settings["memory_fraction"] = 0.9

            # Recommended batch size based on memory
            if memory_mb < 4096:
                settings["recommended_batch_size"] = 500
            elif memory_mb < 8192:
                settings["recommended_batch_size"] = 1000
            elif memory_mb < 16384:
                settings["recommended_batch_size"] = 2000
            else:
                settings["recommended_batch_size"] = 5000

            # XLA optimization flags
            settings["xla_flags"] = [
                f"--xla_gpu_cuda_data_dir={os.environ.get('CUDA_HOME', '/usr/local/cuda')}",
                "--xla_gpu_enable_triton_softmax_fusion=true",
                "--xla_gpu_triton_gemm_any=true",
                "--xla_gpu_enable_async_collectives=true",
                "--xla_gpu_enable_latency_hiding_scheduler=true",
                "--xla_gpu_enable_highest_priority_async_stream=true",
            ]

            # Compute capability specific optimizations
            compute_cap = device.get("compute_capability", "7.0")
            if float(compute_cap) >= 8.0:  # Ampere and newer
                settings["xla_flags"].append("--xla_gpu_enable_triton_gemm=true")
                settings["jax_settings"]["jax_enable_x64"] = True
            elif float(compute_cap) >= 7.0:  # Volta/Turing
                settings["jax_settings"][
                    "jax_enable_x64"
                ] = False  # Use float32 for better performance

        self.optimal_settings = settings
        return settings

    def save_optimization_cache(self):
        """Save optimization results to cache."""
        cache_data = {
            "timestamp": time.time(),
            "gpu_info": self.gpu_info,
            "optimal_settings": self.optimal_settings,
        }

        try:
            with open(self.cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save GPU optimization cache: {e}")

    def load_optimization_cache(self) -> bool:
        """Load cached optimization if recent (within 7 days)."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file) as f:
                    cache_data = json.load(f)

                # Check if cache is recent (7 days)
                if time.time() - cache_data.get("timestamp", 0) < 7 * 24 * 3600:
                    self.gpu_info = cache_data.get("gpu_info", {})
                    self.optimal_settings = cache_data.get("optimal_settings", {})
                    return True
        except Exception:
            pass

        return False

    def apply_optimal_settings(self):
        """Apply optimal settings to environment."""
        if not self.optimal_settings.get("use_gpu"):
            print("ℹ️  GPU optimization not available or not beneficial")
            return

        # Set XLA flags
        if self.optimal_settings.get("xla_flags"):
            os.environ["XLA_FLAGS"] = " ".join(self.optimal_settings["xla_flags"])

        # Set JAX settings
        for key, value in self.optimal_settings.get("jax_settings", {}).items():
            os.environ[key.upper()] = str(value)

        # Set memory fraction
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(
            self.optimal_settings["memory_fraction"]
        )

        print("✅ GPU optimization applied:")
        print(f"   Memory fraction: {self.optimal_settings['memory_fraction']}")
        print(
            f"   Recommended batch size: {self.optimal_settings['recommended_batch_size']}"
        )

    def generate_report(self) -> str:
        """Generate a detailed GPU optimization report."""
        report = []
        report.append("=" * 60)
        report.append("🚀 GPU Optimization Report")
        report.append("=" * 60)

        # Hardware info
        report.append("\n📊 Hardware Detection:")
        if self.gpu_info.get("devices"):
            for i, device in enumerate(self.gpu_info["devices"]):
                report.append(f"   GPU {i}: {device['name']}")
                report.append(f"      Memory: {device['memory_mb']} MB")
                report.append(
                    f"      Compute Capability: {device['compute_capability']}"
                )
        else:
            report.append("   No NVIDIA GPU detected")

        report.append(
            f"\n   CUDA Available: {self.gpu_info.get('cuda_available', False)}"
        )
        if self.gpu_info.get("cuda_version"):
            report.append(f"   CUDA Version: {self.gpu_info['cuda_version']}")

        report.append(
            f"   JAX GPU Support: {self.gpu_info.get('jax_gpu_available', False)}"
        )

        # Optimal settings
        report.append("\n⚙️  Optimal Settings:")
        if self.optimal_settings.get("use_gpu"):
            report.append("   ✅ GPU acceleration recommended")
            report.append(
                f"   Memory fraction: {self.optimal_settings['memory_fraction']}"
            )
            report.append(
                f"   Batch size: {self.optimal_settings['recommended_batch_size']}"
            )

            if self.optimal_settings.get("xla_flags"):
                report.append("\n   XLA Optimizations:")
                for flag in self.optimal_settings["xla_flags"][:3]:  # Show first 3
                    report.append(f"      {flag}")
                if len(self.optimal_settings["xla_flags"]) > 3:
                    report.append(
                        f"      ... and {len(self.optimal_settings['xla_flags']) - 3} more"
                    )
        else:
            report.append("   ❌ GPU acceleration not recommended")
            report.append("   Use CPU mode for better compatibility")

        report.append("\n" + "=" * 60)
        return "\n".join(report)


def main():
    """Main function for GPU optimization CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="GPU Optimizer for Homodyne",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  homodyne-gpu-optimize              # Auto-detect and optimize
  homodyne-gpu-optimize --benchmark  # Run benchmarks
  homodyne-gpu-optimize --apply      # Apply optimal settings
  homodyne-gpu-optimize --report     # Generate detailed report
        """,
    )

    parser.add_argument("--benchmark", action="store_true", help="Run GPU benchmarks")
    parser.add_argument("--apply", action="store_true", help="Apply optimal settings")
    parser.add_argument(
        "--report", action="store_true", help="Generate optimization report"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-detection (ignore cache)"
    )

    args = parser.parse_args()

    optimizer = GPUOptimizer()

    # Load cache or detect hardware
    if not args.force and optimizer.load_optimization_cache():
        print("📦 Loaded cached GPU optimization")
    else:
        print("🔍 Detecting GPU hardware...")
        optimizer.detect_gpu_hardware()

        if args.benchmark and optimizer.gpu_info.get("jax_gpu_available"):
            print("⏱️  Running GPU benchmarks...")
            benchmarks = optimizer.benchmark_gpu()

            if benchmarks.get("matrix_multiply"):
                print("\n📊 Benchmark Results:")
                for size, result in benchmarks["matrix_multiply"].items():
                    print(
                        f"   Matrix {size}x{size}: {result['time_ms']:.2f}ms ({result['gflops']:.1f} GFLOPS)"
                    )

                if "memory_bandwidth_gb_s" in benchmarks:
                    print(
                        f"   Memory Bandwidth: {benchmarks['memory_bandwidth_gb_s']:.1f} GB/s"
                    )

        optimizer.determine_optimal_settings()
        optimizer.save_optimization_cache()

    if args.apply:
        optimizer.apply_optimal_settings()

    if args.report or not any([args.benchmark, args.apply]):
        print(optimizer.generate_report())


if __name__ == "__main__":
    main()
