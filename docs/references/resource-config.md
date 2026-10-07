# Hardware Resource Configuration

Marin dispatches jobs through Fray. Fray's only backend, `LocalClient`, runs each job on the current machine. A `ResourceConfig` describes the hardware a job expects; the local backend does not check it against the machine, but `train_lm` reads its device type to set up the training environment.

## ResourceConfig

The main entry point for resource configuration. Use the static factory methods to create configurations:

```python
from poorin_fray.cluster import ResourceConfig

# TPU configuration
tpu_config = ResourceConfig.with_tpu("v4-8")
tpu_multislice = ResourceConfig.with_tpu("v4-8", slice_count=2)

# GPU configuration
gpu_config = ResourceConfig.with_gpu("H100", count=8)
single_gpu = ResourceConfig.with_gpu("H100")  # count defaults to 1

# CPU-only configuration
cpu_config = ResourceConfig.with_cpu()
```

::: poorin_fray.types.ResourceConfig

## Device Configurations

These are the underlying device types wrapped by `ResourceConfig`:

### CPU

::: poorin_fray.types.CpuConfig

### GPU

::: poorin_fray.types.GpuConfig

### TPU

::: poorin_fray.types.TpuConfig
