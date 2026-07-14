package host

import (
	"context"

	"github.com/dstackai/dstack/runner/internal/common/gpu"
	"github.com/dstackai/dstack/runner/internal/common/log"
	"github.com/dstackai/dstack/runner/internal/shim/nvml"
)

// newNVML constructs the NVML API. It is a package variable so tests can
// substitute a fake implementation.
var newNVML = nvml.New

// getNvidiaGpuInfo enumerates NVIDIA GPUs via NVML (github.com/NVIDIA/go-nvml).
//
// NVML is used instead of parsing nvidia-smi output so that detection works
// across arbitrary GPU models and driver versions without relying on the exact
// textual format of the CLI.
func getNvidiaGpuInfo(ctx context.Context) []GpuInfo {
	api := newNVML()
	if err := api.Init(); err != nil {
		log.Error(ctx, "failed to initialize NVML", "err", err)
		return []GpuInfo{}
	}
	defer func() {
		if err := api.Shutdown(); err != nil {
			log.Warning(ctx, "failed to shut down NVML", "err", err)
		}
	}()

	gpus, err := collectNvidiaGpuInfo(ctx, api)
	if err != nil {
		log.Error(ctx, "failed to collect NVIDIA GPU info", "err", err)
	}
	return gpus
}

// collectNvidiaGpuInfo walks the NVML device list and builds GpuInfo entries.
// Separated from getNvidiaGpuInfo (which owns NVML lifecycle/top-level error
// logging) so it can be unit-tested against a fake nvml.API.
func collectNvidiaGpuInfo(ctx context.Context, api nvml.API) ([]GpuInfo, error) {
	gpus := []GpuInfo{}

	count, err := api.DeviceCount()
	if err != nil {
		return gpus, err
	}

	for i := 0; i < count; i++ {
		device, err := api.DeviceByIndex(i)
		if err != nil {
			log.Error(ctx, "failed to get NVIDIA device handle", "index", i, "err", err)
			continue
		}

		name, err := device.Name()
		if err != nil {
			log.Error(ctx, "failed to get NVIDIA device name", "index", i, "err", err)
		}
		uuid, err := device.UUID()
		if err != nil {
			log.Error(ctx, "failed to get NVIDIA device UUID", "index", i, "err", err)
		}

		vram, err := device.MemoryInfo()
		if err != nil {
			log.Error(ctx, "failed to get NVIDIA device memory", "index", i, "name", name, "uuid", uuid, "err", err)
		}

		gpus = append(gpus, GpuInfo{
			Vendor: gpu.GpuVendorNvidia,
			Name:   name,
			Vram:   bytesToMiB(vram.Total),
			ID:     uuid,
		})
	}

	return gpus, nil
}

// bytesToMiB converts a byte count (as reported by NVML) to mebibytes, the unit
// used by GpuInfo.Vram.
func bytesToMiB(b uint64) int {
	return int(b / (1024 * 1024))
}
