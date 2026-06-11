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
// When MIG (Multi-Instance GPU) is disabled, each physical GPU is reported as a
// single GpuInfo, as before. When MIG is enabled on a GPU, each of its MIG
// instances is reported as its own GpuInfo, identified by the MIG UUID
// ("MIG-<uuid>"), which the NVIDIA container runtime accepts in DeviceIDs. This
// lets dstack's blocks feature split a MIG-partitioned GPU into N schedulable
// units (e.g., 4x 1g.24gb instances => 4 GPUs).
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
		// Return whatever was collected before the error; partial information
		// is more useful than none.
		log.Error(ctx, "failed to collect NVIDIA GPU info", "err", err)
	}
	return gpus
}

// collectNvidiaGpuInfo walks the NVML device tree and builds GpuInfo entries.
// It is separated from getNvidiaGpuInfo (which owns NVML lifecycle and logging
// of the top-level failure) so it can be unit-tested against a fake nvml.API.
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

		migEnabled, err := device.MIGEnabled()
		if err != nil {
			// MIGEnabled already maps "not supported" to (false, nil), so any
			// error here is unexpected; log it and treat the GPU as non-MIG.
			log.Error(ctx, "failed to query MIG mode", "index", i, "name", name, "uuid", uuid, "err", err)
			migEnabled = false
		}

		if migEnabled {
			migGpus, err := collectNvidiaMIGDevices(ctx, device, name)
			if err != nil {
				// We couldn't enumerate the MIG instances. We must not fall back
				// to reporting the parent as a whole GPU: with MIG enabled,
				// workloads can only run on MIG instances, not the raw device.
				// Skip it so the scheduler never tries to place a container on a
				// GPU it can't actually use.
				log.Error(ctx, "failed to enumerate MIG instances; GPU is unschedulable, skipping",
					"index", i, "name", name, "uuid", uuid, "err", err)
				continue
			}
			if len(migGpus) == 0 {
				// MIG mode is on but no instances are configured. The GPU cannot
				// run any container until it is partitioned (e.g. via
				// `nvidia-smi mig -cgi ... -C`), and its full memory is not
				// allocatable as-is, so report nothing for it.
				log.Warning(ctx, "GPU has MIG mode enabled but no MIG instances configured; it cannot run workloads until partitioned, skipping",
					"index", i, "name", name, "uuid", uuid)
				continue
			}
			gpus = append(gpus, migGpus...)
			continue
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

// collectNvidiaMIGDevices builds a GpuInfo for each MIG instance under device.
// parentName is the physical GPU's product name, reused for each instance so
// MIG entries carry a recognizable GPU name (matching prior behavior).
//
// It returns an error only when the instances cannot be enumerated at all; a
// successful call with zero instances returns (nil, nil), which the caller
// treats as "MIG enabled but unpartitioned". Individual instances missing a
// UUID are skipped, since without a UUID they cannot be passed to the container
// runtime and so cannot be scheduled.
func collectNvidiaMIGDevices(ctx context.Context, device nvml.Device, parentName string) ([]GpuInfo, error) {
	migDevices, err := device.MIGDevices()
	if err != nil {
		return nil, err
	}

	gpus := make([]GpuInfo, 0, len(migDevices))
	for _, mig := range migDevices {
		uuid, err := mig.UUID()
		if err != nil {
			log.Error(ctx, "failed to get MIG instance UUID; skipping instance", "parent", parentName, "err", err)
			continue
		}
		vram, err := mig.MemoryInfo()
		if err != nil {
			log.Error(ctx, "failed to get MIG instance memory", "uuid", uuid, "err", err)
		}
		gpus = append(gpus, GpuInfo{
			Vendor: gpu.GpuVendorNvidia,
			Name:   parentName,
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
