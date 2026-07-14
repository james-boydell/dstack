// Package nvml is a thin, testable wrapper around NVIDIA's NVML library
// (github.com/NVIDIA/go-nvml, which in turn dynamically loads libnvidia-ml.so).
//
// It exposes only the small subset of NVML that dstack needs for GPU
// discovery, decoupled from the upstream go-nvml types so that the rest of the
// codebase does not depend on go-nvml directly and so that the enumeration
// logic can be unit-tested with fakes.
//
// go-nvml requires cgo. The real implementation lives in nvml_cgo.go (built
// with cgo); nvml_nocgo.go provides a stub for CGO_ENABLED=0 builds (e.g.,
// cross-compilation). New() returns the appropriate implementation for the
// current build.
package nvml

import "errors"

// ErrNotSupported indicates that the requested NVML operation is not supported
// by the current GPU or driver. Callers typically treat this as "feature
// absent" rather than a hard error.
var ErrNotSupported = errors.New("nvml: operation not supported")

// ErrUnavailable indicates that NVML is not available in this build, for
// example because the binary was compiled with CGO_ENABLED=0.
var ErrUnavailable = errors.New("nvml: unavailable (built without cgo)")

// Memory holds framebuffer memory information, in bytes.
type Memory struct {
	Total uint64
	Free  uint64
	Used  uint64
}

// Utilization holds GPU and memory utilization rates, as percentages (0-100).
type Utilization struct {
	GPU    uint32
	Memory uint32
}

// Device represents an NVML device handle for a physical GPU.
type Device interface {
	// Name returns the product name of the device, e.g.
	// "NVIDIA RTX PRO 6000 Blackwell".
	Name() (string, error)
	// UUID returns the globally unique identifier of the device, in the form
	// "GPU-<uuid>".
	UUID() (string, error)
	// MemoryInfo returns the device's framebuffer memory in bytes.
	MemoryInfo() (Memory, error)
	// UtilizationRates returns current GPU/memory utilization. It may return
	// ErrNotSupported on some drivers.
	UtilizationRates() (Utilization, error)
}

// API is the entry point to NVML. Init must be called before any other method,
// and Shutdown should be called (typically deferred) when done.
type API interface {
	Init() error
	Shutdown() error
	// DeviceCount returns the number of physical NVIDIA GPUs on the host.
	DeviceCount() (int, error)
	// DeviceByIndex returns a handle to the physical GPU at the given index,
	// where 0 <= index < DeviceCount().
	DeviceByIndex(index int) (Device, error)
}
