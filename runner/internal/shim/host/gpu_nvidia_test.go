package host

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/dstackai/dstack/runner/internal/common/gpu"
	"github.com/dstackai/dstack/runner/internal/shim/nvml"
)

const gib = uint64(1024 * 1024 * 1024)

// fakeDevice is an in-memory nvml.Device used to exercise the enumeration
// logic without a real NVIDIA library or hardware.
type fakeDevice struct {
	name string
	uuid string
	mem  nvml.Memory
	util nvml.Utilization

	nameErr error
	uuidErr error
	memErr  error
}

func (d *fakeDevice) Name() (string, error)                       { return d.name, d.nameErr }
func (d *fakeDevice) UUID() (string, error)                       { return d.uuid, d.uuidErr }
func (d *fakeDevice) MemoryInfo() (nvml.Memory, error)            { return d.mem, d.memErr }
func (d *fakeDevice) UtilizationRates() (nvml.Utilization, error) { return d.util, nil }

// fakeAPI is an in-memory nvml.API.
type fakeAPI struct {
	initErr        error
	countErr       error
	devices        []nvml.Device
	shutdownCalled bool
}

func (a *fakeAPI) Init() error     { return a.initErr }
func (a *fakeAPI) Shutdown() error { a.shutdownCalled = true; return nil }
func (a *fakeAPI) DeviceCount() (int, error) {
	if a.countErr != nil {
		return 0, a.countErr
	}
	return len(a.devices), nil
}
func (a *fakeAPI) DeviceByIndex(i int) (nvml.Device, error) {
	if i < 0 || i >= len(a.devices) {
		return nil, errors.New("index out of range")
	}
	return a.devices[i], nil
}

func physicalDevice(name, uuid string, vramGiB uint64) *fakeDevice {
	return &fakeDevice{
		name: name,
		uuid: uuid,
		mem:  nvml.Memory{Total: vramGiB * gib},
	}
}

func TestCollectNvidiaGpuInfo_Basic(t *testing.T) {
	api := &fakeAPI{devices: []nvml.Device{
		physicalDevice("NVIDIA H100", "GPU-1111", 80),
		physicalDevice("NVIDIA H100", "GPU-2222", 80),
	}}

	gpus, err := collectNvidiaGpuInfo(context.Background(), api)
	require.NoError(t, err)

	expected := []GpuInfo{
		{Vendor: gpu.GpuVendorNvidia, Name: "NVIDIA H100", Vram: 80 * 1024, ID: "GPU-1111"},
		{Vendor: gpu.GpuVendorNvidia, Name: "NVIDIA H100", Vram: 80 * 1024, ID: "GPU-2222"},
	}
	assert.Equal(t, expected, gpus)
}

func TestCollectNvidiaGpuInfo_DeviceCountError(t *testing.T) {
	api := &fakeAPI{countErr: errors.New("boom")}
	gpus, err := collectNvidiaGpuInfo(context.Background(), api)
	require.Error(t, err)
	assert.Empty(t, gpus)
}

func TestGetNvidiaGpuInfo_InitFailureReturnsEmpty(t *testing.T) {
	orig := newNVML
	t.Cleanup(func() { newNVML = orig })

	api := &fakeAPI{initErr: errors.New("driver not loaded")}
	newNVML = func() nvml.API { return api }

	gpus := getNvidiaGpuInfo(context.Background())
	assert.Empty(t, gpus)
}

func TestGetNvidiaGpuInfo_ShutsDownNVML(t *testing.T) {
	orig := newNVML
	t.Cleanup(func() { newNVML = orig })

	api := &fakeAPI{devices: []nvml.Device{physicalDevice("NVIDIA L4", "GPU-L4", 24)}}
	newNVML = func() nvml.API { return api }

	gpus := getNvidiaGpuInfo(context.Background())
	require.Len(t, gpus, 1)
	assert.Equal(t, "GPU-L4", gpus[0].ID)
	assert.True(t, api.shutdownCalled, "NVML must be shut down even on the success path")
}

func TestBytesToMiB(t *testing.T) {
	assert.Equal(t, 0, bytesToMiB(0))
	assert.Equal(t, 1, bytesToMiB(1024*1024))
	assert.Equal(t, 24*1024, bytesToMiB(24*gib))
}
