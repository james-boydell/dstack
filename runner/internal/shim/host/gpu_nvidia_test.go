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
	name       string
	uuid       string
	mem        nvml.Memory
	util       nvml.Utilization
	migEnabled bool
	migDevices []nvml.Device

	nameErr       error
	uuidErr       error
	memErr        error
	migEnabledErr error
	migDevicesErr error
}

func (d *fakeDevice) Name() (string, error)                  { return d.name, d.nameErr }
func (d *fakeDevice) UUID() (string, error)                  { return d.uuid, d.uuidErr }
func (d *fakeDevice) MemoryInfo() (nvml.Memory, error)       { return d.mem, d.memErr }
func (d *fakeDevice) UtilizationRates() (nvml.Utilization, error) {
	return d.util, nil
}
func (d *fakeDevice) MIGEnabled() (bool, error)        { return d.migEnabled, d.migEnabledErr }
func (d *fakeDevice) MIGDevices() ([]nvml.Device, error) { return d.migDevices, d.migDevicesErr }

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

func migDevice(uuid string, vramGiB uint64) *fakeDevice {
	return &fakeDevice{uuid: uuid, mem: nvml.Memory{Total: vramGiB * gib}}
}

func TestCollectNvidiaGpuInfo_NoMIG(t *testing.T) {
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

func TestCollectNvidiaGpuInfo_MIGEnabled(t *testing.T) {
	// A single physical GPU partitioned into 4x 1g.24gb MIG instances,
	// mirroring the RTX PRO 6000 Blackwell test setup.
	migInstance := func(uuid string) *fakeDevice {
		return &fakeDevice{uuid: uuid, mem: nvml.Memory{Total: 24 * gib}}
	}
	parent := &fakeDevice{
		name:       "NVIDIA RTX PRO 6000 Blackwell",
		uuid:       "GPU-PARENT",
		mem:        nvml.Memory{Total: 96 * gib},
		migEnabled: true,
		migDevices: []nvml.Device{
			migInstance("MIG-aaaa"),
			migInstance("MIG-bbbb"),
			migInstance("MIG-cccc"),
			migInstance("MIG-dddd"),
		},
	}
	api := &fakeAPI{devices: []nvml.Device{parent}}

	gpus, err := collectNvidiaGpuInfo(context.Background(), api)
	require.NoError(t, err)

	require.Len(t, gpus, 4)
	for i, g := range gpus {
		assert.Equal(t, gpu.GpuVendorNvidia, g.Vendor)
		assert.Equal(t, "NVIDIA RTX PRO 6000 Blackwell", g.Name, "MIG entry keeps parent GPU name")
		assert.Equal(t, 24*1024, g.Vram, "MIG VRAM comes from the instance, not the parent")
		assert.NotEmpty(t, g.ID, "entry %d must have a MIG UUID", i)
	}
	// The MIG UUIDs (not the parent GPU UUID) are what the container runtime needs.
	ids := []string{gpus[0].ID, gpus[1].ID, gpus[2].ID, gpus[3].ID}
	assert.Equal(t, []string{"MIG-aaaa", "MIG-bbbb", "MIG-cccc", "MIG-dddd"}, ids)
}

func TestCollectNvidiaGpuInfo_MIGUnsupportedReportedAsPhysical(t *testing.T) {
	// On GPUs/drivers without MIG, MIGEnabled returns (false, nil); the GPU
	// should be reported as a single physical device.
	api := &fakeAPI{devices: []nvml.Device{
		physicalDevice("NVIDIA RTX 4090", "GPU-4090", 24),
	}}

	gpus, err := collectNvidiaGpuInfo(context.Background(), api)
	require.NoError(t, err)

	expected := []GpuInfo{
		{Vendor: gpu.GpuVendorNvidia, Name: "NVIDIA RTX 4090", Vram: 24 * 1024, ID: "GPU-4090"},
	}
	assert.Equal(t, expected, gpus)
}

func TestCollectNvidiaGpuInfo_MIGEnabledNoInstancesSkipped(t *testing.T) {
	// MIG mode on, but no instances configured yet: the GPU is skipped rather
	// than reported as an unallocatable whole device.
	parent := &fakeDevice{
		name:       "NVIDIA A100",
		uuid:       "GPU-A100",
		mem:        nvml.Memory{Total: 80 * gib},
		migEnabled: true,
		migDevices: nil,
	}
	api := &fakeAPI{devices: []nvml.Device{parent}}

	gpus, err := collectNvidiaGpuInfo(context.Background(), api)
	require.NoError(t, err)
	assert.Empty(t, gpus)
}

func TestCollectNvidiaGpuInfo_MIGEnumerationErrorSkipsGPU(t *testing.T) {
	// MIG mode is on but enumerating instances fails. We must not fall back to
	// reporting the parent as a whole GPU (it can't run containers with MIG on),
	// and the overall call should still succeed for any other GPUs.
	parent := &fakeDevice{
		name:          "NVIDIA A100",
		uuid:          "GPU-A100",
		mem:           nvml.Memory{Total: 80 * gib},
		migEnabled:    true,
		migDevicesErr: errors.New("nvmlDeviceGetMigDeviceHandleByIndex failed"),
	}
	api := &fakeAPI{devices: []nvml.Device{
		parent,
		physicalDevice("NVIDIA T4", "GPU-T4", 16),
	}}

	gpus, err := collectNvidiaGpuInfo(context.Background(), api)
	require.NoError(t, err)

	// Only the healthy non-MIG GPU is reported; the MIG GPU is dropped.
	expected := []GpuInfo{
		{Vendor: gpu.GpuVendorNvidia, Name: "NVIDIA T4", Vram: 16 * 1024, ID: "GPU-T4"},
	}
	assert.Equal(t, expected, gpus)
}

func TestCollectNvidiaGpuInfo_MIGInstanceWithoutUUIDSkipped(t *testing.T) {
	// A single MIG instance that can't report a UUID is skipped (it can't be
	// passed to the container runtime), while its siblings are still reported.
	bad := &fakeDevice{mem: nvml.Memory{Total: 24 * gib}, uuidErr: errors.New("no uuid")}
	parent := &fakeDevice{
		name:       "NVIDIA RTX PRO 6000 Blackwell",
		uuid:       "GPU-PARENT",
		mem:        nvml.Memory{Total: 96 * gib},
		migEnabled: true,
		migDevices: []nvml.Device{
			migDevice("MIG-aaaa", 24),
			bad,
			migDevice("MIG-cccc", 24),
		},
	}
	api := &fakeAPI{devices: []nvml.Device{parent}}

	gpus, err := collectNvidiaGpuInfo(context.Background(), api)
	require.NoError(t, err)

	require.Len(t, gpus, 2)
	assert.Equal(t, []string{"MIG-aaaa", "MIG-cccc"}, []string{gpus[0].ID, gpus[1].ID})
}

func TestCollectNvidiaGpuInfo_MixedMIGAndPhysical(t *testing.T) {
	// One MIG-partitioned GPU plus one ordinary GPU on the same host.
	parent := &fakeDevice{
		name:       "NVIDIA A100",
		uuid:       "GPU-A100",
		mem:        nvml.Memory{Total: 80 * gib},
		migEnabled: true,
		migDevices: []nvml.Device{
			migDevice("MIG-1", 40),
			migDevice("MIG-2", 40),
		},
	}
	api := &fakeAPI{devices: []nvml.Device{
		parent,
		physicalDevice("NVIDIA A100", "GPU-PLAIN", 80),
	}}

	gpus, err := collectNvidiaGpuInfo(context.Background(), api)
	require.NoError(t, err)

	expected := []GpuInfo{
		{Vendor: gpu.GpuVendorNvidia, Name: "NVIDIA A100", Vram: 40 * 1024, ID: "MIG-1"},
		{Vendor: gpu.GpuVendorNvidia, Name: "NVIDIA A100", Vram: 40 * 1024, ID: "MIG-2"},
		{Vendor: gpu.GpuVendorNvidia, Name: "NVIDIA A100", Vram: 80 * 1024, ID: "GPU-PLAIN"},
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
