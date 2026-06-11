//go:build cgo

package nvml

import (
	"fmt"

	gonvml "github.com/NVIDIA/go-nvml/pkg/nvml"
)

// New returns the real NVML-backed API. It does not load the library yet;
// call Init first.
func New() API {
	return &nvmlAPI{lib: gonvml.New()}
}

// retErr converts an NVML Return code into a Go error. SUCCESS maps to nil and
// ERROR_NOT_SUPPORTED maps to ErrNotSupported (wrapped) so callers can detect
// it with errors.Is.
func retErr(op string, ret gonvml.Return) error {
	switch ret {
	case gonvml.SUCCESS:
		return nil
	case gonvml.ERROR_NOT_SUPPORTED:
		return fmt.Errorf("%s: %w", op, ErrNotSupported)
	default:
		return fmt.Errorf("%s: %s", op, gonvml.ErrorString(ret))
	}
}

type nvmlAPI struct {
	lib gonvml.Interface
}

func (a *nvmlAPI) Init() error {
	return retErr("nvmlInit", a.lib.Init())
}

func (a *nvmlAPI) Shutdown() error {
	return retErr("nvmlShutdown", a.lib.Shutdown())
}

func (a *nvmlAPI) DeviceCount() (int, error) {
	count, ret := a.lib.DeviceGetCount()
	return count, retErr("nvmlDeviceGetCount", ret)
}

func (a *nvmlAPI) DeviceByIndex(index int) (Device, error) {
	dev, ret := a.lib.DeviceGetHandleByIndex(index)
	if err := retErr("nvmlDeviceGetHandleByIndex", ret); err != nil {
		return nil, err
	}
	return &nvmlDevice{dev: dev}, nil
}

type nvmlDevice struct {
	dev gonvml.Device
}

func (d *nvmlDevice) Name() (string, error) {
	name, ret := d.dev.GetName()
	return name, retErr("nvmlDeviceGetName", ret)
}

func (d *nvmlDevice) UUID() (string, error) {
	uuid, ret := d.dev.GetUUID()
	return uuid, retErr("nvmlDeviceGetUUID", ret)
}

func (d *nvmlDevice) MemoryInfo() (Memory, error) {
	mem, ret := d.dev.GetMemoryInfo()
	if err := retErr("nvmlDeviceGetMemoryInfo", ret); err != nil {
		return Memory{}, err
	}
	return Memory{Total: mem.Total, Free: mem.Free, Used: mem.Used}, nil
}

func (d *nvmlDevice) UtilizationRates() (Utilization, error) {
	util, ret := d.dev.GetUtilizationRates()
	if err := retErr("nvmlDeviceGetUtilizationRates", ret); err != nil {
		return Utilization{}, err
	}
	return Utilization{GPU: util.Gpu, Memory: util.Memory}, nil
}

// migUnsupported reports whether a Return code means MIG is simply not
// available on this GPU/driver, as opposed to a genuine failure. GPUs without
// MIG return ERROR_NOT_SUPPORTED; very old drivers whose libnvidia-ml.so
// predates the MIG API return ERROR_FUNCTION_NOT_FOUND. Both are benign.
func migUnsupported(ret gonvml.Return) bool {
	return ret == gonvml.ERROR_NOT_SUPPORTED || ret == gonvml.ERROR_FUNCTION_NOT_FOUND
}

func (d *nvmlDevice) MIGEnabled() (bool, error) {
	currentMode, _, ret := d.dev.GetMigMode()
	if migUnsupported(ret) {
		return false, nil
	}
	if err := retErr("nvmlDeviceGetMigMode", ret); err != nil {
		return false, err
	}
	return currentMode == gonvml.DEVICE_MIG_ENABLE, nil
}

func (d *nvmlDevice) MIGDevices() ([]Device, error) {
	maxCount, ret := d.dev.GetMaxMigDeviceCount()
	if migUnsupported(ret) {
		return nil, nil
	}
	if err := retErr("nvmlDeviceGetMaxMigDeviceCount", ret); err != nil {
		return nil, err
	}

	devices := make([]Device, 0, maxCount)
	for i := 0; i < maxCount; i++ {
		migDev, ret := d.dev.GetMigDeviceHandleByIndex(i)
		// Indices without a configured MIG instance return ERROR_NOT_FOUND;
		// skip them. (MaxMigDeviceCount is an upper bound, not the actual count.)
		if ret == gonvml.ERROR_NOT_FOUND {
			continue
		}
		if err := retErr("nvmlDeviceGetMigDeviceHandleByIndex", ret); err != nil {
			return devices, err
		}
		devices = append(devices, &nvmlDevice{dev: migDev})
	}
	return devices, nil
}
