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
