//go:build !cgo

package nvml

// New returns a stub API for builds without cgo. go-nvml relies on cgo to
// dynamically load libnvidia-ml.so, so NVML is unavailable here (this path is
// used for static cross-compiled binaries). All operations report
// ErrUnavailable so callers can degrade gracefully.
func New() API {
	return unsupportedAPI{}
}

type unsupportedAPI struct{}

func (unsupportedAPI) Init() error                       { return ErrUnavailable }
func (unsupportedAPI) Shutdown() error                   { return nil }
func (unsupportedAPI) DeviceCount() (int, error)         { return 0, ErrUnavailable }
func (unsupportedAPI) DeviceByIndex(int) (Device, error) { return nil, ErrUnavailable }
