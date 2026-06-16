package metrics

import (
	"testing"

	"github.com/dstackai/dstack/runner/internal/runner/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGetAMDGPUMetrics_OK(t *testing.T) {
	collector, err := NewMetricsCollector(t.Context())
	assert.NoError(t, err)

	cases := []struct {
		csv      string
		expected []schemas.GPUMetrics
	}{
		// AMDSMI Tool: 24.7.1+0012a68 | AMDSMI Library version: 24.7.1.0 | ROCm version: 6.3.1
		{
			csv: "gpu,gfx,gfx_clock,vram_used,vram_total\n0,10,132,283,196300\n",
			expected: []schemas.GPUMetrics{
				{GPUUtil: 10, GPUMemoryUsage: 296747008},
			},
		},
		// AMDSMI Tool: 25.3.0+ede62f2 | AMDSMI Library version: 25.3.0 | ROCm version: 6.4.0
		{
			csv: "gpu,gfx_clk,gfx,vram_used,vram_free,vram_total,vram_percent\n0,132,10,283,196309,196592,0.0\n",
			expected: []schemas.GPUMetrics{
				{GPUUtil: 10, GPUMemoryUsage: 296747008},
			},
		},
	}

	for _, tc := range cases {
		metrics, err := collector.getAMDGPUMetrics(tc.csv)
		assert.NoError(t, err)
		assert.Equal(t, tc.expected, metrics)
	}
}

func TestGetAMDGPUMetrics_ErrorGPUUtilNA(t *testing.T) {
	collector, err := NewMetricsCollector(t.Context())
	assert.NoError(t, err)
	metrics, err := collector.getAMDGPUMetrics("gpu,gfx,gfx_clock,vram_used,vram_total\n0,N/A,N/A,283,196300\n")
	assert.ErrorContains(t, err, "GPU utilization is N/A")
	assert.Nil(t, metrics)
}

const mib = uint64(1024 * 1024)

// migXML is a representative subset of `nvidia-smi -q -x` output for a
// MIG-enabled GPU partitioned into two instances. The parent GPU reports N/A
// for fb used (expected under MIG), while each MIG device reports real usage.
const migXML = `<?xml version="1.0" ?>
<!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v12.dtd">
<nvidia_smi_log>
  <gpu id="00000000:55:00.0">
    <product_name>NVIDIA RTX PRO 6000 Blackwell</product_name>
    <mig_mode>
      <current_mig>Enabled</current_mig>
    </mig_mode>
    <fb_memory_usage>
      <total>97871 MiB</total>
      <used>N/A</used>
      <free>N/A</free>
    </fb_memory_usage>
    <utilization>
      <gpu_util>N/A</gpu_util>
    </utilization>
    <mig_devices>
      <mig_device>
        <index>0</index>
        <gpu_instance_id>3</gpu_instance_id>
        <fb_memory_usage>
          <total>24576 MiB</total>
          <used>1200 MiB</used>
          <free>23376 MiB</free>
        </fb_memory_usage>
      </mig_device>
      <mig_device>
        <index>1</index>
        <gpu_instance_id>4</gpu_instance_id>
        <fb_memory_usage>
          <total>24576 MiB</total>
          <used>2048 MiB</used>
          <free>22528 MiB</free>
        </fb_memory_usage>
      </mig_device>
    </mig_devices>
  </gpu>
</nvidia_smi_log>`

func TestParseNvidiaMIGMemoryMetrics(t *testing.T) {
	metrics, err := parseNvidiaMIGMemoryMetrics([]byte(migXML))
	require.NoError(t, err)
	require.Len(t, metrics, 2, "one entry per MIG device, in order")

	assert.Equal(t, 1200*mib, metrics[0].GPUMemoryUsage)
	assert.Equal(t, 2048*mib, metrics[1].GPUMemoryUsage)
	// Utilization is not available per-MIG via nvidia-smi.
	assert.Equal(t, uint64(0), metrics[0].GPUUtil)
	assert.Equal(t, uint64(0), metrics[1].GPUUtil)
}

func TestParseNvidiaMIGMemoryMetrics_NAUsedTreatedAsZero(t *testing.T) {
	const x = `<nvidia_smi_log><gpu><mig_devices>
		<mig_device><fb_memory_usage><used>N/A</used></fb_memory_usage></mig_device>
	</mig_devices></gpu></nvidia_smi_log>`

	metrics, err := parseNvidiaMIGMemoryMetrics([]byte(x))
	require.NoError(t, err)
	require.Len(t, metrics, 1)
	assert.Equal(t, uint64(0), metrics[0].GPUMemoryUsage)
}

func TestParseNvidiaMIGMemoryMetrics_NoMIGDevices(t *testing.T) {
	// MIG mode reported but no instances configured: no per-MIG metrics.
	const x = `<nvidia_smi_log><gpu><product_name>x</product_name></gpu></nvidia_smi_log>`
	metrics, err := parseNvidiaMIGMemoryMetrics([]byte(x))
	require.NoError(t, err)
	assert.Empty(t, metrics)
}

func TestParseNvidiaMIGMemoryMetrics_InvalidXML(t *testing.T) {
	_, err := parseNvidiaMIGMemoryMetrics([]byte("not xml"))
	require.Error(t, err)
}

func TestParseMiBValue(t *testing.T) {
	cases := []struct {
		in   string
		want uint64
		ok   bool
	}{
		{"1234 MiB", 1234, true},
		{"  512 MiB ", 512, true},
		{"0 MiB", 0, true},
		{"N/A", 0, false},
		{"", 0, false},
		{"garbage", 0, false},
	}
	for _, c := range cases {
		got, ok := parseMiBValue(c.in)
		assert.Equalf(t, c.want, got, "value for %q", c.in)
		assert.Equalf(t, c.ok, ok, "ok for %q", c.in)
	}
}
