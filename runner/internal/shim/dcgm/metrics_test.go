package dcgm

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestFilterMetrics(t *testing.T) {
	body := []byte(`
# Comment
# HELP DCGM_FI_DEV_SM_CLOCK SM clock frequency (in MHz).
# TYPE DCGM_FI_DEV_SM_CLOCK gauge
DCGM_FI_DEV_SM_CLOCK{gpu="0",UUID="GPU-0781f3bb-da15-f334-d5db-37b3f19542d0",pci_bus_id="00000000:00:1B.0",device="nvidia0",modelName="Tesla T4",Hostname="ip-172-31-16-106",DCGM_FI_DRIVER_VERSION="535.183.06"} 1365
DCGM_FI_DEV_SM_CLOCK{gpu="1",UUID="GPU-41cc2907-3249-5a6b-f0e4-d04063b183a9",pci_bus_id="00000000:00:1C.0",device="nvidia1",modelName="Tesla T4",Hostname="ip-172-31-16-106",DCGM_FI_DRIVER_VERSION="535.183.06"} 300
DCGM_FI_DEV_SM_CLOCK{gpu="2",UUID="GPU-cc8e8c03-ebaa-f217-8e4c-d9cd98e20aed",pci_bus_id="00000000:00:1D.0",device="nvidia2",modelName="Tesla T4",Hostname="ip-172-31-16-106",DCGM_FI_DRIVER_VERSION="535.183.06"} 300
DCGM_FI_DEV_SM_CLOCK{gpu="3",UUID="GPU-fb615fb7-3f5a-5600-0ab1-debad8dc80ee",pci_bus_id="00000000:00:1E.0",device="nvidia3",modelName="Tesla T4",Hostname="ip-172-31-16-106",DCGM_FI_DRIVER_VERSION="535.183.06"} 300

# HELP DCGM_FI_DEV_MEM_CLOCK Memory clock frequency (in MHz).
DCGM_FI_DEV_MEM_CLOCK{gpu="0",UUID="GPU-0781f3bb-da15-f334-d5db-37b3f19542d0",pci_bus_id="00000000:00:1B.0",device="nvidia0",modelName="Tesla T4",Hostname="ip-172-31-16-106",DCGM_FI_DRIVER_VERSION="535.183.06"} 5000
DCGM_FI_DEV_MEM_CLOCK{gpu="1",UUID="GPU-41cc2907-3249-5a6b-f0e4-d04063b183a9",pci_bus_id="00000000:00:1C.0",device="nvidia1",modelName="Tesla T4",Hostname="ip-172-31-16-106",DCGM_FI_DRIVER_VERSION="535.183.06"} 405
# comment
DCGM_FI_DEV_MEM_CLOCK{gpu="2",UUID="GPU-cc8e8c03-ebaa-f217-8e4c-d9cd98e20aed",pci_bus_id="00000000:00:1D.0",device="nvidia2",modelName="Tesla T4",Hostname="ip-172-31-16-106",DCGM_FI_DRIVER_VERSION="535.183.06"} 405
DCGM_FI_DEV_MEM_CLOCK{gpu="3",UUID="GPU-fb615fb7-3f5a-5600-0ab1-debad8dc80ee",pci_bus_id="00000000:00:1E.0",device="nvidia3",modelName="Tesla T4",Hostname="ip-172-31-16-106",DCGM_FI_DRIVER_VERSION="535.183.06"} 405
DCGM_FI_DEV_MEMORY_TEMP{gpu="0",UUID="GPU-0781f3bb-da15-f334-d5db-37b3f19542d0",pci_bus_id="00000000:00:1B.0",device="nvidia0",modelName="Tesla T4",Hostname="ip-172-31-16-106",DCGM_FI_DRIVER_VERSION="535.183.06"} 0
DCGM_FI_DEV_MEMORY_TEMP{gpu="1",UUID="GPU-41cc2907-3249-5a6b-f0e4-d04063b183a9",pci_bus_id="00000000:00:1C.0",device="nvidia1",modelName="Tesla T4",Hostname="ip-172-31-16-106",DCGM_FI_DRIVER_VERSION="535.183.06"} 0

DCGM_FI_DEV_MEMORY_TEMP{gpu="2",UUID="GPU-cc8e8c03-ebaa-f217-8e4c-d9cd98e20aed",pci_bus_id="00000000:00:1D.0",device="nvidia2",modelName="Tesla T4",Hostname="ip-172-31-16-106",DCGM_FI_DRIVER_VERSION="535.183.06"} 0
DCGM_FI_DEV_MEMORY_TEMP{gpu="3",UUID="GPU-fb615fb7-3f5a-5600-0ab1-debad8dc80ee",pci_bus_id="00000000:00:1E.0",device="nvidia3",modelName="Tesla T4",Hostname="ip-172-31-16-106",DCGM_FI_DRIVER_VERSION="535.183.06"} 0
	`)
	filtered := FilterMetrics(body, [][]string{
		{"GPU-41cc2907-3249-5a6b-f0e4-d04063b183a9"},
		{"GPU-fb615fb7-3f5a-5600-0ab1-debad8dc80ee"},
	})
	expected := []byte(`# HELP DCGM_FI_DEV_SM_CLOCK SM clock frequency (in MHz).
# TYPE DCGM_FI_DEV_SM_CLOCK gauge
DCGM_FI_DEV_SM_CLOCK{gpu="1",UUID="GPU-41cc2907-3249-5a6b-f0e4-d04063b183a9",pci_bus_id="00000000:00:1C.0",device="nvidia1",modelName="Tesla T4",Hostname="ip-172-31-16-106",DCGM_FI_DRIVER_VERSION="535.183.06"} 300
DCGM_FI_DEV_SM_CLOCK{gpu="3",UUID="GPU-fb615fb7-3f5a-5600-0ab1-debad8dc80ee",pci_bus_id="00000000:00:1E.0",device="nvidia3",modelName="Tesla T4",Hostname="ip-172-31-16-106",DCGM_FI_DRIVER_VERSION="535.183.06"} 300
# HELP DCGM_FI_DEV_MEM_CLOCK Memory clock frequency (in MHz).
DCGM_FI_DEV_MEM_CLOCK{gpu="1",UUID="GPU-41cc2907-3249-5a6b-f0e4-d04063b183a9",pci_bus_id="00000000:00:1C.0",device="nvidia1",modelName="Tesla T4",Hostname="ip-172-31-16-106",DCGM_FI_DRIVER_VERSION="535.183.06"} 405
DCGM_FI_DEV_MEM_CLOCK{gpu="3",UUID="GPU-fb615fb7-3f5a-5600-0ab1-debad8dc80ee",pci_bus_id="00000000:00:1E.0",device="nvidia3",modelName="Tesla T4",Hostname="ip-172-31-16-106",DCGM_FI_DRIVER_VERSION="535.183.06"} 405
DCGM_FI_DEV_MEMORY_TEMP{gpu="1",UUID="GPU-41cc2907-3249-5a6b-f0e4-d04063b183a9",pci_bus_id="00000000:00:1C.0",device="nvidia1",modelName="Tesla T4",Hostname="ip-172-31-16-106",DCGM_FI_DRIVER_VERSION="535.183.06"} 0
DCGM_FI_DEV_MEMORY_TEMP{gpu="3",UUID="GPU-fb615fb7-3f5a-5600-0ab1-debad8dc80ee",pci_bus_id="00000000:00:1E.0",device="nvidia3",modelName="Tesla T4",Hostname="ip-172-31-16-106",DCGM_FI_DRIVER_VERSION="535.183.06"} 0
`)
	assert.Equal(t, expected, filtered)
}

// TestFilterMetricsMIG verifies AND-matching on (gpu, GPU_I_ID) labels, since
// dcgm-exporter labels MIG rows with the physical GPU UUID + GPU_I_ID, never the
// MIG UUID. Only the requested instance (gpu="0", GPU_I_ID="5") must be kept.
func TestFilterMetricsMIG(t *testing.T) {
	body := []byte(`# HELP DCGM_FI_PROF_GR_ENGINE_ACTIVE graphics/compute engine active ratio
# TYPE DCGM_FI_PROF_GR_ENGINE_ACTIVE gauge
DCGM_FI_PROF_GR_ENGINE_ACTIVE{gpu="0",UUID="GPU-b0e99009",GPU_I_PROFILE="1g.24gb",GPU_I_ID="5",Hostname="h"} 0.174817
DCGM_FI_PROF_GR_ENGINE_ACTIVE{gpu="0",UUID="GPU-b0e99009",GPU_I_PROFILE="1g.24gb",GPU_I_ID="6",Hostname="h"} 0.000000
DCGM_FI_PROF_GR_ENGINE_ACTIVE{gpu="0",UUID="GPU-b0e99009",GPU_I_PROFILE="1g.24gb",GPU_I_ID="15",Hostname="h"} 0.500000
`)
	filtered := FilterMetrics(body, [][]string{{`gpu="0"`, `GPU_I_ID="5"`}})
	expected := []byte(`# HELP DCGM_FI_PROF_GR_ENGINE_ACTIVE graphics/compute engine active ratio
# TYPE DCGM_FI_PROF_GR_ENGINE_ACTIVE gauge
DCGM_FI_PROF_GR_ENGINE_ACTIVE{gpu="0",UUID="GPU-b0e99009",GPU_I_PROFILE="1g.24gb",GPU_I_ID="5",Hostname="h"} 0.174817
`)
	// GPU_I_ID="5" must NOT match GPU_I_ID="15" (closing quote guards the boundary).
	assert.Equal(t, expected, filtered)
}
