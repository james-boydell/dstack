package dcgm

import (
	"bufio"
	"bytes"
	"strings"
)

// FilterMetrics returns the subset of metrics whose label set matches any of the
// given matchers.
//
// A matcher is a set of label substrings that must ALL appear in a metric line
// (logical AND); a line is kept if it satisfies ANY matcher (logical OR). This
// supports both:
//
//   - physical GPUs / non-MIG: a single-element matcher with the GPU UUID, e.g.
//     {`GPU-2b79...`}, matching `... UUID="GPU-2b79..." ...`;
//   - MIG instances: a two-element matcher with the physical GPU index and the
//     GPU instance ID, e.g. {`gpu="0"`, `GPU_I_ID="5"`} — because dcgm-exporter
//     labels MIG rows with those, never the MIG UUID.
//
// DCGM Exporter returns metrics in the following format:
//
//	# HELP DCGM_FIELD_1 Docstring for field 1
//	# TYPE DCGM_FIELD_1 gauge|counter|...
//	DCGM_FIELD{gpu="0", UUID="..." [...other labels...]} 0.0
//	DCGM_FIELD{gpu="1", UUID="..." [...other labels...]} 0.5
//	...
func FilterMetrics(expfmtBody []byte, matchers [][]string) []byte {
	var buffer bytes.Buffer
	scanner := bufio.NewScanner(bytes.NewReader(expfmtBody))
	helpComment := ""
	typeComment := ""
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if len(line) == 0 {
			continue
		}
		if strings.HasPrefix(line, "# HELP") {
			helpComment = line
			continue
		}
		if strings.HasPrefix(line, "# TYPE") {
			typeComment = line
			continue
		}
		if strings.HasPrefix(line, "#") {
			continue
		}
		if lineMatchesAny(line, matchers) {
			if helpComment != "" {
				buffer.WriteString(helpComment)
				buffer.WriteRune('\n')
				helpComment = ""
			}
			if typeComment != "" {
				buffer.WriteString(typeComment)
				buffer.WriteRune('\n')
				typeComment = ""
			}
			buffer.WriteString(line)
			buffer.WriteRune('\n')
		}
	}
	return buffer.Bytes()
}

// lineMatchesAny reports whether line satisfies any matcher, where a matcher is
// satisfied only if every one of its substrings is present in the line.
func lineMatchesAny(line string, matchers [][]string) bool {
	for _, matcher := range matchers {
		if len(matcher) == 0 {
			continue
		}
		matched := true
		for _, sub := range matcher {
			if !strings.Contains(line, sub) {
				matched = false
				break
			}
		}
		if matched {
			return true
		}
	}
	return false
}
