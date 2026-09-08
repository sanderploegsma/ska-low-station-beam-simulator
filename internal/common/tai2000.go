package common

// UnixToTAI2000Seconds converts a Unix (UTC-based) timestamp to seconds
// since the TAI2000 epoch (2000-01-01T00:00:00 TAI).
//
// UNVERIFIED-GRADE CAVEAT, mirroring the Python codebase's own
// non-astropy fallback path (common.py's unix_to_tai2000_seconds):
// this uses a HARDCODED TAI-UTC leap-second offset (37s), valid from
// 2017-01-01 (the most recent leap second insertion) with no further
// leap second inserted since. Go's standard `time` package has no
// leap-second-aware TAI conversion at all (time.Time deliberately
// ignores leap seconds), so unlike the Python side — which can prefer a
// real astropy/IERS table and only falls back to this — there is no
// "better" path available here without vendoring a leap-second table
// and pulling in a real IERS bulletin source. DO NOT deploy this as-is:
// wire in a proper leap-second table (or a shared source of truth with
// the Python side) before this matters for real timestamps.
func UnixToTAI2000Seconds(unixTime float64) float64 {
	const taiUTCOffsetS = 37.0
	const unixTimeAtTAI2000Epoch = 946684800.0 - 32.0
	return (unixTime + taiUTCOffsetS) - unixTimeAtTAI2000Epoch
}
