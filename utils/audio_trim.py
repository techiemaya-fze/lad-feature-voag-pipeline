from __future__ import annotations

import asyncio
from pathlib import Path


async def _run_subprocess(*cmd: str) -> tuple[bytes, bytes, int]:
	process = await asyncio.create_subprocess_exec(
		*cmd,
		stdout=asyncio.subprocess.PIPE,
		stderr=asyncio.subprocess.PIPE,
	)
	stdout, stderr = await process.communicate()
	return stdout or b"", stderr or b"", process.returncode


async def trim_leading_silence_ffmpeg(
	*,
	input_path: Path,
	output_path: Path,
	start_duration: float,
	start_threshold_db: float,
	leading_buffer_seconds: float = 0.5,
	trim_trailing: bool = True,
	stop_duration: float = 0.5,
	stop_threshold_db: float | None = None,
) -> None:
	"""
	Trim leading and trailing silence from an audio file using ffmpeg.
	
	Leading silence: Removes silence at the start but keeps a buffer of 
	`leading_buffer_seconds` (default 0.5s) before the first audio.
	
	Trailing silence: Removes silence at the end (configurable via trim_trailing).
	Does NOT remove silence in the middle of the audio.
	
	Args:
		input_path: Path to input audio file
		output_path: Path for output audio file
		start_duration: Minimum silence duration to trigger removal at start
		start_threshold_db: Threshold in dB for detecting silence at start
		leading_buffer_seconds: Buffer to keep before first audio (default 0.5s)
		trim_trailing: Whether to also trim trailing silence (default True)
		stop_duration: Minimum silence duration to trigger removal at end
		stop_threshold_db: Threshold in dB for end (defaults to start_threshold_db)
	"""
	input_path = Path(input_path)
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	
	if stop_threshold_db is None:
		stop_threshold_db = start_threshold_db
	
	# Build the silenceremove filter
	# start_periods=1: Remove leading silence once
	# stop_periods=1: Remove trailing silence once (if enabled)
	# detection=peak: Use peak detection instead of RMS for more accurate detection
	filter_parts = [
		f"start_periods=1",
		f"start_duration={max(start_duration, 0.0):.3f}",
		f"start_threshold={start_threshold_db}dB",
	]
	
	if trim_trailing:
		# IMPORTANT: stop_periods=-1 means "only remove silence at the very END of the file"
		# stop_periods=1 would remove silence after EVERY non-silent section (breaking the audio!)
		filter_parts.extend([
			f"stop_periods=-1",
			f"stop_duration={max(stop_duration, 0.0):.3f}",
			f"stop_threshold={stop_threshold_db}dB",
		])
	
	silence_filter = "silenceremove=" + ":".join(filter_parts)
	
	# Add delay filter to insert the buffer at the start
	# adelay takes delay in milliseconds
	delay_ms = int(leading_buffer_seconds * 1000)
	delay_filter = f"adelay={delay_ms}|{delay_ms}"  # Apply to all channels
	
	# Combine filters: first remove silence, then add buffer
	filter_expr = f"{silence_filter},{delay_filter}"
	
	stdout, stderr, returncode = await _run_subprocess(
		"ffmpeg",
		"-hide_banner",
		"-loglevel",
		"error",
		"-y",
		"-i",
		str(input_path),
		"-af",
		filter_expr,
		"-c:a",
		"libopus",
		str(output_path),
	)
	if returncode != 0:
		message = stderr.decode("utf-8", errors="replace").strip()
		if not message:
			message = stdout.decode("utf-8", errors="replace").strip() or f"ffmpeg exited with code {returncode}"
		raise RuntimeError(message)


async def probe_duration(path: Path) -> float | None:
	"""Return the duration of an audio file in seconds using ffprobe."""
	stdout, stderr, returncode = await _run_subprocess(
		"ffprobe",
		"-v",
		"error",
		"-show_entries",
		"format=duration",
		"-of",
		"default=noprint_wrappers=1:nokey=1",
		str(path),
	)
	if returncode != 0:
		raise RuntimeError(stderr.decode("utf-8", errors="replace").strip() or f"ffprobe exited with code {returncode}")
	text = stdout.decode("utf-8", errors="replace").strip()
	if not text:
		return None
	try:
		return float(text)
	except ValueError:
		return None

