# flood_gpu.py
# GPU detection & simple monitoring. Falls back to CPU info if no GPU.


import psutil


try:
	import pynvml
	pynvml.nvmlInit()
	_NVML_OK = True
except Exception:
	pynvml = None
	_NVML_OK = False


def gpu_info():
	"""Return (name, memory_total_MB, memory_used_MB, utilization_percent) or None if no GPU."""
	if not _NVML_OK:
		return None
	try:
		handle = pynvml.nvmlDeviceGetHandleByIndex(0)
		name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
		mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
		util = pynvml.nvmlDeviceGetUtilizationRates(handle)
		return name, mem.total // 1024**2, mem.used // 1024**2, util.gpu
	except Exception:
		return None


def cpu_info():
	return psutil.cpu_percent(interval=None), psutil.virtual_memory()