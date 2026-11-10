from __future__ import annotations

import ctypes
import os
import time
from ctypes import wintypes


TH32CS_SNAPPROCESS = 0x00000002
PROCESS_QUERY_VM_READ = 0x1010
PROCESS_SET_QUOTA_TERMINATE = 0x0101
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000


def windows_host_resources() -> dict:
    if os.name != "nt":
        raise OSError("parallel LOCAL_DEMO resource admission requires Windows")

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
    kernel.GlobalMemoryStatusEx.restype = wintypes.BOOL
    memory = MEMORYSTATUSEX()
    memory.dwLength = ctypes.sizeof(memory)
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise ctypes.WinError(ctypes.get_last_error())
    return {
        "logical_processors": os.cpu_count() or 1,
        "available_memory_bytes": int(memory.ullAvailPhys),
        "total_memory_bytes": int(memory.ullTotalPhys),
        "sampled_at_monotonic": time.monotonic(),
    }


def windows_cpu_load_percent(interval_seconds: float = 1.0) -> float:
    if os.name != "nt":
        raise OSError("parallel LOCAL_DEMO resource admission requires Windows")

    class FILETIME(ctypes.Structure):
        _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetSystemTimes.argtypes = [ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME),
                                      ctypes.POINTER(FILETIME)]
    kernel.GetSystemTimes.restype = wintypes.BOOL

    def read() -> tuple[int, int]:
        idle, kernel_time, user_time = FILETIME(), FILETIME(), FILETIME()
        if not kernel.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel_time), ctypes.byref(user_time)):
            raise ctypes.WinError(ctypes.get_last_error())
        to_int = lambda item: (item.high << 32) | item.low
        return to_int(idle), to_int(kernel_time) + to_int(user_time)

    idle_before, total_before = read()
    time.sleep(interval_seconds)
    idle_after, total_after = read()
    total_delta = total_after - total_before
    if total_delta <= 0:
        raise RuntimeError("Windows host CPU time did not advance during load sample")
    idle_delta = max(0, idle_after - idle_before)
    return max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta)))


def _process_rows() -> list[dict]:
    if os.name != "nt":
        return []

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    kernel.Process32FirstW.restype = wintypes.BOOL
    kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    kernel.Process32NextW.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        raise ctypes.WinError(ctypes.get_last_error())
    rows = []
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        has_entry = kernel.Process32FirstW(handle, ctypes.byref(entry))
        while has_entry:
            rows.append({"pid": int(entry.th32ProcessID), "parent_pid": int(entry.th32ParentProcessID)})
            has_entry = kernel.Process32NextW(handle, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(handle)
    return rows


def process_tree_pids(root_pid: int, rows: list[dict] | None = None) -> list[int]:
    rows = _process_rows() if rows is None else rows
    children: dict[int, list[int]] = {}
    for row in rows:
        children.setdefault(row["parent_pid"], []).append(row["pid"])
    result = []
    pending = [root_pid]
    seen = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        result.append(pid)
        pending.extend(children.get(pid, ()))
    return result


def sample_process_tree(root_pid: int) -> dict | None:
    if os.name != "nt":
        return None

    class FILETIME(ctypes.Structure):
        _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, ctypes.POINTER(FILETIME),
                                       ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME),
                                       ctypes.POINTER(FILETIME)]
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                                           wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    process_rows = []
    for pid in process_tree_pids(root_pid):
        handle = kernel.OpenProcess(PROCESS_QUERY_VM_READ, False, pid)
        if not handle:
            continue
        try:
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            memory_ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
            created, exited, kernel_time, user_time = FILETIME(), FILETIME(), FILETIME(), FILETIME()
            times_ok = kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                              ctypes.byref(kernel_time), ctypes.byref(user_time))
            if not times_ok:
                continue
            created_100ns = (created.high << 32) | created.low
            cpu_ticks = ((kernel_time.high << 32) | kernel_time.low) + ((user_time.high << 32) | user_time.low)
            process_rows.append({
                "pid": pid,
                "creation_time_100ns": created_100ns,
                "rss_bytes": int(counters.WorkingSetSize) if memory_ok else 0,
                "cpu_seconds": cpu_ticks / 10_000_000,
            })
        finally:
            kernel.CloseHandle(handle)
    if not process_rows:
        return None
    return {
        "root_pid": root_pid,
        "process_count": len(process_rows),
        "processes": process_rows,
        "rss_bytes": sum(row["rss_bytes"] for row in process_rows),
        "cpu_seconds": sum(row["cpu_seconds"] for row in process_rows),
        "sampled_at_monotonic": time.monotonic(),
    }


class WindowsKillJob:
    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError("Windows process-tree cleanup requires Windows")
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                        ctypes.c_void_p, wintypes.DWORD]
        self.kernel.SetInformationJobObject.restype = wintypes.BOOL
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        self.kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.kernel.TerminateJobObject.restype = wintypes.BOOL
        self.kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                          ctypes.c_void_p, wintypes.DWORD,
                                                          ctypes.POINTER(wintypes.DWORD)]
        self.kernel.QueryInformationJobObject.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                        ("PerJobUserTimeLimit", ctypes.c_longlong),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]
        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in
                        ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                         "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]
        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                        ("IoInfo", IO_COUNTERS), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]
        limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(
                self.handle, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(limits), ctypes.sizeof(limits)):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())

    def assign(self, pid: int) -> None:
        process = self.kernel.OpenProcess(PROCESS_SET_QUOTA_TERMINATE, False, pid)
        if not process:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not self.kernel.AssignProcessToJobObject(self.handle, process):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            self.kernel.CloseHandle(process)

    def terminate(self, exit_code: int = 1) -> None:
        if self.handle and not self.kernel.TerminateJobObject(self.handle, exit_code):
            error = ctypes.get_last_error()
            if error not in (0, 5, 87):
                raise ctypes.WinError(error)

    def active_processes(self) -> int:
        class JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("TotalUserTime", ctypes.c_longlong), ("TotalKernelTime", ctypes.c_longlong),
                ("ThisPeriodTotalUserTime", ctypes.c_longlong),
                ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
                ("TotalPageFaultCount", wintypes.DWORD), ("TotalProcesses", wintypes.DWORD),
                ("ActiveProcesses", wintypes.DWORD), ("TotalTerminatedProcesses", wintypes.DWORD),
            ]

        info = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
        returned = wintypes.DWORD()
        if not self.kernel.QueryInformationJobObject(
                self.handle, 1, ctypes.byref(info), ctypes.sizeof(info), ctypes.byref(returned)):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(info.ActiveProcesses)

    def wait_empty(self, timeout_seconds: float) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while True:
            if self.active_processes() == 0:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    def close(self) -> None:
        if getattr(self, "handle", None):
            self.kernel.CloseHandle(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
