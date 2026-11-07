from agent.database import (
    DatabaseDetectionReport,
    DatabaseDriverStatus,
    DatabaseSelectionError,
    DatabaseSelectionService,
    DatabaseServiceStatus,
    WorkloadRequest,
    WorkloadType,
)
from agent.device import CpuProfile, DeviceProfile, DiskProfile, MemoryProfile, PythonProfile


class StubProfiler:
    def __init__(self, profile):
        self.value = profile

    def profile(self):
        return self.value


class StubDetector:
    def __init__(self, report):
        self.value = report

    def detect(self):
        return self.value


def device(*, memory=2 * 1024**3, disk=10 * 1024**3):
    return DeviceProfile(
        captured_at="2026-11-07T00:00:00+00:00",
        operating_system="Windows", operating_system_release="11", operating_system_version="test",
        machine="x86_64",
        cpu=CpuProfile("x86_64", None, 4),
        memory=MemoryProfile(memory, memory, "test"),
        disk=DiskProfile(".", ".", disk, disk, 0),
        python=PythonProfile("3.10", "CPython", "python", "prefix", True),
        docker=None, warnings=(),
    )


def detection():
    return DatabaseDetectionReport(
        detected_at="2026-11-07T00:00:00+00:00",
        drivers=(DatabaseDriverStatus("postgresql", "psycopg", True),),
        services=(DatabaseServiceStatus("postgresql", "service", "postgres", True),),
        docker_available=False, warnings=(),
    )


def service(*, profile=None):
    return DatabaseSelectionService(
        device_profiler=StubProfiler(profile or device()),
        database_detector=StubDetector(detection()),
    )


def test_user_must_explicitly_confirm_detected_ready_database():
    assessment = service().assess(WorkloadRequest(
        description="read sales data", workload_type=WorkloadType.TRANSACTIONAL_READ,
    ))

    choice = service().confirm(assessment, "postgresql")

    assert choice.database == "postgresql"
    assert choice.user_confirmed is True
    assert choice.ready_for_connection_attempt is True
    assert "read-only" in choice.next_step


def test_unknown_or_unready_user_choice_is_rejected():
    assessment = service().assess(WorkloadRequest(
        description="read sales data", workload_type=WorkloadType.TRANSACTIONAL_READ,
    ))

    try:
        service().confirm(assessment, "sqlite")
    except DatabaseSelectionError as error:
        assert "detected options" in str(error)
    else:
        raise AssertionError("unknown choice must not be accepted")


def test_low_device_capacity_becomes_an_advisory_blocker():
    assessment = service(profile=device(memory=256 * 1024**2, disk=512 * 1024**2)).assess(
        WorkloadRequest(description="read sales", workload_type=WorkloadType.TRANSACTIONAL_READ)
    )

    assert any("memory" in item.lower() for item in assessment.recommendation.blockers)
    assert any("disk" in item.lower() for item in assessment.recommendation.blockers)
