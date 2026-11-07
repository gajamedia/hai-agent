"""Database discovery, recommendation, and connection lifecycle utilities."""

from .contracts import (
    DatabaseRecommendation,
    WorkloadAnalysis,
    WorkloadRequest,
    WorkloadType,
)
from .connection_manager import (
    ConnectionHealth,
    ConnectionManager,
    ConnectionManagerError,
    ConnectionNotConfiguredError,
    ConnectionPolicyError,
    ConnectionProfile,
)
from .database_detector import (
    DatabaseDetectionReport,
    DatabaseDetector,
    DatabaseDriverStatus,
    DatabaseServiceStatus,
)
from .database_selector import DatabaseSelector
from .selection_service import (
    DatabaseSelectionAssessment,
    DatabaseSelectionError,
    DatabaseSelectionService,
    UserDatabaseSelection,
)
from .sqlite_readonly import (
    SQLiteReadOnlyConfigurationError,
    register_sqlite_read_only_profile,
)
from .sqlite_schema import (
    SQLiteAllowlistError,
    SQLiteColumn,
    SQLiteSchema,
    SQLiteSchemaDiscoveryError,
    SQLiteTable,
    build_readonly_table_specs,
    discover_sqlite_schema,
)
from .workload_analyzer import WorkloadAnalyzer

__all__ = [
    "ConnectionHealth",
    "ConnectionManager",
    "ConnectionManagerError",
    "ConnectionNotConfiguredError",
    "ConnectionPolicyError",
    "ConnectionProfile",
    "DatabaseDetectionReport",
    "DatabaseDetector",
    "DatabaseDriverStatus",
    "DatabaseRecommendation",
    "DatabaseSelector",
    "DatabaseSelectionAssessment",
    "DatabaseSelectionError",
    "DatabaseSelectionService",
    "DatabaseServiceStatus",
    "WorkloadAnalysis",
    "WorkloadAnalyzer",
    "WorkloadRequest",
    "WorkloadType",
    "UserDatabaseSelection",
    "SQLiteReadOnlyConfigurationError",
    "register_sqlite_read_only_profile",
    "SQLiteAllowlistError",
    "SQLiteColumn",
    "SQLiteSchema",
    "SQLiteSchemaDiscoveryError",
    "SQLiteTable",
    "build_readonly_table_specs",
    "discover_sqlite_schema",
]
