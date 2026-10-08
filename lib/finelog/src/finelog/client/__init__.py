# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finelog client APIs.

:class:`LogClient` covers the log and stats RPC surfaces.

The error types live in :mod:`finelog.errors` and are re-exported here so
callers can ``from finelog.client import SchemaConflictError`` etc.
"""

from finelog.client.log_client import FlushResult, LogClient, NamespaceInfo, Table, schema_from_dataclass
from finelog.errors import (
    InvalidNamespaceError,
    NamespaceNotFoundError,
    QueryResultTooLargeError,
    QueryTimeoutError,
    SchemaConflictError,
    SchemaValidationError,
    StatsError,
)
from finelog.policy import StoragePolicy
from finelog.table_spec import (
    BucketTransform,
    IdentityTransform,
    L0Mode,
    OperatingPolicy,
    PartitionField,
    PartitionSpec,
    SourceLayout,
    TableSpec,
    TableStatus,
)
