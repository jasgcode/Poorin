# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finelog client APIs.

:class:`LogClient` covers the log and stats RPC surfaces.
:class:`ObjectQueryClient` executes direct object-store queries, and
:class:`RemoteLogHandler` plugs Python's ``logging`` into a
:class:`LogClient`.

The error types live in :mod:`poorin_finelog.errors` and are re-exported here so
callers can ``from poorin_finelog.client import SchemaConflictError`` etc.
"""

from poorin_finelog.client.log_client import FlushResult, LogClient, NamespaceInfo, Table, schema_from_dataclass
from poorin_finelog.client.object_query_client import ObjectQueryClient
from poorin_finelog.client.remote_log_handler import RemoteLogHandler
from poorin_finelog.errors import (
    InvalidNamespaceError,
    NamespaceNotFoundError,
    QueryResultTooLargeError,
    QueryTimeoutError,
    SchemaConflictError,
    SchemaValidationError,
    StatsError,
)
from poorin_finelog.policy import StoragePolicy
from poorin_finelog.table_spec import (
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
