import asyncio
import json
import re
import uuid
from collections.abc import Mapping
from typing import List, NamedTuple, Optional

from sqlalchemy import Delete, delete, select
from sqlalchemy.orm import joinedload

from dstack._internal.core.consts import DSTACK_RUNNER_HTTP_PORT, DSTACK_SHIM_HTTP_PORT
from dstack._internal.core.models.runs import JobStatus
from dstack._internal.server import settings
from dstack._internal.server.db import get_session_ctx
from dstack._internal.server.models import InstanceModel, JobMetricsPoint, JobModel, ProjectModel
from dstack._internal.server.schemas.runner import MetricsResponse
from dstack._internal.server.services.instances import get_instance_ssh_private_keys
from dstack._internal.server.services.jobs import get_job_provisioning_data, get_job_runtime_data
from dstack._internal.server.services.runner import client
from dstack._internal.server.services.runner.ssh import runner_ssh_tunnel
from dstack._internal.server.utils import sentry_utils
from dstack._internal.utils.common import batched, get_current_datetime, get_or_error, run_async
from dstack._internal.utils.logging import get_logger

logger = get_logger(__name__)


MAX_JOBS_FETCHED = 100
BATCH_SIZE = 10
MIN_COLLECT_INTERVAL_SECONDS = 9


@sentry_utils.instrument_scheduled_task
async def collect_metrics():
    async with get_session_ctx() as session:
        res = await session.execute(
            select(JobModel)
            .where(JobModel.status.in_([JobStatus.RUNNING]))
            .options(
                joinedload(JobModel.instance)
                .joinedload(InstanceModel.project)
                .load_only(ProjectModel.ssh_private_key)
            )
            .order_by(JobModel.last_processed_at.asc())
            .limit(MAX_JOBS_FETCHED)
        )
        job_models = res.unique().scalars().all()

    for batch in batched(job_models, BATCH_SIZE):
        await _collect_jobs_metrics(batch)


@sentry_utils.instrument_scheduled_task
async def delete_metrics():
    now_timestamp_micro = int(get_current_datetime().timestamp() * 1_000_000)
    running_timestamp_micro_cutoff = (
        now_timestamp_micro - settings.SERVER_METRICS_RUNNING_TTL_SECONDS * 1_000_000
    )
    finished_timestamp_micro_cutoff = (
        now_timestamp_micro - settings.SERVER_METRICS_FINISHED_TTL_SECONDS * 1_000_000
    )
    await asyncio.gather(
        _execute_delete_statement(
            delete(JobMetricsPoint).where(
                JobMetricsPoint.job_id.in_(
                    select(JobModel.id).where(JobModel.status.in_([JobStatus.RUNNING]))
                ),
                JobMetricsPoint.timestamp_micro < running_timestamp_micro_cutoff,
            )
        ),
        _execute_delete_statement(
            delete(JobMetricsPoint).where(
                JobMetricsPoint.job_id.in_(
                    select(JobModel.id).where(JobModel.status.in_(JobStatus.finished_statuses()))
                ),
                JobMetricsPoint.timestamp_micro < finished_timestamp_micro_cutoff,
            )
        ),
    )


async def _execute_delete_statement(stmt: Delete) -> None:
    async with get_session_ctx() as session:
        await session.execute(stmt)
        await session.commit()


async def _collect_jobs_metrics(job_models: List[JobModel]):
    filtered_job_models = await _filter_recently_collected_jobs(job_models)
    tasks = []
    for job_model in filtered_job_models:
        tasks.append(_collect_job_metrics(job_model))
    points = await asyncio.gather(*tasks)
    async with get_session_ctx() as session:
        for point in points:
            if point is not None:
                session.add(point)
        await session.commit()


async def _filter_recently_collected_jobs(job_models: List[JobModel]) -> List[JobModel]:
    # Skip metrics collection if another replica collected it recently.
    # Two replicas can still collect metrics simultaneously – that's fine since
    # we'll just store some extra metric points in the db.
    async with get_session_ctx() as session:
        res = await session.execute(
            select(JobMetricsPoint).where(
                JobMetricsPoint.job_id.in_([j.id for j in job_models]),
                JobMetricsPoint.timestamp_micro > _get_recently_collected_metric_cutoff(),
            )
        )
        recent_points = res.scalars().all()
        recent_job_ids = [p.job_id for p in recent_points]
    return [j for j in job_models if j.id not in recent_job_ids]


def _get_recently_collected_metric_cutoff() -> int:
    now = int(get_current_datetime().timestamp() * 1_000_000)
    cutoff = now - (MIN_COLLECT_INTERVAL_SECONDS * 1_000_000)
    return cutoff


async def _collect_job_metrics(job_model: JobModel) -> Optional[JobMetricsPoint]:
    ssh_private_keys = get_instance_ssh_private_keys(get_or_error(job_model.instance))
    jpd = get_job_provisioning_data(job_model)
    jrd = get_job_runtime_data(job_model)
    if jpd is None:
        return None
    try:
        res = await run_async(
            _pull_runner_metrics,
            ssh_private_keys,
            jpd,
            jrd,
            job_model.id,
        )
    except Exception:
        logger.exception("Failed to collect job %s metrics", job_model.job_name)
        return None

    if isinstance(res, bool):
        # The job may already be terminated when collecting metrics - that's ok.
        logger.warning("Failed to connect to job %s to collect metrics", job_model.job_name)
        return None

    metrics, dcgm_text = res
    if metrics is None:
        logger.debug(
            (
                "Failed to collect job %s metrics."
                " Either runner version does not support metrics API"
                " or metrics collector is not available."
            ),
            job_model.job_name,
        )
        return None

    gpus_memory_usage_bytes = [g.gpu_memory_usage_bytes for g in metrics.gpus]
    gpus_util_percent = [g.gpu_util_percent for g in metrics.gpus]

    # Under MIG, the runner's nvidia-smi cannot report per-instance GPU utilization
    # (it's only available via DCGM/GPM). When the shim provides per-MIG DCGM
    # metrics, use them for both memory and utilization so the two stay aligned
    # per instance (the DCGM output is labeled by GPU_I_ID, not by list position).
    mig_points = _parse_dcgm_mig_metrics(dcgm_text) if dcgm_text else []
    if mig_points:
        gpus_memory_usage_bytes = [p.memory_usage_bytes for p in mig_points]
        gpus_util_percent = [p.util_percent for p in mig_points]

    return JobMetricsPoint(
        job_id=job_model.id,
        timestamp_micro=metrics.timestamp_micro,
        cpu_usage_micro=metrics.cpu_usage_micro,
        memory_usage_bytes=metrics.memory_usage_bytes,
        memory_working_set_bytes=metrics.memory_working_set_bytes,
        gpus_memory_usage_bytes=json.dumps(gpus_memory_usage_bytes),
        gpus_util_percent=json.dumps(gpus_util_percent),
    )


class _MIGPoint(NamedTuple):
    memory_usage_bytes: int
    util_percent: int


# Matches a DCGM exporter sample line: NAME{label="v",...} <number>
_DCGM_LINE_RE = re.compile(r"^(?P<name>DCGM_FI_\w+)\{(?P<labels>[^}]*)\}\s+(?P<value>[-\d.eE+]+)")
_DCGM_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


def _parse_dcgm_mig_metrics(dcgm_text: str) -> List[_MIGPoint]:
    """
    Parse per-MIG-instance memory and utilization from the shim's filtered
    dcgm-exporter output.

    Only MIG instances (lines carrying a ``GPU_I_ID`` label) are considered;
    physical-GPU lines are ignored, so non-MIG jobs yield an empty list and fall
    back to the runner's metrics. Instances are returned sorted by
    ``(gpu, GPU_I_ID)`` for a stable per-instance order.
    """
    by_instance: dict[tuple[int, int], dict] = {}
    for raw_line in dcgm_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _DCGM_LINE_RE.match(line)
        if match is None:
            continue
        name = match.group("name")
        if name not in ("DCGM_FI_DEV_FB_USED", "DCGM_FI_PROF_SM_ACTIVE"):
            continue
        labels = dict(_DCGM_LABEL_RE.findall(match.group("labels")))
        gi_raw = labels.get("GPU_I_ID")
        if not gi_raw:
            # Not a MIG instance (physical GPU line) — ignore.
            continue
        try:
            key = (int(labels.get("gpu", "0")), int(gi_raw))
            value = float(match.group("value"))
        except ValueError:
            continue
        entry = by_instance.setdefault(key, {})
        if name == "DCGM_FI_DEV_FB_USED":
            entry["memory_usage_bytes"] = int(value) * 1024 * 1024  # MiB -> bytes
        else:
            # DCGM_FI_PROF_SM_ACTIVE: ratio (0..1) of cycles the slice's SMs have
            # a warp assigned, averaged over the slice's SMs. Unlike
            # GR_ENGINE_ACTIVE, this is normalized to the MIG instance, so a fully
            # busy slice reads ~1.0 rather than capping at its fraction of the GPU.
            entry["util_percent"] = round(value * 100)

    points = []
    for key in sorted(by_instance):
        entry = by_instance[key]
        points.append(
            _MIGPoint(
                memory_usage_bytes=entry.get("memory_usage_bytes", 0),
                util_percent=entry.get("util_percent", 0),
            )
        )
    return points


@runner_ssh_tunnel
def _pull_runner_metrics(
    addresses: Mapping[int, client.LocalAddress],
    task_id: uuid.UUID,
) -> tuple[Optional[MetricsResponse], Optional[str]]:
    runner_client = client.RunnerClient.from_address(addresses[DSTACK_RUNNER_HTTP_PORT])
    metrics = runner_client.get_metrics()
    # On VM-based backends the shim port is also forwarded; fetch per-task DCGM
    # metrics so MIG utilization (unavailable via the runner) can be filled in.
    # Container-based backends have no shim, so the port is absent and we skip it.
    dcgm_text: Optional[str] = None
    shim_address = addresses.get(DSTACK_SHIM_HTTP_PORT)
    if shim_address is not None:
        try:
            shim_client = client.ShimClient.from_address(shim_address)
            dcgm_text = shim_client.get_task_metrics(task_id)
        except Exception:
            logger.debug("Failed to fetch DCGM metrics for task %s", task_id, exc_info=True)
    return metrics, dcgm_text
