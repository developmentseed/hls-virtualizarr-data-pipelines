import json
from typing import Any

import aws_cdk as cdk
import aws_cdk.aws_s3 as s3
from aws_cdk.assertions import Match, Template
from stack_constructs.backfill_pipeline import BackfillPipeline


def _template(**overrides: Any) -> Template:
    app = cdk.App()
    stack = cdk.Stack(
        app,
        "TestStack",
        env=cdk.Environment(account="111111111111", region="us-east-1"),
    )
    bucket = s3.Bucket(stack, "IceBucket")
    artifacts = s3.Bucket(stack, "BackfillBucket")
    BackfillPipeline(
        stack,
        "Backfill",
        icechunk_bucket=bucket,
        backfill_bucket=artifacts,
        data_bucket_name="my-data-bucket",
        partition_size=500,
        max_items_per_batch=10,
        max_concurrency=50,
        **overrides,
    )
    return Template.from_stack(stack)


def _memory(template: Template, action: str) -> int:
    (fn,) = [
        r
        for r in template.find_resources("AWS::Lambda::Function").values()
        if r["Properties"]["ImageConfig"]["Command"]
        == [f"backfill_handlers.{action}.handler"]
    ]
    return int(fn["Properties"]["MemorySize"])


def test_worker_memory_defaults_above_the_other_functions() -> None:
    template = _template()
    assert _memory(template, "worker") == 4096
    for action in ["partition", "init", "fork", "reduce", "promote"]:
        assert _memory(template, action) == 2048, action


def test_worker_memory_is_configurable() -> None:
    template = _template(worker_memory_size=6144)
    assert _memory(template, "worker") == 6144
    assert _memory(template, "reduce") == 2048


def test_six_functions_with_cmd_overrides() -> None:
    template = _template()
    template.resource_count_is("AWS::Lambda::Function", 6)
    for action in ["partition", "init", "fork", "worker", "reduce", "promote"]:
        template.has_resource_properties(
            "AWS::Lambda::Function",
            Match.object_like(
                {"ImageConfig": {"Command": [f"backfill_handlers.{action}.handler"]}}
            ),
        )


def test_worker_has_data_bucket_read_and_list() -> None:
    template = _template()
    template.has_resource_properties(
        "AWS::IAM::Policy",
        Match.object_like(
            {
                "PolicyDocument": {
                    "Statement": Match.array_with(
                        [
                            Match.object_like(
                                {
                                    "Action": ["s3:GetObject", "s3:ListBucket"],
                                    "Resource": [
                                        "arn:aws:s3:::my-data-bucket/*",
                                        "arn:aws:s3:::my-data-bucket",
                                    ],
                                }
                            )
                        ]
                    )
                }
            }
        ),
    )


def _state_machine_asl() -> str:
    app = cdk.App()
    stack = cdk.Stack(
        app,
        "TestStack",
        env=cdk.Environment(account="111111111111", region="us-east-1"),
    )
    bucket = s3.Bucket(stack, "IceBucket")
    artifacts = s3.Bucket(stack, "BackfillBucket")
    BackfillPipeline(
        stack,
        "Backfill",
        icechunk_bucket=bucket,
        backfill_bucket=artifacts,
        data_bucket_name="my-data-bucket",
        partition_size=500,
        max_items_per_batch=10,
        max_concurrency=50,
    )
    tmpl = app.synth().get_stack_by_name("TestStack").template
    for res in tmpl["Resources"].values():
        if res["Type"] == "AWS::StepFunctions::StateMachine":
            parts = res["Properties"]["DefinitionString"]["Fn::Join"][1]
            return "".join(p if isinstance(p, str) else "<REF>" for p in parts)
    raise AssertionError("no state machine synthesized")


def test_state_machine_shape() -> None:
    template = _template()
    template.resource_count_is("AWS::StepFunctions::StateMachine", 1)

    asl = _state_machine_asl()
    # inner Distributed Map with a dynamic per-partition ItemReader key
    # (manifest_key comes from the partition item, not the fork result)
    assert '"Mode":"DISTRIBUTED"' in asl
    assert '"Key.$":"$.manifest_key"' in asl
    assert '"MaxItemsPerBatch":10' in asl
    # outer Map is serial
    assert '"MaxConcurrency":1' in asl
    # worker event reshape (Items -> file_keys, BatchInput carries the constants)
    assert '"file_keys.$":"$.Items"' in asl
    assert "$.BatchInput.fork_in_uri" in asl
    # ItemBatcher.BatchInput must use ".$" path keys so the fork URIs resolve at
    # runtime rather than being passed as the literal string "$.forkResult...".
    assert '"fork_in_uri.$":"$.forkResult.fork_in_uri"' in asl
    # reduce is reshaped to the flat event its handler expects
    assert '"forks_out_prefix.$":"$.forkResult.forks_out_prefix"' in asl
    # run_prefix derives from the execution name
    assert "Execution.Name" in asl


def test_state_machine_passes_inventory_to_init() -> None:
    asl = _state_machine_asl()
    init_state = json.loads(asl)["States"]["InitTask"]
    assert init_state["Parameters"] == {"inventory_uri.$": "$.inventory_uri"}


def test_state_machine_passes_skip_prefix() -> None:
    asl = _state_machine_asl()
    assert '"skipped_out_prefix.$":"$.forkResult.skipped_out_prefix"' in asl
    states = json.loads(asl)["States"]
    reduce_payload = states["OuterMap"]["ItemProcessor"]["States"]["ReduceTask"][
        "Parameters"
    ]
    assert reduce_payload["run_prefix.$"] == "$.run_prefix"
    assert reduce_payload["skipped_out_prefix.$"] == "$.forkResult.skipped_out_prefix"


def test_worker_task_retries_all_errors() -> None:
    """R15: a raising worker (e.g. a partial write) is retried by Step Functions."""
    states = json.loads(_state_machine_asl())["States"]
    inner = states["OuterMap"]["ItemProcessor"]["States"]["InnerMap"]
    worker = inner["ItemProcessor"]["States"]["WorkerTask"]
    retry = [r for r in worker["Retry"] if r["ErrorEquals"] == ["States.ALL"]]
    assert retry == [
        {
            "ErrorEquals": ["States.ALL"],
            "IntervalSeconds": 10,
            "MaxAttempts": 3,
            "BackoffRate": 2,
        }
    ]
