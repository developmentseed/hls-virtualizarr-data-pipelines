"""HLS tile settings reach every processor Lambda; the Earthdata secret is only
granted to the functions that read COGs."""

from typing import Any

import aws_cdk as cdk
from aws_cdk.assertions import Template
from settings import StackSettings
from stack import VirtualizarrSqsStack

SECRET_ARN = "arn:aws:secretsmanager:us-west-2:111111111111:secret:edl-AbCdEf"
READERS = {"init", "worker", "process_messages"}


def _template(*, backfill: bool = True, **overrides: Any) -> Template:
    settings = StackSettings(
        STAGE="dev",
        ACCOUNT_ID="111111111111",
        ACCOUNT_REGION="us-west-2",
        ICECHUNK_BUCKET_NAME="ice-test",
        DATA_BUCKET_NAME="data-test",
        BACKFILL_ENABLED=backfill,
        **overrides,
    )
    stack = VirtualizarrSqsStack(
        cdk.App(),
        settings.STACK_NAME,
        settings=settings,
        env={"account": settings.ACCOUNT_ID, "region": settings.ACCOUNT_REGION},
    )
    return Template.from_stack(stack)


def _processor_functions(template: Template) -> dict[str, dict[str, Any]]:
    """Processor Lambdas by name (backfill action, or the template's function)."""
    found = {}
    for logical_id, res in template.find_resources("AWS::Lambda::Function").items():
        env = res["Properties"].get("Environment", {}).get("Variables", {})
        if "ICECHUNK_BUCKET" not in env:
            continue
        command = res["Properties"].get("ImageConfig", {}).get("Command")
        if command:
            name = command[0].split(".")[1]
        elif "processmessages" in logical_id.lower():
            name = "process_messages"
        else:
            name = "initialize"
        found[name] = res
    return found


def _granted_roles(template: Template) -> set[str]:
    roles = set()
    for pol in template.find_resources("AWS::IAM::Policy").values():
        for st in pol["Properties"]["PolicyDocument"]["Statement"]:
            actions = st["Action"] if isinstance(st["Action"], list) else [st["Action"]]
            if "secretsmanager:GetSecretValue" in actions:
                assert st["Resource"] == SECRET_ARN
                roles.update(r["Ref"] for r in pol["Properties"]["Roles"])
    return roles


def test_processor_lambdas_get_hls_env() -> None:
    fns = _processor_functions(_template())
    assert set(fns) == {
        "partition", "init", "fork", "worker", "reduce", "promote",
        "process_messages",
    }  # fmt: skip
    for name, res in fns.items():
        env = res["Properties"]["Environment"]["Variables"]
        assert env["MGRS_TILE"] == "T13SDA", name
        assert env["HLS_PRODUCT"] == "L30", name
        assert env["LPDAAC_ACCESS"] == "s3", name
        assert env["ICECHUNK_NUM_CHUNK_REFS"] == "5000000", name


def test_initialize_lambda_gets_hls_env() -> None:
    fns = _processor_functions(_template(backfill=False))
    env = fns["initialize"]["Properties"]["Environment"]["Variables"]
    assert env["MGRS_TILE"] == "T13SDA"
    assert env["HLS_PRODUCT"] == "L30"


def test_env_follows_settings() -> None:
    fns = _processor_functions(
        _template(MGRS_TILE="T10SGD", LPDAAC_ACCESS="https", ICECHUNK_NUM_CHUNK_REFS=7)
    )
    for res in fns.values():
        env = res["Properties"]["Environment"]["Variables"]
        assert env["MGRS_TILE"] == "T10SGD"
        assert env["LPDAAC_ACCESS"] == "https"
        assert env["ICECHUNK_NUM_CHUNK_REFS"] == "7"


def test_earthdata_secret_granted_to_readers() -> None:
    template = _template(EARTHDATA_SECRET_ARN=SECRET_ARN)
    fns = _processor_functions(template)
    granted = _granted_roles(template)
    for name, res in fns.items():
        env = res["Properties"]["Environment"]["Variables"]
        role = res["Properties"]["Role"]["Fn::GetAtt"][0]
        if name in READERS:
            assert env["EARTHDATA_SECRET_ARN"] == SECRET_ARN, name
            assert role in granted, name
        else:
            # The opener fetches the secret whenever the ARN is set, so a
            # function without the grant must not be given the ARN either.
            assert "EARTHDATA_SECRET_ARN" not in env, name
            assert role not in granted, name


def test_no_secret_without_arn() -> None:
    template = _template()
    assert not _granted_roles(template)
    for res in _processor_functions(template).values():
        assert (
            "EARTHDATA_SECRET_ARN" not in res["Properties"]["Environment"]["Variables"]
        )


def test_initialize_lambda_gets_no_secret() -> None:
    template = _template(backfill=False, EARTHDATA_SECRET_ARN=SECRET_ARN)
    env = _processor_functions(template)["initialize"]["Properties"]["Environment"][
        "Variables"
    ]
    assert "EARTHDATA_SECRET_ARN" not in env


def test_no_sns_subscription_by_default() -> None:
    _template().resource_count_is("AWS::SNS::Subscription", 0)
