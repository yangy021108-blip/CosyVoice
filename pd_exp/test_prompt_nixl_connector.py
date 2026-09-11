from types import SimpleNamespace

from pd_exp.prompt_nixl_connector_0251 import CosyPromptNixlPullConnectorScheduler


def test_remote_prefill_uses_prompt_embedding_length() -> None:
    scheduler = object.__new__(CosyPromptNixlPullConnectorScheduler)
    scheduler._has_mamba = False
    request = SimpleNamespace(
        prompt_token_ids=None,
        num_prompt_tokens=137,
        kv_transfer_params={"do_remote_prefill": True},
    )
    assert scheduler.get_num_new_matched_tokens(request, 9) == (128, True)


def test_local_request_has_no_external_tokens() -> None:
    scheduler = object.__new__(CosyPromptNixlPullConnectorScheduler)
    scheduler._has_mamba = False
    request = SimpleNamespace(
        prompt_token_ids=None,
        num_prompt_tokens=137,
        kv_transfer_params=None,
    )
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
