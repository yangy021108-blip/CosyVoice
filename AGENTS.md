# Repository Instructions

- Keep every CosyVoice/SDAA API source file, script, configuration, document,
  and test artifact under this repository root.
- Do not create persistent project files in the parent directory, `/tmp`, or
  another checkout. Temporary transfer files must be removed before handoff.
- Do not add general archive files such as ZIP, TAR, TGZ, 7Z, RAR, BZ2, or
  XZ. Compressed profiler traces ending in `.json.gz` are allowed only under
  the ignored `cosyvoice_api_outputs/` tree; delete traces from attempts that
  do not show a meaningful improvement.
- Store generated audio and diagnostics under `cosyvoice_api_outputs/`.
- Store the writable SDAA vLLM export under
  `cosyvoice_sdaa_vllm_model/`.
- Keep `cosyvoice_sdaa_vllm_api.env` local and permissioned as `0600`. Never
  commit API keys or other secrets.
- The runtime directories and secret file above must remain excluded by
  `.gitignore`.
