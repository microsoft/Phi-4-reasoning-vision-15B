FROM vllm/vllm-openai:v0.16.0

COPY phi_4_rv_vllm_plugin/ /app/phi_4_rv_vllm_plugin/
WORKDIR /app/phi_4_rv_vllm_plugin
RUN pip install --no-deps --no-build-isolation /app/phi_4_rv_vllm_plugin/

ENV VLLM_LOGGING_LEVEL=WARN
