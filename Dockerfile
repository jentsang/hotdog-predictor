FROM condaforge/miniforge3:24.3.0-0

COPY conda-linux-64.lock tmp/conda-linux-64.lock


RUN mamba install conda-lock \
    && conda-lock install -n dsci522proj /tmp/conda-linux-64.lock \
    && conda clean --all -y -f

ENV PATH=/opt/conda/envs/dsci522proj/bin:$PATH

RUN mkdir -p /home/dog_or_not

WORKDIR /home/dog_or_not

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--IdentityProvider.token=", "--ServerApp.password="]
