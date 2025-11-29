FROM condaforge/miniforge3:24.3.0-0

COPY . /home/dog_or_not/

RUN mamba install conda-lock \
    && conda-lock install -n dsci522proj /home/dog_or_not/conda-linux-64.lock \
    && conda clean --all -y -f

ENV PATH=/opt/conda/envs/dsci522proj/bin:$PATH

EXPOSE 8888

WORKDIR /home/dog_or_not

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--IdentityProvider.token=", "--ServerApp.password="]


