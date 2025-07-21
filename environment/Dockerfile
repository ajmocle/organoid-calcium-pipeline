FROM continuumio/miniconda3

COPY calcium_env.yml .

RUN conda env create -f calcium_env.yml -n calcium_env

SHELL ["conda", "activate", "calcium_env"]