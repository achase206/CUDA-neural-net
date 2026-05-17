# Child image: course base + RDKit (base does not include RDKit).
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /work

# Install pip if the base image is Debian/Ubuntu without it, then RDKit from PyPI.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir rdkit

# Fail the build early if the install did not work.
RUN python3 -c "from rdkit import Chem; from rdkit.Chem import AllChem"
