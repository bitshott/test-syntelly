SCRIPTS:=src.scripts

.PHONY: setup_conda remove_conda setup_deepchem setup_torch run_cv train_model

setup_conda:
	@echo "||Creating conda environment||"
	@conda create -p .conda python=3.11 -y
	@.conda/bin/pip install -r requirements.txt
	@echo "Done."

remove_conda:
	@echo "||Deleting conda environment||"
	@rm -rf .conda
	@echo "Done."

setup_deepchem:
	@conda create -p .deepchem python=3.11 -y
	@.deepchem/bin/pip install rdkit deepchem dgllife ipykernel dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html 

setup_torch:
	@conda env create -f environment.yml 

run_cv:
	@cd notebooks && \
	../.torch/bin/python3 gnn_melt_points.py && \
	cd -

train_model:
	@.conda/bin/python3 -m $(SCRIPTS).ingest && \
		.conda/bin/python3 -m $(SCRIPTS).train
