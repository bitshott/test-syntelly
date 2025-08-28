.PHONY: setup_conda remove_conda

setup_conda:
	@echo "||Creating conda environment||"
	@conda create -p .conda python=3.11 -y
	@.conda/bin/pip -r requirements.txt
	@echo "Done."

remove_conda:
	@echo "||Deleting conda environment||"
	@rm -rf .conda
	@echo "Done."
