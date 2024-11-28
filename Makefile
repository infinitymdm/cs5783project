
.PHONY: all

all: dependencies

dependencies:
	pip3 install einops
	pip3 install omegaconf
	#pip3 install taming-transformers
	pip3 install taming-transformers-rom1504
