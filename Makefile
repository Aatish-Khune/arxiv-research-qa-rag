PY ?= python

.PHONY: data index eval test app all

data:
	$(PY) run.py data

index:
	$(PY) run.py index

eval:
	$(PY) run.py eval

test:
	$(PY) run.py test

app:
	$(PY) run.py app

all:
	$(PY) run.py all
