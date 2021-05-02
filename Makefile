setup:
	python3 -m venv ~/.financial-tools
	
install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt
		
test:
	pytest -vv --cov=project
	
lint:
	black project/domain/*.py
	pylint --disable=R,C,W project/domain/*.py
	
all: setup install test lint
