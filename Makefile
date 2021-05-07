# setup:
# 	python3 -m venv ~/.financial-tools
# 	. ~/.financial-tools/bin/activate
# 	pip install --upgrade pip
# 	pip install -r requirements.txt

# test:
# 	pytest -vv --cov=project
	
# lint:
# 	black project/domain/*.py
# 	pylint --disable=R,C,W project/domain/*.py

setup:
	python3 -m venv ~/.financial-tools
	
install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt
		
test:
	python -m pytest -vv --cov=project/domain project/tests/*.py
	
lint:
	black */*.py
	pylint --disable=R,C */*.py
	
all: setup 	\
	install	\
	test 	\
	lint