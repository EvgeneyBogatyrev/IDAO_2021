all: run
	
build:
	echo "Building..."
run:
	bash main.sh

clear:
	rm -r __pycache__
