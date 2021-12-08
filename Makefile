install: export PYTHONHOME =
install: VIRTUAL_ENV = ${CURDIR}/.env
install: export PATH := ${VIRTUAL_ENV}/bin:${PATH}
install: .env
	pip3 install matplotlib
	pip3 install pillow
	pip3 install numpy
	pip3 install sklearn
	pip3 install scikit-image
	pip3 install jupyter

.env:
	virtualenv .env

clean:
	cd environment && make clean
