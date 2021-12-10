#PYTHON=C:\Python36
PYTHON_INCLUDE=/usr/include/python3.8

LIBS=
ODIR=.tmp
CC=g++
CFLAGS=-I. -I$(PYTHON_INCLUDE) -I../.env/lib/python3.8/site-packages -fPIC
_OBJ = file.o module.o model/country_side.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))
DEPS = model/country_side.h model/matrix.h

model.so: $(ODIR) $(OBJ)
	$(CC) -o $@ $(OBJ) --shared -lpython3.8 $(LIBS)

$(ODIR):
	mkdir -p $(ODIR)

$(ODIR)/%.o: %.c $(DEPS)
	mkdir -p `dirname $@`
	$(CC) -c -o $@ $< $(CFLAGS)

$(ODIR)/%.o: %.cpp $(DEPS)
	mkdir -p `dirname $@`
	$(CC) -c -o $@ $< $(CFLAGS)

clean:
	-rm -rf $(ODIR) *.so
