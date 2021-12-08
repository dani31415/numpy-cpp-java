#PYTHON=C:\Python36
PYTHON_INCLUDE=/usr/include/python3.8

LIBS=
ODIR=.tmp
CC=gcc
CFLAGS=-I. -I$(PYTHON_INCLUDE) -I../.env/lib/python3.8/site-packages -fPIC
_OBJ = file.o module.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))
DEPS =

spam.so: $(ODIR) $(OBJ)
	$(CC) -o $@ $(OBJ) --shared -lpython3.8 $(LIBS)

$(ODIR):
	mkdir -p $(ODIR)

$(ODIR)/%.o: %.c $(DEPS) 
	$(CC) -c -o $@ $< $(CFLAGS)

clean:
	-rm -rf $(ODIR) *.so
