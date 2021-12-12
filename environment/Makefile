#PYTHON=C:\Python36
PYTHON_INCLUDE=-I/usr/include/python3.8
JAVA_INCLUDE=-I/usr/lib/jvm/java-17-openjdk-amd64/include \
	-I/usr/lib/jvm/java-17-openjdk-amd64/include/linux

LIBS=-L/usr/lib/jvm/java-17-openjdk-amd64/lib/server
ODIR=.o
CC=g++
CFLAGS=-I. $(PYTHON_INCLUDE) $(JAVA_INCLUDE) -I../.env/lib/python3.8/site-packages -fPIC -fpermissive
_OBJ = file.o module.o java.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))
DEPS = java.h

all: model.so .class/dev/damaso/Main.class

model.so: $(ODIR) $(OBJ)
	$(CC) -o $@ $(OBJ) --shared -lpython3.8 -ljvm $(LIBS)

.class/dev/damaso/Main.class: model/src/dev/damaso/Main.java
	mkdir -p .class
	cd model/src && javac -d ../../.class dev/damaso/Main.java

$(ODIR):
	mkdir -p $(ODIR)

$(ODIR)/%.o: %.c $(DEPS)
	mkdir -p `dirname $@`
	$(CC) -c -o $@ $< $(CFLAGS)

$(ODIR)/%.o: %.cpp $(DEPS)
	mkdir -p `dirname $@`
	$(CC) -c -o $@ $< $(CFLAGS)

clean:
	-rm -rf $(ODIR) *.so .class
