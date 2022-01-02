## Compiler, tools and options
CC = cc
LINK = cc
OPT = -O3

CCFLAGS = $(OPT) -mavx
LDFLAGS = $(OPT) -mavx

## PAPI setup
PAPI_ROOT = /opt/cray/pe/papi/6.0.0.7
#PAPI_LIB = $(PAPI_ROOT)/lib/libpapi.a

## Libraries
LIBS = -L $(PAPI_ROOT)/lib -lpapi
INC = -I $(PAPI_ROOT)/include

## Files
OBJECTS = main.o
TARGET = main

## Implicit rules
.SUFFIXES: .c
.c.o:
	$(CC) -c $(CCFLAGS) $(INC) $<

## Build Rules
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(LINK) -o $@ $(OBJECTS) $(LDFLAGS) $(LIBS)

clean:
	rm -f $(OBJECTS) $(TARGET)
	rm -f *~ core

run:
	./main



