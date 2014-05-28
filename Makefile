CC = gcc
CCPP = g++
NVCC = nvcc
AR = ar
LIBTOOL = libtool
SOURCEDIR = src

EXE   = cuda_take
LIBOUT = libcuda_take.a
#SOURCES  = $(SOURCEDIR)/cuda_take.c $(SOURCEDIR)/constant_filter.cu
SOURCES = main.cpp constant_filter.cu dark_subtraction_filter.cu take_object.cpp frame.cpp


vpath %.c $(SOURCEDIR)
vpath %.cu $(SOURCEDIR)
vpath %.cpp $(SOURCEDIR)

#SOURCES = $(wildcard src/*.c)
#SOURCES = $(wildcard src/*.cpp)
#SOURCES = $(wildcard src/*.cu)

objects = $(patsubst %.c,obj/%.o,$(SOURCES)) 
objects += $(patsubst %.cpp,obj/%.o,$(SOURCES)) 
objects += $(patsubst %.cu,obj/%.o,$(SOURCES)) 

IDIR      = -Iinclude -IEDT_include

OBJS        = $(SOURCES:%.c=%.o)
OBJS		+= $(SOURCES:%.cpp=%.o)
OBJS		+= $(SOURCES:%.cu=%.o)

#OBJS = $(shell ls src/*.o)
OBJDIR = obj


#NOTE, NVCC does not support C++11, therefore -std=c++11 cpp files must be split up from cu files
CFLAGS     = -g

CPPFLAGS = -std=c++11
CPPFLAGS += $(CFLAGS)

NVCCFLAGS  = -arch=sm_20 -G
NVCCFLAGS += $(CFLAGS)

LINKDIR 	= lib
LFLAGS      = -L$(LINKDIR) -lm -lpdv -lboost_thread
AR_COMBINE_SCRIPT = combine_libs_script.ar
STATIC_COMPILE_SYSTEM_LIBS = 1
all : $(EXE) $(LIBOUT)
#	@echo $(SOURCES)
#	@echo $(objects)
#	@echo $(OBJS)
$(EXE) : $(objects)
	$(NVCC) $(NVCCFLAGS) -o $@ $(wildcard obj/*.o) $(LFLAGS)
	
$(LIBOUT) : $(objects)
ifeq ($(STATIC_COMPILE_SYSTEM_LIBS), 1)
	$(AR) rcs thin_$@ $(filter-out obj/main.o, $(wildcard obj/*.o))		
	$(AR) -M <$(AR_COMBINE_SCRIPT)
else
	$(AR) rcs $@ $(filter-out obj/main.o, $(wildcard obj/*.o))
endif
$(objects): | obj

obj:
	@mkdir -p $@
	
#what to do to build c files -> o files
$(OBJDIR)/%.o : %.c
	$(CC) $(CFLAGS) $(IDIR) -c -o $@ $<

#what to do to build cpp files -> o files
$(OBJDIR)/%.o : %.cpp
	$(CCPP) $(CPPFLAGS) $(IDIR) -c -o $@ $<

#What to do to build cu files -> o files
$(OBJDIR)/%.o : %.cu 
	$(NVCC) $(CFLAGS) $(IDIR) -c -o $@ $<
clean:
	rm -rf $(OBJDIR) $(EXE) $(LIBOUT) thin_$(LIBOUT)