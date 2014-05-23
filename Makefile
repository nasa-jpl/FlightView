CC = nvcc

SOURCEDIR = src

EXE   = cuda_take

#SOURCES  = $(SOURCEDIR)/cuda_take.c $(SOURCEDIR)/constant_filter.cu
SOURCES = cuda_take.c constant_filter.cu dark_subtraction_filter.cu
vpath %.c $(SOURCEDIR)
vpath %.cu $(SOURCEDIR)
vpath %.cpp $(SOURCEDIR)

objects = $(patsubst %.c,obj/%.o,$(SOURCES)) 
objects += $(patsubst %.cpp,obj/%.o,$(SOURCES)) 
objects += $(patsubst %.cu,obj/%.o,$(SOURCES)) 

IDIR      = -Iinclude -IEDT_include

OBJS        = $(SOURCES:%.c=%.o)
OBJS		+= $(SOURCES:%.cpp=%.o)
OBJS		+= $(SOURCES:%.cu=%.o)

#OBJS = $(shell ls src/*.o)
OBJDIR = obj
H_FILES = include/constant_filter.cuh
GPUFLAGS = -G

CFLAGS     = -g
CFLAGS += $(GPUFLAGS)
NVCCFLAGS  = -arch=sm_20
NVCCFLAGS += $(CFLAGS)
LINKDIR 	= lib
LFLAGS      = -L$(LINKDIR) -lm -lpdv


all : cuda_take
#	@echo $(SOURCES)
#	@echo $(objects)
#	@echo $(OBJS)
$(EXE) : $(objects)

#	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)
#	$(CC) $(CFLAGS) $@ -c src/cuda_take.c -o   obj/cuda_take.o obj/constant_filter.o $(LFLAGS)

	$(CC) $(CFLAGS) -o $@ obj/cuda_take.o obj/constant_filter.o obj/dark_subtraction_filter.o $(LFLAGS)
$(objects): | obj

obj:
	@mkdir -p $@
	
#what to do to build c files -> o files
$(OBJDIR)/%.o : %.c
	$(CC) $(NVCCFLAGS) $(IDIR) -c -o $@ $<

#what to do to build cpp files -> o files
$(OBJDIR)/%.o : %.cpp
	$(CC) $(NVCCFLAGS) $(IDIR) -c -o $@ $<

#What to do to build cu files -> o files
$(OBJDIR)/%.o : %.cu $(H_FILES)
	$(CC) $(NVCCFLAGS) $(IDIR) -c -o $@ $<
clean:
	rm -rf $(OBJDIR)