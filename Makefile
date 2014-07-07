######################################
#Set which command line tools the makefile should use in this block
CC = gcc
CCPP = g++
NVCC = nvcc
AR = ar
LIBTOOL = libtool
SOURCEDIR = src
######################################





######################################
#This makefile will produce a unix executable and a static library. Set there names here
EXE   = cuda_take
LIBOUT = libcuda_take.a
######################################





######################################
#Here we specify what source files are needed for the program/library, and we create virtual paths so that we don't have to refer to the source directory all the time
SOURCES = fft.cpp main.cpp dark_subtraction_filter.cu take_object.cpp std_dev_filter.cu chroma_translate_filter.cu mean_filter.cu
#SOURCES  = $(SOURCEDIR)/cuda_take.c $(SOURCEDIR)/constant_filter.cu


vpath %.c $(SOURCEDIR)
vpath %.cu $(SOURCEDIR)
vpath %.cpp $(SOURCEDIR)

#SOURCES = $(wildcard src/*.c)
#SOURCES = $(wildcard src/*.cpp)
#SOURCES = $(wildcard src/*.cu)
######################################








######################################
#Here we specify the names of all the intermediate files (.o, object files, we will need for the final lib and executable)
objects = $(patsubst %.c,obj/%.o,$(SOURCES)) 
objects += $(patsubst %.cpp,obj/%.o,$(SOURCES)) 
objects += $(patsubst %.cu,obj/%.o,$(SOURCES)) 


OBJDIR = obj

######################################







######################################
#Here we set the directories containing necessary header files (.h,.cuh,.hpp); ones specific to this program are kept in include/, ones needed for cameralink stuff in EDT_include/
IDIR      = -Iinclude -IEDT_include

#SECOND NOTE: ONLY BUILD WITH O1! Somehow, someway, O2 optimizes out things that nvcc needs and O0 has linker redefinition errors. #JankCity <- This has been fixed, but still a good idea
CFLAGS     = -g -O1 #This is added to the compilation of every file, enables gdb debugging symbols (-g) and limited optimization (-O1)
CONLYFLAGS = -std=c99
CONLYFLAGS += $(CFLAGS)

CPPFLAGS = -std=c++11 -Wall -Werror #NOTE, NVCC does not support C++11, therefore -std=c++11 cpp files must be split up from cu files
CPPFLAGS += $(CFLAGS)

NVCCFLAGS  = -gencode arch=compute_20,code=sm_20 -G -lineinfo -Xcompiler -rdynamic --compiler-options '-Wall -Werror -Wno-unused-function'#This program is targeted at GT590's which support Nvidia's 2.0 CUDA arch. Therefore that's what we build for
#(-G -lineinfo) are both to enable cuda-gdb debugging, -Xcompiler specifies arguments to get passed directly to g++ (which NVCC is built on), the internet said to do -rdynamic
NVCCFLAGS += $(CFLAGS)

LINKDIR 	= lib #where do the libraries we need to link in go?
LFLAGS      = -L$(LINKDIR) -lm -lpdv -lboost_thread -lz -lcuda -lcudart #Libraries needed to build program, only libpdv.a is not already visible in the path, as a result that is put in linkdir
AR_COMBINE_SCRIPT = combine_libs_script.ar #For building out output library we compine our stuff with libpdv, this script tells ar how to do that
#This switch enables concatenating libpdv.a and libcuda_take.a (and possibly libboost_thread.a)
STATIC_COMPILE_SYSTEM_LIBS = 1

######################################

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
	$(CC) $(CONLYFLAGS) $(IDIR) -c -o $@ $<

#what to do to build cpp files -> o files
$(OBJDIR)/%.o : %.cpp
	$(CCPP) $(CPPFLAGS) $(IDIR) -c -o $@ $<

#What to do to build cu files -> o files
$(OBJDIR)/%.o : %.cu 
	$(NVCC) $(NVCCFLAGS) $(IDIR) -c -o $@ $<
clean:
	rm -rf $(OBJDIR) $(EXE) $(LIBOUT) thin_$(LIBOUT)