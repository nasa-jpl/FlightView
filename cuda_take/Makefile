######################################
#Set which command line tools the makefile should use in this block
CC = gcc
CCPP = g++
NVCC = nvcc
AR = ar
LIBTOOL = libtool
SOURCEDIR = src
HARDWARE = EDT
ifeq ($(HARDWARE),OPALKELLY)
okFP_SDK ?= /usr/lib/ 
endif
######################################





######################################
#This makefile will produce a unix executable and a static library. Set there names here
EXE   = cuda_take
LIBOUT = libcuda_take.a
######################################





######################################
#Here we specify what source files are needed for the program/library, and we create virtual paths so that we don't have to refer to the source directory all the time
SOURCES = fft.cpp main.cpp dark_subtraction_filter.cu take_object.cpp std_dev_filter_device_code.cu std_dev_filter.cpp chroma_translate_filter.cpp mean_filter.cpp xiocamera.cpp rtpcamera.cpp osutils.cpp safestringset.cpp
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
IDIR      = -Iinclude -I/opt/EDTpdv -I/usr/local/cuda/include 
#IDIR      = -Iinclude -I/opt/EDTpdv -I/usr/local/cuda/include -I/usr/include/x86_64-linux-gnu/qt5 -I/usr/include/x86_64-linux-gnu/qt5/QtCore

#SECOND NOTE: ONLY BUILD WITH O1! Somehow, someway, O2 optimizes out things that nvcc needs and O0 has linker redefinition errors. #JankCity <- This has been fixed, but still a good idea
CFLAGS     = -g -O3 -D HOST=\"`hostname`\" -D UNAME=\"`whoami`\" #This is added to the compilation of every file, enables gdb debugging symbols (-g) and limited optimization (-O1) 
# and defines some global constants that interact with the operating 
ifeq ($(HARDWARE),OPALKELLY)
CFLAGS	   += -I$(okFP_SDK)
endif
CONLYFLAGS = -std=c99
CONLYFLAGS += $(CFLAGS)

# CFLAGS    += `pkg-config --cflags --libs gstreamer-1.0`
# -pthread -I/usr/include/gstreamer-1.0 -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include -lgstreamer-1.0 -lgobject-2.0 -lglib-2.0
CFLAGS += -I/usr/include/gstreamer-1.0 -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include -lgstreamer-1.0 -lgobject-2.0 -lglib-2.0
#CFLAGS += -Xcompiler="-pthread"
CPPFLAGS = $(CFLAGS)
CPPFLAGS += -g -O3 -std=c++11 -fopenmp -Wall -Werror -Wno-error=cpp -Wno-unused-function -Wno-unused-variable -Wno-unused-result#NOTE, NVCC does not support C++11, therefore -std=c++11 cpp files must be split up from cu files
CPPFLAGS += -fPIC
CPPFLAGS += -lgsl -lgslcblas

#CPPFLAGS += -isystem /usr/include/x86_64-linux-gnu/qt5 -isystem /usr/include/x86_64-linux-gnu/qt5/QtCore -DQT_NO_VERSION_TAGGING
#CPPFLAGS += -DQT_CORE_LIB -lQt5Core -lQt5Widgets

# See this URL to determine the architecture flags for your GPU: 
# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/  

#This version of NVCCFLAGS will build the fatest binary possible if the version of NVCC >= 6
#NVCCFLAGS  = -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30, -gencode arch=compute_35,code=sm_35, -gencode arch=compute_50,code=sm_50 -G -lineinfo -Xcompiler -rdynamic --compiler-options '-Wall -Werror -Wno-unused-function'#This program is targeted at GT590's which support Nvidia's 2.0 CUDA arch. Therefore that's what we build for

#This only targets the GTX 590 & GTX 780 series cards
#NVCCFLAGS  = -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35 -G -lineinfo -Xcompiler -rdynamic --compiler-options '-Wall -Werror -Wno-unused-function'#This program is targeted at GT590's which support Nvidia's 2.0 CUDA arch. Therefore that's what we build for
#(-G -lineinfo) are both to enable cuda-gdb debugging, -Xcompiler specifies arguments to get passed directly to g++ (which NVCC is built on), the internet said to do -rdynamic

# NVIDIA GeForce RTX 1080: 
#NVCCFLAGS  = -gencode arch=compute_61,code=sm_61 -lineinfo -Xcompiler -rdynamic --compiler-options '-Wall -Werror -Wno-unused-function'

# NVIDIA GeForce RTX 2080: 
#NVCCFLAGS  = -gencode arch=compute_75,code=sm_75 -lineinfo -Xcompiler -rdynamic --compiler-options '-Wall -Werror -Wno-unused-function'

# NVIDIA GeForce RTX 3080: 
#NVCCFLAGS  = -gencode arch=compute_86,code=sm_86 -lineinfo -Xcompiler -rdynamic --compiler-options '-Wall -Werror -Wno-unused-function'

# NVIDIA "native" code generator, for cuda versions 11.6+
NVCCFLAGS  = -arch=native -lineinfo -Xcompiler -pthread -Xcompiler -rdynamic --compiler-options '-Wall -Werror -Wno-unused-function'


# Truly legacy GPUs which aren't supported anymore: 
#NVCCFLAGS  = -gencode arch=compute_30,code=sm_30, -gencode arch=compute_35,code=sm_35 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -lineinfo -Xcompiler -rdynamic --compiler-options '-Wall -Werror -Wno-unused-function'#This program is targeted at GT590's which support Nvidia's 2.0 CUDA arch. Therefore that's what we build for

NVCCFLAGS += $(CFLAGS)

LINKDIR 	= /opt/EDTpdv #where do the libraries we need to link in go?
LFLAGS      = -L$(LINKDIR) -lm -lpdv -lboost_thread -lboost_system -lz -lcuda -lcudart -lgsl -lgslcblas -lgomp -lpthread -ldl -lgstapp-1.0 -lgstbase-1.0 -lgstreamer-1.0 -lgobject-2.0 -lglib-2.0 #Libraries needed to build program, only libpdv.a is not already visible in the path, as a result that is put in linkdir
ifeq ($(HARDWARE),OPALKELLY)
LDFLAGS	   := -L$(okFP_SDK)
okFP_LIBS  := -lokFrontPanel
endif
AR_COMBINE_SCRIPT = combine_libs_script.ar #For building out output library we compine our stuff with libpdv, this script tells ar how to do that
#This switch enables concatenating libpdv.a and libcuda_take.a (and possibly libboost_thread.a)
CONCATENATE_LIBPDV = 1

######################################

all : $(EXE) $(LIBOUT)
#	@echo $(SOURCES)
#	@echo $(objects)
#	@echo $(OBJS)
ifeq ($(HARDWARE),OPALKELLY)
$(EXE) : $(objects)
	$(NVCC) $(NVCCFLAGS) $(okFP_LDFLAGS) $(LDFLAGS) -o $@ $(wildcard obj/*.o) $(LFLAGS) $(okFP_LIBS)
else
$(EXE) : $(objects)
	$(NVCC) $(NVCCFLAGS) -o $@ $(wildcard obj/*.o) $(LFLAGS)
endif
	
$(LIBOUT) : $(objects)
ifeq ($(CONCATENATE_LIBPDV), 1)
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
ifeq ($(HARDWARE),OPALKELLY)
$(OBJDIR)/%.o : %.cpp
	$(CCPP) $(CPPFLAGS) $(okFP_CXXFLAGS) $(CXXFLAGS) $(IDIR) -c -o $@ $<
else
$(OBJDIR)/%.o : %.cpp
	$(CCPP) $(CPPFLAGS) $(IDIR) -c -o $@ $<
endif
#What to do to build cu files -> o files
$(OBJDIR)/%.o : %.cu 
	$(NVCC) $(NVCCFLAGS) $(IDIR) -c -o $@ $<
clean:
	rm -rf $(OBJDIR) $(EXE) $(LIBOUT) thin_$(LIBOUT)
