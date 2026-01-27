######################################
#Set which command line tools the makefile should use in this block
CC = gcc
CCPP = g++
NVCC = nvcc
AR = ar
LIBTOOL = libtool
SOURCEDIR = src
HARDWARE = EDT
USE_CAMERALINK ?= 0

# Detect OS first
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	OS = MACOS
	# Detect Homebrew location
	ifeq ($(shell test -d /opt/homebrew && echo yes),yes)
		HOMEBREW_PREFIX = /opt/homebrew
	else
		HOMEBREW_PREFIX = /usr/local
	endif
	# Use full path to pkg-config
	PKG_CONFIG = $(HOMEBREW_PREFIX)/bin/pkg-config
	# Default to no CUDA on macOS (can be overridden with USE_CUDA=1)
	USE_CUDA ?= 0
	# Metal GPU acceleration (macOS only, can be overridden with USE_METAL=0)
	USE_METAL ?= 1
else
	OS = LINUX
	# Default to CUDA on Linux (can be overridden with USE_CUDA=0)
	USE_CUDA ?= 1
	USE_METAL = 0
endif
######################################





######################################
#This makefile will produce a unix executable and a static library. Set there names here
EXE   = backend
LIBOUT = lib_backend.a
######################################





######################################
#Here we specify what source files are needed for the program/library, and we create virtual paths so that we don't have to refer to the source directory all the time
ifeq ($(USE_CUDA), 1)
SOURCES = fft.cpp dark_subtraction_filter.cu white_ref_filter.cu acquire.cpp std_dev_filter_device_code.cu std_dev_filter.cpp chroma_translate_filter.cpp mean_filter.cpp xiocamera.cpp rtpcamera.cpp rtpnextgen.cpp osutils.cpp safestringset.cpp
else ifeq ($(USE_METAL), 1)
SOURCES = fft.cpp dark_subtraction_filter.cpp white_ref_filter.cpp acquire.cpp std_dev_filter.cpp std_dev_filter_metal.mm chroma_translate_filter.cpp mean_filter.cpp xiocamera.cpp rtpcamera.cpp rtpnextgen.cpp osutils.cpp safestringset.cpp
else
SOURCES = fft.cpp dark_subtraction_filter.cpp white_ref_filter.cpp acquire.cpp std_dev_filter.cpp chroma_translate_filter.cpp mean_filter.cpp xiocamera.cpp rtpcamera.cpp rtpnextgen.cpp osutils.cpp safestringset.cpp
endif
#SOURCES  = $(SOURCEDIR)/cuda_take.c $(SOURCEDIR)/constant_filter.cu


vpath %.c $(SOURCEDIR)
vpath %.cu $(SOURCEDIR)
vpath %.cpp $(SOURCEDIR)
vpath %.mm $(SOURCEDIR)

#SOURCES = $(wildcard src/*.c)
#SOURCES = $(wildcard src/*.cpp)
#SOURCES = $(wildcard src/*.cu)
######################################








######################################
#Here we specify the names of all the intermediate files (.o, object files, we will need for the final lib and executable)
objects = $(patsubst %.c,obj/%.o,$(SOURCES)) 
objects += $(patsubst %.cpp,obj/%.o,$(SOURCES)) 
objects += $(patsubst %.cu,obj/%.o,$(SOURCES))
objects += $(patsubst %.mm,obj/%.o,$(SOURCES)) 


OBJDIR = obj

######################################







######################################
#Here we set the directories containing necessary header files (.h,.cuh,.hpp); ones specific to this program are kept in include/, ones needed for cameralink stuff in EDT_include/
ifeq ($(USE_CAMERALINK), 1)
	CFLAGS   += -DCAMERALINK
	CXXFLAGS += -DCAMERALINK
	IDIR      = -Iinclude -I/opt/EDTpdv -I/usr/local/cuda/include 
else
	IDIR      = -Iinclude -I/usr/local/cuda/include 
endif

#SECOND NOTE: ONLY BUILD WITH O1! Somehow, someway, O2 optimizes out things that nvcc needs and O0 has linker redefinition errors. #JankCity <- This has been fixed, but still a good idea
CFLAGS     = -g -O3 -D HOST=\"`hostname`\" -D UNAME=\"`whoami`\" #This is added to the compilation of every file, enables gdb debugging symbols (-g) and limited optimization (-O1) 
# and defines some global constants that interact with the operating 
ifeq ($(HARDWARE),OPALKELLY)
CFLAGS	   += -I$(okFP_SDK)
endif
CONLYFLAGS = -std=c99
CONLYFLAGS += $(CFLAGS)

# CFLAGS    += `pkg-config --cflags --libs gstreamer-1.0`
# GStreamer configuration (INCLUDES ONLY - libraries go in LFLAGS)
ifeq ($(OS),MACOS)
	CFLAGS += $(shell $(PKG_CONFIG) --cflags gstreamer-1.0 glib-2.0 gobject-2.0)
	# GSL doesn't provide pkg-config on macOS, add manually
	CFLAGS += -I$(HOMEBREW_PREFIX)/include
else
	CFLAGS += -I/usr/include/gstreamer-1.0 -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include
endif

#CFLAGS += -Xcompiler="-pthread"
CPPFLAGS = $(CFLAGS)

ifeq ($(OS),MACOS)
	# macOS: use -Xpreprocessor for OpenMP with clang
	CPPFLAGS += -g -O3 -std=c++11 -Xpreprocessor -fopenmp -Wall -Werror -Wno-error=cpp -Wno-unused-function -Wno-unused-variable -Wno-unused-result
	CPPFLAGS += -fPIC
	CPPFLAGS += -I$(HOMEBREW_PREFIX)/include -I$(HOMEBREW_PREFIX)/opt/libomp/include
	# Metal GPU support (macOS only)
	ifeq ($(USE_METAL), 1)
		CPPFLAGS += -DUSE_METAL
		OBJCPPFLAGS = $(CPPFLAGS)
		METAL_FRAMEWORKS = -framework Metal -framework Foundation
	endif
else
	# Linux
	CPPFLAGS += -g -O3 -march=native -mtune=native -std=c++11 -fopenmp -Wall -Werror -Wno-error=cpp -Wno-unused-function -Wno-unused-variable -Wno-unused-result -Wno-mismatched-new-delete
	CPPFLAGS += -fPIC
endif

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

# Platform and feature-specific linking
ifeq ($(OS),MACOS)
	# macOS linking
	BOOST_MT_SUFFIX =
	BOOST_SYSTEM_LIB =
	GSTREAMER_LIBS = $(shell $(PKG_CONFIG) --libs gstreamer-1.0 gstreamer-app-1.0 glib-2.0 gobject-2.0)
	PLATFORM_LIBS = -L$(HOMEBREW_PREFIX)/lib -lm -lz -lpthread
	OPENMP_LIB = -L$(HOMEBREW_PREFIX)/opt/libomp/lib -lomp
	# macOS doesn't have -lrt or -ldl as separate libraries
	# Modern Boost (1.69+) made boost_system header-only, so don't link it
else
	# Linux linking
	BOOST_MT_SUFFIX =
	BOOST_SYSTEM_LIB = -lboost_system$(BOOST_MT_SUFFIX)
	GSTREAMER_LIBS = -lgstapp-1.0 -lgstbase-1.0 -lgstreamer-1.0 -lgobject-2.0 -lglib-2.0
	PLATFORM_LIBS = -lm -lz -lpthread -ldl -lrt
	OPENMP_LIB = -lgomp
	# Keep boost_system for backward compatibility with older Boost versions
endif

ifeq ($(USE_CAMERALINK), 1)
	LINKDIR 	= /opt/EDTpdv
	ifeq ($(USE_CUDA), 1)
		LFLAGS = -L$(LINKDIR) -lpdv -lboost_thread$(BOOST_MT_SUFFIX) $(BOOST_SYSTEM_LIB) -lcuda -lcudart -lgsl -lgslcblas $(OPENMP_LIB) $(PLATFORM_LIBS) $(GSTREAMER_LIBS)
	else
		LFLAGS = -L$(LINKDIR) -lpdv -lboost_thread$(BOOST_MT_SUFFIX) $(BOOST_SYSTEM_LIB) -lgsl -lgslcblas $(OPENMP_LIB) $(PLATFORM_LIBS) $(GSTREAMER_LIBS)
		ifeq ($(USE_METAL), 1)
			LFLAGS += $(METAL_FRAMEWORKS)
		endif
	endif
	AR_COMBINE_SCRIPT = combine_libs_script_cl.ar
else
	ifeq ($(USE_CUDA), 1)
		LFLAGS = -lboost_thread$(BOOST_MT_SUFFIX) $(BOOST_SYSTEM_LIB) -lcuda -lcudart -lgsl -lgslcblas $(OPENMP_LIB) $(PLATFORM_LIBS) $(GSTREAMER_LIBS)
	else
		LFLAGS = -lboost_thread$(BOOST_MT_SUFFIX) $(BOOST_SYSTEM_LIB) -lgsl -lgslcblas $(OPENMP_LIB) $(PLATFORM_LIBS) $(GSTREAMER_LIBS)
		ifeq ($(USE_METAL), 1)
			LFLAGS += $(METAL_FRAMEWORKS)
		endif
	endif
	AR_COMBINE_SCRIPT = combine_libs_script.ar
endif

# BSD ar (macOS) doesn't support -M scripting mode, only GNU ar (Linux) does
ifeq ($(OS),MACOS)
	CONCATENATE_LIBS = 0
else
	CONCATENATE_LIBS = 1
endif
######################################

all : $(LIBOUT)
#	@echo $(SOURCES)
#	@echo $(objects)
#	@echo $(OBJS)
ifeq ($(HARDWARE),OPALKELLY)
$(EXE) : $(objects)
ifeq ($(USE_CUDA), 1)
	$(NVCC) $(NVCCFLAGS) $(okFP_LDFLAGS) $(LDFLAGS) -o $@ $(wildcard obj/*.o) $(LFLAGS) $(okFP_LIBS)
else
	$(CCPP) $(CPPFLAGS) $(okFP_LDFLAGS) $(LDFLAGS) -o $@ $(wildcard obj/*.o) $(LFLAGS) $(okFP_LIBS)
endif
else
$(EXE) : $(objects)
ifeq ($(USE_CUDA), 1)
	$(NVCC) $(NVCCFLAGS) -o $@ $(wildcard obj/*.o) $(LFLAGS)
else
	$(CCPP) $(CPPFLAGS) -o $@ $(wildcard obj/*.o) $(LFLAGS)
endif
endif
	
$(LIBOUT) : $(objects)
ifeq ($(CONCATENATE_LIBS), 1)
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
#What to do to build mm files (Objective-C++) -> o files (macOS Metal)
$(OBJDIR)/%.o : %.mm
	$(CCPP) $(OBJCPPFLAGS) $(IDIR) -c -o $@ $<
clean:
	rm -rf $(OBJDIR) $(EXE) $(LIBOUT) thin_$(LIBOUT)
	rm -f *.metallib *.air
