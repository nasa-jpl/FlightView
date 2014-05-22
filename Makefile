CC = nvcc

SOURCEDIR = src

EXE   = cuda_take

SOURCES  = $(SOURCEDIR)/cuda_take.c $(SOURCEDIR)/constant_filter.cu

IDIR      = -Iinclude -IEDT_include

OBJS        = $(SOURCES:.c=.o)

CFLAGS     = -O3

NVCCFLAGS  = -arch=sm_20

LINKDIR 	= lib
LFLAGS      = -L$(LINKDIR) -lm -lpdv


all : cuda_take

$(EXE) : $(OBJS) $(SOURCEDIR)/cuda_take.o
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

$(SOURCEDIR)/%.o : $(SOURCEDIR)/%.c
	$(CC) $(NVCCFLAGS) $(IDIR) -c -o $@ $<

$(SOURCEDIR)/%.o : $(SOURCEDIR)/%.cu $(H_FILES)
	$(CC) $(NVCCFLAGS) $(IDIR) -c -o $@ $<

clean:
	rm -f $(OBJS) $(EXE)