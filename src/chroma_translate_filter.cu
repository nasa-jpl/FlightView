#include <stdint.h>
#include <stdlib.h>
#include "chroma_translate_filter.cuh"
#include "cuda_utils.cuh"
#include <iostream>


//MAKING ASSUMPTIONS BECAUSE CHROMA!
#define WIDTH 1280
#define HEIGHT 480
#define BLOCK_SIDE 20
#define PIC_SIZE WIDTH*HEIGHT*2
#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))

__global__ void chroma_filter_kernel(uint16_t * pic_d, uint16_t * pic_out_d)
{
	//int offset = blockIdx.x*blockDim.x +threadIdx.x; //This gives us how far we are into the u_char
	unsigned short col = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned short r = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned short width_eigth = WIDTH/8;
	unsigned short c = col / 8;
	unsigned short i = col % 8;
	//	if(col < WIDTH && row < HEIGHT) //Because we needed an interger grid size, we will have a few threads that don't correspond to a location in the image.
	pic_out_d[c+width_eigth*i + r*WIDTH] = pic_d[c*8+i + r*WIDTH];

}
uint16_t * chroma_translate_filter::apply_chroma_translate_filter(uint16_t * picture_in)
{

	HANDLE_ERROR(cudaSetDevice(CTF_DEVICE_NUM));
	memcpy(pic_in_host,picture_in, PIC_SIZE); //If we stage ourselves it allows for cuda kernel concurrency
	HANDLE_ERROR(cudaMemcpyAsync(picture_device, pic_in_host, PIC_SIZE, cudaMemcpyHostToDevice, chroma_translate_stream));
	//HANDLE_ERROR(cudaMemcpy(picture_device, picture_in, PIC_SIZE, cudaMemcpyHostToDevice));


	dim3 blockDims(BLOCK_SIDE,BLOCK_SIDE,1);
	dim3 gridDims(WIDTH/BLOCK_SIDE, HEIGHT/BLOCK_SIDE,1);

	chroma_filter_kernel<<<gridDims,blockDims,0,chroma_translate_stream>>>(picture_device, pic_out_d);
	HANDLE_ERROR(cudaMemcpyAsync(picture_out,pic_out_d,PIC_SIZE,cudaMemcpyDeviceToHost, chroma_translate_stream));
	HANDLE_ERROR(cudaStreamSynchronize(chroma_translate_stream)); //blocks until done
	//HANDLE_ERROR( cudaPeekAtLastError() );
	//Serial Algorithm
	/*
	unsigned short width_eigth = WIDTH/8;

	for(int r = 0; r < HEIGHT; r++)
	{

		for(int c = 0; c < width_eigth; c++)
		 {
			 for(int i = 0; i < 8; i++)
			 {
				 picture_out[c+width_eigth*i + r*WIDTH] = picture_in[c*8+i + r*WIDTH];
			 }

		 }
	}
	*/

	return picture_out;
}
chroma_translate_filter::chroma_translate_filter()
{
	HANDLE_ERROR(cudaSetDevice(CTF_DEVICE_NUM));


	picture_out = (uint16_t * )malloc(PIC_SIZE); //Create buffer for CPU memory output

	HANDLE_ERROR(cudaMalloc( (void **)&picture_device, PIC_SIZE));
	HANDLE_ERROR(cudaMalloc( (void **)&pic_out_d, PIC_SIZE));
	HANDLE_ERROR(cudaMallocHost((void **) &pic_in_host, PIC_SIZE));

	HANDLE_ERROR(cudaStreamCreate(&chroma_translate_stream));

	//std::cout << "done alloc" << std::endl;
}
chroma_translate_filter::~chroma_translate_filter()
{
	HANDLE_ERROR(cudaSetDevice(CTF_DEVICE_NUM));

	HANDLE_ERROR(cudaStreamDestroy(chroma_translate_stream));
	HANDLE_ERROR(cudaFree(picture_device));
	HANDLE_ERROR(cudaFree(pic_out_d));
	HANDLE_ERROR(cudaFreeHost(pic_in_host));
	free(picture_out);
}
