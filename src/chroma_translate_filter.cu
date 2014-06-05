#include <stdint.h>
#include "chroma_translate_filter.cuh"

//MAKING ASSUMPTIONS BECAUSE CHROMA!
#define WIDTH 1280
#define HEIGHT 480
#define BLOCK_SIDE 20
#define PIC_SIZE WIDTH*HEIGHT*2
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

	cudaMemcpy(picture_device, picture_in, PIC_SIZE, cudaMemcpyHostToDevice);

	//dim3 blockDims(block_length,block_length,1);
	//dim3 blockDims(512,1,1);
	//dim3 gridDims(ceil((float)width*height/blockDims.x),1,1);
	dim3 blockDims(BLOCK_SIDE,BLOCK_SIDE,1);
	dim3 gridDims(WIDTH/BLOCK_SIDE, HEIGHT/BLOCK_SIDE,1);

	chroma_filter_kernel<<<gridDims,blockDims>>>(picture_device, pic_out_d);
	cudaMemcpy(picture_out,pic_out_d,PIC_SIZE,cudaMemcpyDeviceToHost);

	//Serial Algornthim
	/*
	unsigned short width_eigth = WIDTH/8;

	for(int r = 0; r < HEIGHT; r++)
	{

		for(int c = 0; c < WIDTH/8; c++)
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

	picture_out = (uint16_t * )malloc(PIC_SIZE); //Create buffer for CPU memory output
	cudaMalloc( (void **)&picture_device, PIC_SIZE);
	cudaMalloc( (void **)&pic_out_d, PIC_SIZE);
}
chroma_translate_filter::~chroma_translate_filter()
{
	cudaFree(picture_device);
	free(picture_out);
}
